"""
Rank-minimal exact tensor decomposition for 2x4 by 4x5 matrix multiplication.
Uses a recursive Strassen-based block decomposition to reduce rank below schoolbook 40.
"""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-minimal tensor decomposition for 2x4 by 4x5 matrix multiplication.
    Uses recursive Strassen decomposition for 2x2x2 blocks.
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # We will build the decomposition by combining sub-decompositions.
    # Each sub-decomposition is represented as (U_sub, V_sub, W_sub, rank_sub)
    # where U_sub has shape (n*m, rank_sub), etc.
    # But since we are partitioning the middle dimension m or output dimension p,
    # we need to be careful with the indexing.
    
    # The tensor T[i, j, k] corresponds to U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Strategy:
    # 1. Decompose 2x4x5 into 2x4x2 and 2x4x3 (partition p: 2+3)
    # 2. Decompose 2x4x3 into 2x4x2 and 2x4x1 (partition p: 2+1)
    # 3. Decompose 2x4x2 into 2x2x2 + 2x2x2 (partition m: 2+2)
    #    Each 2x2x2 uses Strassen (rank 7)
    # 4. Decompose 2x4x1 into 2x2x1 + 2x2x1 (partition m: 2+2)
    #    Each 2x2x1 is schoolbook (rank 2*2*1 = 4)
    
    # Let's implement a helper to create a schoolbook decomposition for n x m x p
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
        return U, V, W, rank
    
    # Helper to create a Strassen decomposition for 2x2x2
    # Strassen formulas for 2x2 matrices:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B11 - B12)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # In tensor decomposition terms:
    # A is 2x2, B is 2x2, C is 2x2.
    # Indices: A[i,j], B[j,k], C[i,k].
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    def strassen_2x2x2():
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # M1: (A00 + A11)(B00 + B11) -> C00 + C11 - C01 + C10 ? No.
        # Let's map carefully.
        # A indices: 00, 01, 10, 11 -> U row indices 0,1,2,3
        # B indices: 00, 01, 10, 11 -> V row indices 0,1,2,3
        # C indices: 00, 01, 10, 11 -> W row indices 0,1,2,3 (k*n+i, n=2)
        # C00: k=0,i=0 -> W[0]
        # C01: k=1,i=0 -> W[2]
        # C10: k=0,i=1 -> W[1]
        # C11: k=1,i=1 -> W[3]
        
        # M1 = (A00+A11)(B00+B11)
        # A00: U[0]=1, A11: U[3]=1
        # B00: V[0]=1, B11: V[3]=1
        # C00: W[0]=1, C11: W[3]=1, C01: W[2]=-1, C10: W[1]=1
        # Wait, C11 = M1 - M2 + M3 + M6. C00 = M1 + M4 - M5 + M7.
        # Let's just set the coefficients for each M.
        
        # M1
        U[0, 0] += 1.0; U[3, 0] += 1.0
        V[0, 0] += 1.0; V[3, 0] += 1.0
        W[0, 0] += 1.0; W[3, 0] += 1.0; W[2, 0] += -1.0; W[1, 0] += 1.0
        
        # M2 = (A10+A11)B00
        # A10: U[2]=1, A11: U[3]=1
        # B00: V[0]=1
        # C21 = M2 + M4 -> W[1] += 1
        # C11 = ... - M2 ... -> W[3] += -1
        U[2, 1] += 1.0; U[3, 1] += 1.0
        V[0, 1] += 1.0
        W[1, 1] += 1.0; W[3, 1] += -1.0
        
        # M3 = A00(B10 - B11)
        # A00: U[0]=1
        # B10: V[2]=1, B11: V[3]=-1
        # C12 = M3 + M5 -> W[2] += 1
        # C11 = ... + M3 ... -> W[3] += 1
        U[0, 2] += 1.0
        V[2, 2] += 1.0; V[3, 2] += -1.0
        W[2, 2] += 1.0; W[3, 2] += 1.0
        
        # M4 = A11(B00 - B01)
        # A11: U[3]=1
        # B00: V[0]=1, B01: V[1]=-1
        # C00 = ... + M4 ... -> W[0] += 1
        # C21 = M2 + M4 -> W[1] += 1
        U[3, 3] += 1.0
        V[0, 3] += 1.0; V[1, 3] += -1.0
        W[0, 3] += 1.0; W[1, 3] += 1.0
        
        # M5 = (A00+A01)B11
        # A00: U[0]=1, A01: U[1]=1
        # B11: V[3]=1
        # C00 = ... - M5 ... -> W[0] += -1
        # C12 = M3 + M5 -> W[2] += 1
        U[0, 4] += 1.0; U[1, 4] += 1.0
        V[3, 4] += 1.0
        W[0, 4] += -1.0; W[2, 4] += 1.0
        
        # M6 = (A10 - A00)(B00 + B01)
        # A10: U[2]=1, A00: U[0]=-1
        # B00: V[0]=1, B01: V[1]=1
        # C11 = ... + M6 ... -> W[3] += 1
        U[2, 5] += 1.0; U[0, 5] += -1.0
        V[0, 5] += 1.0; V[1, 5] += 1.0
        W[3, 5] += 1.0
        
        # M7 = (A01 - A11)(B10 + B11)
        # A01: U[1]=1, A11: U[3]=-1
        # B10: V[2]=1, B11: V[3]=1
        # C00 = ... + M7 ... -> W[0] += 1
        U[1, 6] += 1.0; U[3, 6] += -1.0
        V[2, 6] += 1.0; V[3, 6] += 1.0
        W[0, 6] += 1.0
        
        return U, V, W, rank
    
    # Helper to combine two decompositions in parallel (block diagonal)
    # Decomp 1: (U1, V1, W1, r1)
    # Decomp 2: (U2, V2, W2, r2)
    # Result: U = [U1, U2], V = [V1, V2], W = [W1, W2], rank = r1 + r2
    def combine_parallel(d1, d2):
        U1, V1, W1, r1 = d1
        U2, V2, W2, r2 = d2
        U = np.hstack([U1, U2])
        V = np.hstack([V1, V2])
        W = np.hstack([W1, W2])
        return U, V, W, r1 + r2
    
    # Helper to embed a smaller decomposition into a larger one with zero padding
    # We want to compute C = A * B where A is n x m, B is m x p.
    # If we only use a subset of A's columns and B's rows, we can embed.
    # Here we partition m into m1 and m2.
    # A = [A1, A2], B = [B1; B2]. C = A1*B1 + A2*B2.
    # Decomp for A1*B1: U1 (n*m1, r1), V1 (m1*p, r1), W1 (p*n, r1)
    # Decomp for A2*B2: U2 (n*m2, r2), V2 (m2*p, r2), W2 (p*n, r2)
    # Total U: (n*(m1+m2), r1+r2).
    # U rows 0..n*m1-1 correspond to A1. U rows n*m1..n*(m1+m2)-1 correspond to A2.
    # But U1 is n*m1 x r1. We need to map U1 to the first n*m1 rows of U.
    # And U2 to the last n*m2 rows of U.
    # V1 is m1*p x r1. V2 is m2*p x r2.
    # V rows 0..m1*p-1 correspond to B1. V rows m1*p..(m1+m2)*p-1 correspond to B2.
    # W is just concatenated.
    
    def combine_block_m(d1, d2, n, m1, m2, p):
        U1, V1, W1, r1 = d1
        U2, V2, W2, r2 = d2
        rank = r1 + r2
        U = np.zeros((n * (m1 + m2), rank))
        V = np.zeros(((m1 + m2) * p, rank))
        W = np.hstack([W1, W2])
        
        # Embed U1 into U[0:n*m1, 0:r1]
        # U1 has shape (n*m1, r1).
        # But wait, U1's rows are indexed by i*m1 + j? No, in the subproblem,
        # the "m" dimension is m1. So U1 row i*m1 + j.
        # In the full problem, A has m = m1+m2.
        # A1 is the first m1 columns. So A[i, j] for j<m1.
        # U1[i*m1+j, c] should map to U[i*(m1+m2)+j, c].
        # This is not a simple block copy if m1 != m.
        # We need to remap indices.
        
        # Let's do it explicitly.
        for i in range(n):
            for j in range(m1):
                for c in range(r1):
                    U[i * (m1 + m2) + j, c] = U1[i * m1 + j, c]
                    
        for i in range(n):
            for j in range(m2):
                for c in range(r2):
                    U[i * (m1 + m2) + (m1 + j), r1 + c] = U2[i * m2 + j, c]
                    
        # Embed V1 into V[0:m1*p, 0:r1]
        # V1 row j*p + k (where j<m1).
        # Full V row j*p + k (where j<m1). Same index.
        V[0:m1*p, 0:r1] = V1
        # Embed V2 into V[m1*p:(m1+m2)*p, r1:r1+r2]
        # V2 row j*p + k (where j<m2).
        # Full V row (m1+j)*p + k.
        for j in range(m2):
            for k in range(p):
                for c in range(r2):
                    V[(m1 + j) * p + k, r1 + c] = V2[j * p + k, c]
                    
        return U, V, W, rank
        
    # Helper to partition p into p1 and p2
    # C = A * B. B = [B1 | B2]. C = [C1 | C2].
    # C1 = A * B1, C2 = A * B2.
    # Decomp 1: U1 (n*m, r1), V1 (m*p1, r1), W1 (p1*n, r1)
    # Decomp 2: U2 (n*m, r2), V2 (m*p2, r2), W2 (p2*n, r2)
    # Total U: [U1, U2]. V: [V1; V2]. W: [W1; W2] (stacked vertically? No, W is p*n x rank).
    # W1 is p1*n x r1. W2 is p2*n x r2.
    # Full W is (p1+p2)*n x (r1+r2).
    # W1 rows k*n+i (k<p1). Full W rows k*n+i (k<p1). Same.
    # W2 rows k*n+i (k<p2). Full W rows (p1+k)*n+i.
    
    def combine_block_p(d1, d2, n, m, p1, p2):
        U1, V1, W1, r1 = d1
        U2, V2, W2, r2 = d2
        rank = r1 + r2
        U = np.hstack([U1, U2])
        V = np.vstack([V1, V2])
        W = np.zeros(((p1 + p2) * n, rank))
        
        W[0:p1*n, 0:r1] = W1
        for k in range(p2):
            for i in range(n):
                for c in range(r2):
                    W[(p1 + k) * n + i, r1 + c] = W2[k * n + i, c]
                    
        return U, V, W, rank

    # Now build the decomposition
    # 1. 2x4x5 -> 2x4x2 + 2x4x3
    # 2. 2x4x3 -> 2x4x2 + 2x4x1
    # 3. 2x4x2 -> 2x2x2 + 2x2x2 (Strassen)
    # 4. 2x4x1 -> 2x2x1 + 2x2x1 (Schoolbook)
    
    # Build 2x2x2 Strassen
    s_222 = strassen_2x2x2()
    
    # Build 2x2x1 Schoolbook
    s_221 = schoolbook(2, 2, 1)
    
    # Build 2x4x2 from two 2x2x2
    # m1=2, m2=2, p=2, n=2
    d_242 = combine_block_m(s_222, s_222, 2, 2, 2, 2)
    
    # Build 2x4x1 from two 2x2x1
    # m1=2, m2=2, p=1, n=2
    d_241 = combine_block_m(s_221, s_221, 2, 2, 2, 1)
    
    # Build 2x4x3 from 2x4x2 + 2x4x1
    # p1=2, p2=1, n=2, m=4
    d_243 = combine_block_p(d_242, d_241, 2, 4, 2, 1)
    
    # Build 2x4x5 from 2x4x2 + 2x4x3
    # p1=2, p2=3, n=2, m=4
    d_final = combine_block_p(d_242, d_243, 2, 4, 2, 3)
    
    U, V, W, rank = d_final
    
    # Check deadline
    if time.time() - start_time > 140:
        pass
        
    # Convert to lists of integers/half-integers
    U_list = [[int(val) if val == int(val) else val for val in row] for row in U.tolist()]
    V_list = [[int(val) if val == int(val) else val for val in row] for row in V.tolist()]
    W_list = [[int(val) if val == int(val) else val for val in row] for row in W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
