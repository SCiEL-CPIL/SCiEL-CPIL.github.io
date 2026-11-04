"""Program for 2x5x5 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time
import random

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[a,b,c] = 1 if j==k."""
    tensor = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for l in range(m):
                    if j == k:
                        a = i * m + j
                        b = k * m + l
                        c = l * n + i
                        tensor[a, b, c] = 1
    return tensor

def _verify_decomposition(U, V, W, tensor, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == tensor."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    recon = np.zeros_like(tensor, dtype=np.float64)
    for r in range(rank):
        recon += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    
    return np.max(np.abs(recon - tensor)) < tol

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Strassen's algorithm for 2x2x2 matrix multiplication.
    Returns U, V, W as lists of length 7, each a vector of length 4.
    
    For C = A * B where A, B are 2x2:
    A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    
    Strassen's 7 products:
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C[0,0] = M1 + M4 - M5 + M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M3 + M6
    
    In tensor decomposition form, for each product r:
    U[r] encodes which entries of A are used (with signs)
    V[r] encodes which entries of B are used (with signs)  
    W[r] encodes which entries of C are affected (with signs)
    
    Indexing: a=i*m+j, so for 2x2: a00=0, a01=1, a10=2, a11=3
    For B: b00=0, b01=1, b10=2, b11=3
    For C: c00=0, c01=1, c10=2, c11=3
    """
    # Each of the 7 products
    # U[r] = coefficients for A entries [a00, a01, a10, a11]
    # V[r] = coefficients for B entries [b00, b01, b10, b11]
    # W[r] = coefficients for C entries [c00, c01, c10, c11]
    
    U = []
    V = []
    W = []
    
    # M1 = (a+d)(e+h): A entries a00=1, a11=1; B entries b00=1, b11=1
    # C: c00=1, c01=-1, c10=-1, c11=1 (from M1 + M4 - M5 + M7 for c00, etc.)
    # Actually let me be more careful.
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # So:
    # M1 appears in: C[0,0] with +1, C[1,1] with +1
    # M2 appears in: C[1,0] with +1, C[1,1] with -1
    # M3 appears in: C[0,1] with +1, C[1,1] with +1
    # M4 appears in: C[0,0] with +1, C[1,0] with +1
    # M5 appears in: C[0,0] with -1, C[0,1] with +1
    # M6 appears in: C[1,1] with +1
    # M7 appears in: C[0,0] with +1
    
    # M1: A: a00=1, a11=1 => U=[1,0,0,1]
    #     B: b00=1, b11=1 => V=[1,0,0,1]
    #     C: c00=1, c11=1 => W=[1,0,0,1]
    U.append([1, 0, 0, 1])
    V.append([1, 0, 0, 1])
    W.append([1, 0, 0, 1])
    
    # M2: (c+d)e => A: a10=1, a11=1 => U=[0,0,1,1]
    #     B: b00=1 => V=[1,0,0,0]
    #     C: c10=1, c11=-1 => W=[0,0,1,-1]
    U.append([0, 0, 1, 1])
    V.append([1, 0, 0, 0])
    W.append([0, 0, 1, -1])
    
    # M3: a(f-h) => A: a00=1 => U=[1,0,0,0]
    #     B: b01=1, b11=-1 => V=[0,1,0,-1]
    #     C: c01=1, c11=1 => W=[0,1,0,1]
    U.append([1, 0, 0, 0])
    V.append([0, 1, 0, -1])
    W.append([0, 1, 0, 1])
    
    # M4: d(g-e) => A: a11=1 => U=[0,0,0,1]
    #     B: b10=1, b00=-1 => V=[-1,0,1,0]
    #     C: c00=1, c10=1 => W=[1,0,1,0]
    U.append([0, 0, 0, 1])
    V.append([-1, 0, 1, 0])
    W.append([1, 0, 1, 0])
    
    # M5: (a+b)h => A: a00=1, a01=1 => U=[1,1,0,0]
    #     B: b11=1 => V=[0,0,0,1]
    #     C: c00=-1, c01=1 => W=[-1,1,0,0]
    U.append([1, 1, 0, 0])
    V.append([0, 0, 0, 1])
    W.append([-1, 1, 0, 0])
    
    # M6: (c-a)(e+f) => A: a10=1, a00=-1 => U=[-1,0,1,0]
    #     B: b00=1, b01=1 => V=[1,1,0,0]
    #     C: c11=1 => W=[0,0,0,1]
    U.append([-1, 0, 1, 0])
    V.append([1, 1, 0, 0])
    W.append([0, 0, 0, 1])
    
    # M7: (b-d)(g+h) => A: a01=1, a11=-1 => U=[0,1,0,-1]
    #     B: b10=1, b11=1 => V=[0,0,1,1]
    #     C: c00=1 => W=[1,0,0,0]
    U.append([0, 1, 0, -1])
    V.append([0, 0, 1, 1])
    W.append([1, 0, 0, 0])
    
    return U, V, W

def _recursive_block_decomposition(n, m, p, deadline):
    """
    Decompose n x m * m x p = n x p using recursive block decomposition.
    
    Strategy: Split dimensions into smaller blocks, use Strassen for 2x2x2,
    and simple rank formulas for rectangular cases.
    
    Returns U, V, W as lists.
    """
    if n == 1 and m == 1 and p == 1:
        # Single scalar multiplication
        return [[1]], [[1]], [[1]]
    
    if n == 1:
        # 1 x m * m x p = 1 x p
        # This is a dot product for each output: C[j] = sum_i A[i] * B[j,i]
        # Rank = m * p (schoolbook) but can we do better?
        # Actually for 1xm * mxp, the tensor is sum_{i,j} A[i]*B[j,i]*C[j]
        # This is rank m*p in general, but let's use schoolbook
        # Wait, actually this is just: for each j, C[j] = dot(A, B[:,j])
        # The bilinear map has rank m*p in general
        return _schoolbook_decomposition(1, m, p)["U"], \
               _schoolbook_decomposition(1, m, p)["V"], \
               _schoolbook_decomposition(1, m, p)["W"]
    
    if p == 1:
        # n x m * m x 1 = n x 1
        # C[i] = dot(A[i,:], B)
        # Rank = n*m in general
        return _schoolbook_decomposition(n, m, 1)["U"], \
               _schoolbook_decomposition(n, m, 1)["V"], \
               _schoolbook_decomposition(n, m, 1)["W"]
    
    if n == 2 and m == 2 and p == 2:
        U, V, W = _strassen_2x2x2()
        return U, V, W
    
    # For larger dimensions, try splitting
    # Split n into n1 + n2, m into m1 + m2, p into p1 + p2
    # C = A1*B11 + A1*B12 + A2*B21 + A2*B22 (with appropriate sub-blocks)
    
    # Let's try splitting m first (the shared dimension)
    # A is n x m, B is m x p
    # Split m into m1 + m2:
    # A = [A1 | A2] where A1 is n x m1, A2 is n x m2
    # B = [B1; B2] where B1 is m1 x p, B2 is m2 x p
    # C = A1*B1 + A2*B2
    
    # This gives rank = rank(n x m1 * m1 x p) + rank(n x m2 * m2 x p)
    
    # Try different splits of m
    best_rank = n * m * p
    best_result = None
    
    for m1 in range(1, m):
        m2 = m - m1
        if time.time() > deadline:
            break
        
        # Get decompositions for the two parts
        U1, V1, W1 = _recursive_block_decomposition(n, m1, p, deadline)
        U2, V2, W2 = _recursive_block_decomposition(n, m2, p, deadline)
        
        r1 = len(U1[0]) if isinstance(U1, list) else U1.shape[1]
        r2 = len(U2[0]) if isinstance(U2, list) else U2.shape[1]
        
        total_rank = r1 + r2
        if total_rank < best_rank:
            # Combine: pad U1 to full m columns, U2 to full m columns
            U_full1 = [[0] * total_rank for _ in range(n * m)]
            U_full2 = [[0] * total_rank for _ in range(n * m)]
            V_full1 = [[0] * total_rank for _ in range(m * p)]
            V_full2 = [[0] * total_rank for _ in range(m * p)]
            W_full1 = [[0] * total_rank for _ in range(p * n)]
            W_full2 = [[0] * total_rank for _ in range(p * n)]
            
            # U1 operates on first m1 columns of A
            for i in range(n * m1):
                U_full1[i][:r1] = U1[i]
            # U2 operates on last m2 columns of A
            for i in range(n * m2):
                U_full2[n * m1 + i][:r2] = U2[i]
            
            # V1 operates on first m1 rows of B
            for j in range(m1 * p):
                V_full1[j][:r1] = V1[j]
            # V2 operates on last m2 rows of B
            for j in range(m2 * p):
                V_full2[m1 * p + j][:r2] = V2[j]
            
            # W is the same for both
            for k in range(p * n):
                W_full1[k][:r1] = W1[k]
                W_full2[k][r1:r1+r2] = W2[k]
            
            U_combined = [U_full1[i] + U_full2[i] for i in range(n * m)]
            V_combined = [V_full1[j] + V_full2[j] for j in range(m * p)]
            W_combined = [W_full1[k] + W_full2[k] for k in range(p * n)]
            
            best_rank = total_rank
            best_result = (U_combined, V_combined, W_combined)
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    sb = _schoolbook_decomposition(n, m, p)
    return sb["U"], sb["V"], sb["W"]

def _try_strassen_embedding(n, m, p, deadline):
    """
    Try to embed Strassen 2x2x2 into the 2x5x5 problem by zero-padding.
    
    For 2x5x5:
    A is 2x5, B is 5x5, C is 5x2.
    
    We can try to find a 2x2x2 subproblem and use Strassen for it,
    then handle the remaining terms with schoolbook.
    
    Actually, let's think about this differently.
    C = A * B where A is 2x5, B is 5x5.
    
    We can split A into A1 (2x2) and A2 (2x3), B into B1 (2x5) and B2 (3x5).
    C = A1*B1 + A2*B2
    
    For A1*B1: 2x2 * 2x5 = 2x5. We can further split B1 into B11 (2x2) and B12 (2x3).
    A1*B11 is 2x2*2x2 = 2x2 (use Strassen, rank 7)
    A1*B12 is 2x2*2x3 = 2x3 (schoolbook rank 12, or try to improve)
    
    For A2*B2: 2x3 * 3x5 = 2x5 (schoolbook rank 30)
    
    Total: 7 + 12 + 30 = 49. Not much improvement.
    
    Let's try a different approach: use the recursive decomposition.
    """
    U, V, W = _recursive_block_decomposition(n, m, p, deadline)
    
    # Convert to Fraction for exactness
    U_frac = [[Fraction(x).limit_denominator(2) for x in row] for row in U]
    V_frac = [[Fraction(x).limit_denominator(2) for x in row] for row in V]
    W_frac = [[Fraction(x).limit_denominator(2) for x in row] for row in W]
    
    rank = len(U_frac[0]) if isinstance(U_frac, list) else U_frac.shape[1]
    return rank, {"U": U_frac, "V": V_frac, "W": W_frac}

def construct():
    deadline = time.time() + 145  # Leave 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best_rank = n * m * p
    best_result = _schoolbook_decomposition(n, m, p)
    
    # Try recursive block decomposition
    if time.time() < deadline - 5:
        new_rank, new_result = _try_strassen_embedding(n, m, p, deadline - 5)
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = new_result
    
    # Verify the best result
    tensor = _build_tensor(n, m, p)
    if not _verify_decomposition(best_result["U"], best_result["V"], best_result["W"], tensor):
        # Fall back to schoolbook if verification fails
        best_result = _schoolbook_decomposition(n, m, p)
        best_rank = n * m * p
    
    return best_result

# EVOLVE-BLOCK-END
