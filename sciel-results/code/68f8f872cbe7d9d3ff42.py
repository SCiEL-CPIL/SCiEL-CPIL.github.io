import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 4]

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    if len(U) == 0:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                # Check for integer or half-integer
                # val * 2 should be close to an integer
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return _verify_full(U, V, W)

def _verify_full(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Build tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    # Compute sum_c U[:,c] * V[:,c] * W[:,c]
    # This is a 3-way sum: result[u,v,w] = sum_c U[u,c]*V[v,c]*W[w,c]
    result = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for c in range(rank):
        # U, V, W are lists of lists. Extract column c
        uc = np.array([row[c] for row in U], dtype=np.float64)
        vc = np.array([row[c] for row in V], dtype=np.float64)
        wc = np.array([row[c] for row in W], dtype=np.float64)
        result += np.einsum('u,v,w->uvw', uc, vc, wc)
    
    if np.max(np.abs(result - T)) > 1e-9:
        return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return U, V, W, rank

def _block_partition_search():
    """
    Partition the 3x4 matrix A and 4x4 matrix B into overlapping 2x2 blocks.
    A is 3x4: split rows into [0:2] and [1:3], cols into [0:2] and [2:4]
    B is 4x4: split rows into [0:2] and [2:4], cols into [0:2] and [2:4]
    
    The product C = A*B is 3x4.
    We use 2x2 block structure where each block multiplication of 2x2 by 2x2
    can be done with 7 multiplications (Strassen) instead of 8.
    
    Block layout:
    A = [[A11, A12],     B = [[B11, B12],
         [A21, A22]]          [B21, B22]]
    
    where A11 is 2x2 (rows 0-1, cols 0-1)
          A12 is 2x2 (rows 0-1, cols 2-3)
          A21 is 1x2 (row 2, cols 0-1)  -- note: A is 3x4, so bottom row has 2 cols
          A22 is 1x2 (row 2, cols 2-3)
    
    B11 is 2x2 (rows 0-1, cols 0-1)
    B12 is 2x2 (rows 0-1, cols 2-3)
    B21 is 2x2 (rows 2-3, cols 0-1)
    B22 is 2x2 (rows 2-3, cols 2-3)
    
    C = [[C11, C12],
         [C21, C22]]
    
    C11 = A11*B11 + A12*B21  (2x2 result)
    C12 = A11*B12 + A12*B22  (2x2 result)
    C21 = A21*B11 + A22*B21  (1x2 result)
    C22 = A21*B12 + A22*B22  (1x2 result)
    
    For the 2x2 block multiplications, we can use Strassen (7 mults each).
    For the 1x2 by 2x2 multiplications, we need 2 mults each (each element of 1x2
    times each element of 2x2, but actually it's a dot product per column).
    
    Actually, let me reconsider. The A21 and A22 are 1x2 vectors, and B21, B22 are 2x2.
    A21 * B11 means (1x2) * (2x2) = 1x2. This requires 2 multiplications per column
    of the result, so 2*2 = 4 mults total for A21*B11.
    
    Let me count more carefully:
    - A11*B11: 2x2 * 2x2 = 2x2, needs 8 schoolbook or 7 Strassen
    - A12*B21: 2x2 * 2x2 = 2x2, needs 8 or 7
    - A11*B12: 2x2 * 2x2 = 2x2, needs 8 or 7
    - A12*B22: 2x2 * 2x2 = 2x2, needs 8 or 7
    - A21*B11: 1x2 * 2x2 = 1x2, needs 4 mults (2 per output element)
    - A22*B21: 1x2 * 2x2 = 1x2, needs 4 mults
    - A21*B12: 1x2 * 2x2 = 1x2, needs 4 mults
    - A22*B22: 1x2 * 2x2 = 1x2, needs 4 mults
    
    Total schoolbook: 4*8 + 4*4 = 32 + 16 = 48 (matches n*m*p = 3*4*4 = 48)
    
    With Strassen for the four 2x2*2x2 products: 4*7 + 4*4 = 28 + 16 = 44
    
    So we can potentially get 44 with this block structure!
    
    Let me implement this properly.
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # We'll build the decomposition using Strassen for 2x2 blocks
    # and direct multiplication for 1x2 by 2x2 blocks.
    
    # Strassen for 2x2:
    # Given P (2x2) and Q (2x2), compute R = P*Q
    # P = [[p00, p01],     Q = [[q00, q01],
    #      [p10, p11]]          [q10, q11]]
    #
    # M1 = (p00 + p11) * (q00 + q11)
    # M2 = (p10 + p11) * q00
    # M3 = p00 * (q01 - q11)
    # M4 = p11 * (q10 - q00)
    # M5 = (p00 + p01) * q11
    # M6 = (p10 - p00) * (q00 + q01)
    # M7 = (p01 - p11) * (q10 + q11)
    #
    # R00 = M1 + M4 - M5 + M7
    # R01 = M3 + M5
    # R10 = M2 + M4
    # R11 = M1 - M2 + M3 + M6
    
    # Each Strassen multiplication uses 7 scalar multiplications.
    # In our tensor decomposition, each "multiplication" corresponds to one rank-1 term.
    
    # For the 1x2 * 2x2 case:
    # v (1x2) = [v0, v1], M (2x2) = [[m00, m01], [m10, m11]]
    # result = [v0*m00 + v1*m10, v0*m01 + v1*m11]
    # This is 4 scalar multiplications, each contributing to one output.
    # So we need 4 rank-1 terms.
    
    # Let's enumerate all the rank-1 terms needed.
    
    terms = []  # Each term is (u_indices, v_indices, w_indices) where each is a dict mapping position->coefficient
    
    # Helper: add a Strassen-based 2x2*2x2 product
    # a_block: 2x2 submatrix of A, specified by (row_start, col_start)
    # b_block: 2x2 submatrix of B, specified by (row_start, col_start)
    # The result contributes to C at (row_start, col_start) with a 2x2 block
    # But we need to handle the fact that A is 3x4, so some "2x2" blocks are actually 1x2
    
    def add_2x2_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """Add rank-1 terms for Strassen multiplication of A[a_rows, a_cols] * B[b_rows, b_cols] -> C[c_rows, c_cols]"""
        # a_rows, a_cols, b_rows, b_cols, c_rows, c_cols are lists of indices
        # For 2x2: len=2 for each
        # For 1x2: a_rows has len 1, a_cols has len 2, etc.
        
        # We handle the general case by decomposing into scalar products
        # Actually, let me just do direct multiplication for simplicity first,
        # then optimize with Strassen where applicable.
        pass
    
    # Let me take a different approach: directly enumerate all the scalar multiplications
    # needed, then try to combine them using Strassen identities.
    
    # Actually, the cleanest approach: build the full tensor decomposition by
    # enumerating all (i,j,k) triples and then apply Strassen to reduce.
    
    # For now, let me implement the block structure with Strassen for the 2x2*2x2 parts.
    
    # The four 2x2*2x2 products:
    # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # 2. A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # 3. A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # 4. A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    
    # The four 1x2*2x2 products:
    # 5. A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
    # 6. A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
    # 7. A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4]
    # 8. A[2:3, 2:4] * B[2:4, 2:4] -> C[2:3, 2:4]
    
    # For products 1-4, use Strassen (7 mults each) = 28 mults
    # For products 5-8, use direct (4 mults each) = 16 mults
    # Total: 44
    
    U = [[0] * 44 for _ in range(n * m)]
    V = [[0] * 44 for _ in range(m * p)]
    W = [[0] * 44 for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_2x2(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        """Add 7 rank-1 terms for Strassen multiplication of 2x2 blocks.
        A block: rows [a_r0, a_r0+1], cols [a_c0, a_c0+1]
        B block: rows [b_r0, b_r0+1], cols [b_c0, b_c0+1]
        C block: rows [c_r0, c_r0+1], cols [c_c0, c_c0+1]
        """
        nonlocal col
        # P indices in A: (a_r0, a_c0)=p00, (a_r0, a_c0+1)=p01, (a_r0+1, a_c0)=p10, (a_r0+1, a_c0+1)=p11
        # Q indices in B: (b_r0, b_c0)=q00, (b_r0, b_c0+1)=q01, (b_r0+1, b_c0)=q10, (b_r0+1, b_c0+1)=q11
        # R indices in C: (c_r0, c_c0)=r00, (c_r0, c_c0+1)=r01, (c_r0+1, c_c0)=r10, (c_r0+1, c_c0+1)=r11
        
        def A_idx(r, c):
            return r * m + c
        
        def B_idx(r, c):
            return r * p + c
        
        def C_idx(r, c):
            return r * n + c
        
        # M1 = (p00 + p11) * (q00 + q11)
        # U contribution: A[a_r0, a_c0] and A[a_r0+1, a_c0+1] get coefficient 1
        # V contribution: B[b_r0, b_c0] and B[b_r0+1, b_c0+1] get coefficient 1
        # W contribution: C[c_r0, c_c0] += 1, C[c_r0, c_c0+1] += 0, C[c_r0+1, c_c0] += 0, C[c_r0+1, c_c0+1] += 1
        # Wait, R00 = M1 + M4 - M5 + M7, so M1 contributes +1 to R00 and +1 to R11
        
        # Let me be more careful. In Strassen:
        # M1 = (P00+P11)(Q00+Q11)
        # M2 = (P10+P11)Q00
        # M3 = P00(Q01-Q11)
        # M4 = P11(Q10-Q00)
        # M5 = (P00+P01)Q11
        # M6 = (P10-P00)(Q00+Q01)
        # M7 = (P01-P11)(Q10+Q11)
        #
        # R00 = M1 + M4 - M5 + M7
        # R01 = M3 + M5
        # R10 = M2 + M4
        # R11 = M1 - M2 + M3 + M6
        
        # Each M is a product of a linear combination of A entries and a linear combination of B entries.
        # In the tensor decomposition, each M corresponds to one rank-1 term where:
        # U has coefficients for the A entries involved
        # V has coefficients for the B entries involved
        # W has coefficients for the C entries that this M contributes to
        
        # M1: A coeffs: A[a_r0,a_c0]=1, A[a_r0+1,a_c0+1]=1; B coeffs: B[b_r0,b_c0]=1, B[b_r0+1,b_c0+1]=1
        #     C coeffs: C[c_r0,c_c0]=1, C[c_r0+1,c_c0+1]=1
        U[A_idx(a_r0, a_c0)][col] += 1
        U[A_idx(a_r0+1, a_c0+1)][col] += 1
        V[B_idx(b_r0, b_c0)][col] += 1
        V[B_idx(b_r0+1, b_c0+1)][col] += 1
        W[C_idx(c_r0, c_c0)][col] += 1
        W[C_idx(c_r0+1, c_c0+1)][col] += 1
        col += 1
        
        # M2: A coeffs: A[a_r0+1,a_c0]=1, A[a_r0+1,a_c0+1]=1; B coeffs: B[b_r0,b_c0]=1
        #     C coeffs: C[c_r0+1,c_c0]=1, C[c_r0+1,c_c0+1]=-1
        U[A_idx(a_r0+1, a_c0)][col] += 1
        U[A_idx(a_r0+1, a_c0+1)][col] += 1
        V[B_idx(b_r0, b_c0)][col] += 1
        W[C_idx(c_r0+1, c_c0)][col] += 1
        W[C_idx(c_r0+1, c_c0+1)][col] -= 1
        col += 1
        
        # M3: A coeffs: A[a_r0,a_c0]=1; B coeffs: B[b_r0,b_c0+1]=1, B[b_r0+1,b_c0+1]=-1
        #     C coeffs: C[c_r0,c_c0+1]=1, C[c_r0+1,c_c0+1]=1
        U[A_idx(a_r0, a_c0)][col] += 1
        V[B_idx(b_r0, b_c0+1)][col] += 1
        V[B_idx(b_r0+1, b_c0+1)][col] -= 1
        W[C_idx(c_r0, c_c0+1)][col] += 1
        W[C_idx(c_r0+1, c_c0+1)][col] += 1
        col += 1
        
        # M4: A coeffs: A[a_r0+1,a_c0+1]=1; B coeffs: B[b_r0+1,b_c0]=1, B[b_r0,b_c0]=-1
        #     C coeffs: C[c_r0,c_c0]=1, C[c_r0+1,c_c0]=1
        U[A_idx(a_r0+1, a_c0+1)][col] += 1
        V[B_idx(b_r0+1, b_c0)][col] += 1
        V[B_idx(b_r0, b_c0)][col] -= 1
        W[C_idx(c_r0, c_c0)][col] += 1
        W[C_idx(c_r0+1, c_c0)][col] += 1
        col += 1
        
        # M5: A coeffs: A[a_r0,a_c0]=1, A[a_r0,a_c0+1]=1; B coeffs: B[b_r0+1,b_c0+1]=1
        #     C coeffs: C[c_r0,c_c0]=-1, C[c_r0,c_c0+1]=1
        U[A_idx(a_r0, a_c0)][col] += 1
        U[A_idx(a_r0, a_c0+1)][col] += 1
        V[B_idx(b_r0+1, b_c0+1)][col] += 1
        W[C_idx(c_r0, c_c0)][col] -= 1
        W[C_idx(c_r0, c_c0+1)][col] += 1
        col += 1
        
        # M6: A coeffs: A[a_r0+1,a_c0]=1, A[a_r0,a_c0]=-1; B coeffs: B[b_r0,b_c0]=1, B[b_r0,b_c0+1]=1
        #     C coeffs: C[c_r0+1,c_c0+1]=1
        U[A_idx(a_r0+1, a_c0)][col] += 1
        U[A_idx(a_r0, a_c0)][col] -= 1
        V[B_idx(b_r0, b_c0)][col] += 1
        V[B_idx(b_r0, b_c0+1)][col] += 1
        W[C_idx(c_r0+1, c_c0+1)][col] += 1
        col += 1
        
        # M7: A coeffs: A[a_r0,a_c0+1]=1, A[a_r0+1,a_c0+1]=-1; B coeffs: B[b_r0+1,b_c0]=1, B[b_r0+1,b_c0+1]=1
        #     C coeffs: C[c_r0,c_c0]=1
        U[A_idx(a_r0, a_c0+1)][col] += 1
        U[A_idx(a_r0+1, a_c0+1)][col] -= 1
        V[B_idx(b_r0+1, b_c0)][col] += 1
        V[B_idx(b_r0+1, b_c0+1)][col] += 1
        W[C_idx(c_r0, c_c0)][col] += 1
        col += 1
    
    def add_direct_1x2(a_r, a_c0, b_r0, b_c0, c_r, c_c0):
        """Add 4 rank-1 terms for direct multiplication of 1x2 * 2x2 -> 1x2.
        A entry: A[a_r, a_c0] and A[a_r, a_c0+1]
        B block: rows [b_r0, b_r0+1], cols [b_c0, b_c0+1]
        C entries: C[c_r, c_c0] and C[c_r, c_c0+1]
        """
        nonlocal col
        # C[c_r, c_c0] = A[a_r, a_c0]*B[b_r0, b_c0] + A[a_r, a_c0+1]*B[b_r0+1, b_c0]
        # C[c_r, c_c0+1] = A[a_r, a_c0]*B[b_r0, b_c0+1] + A[a_r, a_c0+1]*B[b_r0+1, b_c0+1]
        
        # Term 1: A[a_r,a_c0] * B[b_r0,b_c0] -> C[c_r,c_c0]
        U[a_r * m + a_c0][col] += 1
        V[b_r0 * p + b_c0][col] += 1
        W[c_r * n + c_c0][col] += 1
        col += 1
        
        # Term 2: A[a_r,a_c0+1] * B[b_r0+1,b_c0] -> C[c_r,c_c0]
        U[a_r * m + (a_c0+1)][col] += 1
        V[(b_r0+1) * p + b_c0][col] += 1
        W[c_r * n + c_c0][col] += 1
        col += 1
        
        # Term 3: A[a_r,a_c0] * B[b_r0,b_c0+1] -> C[c_r,c_c0+1]
        U[a_r * m + a_c0][col] += 1
        V[b_r0 * p + (b_c0+1)][col] += 1
        W[c_r * n + (c_c0+1)][col] += 1
        col += 1
        
        # Term 4: A[a_r,a_c0+1] * B[b_r0+1,b_c0+1] -> C[c_r,c_c0+1]
        U[a_r * m + (a_c0+1)][col] += 1
        V[(b_r0+1) * p + (b_c0+1)][col] += 1
        W[c_r * n + (c_c0+1)][col] += 1
        col += 1
    
    # Product 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_2x2(0, 0, 0, 0, 0, 0)
    
    # Product 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_2x2(0, 2, 2, 0, 0, 0)
    
    # Product 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_2x2(0, 0, 0, 2, 0, 2)
    
    # Product 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_2x2(0, 2, 2, 2, 0, 2)
    
    # Product 5: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
    add_direct_1x2(2, 0, 0, 0, 2, 0)
    
    # Product 6: A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
    add_direct_1x2(2, 2, 2, 0, 2, 0)
    
    # Product 7: A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4]
    add_direct_1x2(2, 0, 0, 2, 2, 2)
    
    # Product 8: A[2:3, 2:4] * B[2:4, 2:4] -> C[2:3, 2:4]
    add_direct_1x2(2, 2, 2, 2, 2, 2)
    
    return U, V, W, col

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s margin
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school, rank_school = _schoolbook()
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try block partition with Strassen
    if time.time() < deadline:
        U_block, V_block, W_block, rank_block = _block_partition_search()
        if _verify_full(U_block, V_block, W_block):
            best_U, best_V, best_W = U_block, V_block, W_block
            best_rank = rank_block
    
    # Try to find even lower rank with additional search
    if time.time() < deadline:
        # Greedy search: try to reduce rank by finding linear dependencies
        # For now, just return the best we have
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
