import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 4, 6
    rank = 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 terms
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def add_strassen_block(A_idx, B_idx, C_idx, start_col):
        a, b, c, d = A_idx
        e, f, g, h = B_idx
        c11, c12, c21, c22 = C_idx
        
        # M1: (a+d)(e+h)
        U[a][start_col] += 1
        U[d][start_col] += 1
        V[e][start_col] += 1
        V[h][start_col] += 1
        W[c11][start_col] += 1
        W[c22][start_col] += 1
        start_col += 1
        
        # M2: (c+d)e
        U[c][start_col] += 1
        U[d][start_col] += 1
        V[e][start_col] += 1
        W[c21][start_col] += 1
        W[c22][start_col] -= 1
        start_col += 1
        
        # M3: a(f-h)
        U[a][start_col] += 1
        V[f][start_col] += 1
        V[h][start_col] -= 1
        W[c12][start_col] += 1
        W[c22][start_col] += 1
        start_col += 1
        
        # M4: d(g-e)
        U[d][start_col] += 1
        V[g][start_col] += 1
        V[e][start_col] -= 1
        W[c11][start_col] += 1
        W[c21][start_col] += 1
        start_col += 1
        
        # M5: (a+b)h
        U[a][start_col] += 1
        U[b][start_col] += 1
        V[h][start_col] += 1
        W[c11][start_col] -= 1
        W[c12][start_col] += 1
        start_col += 1
        
        # M6: (c-a)(e+f)
        U[c][start_col] += 1
        U[a][start_col] -= 1
        V[e][start_col] += 1
        V[f][start_col] += 1
        W[c22][start_col] += 1
        start_col += 1
        
        # M7: (b-d)(g+h)
        U[b][start_col] += 1
        U[d][start_col] -= 1
        V[g][start_col] += 1
        V[h][start_col] += 1
        W[c11][start_col] += 1
        start_col += 1
        
        return start_col
    
    # We decompose the 2x4 * 4x6 multiplication.
    # A is 2x4, B is 4x6, C = AB is 2x6.
    # 
    # We can split B into two 2x6 halves (rows 0,1 and rows 2,3) and A into two 2x2 halves (cols 0,1 and cols 2,3).
    # C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
    # 
    # Each term is a 2x2 * 2x6 multiplication.
    # We can further split the 2x6 into three 2x2 blocks.
    # So we have 2 (A halves) * 3 (B column pairs) = 6 independent 2x2*2x2 Strassen multiplications.
    # Each uses 7 terms, so 6 * 7 = 42. But we can potentially share some terms.
    #
    # Actually, let's think more carefully. The standard approach for 2x4 * 4x6:
    # Split into 2x2 blocks. A has 1x2 blocks, B has 2x3 blocks, C has 1x3 blocks.
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # 
    # Alternative: Use the fact that we can combine the two A-halves in a smarter way.
    # Let A1 = A[:,0:2], A2 = A[:,2:4], B1 = B[0:2,:], B2 = B[2:4,:]
    # C = A1*B1 + A2*B2
    #
    # We can compute A1*B1 and A2*B2 independently, each requiring 3*7=21 operations (3 Strassen blocks of 7).
    # Total = 42.
    #
    # To get rank 40, we need to save 2 operations. One way is to use a 2x2x2 algorithm with rank 7 
    # but share some computations across the 6 blocks.
    #
    # Actually, a known result: the tensor rank of (2,4,6) matrix multiplication is at most 42 (trivial) 
    # and it's known to be at most 40 using specific techniques.
    #
    # Let me try a different decomposition. Instead of 6 independent 2x2 Strassen, 
    # we can group differently. 
    #
    # Consider: A is 2x4, B is 4x6.
    # Split A into A_left (2x2) and A_right (2x2).
    # Split B into B_top (2x6) and B_bottom (2x6).
    # C = A_left * B_top + A_right * B_bottom.
    #
    # For each 2x2 * 2x6 product, we can use 3 Strassen blocks = 7*3 = 21 ops.
    # Two such products = 42 ops.
    #
    # To reduce to 40, we need to find a way to share 2 operations.
    # One approach: use a 2x2 * 2x4 algorithm with rank 13 (instead of 14=2*7),
    # then we'd have 2 * 13 = 26... no that's for 2x2*2x4.
    #
    # Let me just use the straightforward 42-rank and see if I can find 2 savings.
    # Actually, the task says score is rank, and current best is 42 (score 1.1429 = 48/42).
    # Wait, the seed was probably 48 (naive). 48/42 ≈ 1.1429. So we need rank < 42 to improve.
    #
    # Known: rank of (2,4,6) matmul is at most 40. Let me implement a rank-40 scheme.
    #
    # One known approach: Use the fact that (2,4,6) can be embedded in a larger structure.
    # Alternatively, use a hybrid: one 2x2 Strassen (7 ops) for some part and 
    # a custom decomposition for the rest.
    #
    # Let me try: split into 2x2 blocks differently.
    # A: [[A11, A12], [A21, A22]] where each is 2x2.
    # B: [[B11, B12, B13], [B21, B22, B23]] where each is 2x2.
    # C: [[C11, C12, C13], [C21, C22, C23]] where each is 2x2.
    #
    # C_ij = A_i1 * B_1j + A_i2 * B_2j for j=1,2,3.
    #
    # For fixed i, C_i = A_i1 * B_1 + A_i2 * B_2 where B_1, B_2 are 2x6.
    #
    # I'll stick with 42 for now but verify correctness. If I can find a rank-40, great.
    #
    # Let me just output the 42-rank solution correctly.

    # Loop over 3 column pairs of B (and C)
    for k in range(3):
        # B_Tk: rows 0,1; cols 2k, 2k+1
        # A_L: cols 0,1
        # C cols 2k, 2k+1
        
        A_L_idx = [0, 1, 4, 5]  # a,b,c,d for A[0,0],A[0,1],A[1,0],A[1,1]
        B_Tk_idx = [0*6 + 2*k, 0*6 + 2*k + 1, 1*6 + 2*k, 1*6 + 2*k + 1]
        C_idx = [2*k*2 + 0, (2*k+1)*2 + 0, 2*k*2 + 1, (2*k+1)*2 + 1]
        
        col = add_strassen_block(A_L_idx, B_Tk_idx, C_idx, col)
        
        # B_Bk: rows 2,3; cols 2k, 2k+1
        # A_R: cols 2,3
        
        A_R_idx = [2, 3, 6, 7]  # a,b,c,d for A[0,2],A[0,3],A[1,2],A[1,3]
        B_Bk_idx = [2*6 + 2*k, 2*6 + 2*k + 1, 3*6 + 2*k, 3*6 + 2*k + 1]
        
        col = add_strassen_block(A_R_idx, B_Bk_idx, C_idx, col)
    
    # col should be 42
    return {"U": U, "V": V, "W": W}
