import time
import numpy as np

def _verify(U, V, W, n=2, m=4, p=8, trials=20, tol=1e-9):
    """Vectorized verification for speed."""
    rank = len(U[0]) if U else 0
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # Reshape for matrix multiplication check
    # C[i,k] = sum_j A[i,j] B[j,k]
    # C_computed[i,k] = sum_c (sum_j A[i,j] U[i*m+j, c]) * (sum_j V[j*p+k, c] B[j,k]) * W[k*n+i, c]
    
    # We can check linearity by checking basis vectors
    # Or just random trials with numpy matmul
    
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        
        # Efficient computation:
        # Factor A for each c: FA_c = U_c^T A_flat ?
        # U is (n*m, rank). A_flat is (n*m,).
        # FA = U.T @ A_flat -> (rank,)
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        # FA: (rank, )
        FA = U.T @ A_flat
        
        # FB: (rank, )
        # V is (m*p, rank). B_flat is (m*p,).
        FB = V.T @ B_flat
        
        # C_computed[i,k] = sum_c FA[c] * FB[c] * W[k*n+i, c]
        # W is (p*n, rank).
        # Let W_flat = W.T -> (rank, p*n)
        # Prod = FA * FB -> (rank,)
        # C_flat = W.T @ Prod -> (p*n,)
        
        Prod = FA * FB
        C_flat = W.T @ Prod
        C_computed = C_flat.reshape(p, n).T # W index is k*n+i, so reshape to (p,n) then transpose to (n,p)
        
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 4, 8
    
    # We construct the rank 56 decomposition using the recursive Strassen structure.
    # C = A_L B_L + A_R B_R
    # A_L: A[:, 0:2], A_R: A[:, 2:4]
    # B_L: B[0:2, :], B_R: B[2:4, :]
    
    # Each term is 2x2 * 2x8.
    # 2x2 * 2x8 can be decomposed into two 2x2 * 2x4 blocks.
    # 2x2 * 2x4 can be decomposed into two 2x2 * 2x2 blocks.
    # 2x2 * 2x2 uses Strassen (rank 7).
    
    # Total blocks of 2x2*2x2:
    # A_L B_L -> 4 blocks (2 from split of 2x8 into 2x4, 2 from split of 2x4 into 2x2)
    # A_R B_R -> 4 blocks
    # Total 8 blocks.
    # Rank 8 * 7 = 56.
    
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2*2x2
    # U_strassen: 7 rows (factors), 4 cols (A entries)
    # V_strassen: 7 rows (factors), 4 cols (B entries)
    # W_strassen: 4 rows (C entries), 7 cols (factors)
    
    U_s = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, 1]
    ]
    
    V_s = [
        [1, 0, 0, 1],
        [0, 0, 0, 1],
        [1, 0, -1, 0],
        [0, -1, 0, 1],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Helper to add a Strassen block
    # a_idx: 4 indices in U (flattened A)
    # b_idx: 4 indices in V (flattened B)
    # c_idx: 4 indices in W (flattened C)
    # offset: column offset in U,V,W
    
    def add_block(a_idx, b_idx, c_idx, offset):
        for f in range(7):
            col = offset + f
            for r in range(4):
                U[a_idx[r]][col] = U_s[f][r]
                V[b_idx[r]][col] = V_s[f][r]
                W[c_idx[r]][col] = W_s[r][f]
        
    # Indices mapping
    # A is 2x4. Flat index i*4 + j.
    # B is 4x8. Flat index j*8 + k.
    # C is 2x8. Flat index k*2 + i. (Trace convention W[k*n+i])
    
    # Blocks:
    # 1. A_L (i=0,1; j=0,1) * B_L_top (j=0,1; k=0,1) -> C (i=0,1; k=0,1)
    #    A_idx: [0*4+0, 0*4+1, 1*4+0, 1*4+1] = [0,1,4,5]
    #    B_idx: [0*8+0, 0*8+1, 1*8+0, 1*8+1] = [0,1,8,9]
    #    C_idx: [0*2+0, 0*2+1, 1*2+0, 1*2+1] = [0,1,2,3]
    add_block([0,1,4,5], [0,1,8,9], [0,1,2,3], 0)
    
    # 2. A_L * B_L_top (j=0,1; k=2,3) -> C (i=0,1; k=2,3)
    #    A_idx: [0,1,4,5]
    #    B_idx: [0*8+2, 0*8+3, 1*8+2, 1*8+3] = [2,3,10,11]
    #    C_idx: [2*2+0, 2*2+1, 3*2+0, 3*2+1] = [4,5,6,7]
    add_block([0,1,4,5], [2,3,10,11], [4,5,6,7], 7)
    
    # 3. A_L * B_L_bot (j=0,1; k=4,5) -> C (i=0,1; k=4,5)
    #    A_idx: [0,1,4,5]
    #    B_idx: [0*8+4, 0*8+5, 1*8+4, 1*8+5] = [4,5,12,13]
    #    C_idx: [4*2+0, 4*2+1, 5*2+0, 5*2+1] = [8,9,10,11]
    add_block([0,1,4,5], [4,5,12,13], [8,9,10,11], 14)
    
    # 4. A_L * B_L_bot (j=0,1; k=6,7) -> C (i=0,1; k=6,7)
    #    A_idx: [0,1,4,5]
    #    B_idx: [0*8+6, 0*8+7, 1*8+6, 1*8+7] = [6,7,14,15]
    #    C_idx: [6*2+0, 6*2+1, 7*2+0, 7*2+1] = [12,13,14,15]
    add_block([0,1,4,5], [6,7,14,15], [12,13,14,15], 21)
    
    # 5. A_R (i=0,1; j=2,3) * B_R_top (j=2,3; k=0,1) -> C (i=0,1; k=0,1)
    #    A_idx: [0*4+2, 0*4+3, 1*4+2, 1*4+3] = [2,3,6,7]
    #    B_idx: [2*8+0, 2*8+1, 3*8+0, 3*8+1] = [16,17,24,25]
    #    C_idx: [0,1,2,3]
    add_block([2,3,6,7], [16,17,24,25], [0,1,2,3], 28)
    
    # 6. A_R * B_R_top (j=2,3; k=2,3) -> C (i=0,1; k=2,3)
    #    A_idx: [2,3,6,7]
    #    B_idx: [2*8+2, 2*8+3, 3*8+2, 3*8+3] = [18,19,26,27]
    #    C_idx: [4,5,6,7]
    add_block([2,3,6,7], [18,19,26,27], [4,5,6,7], 35)
    
    # 7. A_R * B_R_bot (j=2,3; k=4,5) -> C (i=0,1; k=4,5)
    #    A_idx: [2,3,6,7]
    #    B_idx: [2*8+4, 2*8+5, 3*8+4, 3*8+5] = [20,21,28,29]
    #    C_idx: [8,9,10,11]
    add_block([2,3,6,7], [20,21,28,29], [8,9,10,11], 42)
    
    # 8. A_R * B_R_bot (j=2,3; k=6,7) -> C (i=0,1; k=6,7)
    #    A_idx: [2,3,6,7]
    #    B_idx: [2*8+6, 2*8+7, 3*8+6, 3*8+7] = [22,23,30,31]
    #    C_idx: [12,13,14,15]
    add_block([2,3,6,7], [22,23,30,31], [12,13,14,15], 49)
    
    # Verify
    if _verify(U, V, W, n, m, p, trials=10, tol=1e-9):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
