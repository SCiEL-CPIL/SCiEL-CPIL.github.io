import numpy as np
import time

def construct():
    t_start = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2 factors
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)(Y00)
    # M3 = (X00)(Y01-Y11)
    # M4 = (X11)(Y10-Y00)
    # M5 = (X00+X01)(Y11)
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    
    # U coeffs for 2x2 block (rows of A)
    # Local indices (lr, lc)
    s_u = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # V coeffs for 2x2 block (cols of B)
    s_v = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    
    # W coeffs for 2x2 block (rows/cols of C)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    s_w = [
        [1, 0, 0, 1],  # M1: Z00, Z11
        [0, 0, 1, -1], # M2: Z10, Z11
        [0, 1, 0, 1],  # M3: Z01, Z11
        [1, 0, 1, 0],  # M4: Z00, Z10
        [-1, 1, 0, 0], # M5: Z00, Z01
        [0, 0, 0, 1],  # M6: Z11
        [1, 0, 0, 0]   # M7: Z00
    ]
    
    def add_strassen_block(a_r, a_c, b_r, b_c, c_r, c_c, col_off):
        for t in range(7):
            col = col_off + t
            for lr in range(2):
                for lc in range(2):
                    val = s_u[t][lr*2+lc]
                    if val != 0:
                        U[(a_r + lr) * m + (a_c + lc), col] += val
                    val = s_v[t][lr*2+lc]
                    if val != 0:
                        V[(b_r + lr) * p + (b_c + lc), col] += val
                    val = s_w[t][lr*2+lc]
                    if val != 0:
                        W[(c_c + lc) * n + (c_r + lr), col] += val
                        
    # Part 1: Row 0 of A (1x4) * B (4x4) -> Row 0 of C (1x4)
    # Schoolbook: 4*4 = 16 terms
    # U index: (0, j)
    # V index: (j, k)
    # W index: (k, 0)
    col_off = 0
    for j in range(4):
        for k in range(4):
            col = col_off + j * 4 + k
            U[0 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 0, col] = 1
            
    col_off = 16
    
    # Part 2: Rows 1-2 of A (2x4) * B (4x4) -> Rows 1-2 of C (2x4)
    # Split into four 2x2 blocks:
    # A[1:3, 0:2] * B[0:2, 0:2] -> C[1:3, 0:2]
    add_strassen_block(1, 0, 0, 0, 1, 0, col_off)
    col_off += 7
    
    # A[1:3, 0:2] * B[0:2, 2:4] -> C[1:3, 2:4]
    add_strassen_block(1, 0, 0, 2, 1, 2, col_off)
    col_off += 7
    
    # A[1:3, 2:4] * B[2:4, 0:2] -> C[1:3, 0:2]
    add_strassen_block(1, 2, 2, 0, 1, 0, col_off)
    col_off += 7
    
    # A[1:3, 2:4] * B[2:4, 2:4] -> C[1:3, 2:4]
    add_strassen_block(1, 2, 2, 2, 1, 2, col_off)
    col_off += 7
    
    # Verify correctness
    def verify(U_mat, V_mat, W_mat):
        r = U_mat.shape[1]
        U_r = U_mat.reshape(n, m, r)
        V_r = V_mat.reshape(m, p, r)
        W_r = W_mat.reshape(p, n, r)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=1e-10)
        
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
