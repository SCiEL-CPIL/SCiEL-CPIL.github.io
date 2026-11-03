import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def _strassen_2x2_factors():
    """Return (alpha, beta, recon) for 2x2 * 2x2 Strassen."""
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1},
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]
    return alpha, beta, recon

def construct():
    _t0 = time.time()
    n, m, p = 4, 4, 6
    
    # Strategy: split B into three 4x2 blocks, multiply each with A using
    # 4 independent 2x2*2x2 Strassen products (rank 7 each).
    # Total rank = 3 * 4 * 7 = 84
    
    R = 84
    U = np.zeros((n * m, R))  # (16, 84)
    V = np.zeros((m * p, R))  # (24, 84)
    W = np.zeros((p * n, R))  # (24, 84)
    
    alpha, beta, recon = _strassen_2x2_factors()
    
    col = 0
    for b_col in range(3):  # B column block: 0, 1, 2
        for a_row in range(2):  # A row block: 0, 1
            # A_block is 2x2: rows [a_row*2, a_row*2+1], cols [0,1,2,3] -> split into 2x2
            # Actually A is 4x4, split into 2x2 blocks: A[ar:ar+2, ac:ac+2] for ar,ac in {0,1}
            # We need A * B[:, b_col*2 : b_col*2+2]
            # A is 4x4, B_block is 4x2
            # Split A into 2x2 blocks: A_00, A_01, A_10, A_11 (each 2x2)
            # Split B_block into 2x2 blocks: B_00, B_10 (each 2x2, since B_block is 4x2)
            # Wait, B_block is 4x2, so it has 2 row-blocks of 2x2: B_block[0:2, :] and B_block[2:4, :]
            
            # Output C_block is 4x2: rows 0-3, cols b_col*2 to b_col*2+1
            # C_block = A * B_block
            # C_block[0:2, :] = A[0:2, :] * B_block  (A rows 0-1 times B_block)
            # C_block[2:4, :] = A[2:4, :] * B_block  (A rows 2-3 times B_block)
            
            # For A rows a_row*2 to a_row*2+1:
            # A_sub = A[a_row*2 : a_row*2+2, :]  (2x4)
            # A_sub * B_block (4x2) = (2x4)*(4x2) = 2x2
            # Split A_sub into 2x2: A_sub[:, 0:2] and A_sub[:, 2:4]
            # Split B_block into 2x2: B_block[0:2, :] and B_block[2:4, :]
            
            # (2x4)*(4x2) = (2x2)*(2x2) + (2x2)*(2x2)
            # = A_sub[:,0:2] * B_block[0:2,:] + A_sub[:,2:4] * B_block[2:4,:]
            
            for a_col in range(2):  # A column block within the 2x4
                # A_sub[:, a_col*2 : a_col*2+2] is 2x2
                # B_block[2*(a_col+1)-2 : 2*(a_col+1), :] is 2x2 (rows 2a_col+... wait)
                # B_block is 4x2. Split into rows: B_block[0:2,:] and B_block[2:4,:]
                # A_sub[:, a_col*2:a_col*2+2] * B_block[2*a_col:2*a_col+2, :]
                
                # For each Strassen term s (7 terms):
                for s in range(7):
                    # U part: A entries
                    # A_sub[:, a_col*2:a_col*2+2] = A[a_row*2:a_row*2+2, a_col*2:a_col*2+2]
                    # This is a 2x2 matrix. Strassen combines its entries.
                    # alpha[s][a] where a=0,1,2,3 indexes the 2x2 matrix entries
                    # a=0: (0,0), a=1: (0,1), a=2: (1,0), a=3: (1,1)
                    for a_idx in range(4):
                        if alpha[s][a_idx] != 0:
                            ai, aj = a_idx // 2, a_idx % 2
                            # Global A index: row = a_row*2 + ai, col = a_col*2 + aj
                            u_idx = (a_row * 2 + ai) * 4 + (a_col * 2 + aj)
                            U[u_idx, col] += alpha[s][a_idx]
                    
                    # V part: B entries
                    # B_block[2*a_col : 2*a_col+2, :] where B_block = B[:, b_col*2 : b_col*2+2]
                    # So B entries are B[2*a_col : 2*a_col+2, b_col*2 : b_col*2+2]
                    # beta[s][b] where b=0,1,2,3
                    for b_idx in range(4):
                        if beta[s][b_idx] != 0:
                            bi, bj = b_idx // 2, b_idx % 2
                            # Global B index: row = 2*a_col + bi, col = b_col*2 + bj
                            v_idx = (2 * a_col + bi) * 6 + (b_col * 2 + bj)
                            V[v_idx, col] += beta[s][b_idx]
                    
                    # W part: output entries
                    # Output is C[a_row*2:a_row*2+2, b_col*2:b_col*2+2]
                    # recon[s] gives (out_row, out_col, gamma)
                    for out_r, out_c, gamma in recon[s]:
                        # Global C index: row = a_row*2 + out_r, col = b_col*2 + out_c
                        w_idx = (b_col * 2 + out_c) * 4 + (a_row * 2 + out_r)
                        W[w_idx, col] += gamma
                    
                    col += 1
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
    
    U_list = to_list(U)
    V_list = to_list(V)
    W_list = to_list(W)
    
    # Validate
    if (time.time() - _t0) < 130.0:
        np.random.seed(12345)
        A_test = np.random.randn(4, 4)
        B_test = np.random.randn(4, 6)
        
        U_arr = np.array(U_list, dtype=float)
        V_arr = np.array(V_list, dtype=float)
        W_arr = np.array(W_list, dtype=float)
        
        A_flat = A_test.flatten()
        B_flat = B_test.flatten()
        
        C_computed = np.zeros((4, 6))
        for r in range(R):
            u_dot_A = np.dot(U_arr[:, r], A_flat)
            v_dot_B = np.dot(V_arr[:, r], B_flat)
            if u_dot_A != 0 and v_dot_B != 0:
                C_computed += (u_dot_A * v_dot_B) * W_arr[:, r].reshape(6, 4).T
        
        C_expected = A_test @ B_test
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback to schoolbook rank 96
            R_s = 96
            U_s = np.zeros((16, R_s))
            V_s = np.zeros((24, R_s))
            W_s = np.zeros((24, R_s))
            idx = 0
            for i in range(4):
                for k in range(4):
                    for j in range(6):
                        U_s[i * 4 + k, idx] = 1
                        V_s[k * 6 + j, idx] = 1
                        W_s[j * 4 + i, idx] = 1
                        idx += 1
            return {"U": to_list(U_s), "V": to_list(V_s), "W": to_list(W_s)}
    
    return {"U": U_list, "V": V_list, "W": W_list}
