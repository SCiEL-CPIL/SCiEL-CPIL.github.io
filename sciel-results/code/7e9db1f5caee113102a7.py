import numpy as np

def construct():
    n, m, p = 4, 4, 8
    rank = 98
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # c=0
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    # c=1
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    # c=2
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    # c=3
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    # c=4
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    # c=5
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    # c=6
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1
    
    # Block Strassen for 4x4x4
    # A_specs: list of 7 lists, each containing (coeff, block_idx)
    # block_idx 0..3 for 2x2 blocks of 4x4 matrix
    A_specs = [
        [(1, 0), (1, 3)], # P0
        [(1, 2), (1, 3)], # P1
        [(1, 0)],         # P2
        [(1, 3)],         # P3
        [(1, 0), (1, 1)], # P4
        [(1, 2), (-1, 0)],# P5
        [(1, 1), (-1, 3)],# P6
    ]
    
    B_specs = [
        [(1, 0), (1, 3)], # P0
        [(1, 0)],         # P1
        [(1, 1), (-1, 3)],# P2
        [(1, 2), (-1, 0)],# P3
        [(1, 3)],         # P4
        [(1, 0), (1, 1)], # P5
        [(1, 2), (1, 3)], # P6
    ]
    
    C_specs = [
        [(1, 0), (1, 3)], # P0
        [(1, 2), (-1, 3)], # P1
        [(1, 1), (1, 3)], # P2
        [(1, 0), (1, 2)], # P3
        [(-1, 0), (1, 1)], # P4
        [(1, 3)], # P5
        [(1, 0)], # P6
    ]
    
    # We have two halves for the 8 columns of B
    # half 0: cols 0-3, half 1: cols 4-7
    for half in range(2):
        base_col = half * 49
        for k_p in range(7):
            for m_s in range(7):
                col = base_col + k_p * 7 + m_s
                
                # U (A part)
                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_a_idx = global_r * 4 + global_c
                            a_eff_idx = r * 2 + c
                            U[global_a_idx, col] += coeff_a * U_2[a_eff_idx, m_s]
                
                # V (B part)
                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_k = global_c + half * 4
                            global_b_idx = global_r * 8 + global_k
                            b_eff_idx = r * 2 + c
                            V[global_b_idx, col] += coeff_b * V_2[b_eff_idx, m_s]
                
                # W (C part)
                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = divmod(c_block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_i = bi * 2 + r
                            global_k = bj * 2 + c + half * 4
                            global_w_idx = global_k * 4 + global_i
                            p_eff_idx = r * 2 + c
                            W[global_w_idx, col] += coeff_c * W_2[p_eff_idx, m_s]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
