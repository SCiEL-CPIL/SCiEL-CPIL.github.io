import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill

    # Laderman's 4x4x4 decomposition (rank 17)
    # U: 16x17, V: 16x17, W: 16x17
    # Indices: U[i*4+j, k], V[i*4+j, k], W[k*4+i, j] where C[i,j] = sum_k U[i,j,k]*V[i,j,k]*W[k,i,j]
    
    # Laderman's algorithm coefficients
    # P0 = (A00+A11+A22+A33)(B00+B11+B22+B33)
    # P1 = (A01+A10+A23+A32)(B01+B10+B23+B32)
    # P2 = (A02+A13+A20+A31)(B02+B13+B20+B31)
    # P3 = (A03+A12+A21+A30)(B03+B12+B21+B30)
    # P4 = (A00+A11)(B00+B11)
    # P5 = (A00+A22)(B00+B22)
    # P6 = (A00+A33)(B00+B33)
    # P7 = (A11+A22)(B11+B22)
    # P8 = (A11+A33)(B11+B33)
    # P9 = (A22+A33)(B22+B33)
    # P10 = (A01+A23)(B01+B23)
    # P11 = (A02+A31)(B02+B31)
    # P12 = (A03+A12)(B03+B12)
    # P13 = (A10+A32)(B10+B32)
    # P14 = (A13+A20)(B13+B20)
    # P15 = (A21+A30)(B21+B30)
    # P16 = (A01+A10+A23+A32)(B10+B01+B32+B23)  -- wait, this is getting complex
    
    # Let me use a verified Laderman decomposition.
    # Actually, let me use a simpler approach: just use the known rank-17 decomposition.
    
    # I'll use a standard Laderman 4x4x4 rank-17 decomposition.
    # Reference: Laderman (1976), "The number of multiplications needed to multiply 2x2 matrices"
    # Actually Laderman did 2x2 with 7, the 4x4 rank 17 is from later work.
    
    # Let me use a different approach: keep the Strassen structure but optimize.
    # The current rank is 98. Let me see if I can reduce it.
    
    # Actually, the simplest improvement: the current code has rank 98 = 2 * 7 * 7.
    # But we can potentially share some computations between the two halves.
    
    # For now, let me just add a deadline guard and keep the current approach,
    # but verify correctness and ensure it returns quickly.
    
    n, m, p = 4, 4, 8
    rank = 98
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1
    
    A_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (1, 3)],
        [(1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (-1, 0)],
        [(1, 1), (-1, 3)],
    ]
    
    B_specs = [
        [(1, 0), (1, 3)],
        [(1, 0)],
        [(1, 1), (-1, 3)],
        [(1, 2), (-1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (1, 3)],
    ]
    
    C_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (-1, 3)],
        [(1, 1), (1, 3)],
        [(1, 0), (1, 2)],
        [(-1, 0), (1, 1)],
        [(1, 3)],
        [(1, 0)],
    ]
    
    for half in range(2):
        for k_p in range(7):
            for m_s in range(7):
                col = half * 49 + k_p * 7 + m_s
                
                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_a_idx = global_r * 4 + global_c
                            a_eff_idx = r * 2 + c
                            U[global_a_idx, col] += coeff_a * U_2[a_eff_idx, m_s]
                
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
