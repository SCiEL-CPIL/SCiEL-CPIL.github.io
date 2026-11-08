import numpy as np
import time

def construct():
    # Deadline guard: The evaluator kills after 150s.
    # We should return the best result found before 120s to be safe.
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    n, m, p = 4, 4, 8
    
    # We use a rank-98 decomposition for 4x4x8.
    # Strategy: Split B into two 4x4 halves.
    # Each 4x4x4 is decomposed using Strassen (rank 49).
    # Total rank = 2 * 49 = 98.
    
    rank = 98
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2 decomposition
    # U_2, V_2, W_2 are 4x7
    # Indices for 2x2 matrix: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # c=0 (P1)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    # c=1 (P2)
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    # c=2 (P3)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    # c=3 (P4)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    # c=4 (P5)
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    # c=5 (P6)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    # c=6 (P7)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1
    
    # Define the 7 P_k for Strassen 4x4x4
    # A_specs[k_p]: list of (coeff, block_idx) for A_eff
    # B_specs[k_p]: list of (coeff, block_idx) for B_eff
    # C_specs[k_p]: list of (coeff, block_idx) for how P_k contributes to C blocks
    
    # Block indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    A_specs = [
        [(1, 0), (1, 3)],       # P0: A00 + A11
        [(1, 2), (1, 3)],       # P1: A10 + A11
        [(1, 0)],               # P2: A00
        [(1, 3)],               # P3: A11
        [(1, 0), (1, 1)],       # P4: A00 + A01
        [(1, 2), (-1, 0)],      # P5: A10 - A00
        [(1, 1), (-1, 3)],      # P6: A01 - A11
    ]
    
    B_specs = [
        [(1, 0), (1, 3)],       # P0: B00 + B11
        [(1, 0)],               # P1: B00
        [(1, 1), (-1, 3)],      # P2: B01 - B11
        [(1, 2), (-1, 0)],      # P3: B10 - B00
        [(1, 3)],               # P4: B11
        [(1, 0), (1, 1)],       # P5: B00 + B01
        [(1, 2), (1, 3)],       # P6: B10 + B11
    ]
    
    C_specs = [
        [(1, 0), (1, 3)],       # P0: C00 + C11
        [(1, 2), (-1, 3)],      # P1: C10 - C11
        [(1, 1), (1, 3)],       # P2: C01 + C11
        [(1, 0), (1, 2)],       # P3: C00 + C10
        [(-1, 0), (1, 1)],      # P4: -C00 + C01
        [(1, 3)],               # P5: C11
        [(1, 0)],               # P6: C00
    ]
    
    # Now fill U, V, W for the full 4x4x8 problem
    for half in range(2):  # 0 for left half, 1 for right half
        for k_p in range(7):  # P_k
            for m_s in range(7):  # S_m
                col = half * 49 + k_p * 7 + m_s
                
                # --- U (Input A) ---
                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_a_idx = global_r * 4 + global_c
                            a_eff_idx = r * 2 + c
                            U[global_a_idx, col] += coeff_a * U_2[a_eff_idx, m_s]
                
                # --- V (Input B) ---
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
                
                # --- W (Output C) ---
                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = divmod(c_block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_i = bi * 2 + r
                            global_k = bj * 2 + c + half * 4
                            global_w_idx = global_k * 4 + global_i
                            p_eff_idx = r * 2 + c
                            W[global_w_idx, col] += coeff_c * W_2[p_eff_idx, m_s]
                
                # Check deadline
                if time.time() > deadline:
                    # If we run out of time, return what we have.
                    # Since the loop is deterministic and fast, this shouldn't happen.
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
