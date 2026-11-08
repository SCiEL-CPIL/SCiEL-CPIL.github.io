import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s deadline, 40s buffer for evaluation overhead

    n, m, p = 4, 4, 8

    # Rank for 4x4x8 matrix multiplication via Strassen (rank 70)
    # Base: 2x2x2 Strassen uses rank 7
    # 4x4x4 uses 2 levels of Strassen: rank = 7^2 = 49
    # 4x4x8: split B into two 4x4 halves
    # We need to check if rank 49 is sufficient for 4x4x4, and 70 for 4x4x8
    # Actually, 4x4x4 with Strassen rank is 49, and 4x4x8 is 2 * 49 = 98? 
    # But wait, we can potentially do better. The current program uses rank 98.
    # Let's check: for 4x4x4, Strassen gives rank 49. For 4x4x8, if we split B into two
    # 4x4 blocks, we need two copies of the rank-49 decomposition, giving 98.
    # But there might be a more efficient construction. Let's stick with 98 for now.
    
    # Actually, let's reconsider. The 4x4x4 Strassen decomposition has rank 49.
    # For 4x4x8, we can use the fact that B is 4x8, which is two 4x4 blocks.
    # So the rank is 2 * 49 = 98. This matches the current program.
    
    # However, there might be a rank-70 construction for 4x4x8 using a smarter
    # approach. Let's try to implement the known optimal or near-optimal ranks.
    
    # For now, let's implement the rank-49 Strassen for 4x4x4 and replicate for 4x4x8.
    
    rank = 98

    U = np.zeros((n * m, rank))  # (16, 98)
    V = np.zeros((m * p, rank))  # (32, 98)
    W = np.zeros((p * n, rank))  # (32, 98)

    # Strassen 2x2x2 base
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

    # For 4x4x4, use two levels of Strassen
    # A is 4x4, split into 2x2 blocks of 2x2
    # B is 4x4, split into 2x2 blocks of 2x2
    # C is 4x4, split into 2x2 blocks of 2x2

    # A_specs[k]: list of (coeff, block_idx) for A
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

    # Fill U, V, W for 4x4x4, then extend to 4x4x8
    # For 4x4x4:
    # A is 4x4, B is 4x4, C is 4x4
    # For 4x4x8:
    # A is 4x4, B is 4x8 (two 4x4 halves), C is 4x8 (two 4x4 halves)

    for half in range(2):
        # Offset for B and C
        # B: rows are 0-3, cols are half*4 to half*4+3
        # C: rows are 0-3, cols are half*4 to half*4+3

        for k in range(7):
            for s in range(7):
                col = half * 49 + k * 7 + s

                # U (A is 4x4, no offset)
                for coeff_a, block_idx in A_specs[k]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_a_idx = global_r * 4 + global_c
                            a_eff_idx = r * 2 + c
                            U[global_a_idx, col] += coeff_a * U_2[a_eff_idx, s]

                # V (B is 4x8, offset by half)
                for coeff_b, block_idx in B_specs[k]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_k = global_c + half * 4
                            global_b_idx = global_r * 8 + global_k
                            b_eff_idx = r * 2 + c
                            V[global_b_idx, col] += coeff_b * V_2[b_eff_idx, s]

                # W (C is 4x8, offset by half)
                # W index is k*n + i = k*4 + i
                for coeff_c, c_block_idx in C_specs[k]:
                    bi, bj = divmod(c_block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_i = bi * 2 + r
                            global_k = bj * 2 + c + half * 4
                            global_w_idx = global_k * 4 + global_i
                            p_eff_idx = r * 2 + c
                            W[global_w_idx, col] += coeff_c * W_2[p_eff_idx, s]

                # Deadline check
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
