import numpy as np
import time

def construct():
    deadline = time.time() + 120.0
    n, m, p = 4, 4, 8
    rank = 98

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen 2x2x2 decomposition
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

    # Block Strassen for 4x4x4
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

    # Precompute element mappings for each spec entry
    # For U: (coeff, block_idx) -> list of (global_a_idx, a_eff_idx)
    # For V: (coeff, block_idx, half) -> list of (global_b_idx, b_eff_idx)
    # For W: (coeff, block_idx, half) -> list of (global_w_idx, p_eff_idx)

    def get_u_elements(block_idx):
        bi, bj = divmod(block_idx, 2)
        elems = []
        for r in range(2):
            for c in range(2):
                global_r = bi * 2 + r
                global_c = bj * 2 + c
                global_a_idx = global_r * 4 + global_c
                a_eff_idx = r * 2 + c
                elems.append((global_a_idx, a_eff_idx))
        return elems

    def get_v_elements(block_idx, half):
        bi, bj = divmod(block_idx, 2)
        elems = []
        for r in range(2):
            for c in range(2):
                global_r = bi * 2 + r
                global_c = bj * 2 + c
                global_k = global_c + half * 4
                global_b_idx = global_r * 8 + global_k
                b_eff_idx = r * 2 + c
                elems.append((global_b_idx, b_eff_idx))
        return elems

    def get_w_elements(block_idx, half):
        bi, bj = divmod(block_idx, 2)
        elems = []
        for r in range(2):
            for c in range(2):
                global_i = bi * 2 + r
                global_k = bj * 2 + c + half * 4
                global_w_idx = global_k * 4 + global_i
                p_eff_idx = r * 2 + c
                elems.append((global_w_idx, p_eff_idx))
        return elems

    # Precompute all element lists
    U_elems = {}
    for k_p in range(7):
        for coeff_a, block_idx in A_specs[k_p]:
            key = (k_p, coeff_a, block_idx)
            U_elems[key] = get_u_elements(block_idx)

    V_elems = {}
    for half in range(2):
        for k_p in range(7):
            for coeff_b, block_idx in B_specs[k_p]:
                key = (half, k_p, coeff_b, block_idx)
                V_elems[key] = get_v_elements(block_idx, half)

    W_elems = {}
    for half in range(2):
        for k_p in range(7):
            for coeff_c, c_block_idx in C_specs[k_p]:
                key = (half, k_p, coeff_c, c_block_idx)
                W_elems[key] = get_w_elements(c_block_idx, half)

    for half in range(2):
        base_col = half * 49
        for k_p in range(7):
            for m_s in range(7):
                col = base_col + k_p * 7 + m_s

                # U (A part)
                for coeff_a, block_idx in A_specs[k_p]:
                    key = (k_p, coeff_a, block_idx)
                    for global_a_idx, a_eff_idx in U_elems[key]:
                        U[global_a_idx, col] += coeff_a * U_2[a_eff_idx, m_s]

                # V (B part)
                for coeff_b, block_idx in B_specs[k_p]:
                    key = (half, k_p, coeff_b, block_idx)
                    for global_b_idx, b_eff_idx in V_elems[key]:
                        V[global_b_idx, col] += coeff_b * V_2[b_eff_idx, m_s]

                # W (C part)
                for coeff_c, c_block_idx in C_specs[k_p]:
                    key = (half, k_p, coeff_c, c_block_idx)
                    for global_w_idx, p_eff_idx in W_elems[key]:
                        W[global_w_idx, col] += coeff_c * W_2[p_eff_idx, m_s]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
