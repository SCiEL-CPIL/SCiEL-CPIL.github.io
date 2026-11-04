import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Rank 98 Strassen-based decomposition
    # This is 7 (Strassen terms for 4x4) * 7 (Strassen terms for 2x2) * 2 (halves of B)
    # Wait, the current logic is:
    # For each of 2 halves of B (4x4 each):
    #   Use Strassen for 4x4x4 (rank 49)
    # Total rank 98.
    
    rank = 98
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Base 2x2x2 Strassen matrices (rank 7)
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))

    # Standard Strassen coefficients for 2x2
    # c0: (A11+A22)(B11+B22) -> C11, C22
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    # c1: (A21+A22)B11 -> C21, C22
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    # c2: A11(B21-B22) -> C12, C22
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    # c3: A22(B12-B11) -> C11, C21
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    # c4: (A11+A12)B22 -> C11, C12
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    # c5: (A21-A11)(B11+B12) -> C22
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    # c6: (A12-A22)(B21+B22) -> C11
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    # Specs for the 4x4 Strassen step
    # Block indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # A_specs, B_specs, C_specs define how the 4x4 blocks are combined
    # This is the standard Strassen recursion for 4x4.
    
    A_specs = [
        [(1, 0), (1, 3)],   # P0: A00 + A11
        [(1, 2), (1, 3)],   # P1: A10 + A11
        [(1, 0)],           # P2: A00
        [(1, 3)],           # P3: A11
        [(1, 0), (1, 1)],   # P4: A00 + A01
        [(1, 2), (-1, 0)],  # P5: A10 - A00
        [(1, 1), (-1, 3)],  # P6: A01 - A11
    ]

    B_specs = [
        [(1, 0), (1, 3)],   # P0: B00 + B11
        [(1, 0)],           # P1: B00
        [(1, 1), (-1, 3)],  # P2: B01 - B11
        [(1, 2), (-1, 0)],  # P3: B10 - B00
        [(1, 3)],           # P4: B11
        [(1, 0), (1, 1)],   # P5: B00 + B01
        [(1, 2), (1, 3)],   # P6: B10 + B11
    ]

    C_specs = [
        [(1, 0), (1, 3)],   # P0: C00 + C11
        [(1, 2), (-1, 3)],  # P1: C10 - C11
        [(1, 1), (1, 3)],   # P2: C01 + C11
        [(1, 0), (1, 2)],   # P3: C00 + C10
        [(-1, 0), (1, 1)],  # P4: -C00 + C01
        [(1, 3)],           # P5: C11
        [(1, 0)],           # P6: C00
    ]

    # Build the decomposition
    # We process two halves of B (left and right 4x4 blocks)
    for half in range(2):
        for k_p in range(7):
            for m_s in range(7):
                col = half * 49 + k_p * 7 + m_s

                # --- U (Input A) ---
                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            gr = bi * 2 + r
                            gc = bj * 2 + c
                            a_idx = gr * 4 + gc
                            eff = r * 2 + c
                            U[a_idx, col] += coeff_a * U_2[eff, m_s]

                # --- V (Input B) ---
                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            gr = bi * 2 + r
                            gc = bj * 2 + c
                            gk = gc + half * 4
                            b_idx = gr * 8 + gk
                            eff = r * 2 + c
                            V[b_idx, col] += coeff_b * V_2[eff, m_s]

                # --- W (Output C) ---
                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = divmod(c_block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bi * 2 + r
                            gk = bj * 2 + c + half * 4
                            w_idx = gk * 4 + gi
                            eff = r * 2 + c
                            W[w_idx, col] += coeff_c * W_2[eff, m_s]

                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
