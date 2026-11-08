import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill

    # We build a rank-98 decomposition using recursive Strassen structure.
    # The 4x4 * 4x8 multiplication is split into two 4x4 * 4x4 multiplications.
    # Each 4x4 * 4x4 uses a rank-49 Strassen decomposition (7x7 from 2x2 blocks).
    # Total rank = 2 * 49 = 98.
    
    rank = 98
    U = np.zeros((n * m, rank))  # 16 x 98
    V = np.zeros((m * p, rank))  # 32 x 98
    W = np.zeros((p * n, rank))  # 32 x 98

    # Strassen 2x2x2 decomposition
    # U_2, V_2, W_2 are 4x7 (flattened 2x2 matrices)
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # P0 = (A00+A11)(B00+B11)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    # P1 = (A10+A11)B00
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    # P2 = A00(B01-B11)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    # P3 = A11(B10-B00)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    # P4 = (A00+A01)B11
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    # P5 = (A10-A00)(B00+B01)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    # P6 = (A01-A11)(B10+B11)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    # Block Strassen coefficients for 4x4x4
    # A is 4x4, B is 4x4 (sub-problem), C is 4x4
    # Blocks are 2x2. Block index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # For each of the 7 Strassen products P_k, define:
    # A_specs[k]: list of (coeff, block_idx) for A_eff
    # B_specs[k]: list of (coeff, block_idx) for B_eff
    # C_specs[k]: list of (coeff, block_idx) for C contribution
    
    A_specs = [
        [(1, 0), (1, 3)],      # P0: A00 + A11
        [(1, 2), (1, 3)],      # P1: A10 + A11
        [(1, 0)],              # P2: A00
        [(1, 3)],              # P3: A11
        [(1, 0), (1, 1)],      # P4: A00 + A01
        [(1, 2), (-1, 0)],     # P5: A10 - A00
        [(1, 1), (-1, 3)],     # P6: A01 - A11
    ]
    
    B_specs = [
        [(1, 0), (1, 3)],      # P0: B00 + B11
        [(1, 0)],              # P1: B00
        [(1, 1), (-1, 3)],     # P2: B01 - B11
        [(1, 2), (-1, 0)],     # P3: B10 - B00
        [(1, 3)],              # P4: B11
        [(1, 0), (1, 1)],      # P5: B00 + B01
        [(1, 2), (1, 3)],      # P6: B10 + B11
    ]
    
    # C_specs: How P_k contributes to C blocks
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    
    C_specs = [
        [(1, 0), (1, 3)],      # P0 -> C00 + C11
        [(1, 2), (-1, 3)],     # P1 -> C10 - C11
        [(1, 1), (1, 3)],      # P2 -> C01 + C11
        [(1, 0), (1, 2)],      # P3 -> C00 + C10
        [(-1, 0), (1, 1)],     # P4 -> -C00 + C01
        [(1, 3)],              # P5 -> C11
        [(1, 0)],              # P6 -> C00
    ]
    
    # Build the decomposition for each half (left and right 4 columns of B)
    for half in range(2):
        b_col_offset = half * 4  # 0 for left, 4 for right
        
        for k_p in range(7):
            for m_s in range(7):
                col = half * 49 + k_p * 7 + m_s
                
                # --- U (Input A, 4x4) ---
                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_row = bi * 2 + r
                            global_col = bj * 2 + c
                            a_global_idx = global_row * 4 + global_col
                            a_eff_idx = r * 2 + c
                            U[a_global_idx, col] += coeff_a * U_2[a_eff_idx, m_s]
                
                # --- V (Input B, 4x8) ---
                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_row = bi * 2 + r
                            global_col = bj * 2 + c + b_col_offset
                            b_global_idx = global_row * 8 + global_col
                            b_eff_idx = r * 2 + c
                            V[b_global_idx, col] += coeff_b * V_2[b_eff_idx, m_s]
                
                # --- W (Output C, 4x8) ---
                # C index: k*n + i where k is column (0..7), i is row (0..3)
                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = divmod(c_block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_row = bi * 2 + r
                            global_col = bj * 2 + c + b_col_offset
                            w_global_idx = global_col * 4 + global_row
                            c_eff_idx = r * 2 + c
                            W[w_global_idx, col] += coeff_c * W_2[c_eff_idx, m_s]
    
    # Verify correctness
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 8)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((4, 8))
    for k in range(rank):
        prod_a = 0.0
        for i in range(16):
            prod_a += U[i, k] * A_test.flat[i]
        prod_b = 0.0
        for i in range(32):
            prod_b += V[i, k] * B_test.flat[i]
        for i in range(32):
            C_computed.flat[i] += W[i, k] * prod_a * prod_b
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback: naive decomposition
        return _naive_decomposition()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _naive_decomposition():
    """Naive rank-112 decomposition as fallback."""
    n, m, p = 4, 4, 8
    rank = n * m * p  # 112
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
