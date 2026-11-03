"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen's algorithm for 4x4x4 embedded in 4x5x5
    # Rank is 49
    
    # 1. Define inner 2x2x2 Strassen decomposition
    # 7 terms
    # u: coeffs for A (2x2), v: coeffs for B (2x2), w: coeffs for C (2x2)
    # Order: [00, 01, 10, 11]
    
    U_inner = np.array([
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ], dtype=float)
    
    V_inner = np.array([
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ], dtype=float)
    
    W_inner = np.array([
        [1, 0, 0, 1],  # M1 -> C00, C11
        [0, 0, 1, -1], # M2 -> C10, -C11
        [0, 1, 0, 1],  # M3 -> C01, C11
        [1, 0, 1, 0],  # M4 -> C00, C10
        [-1, 1, 0, 0], # M5 -> -C00, C01
        [0, 0, 0, 1],  # M6 -> C11
        [1, 0, 0, 0]   # M7 -> C00
    ], dtype=float)

    # 2. Define outer 4x4x4 Strassen coefficients
    # 7 terms
    # Alpha: coeffs for A blocks [00, 01, 10, 11]
    # Beta: coeffs for B blocks [00, 01, 10, 11]
    
    Alpha = np.array([
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [-1, 0, 1, 0], # M6: -A00 + A10
        [0, 1, 0, -1]  # M7: A01 - A11
    ], dtype=float)
    
    Beta = np.array([
        [1, 0, 0, 1],  # M1: B00 + B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [0, 0, 1, -1], # M4: B10 - B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00 + B01
        [0, 0, 1, 1]   # M7: B10 + B11
    ], dtype=float)
    
    # Gamma: coeffs for C blocks [00, 01, 10, 11] from M1..M7
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    Gamma = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ], dtype=float)

    R = 49
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Helper to map block and local index to global index
    # Blocks are 2x2.
    # Block 00: rows 0-1, cols 0-1
    # Block 01: rows 0-1, cols 2-3
    # Block 10: rows 2-3, cols 0-1
    # Block 11: rows 2-3, cols 2-3
    
    def get_global_index(rows, cols, block, r_loc, c_loc):
        # block: 0, 1, 2, 3 corresponding to 00, 01, 10, 11
        if block == 0: # 00
            r = r_loc
            c = c_loc
        elif block == 1: # 01
            r = r_loc
            c = c_loc + 2
        elif block == 2: # 10
            r = r_loc + 2
            c = c_loc
        elif block == 3: # 11
            r = r_loc + 2
            c = c_loc + 2
        else:
            raise ValueError("Invalid block")
        return r * cols + c

    # Construct U, V, W
    for k in range(7): # Outer term
        for j in range(7): # Inner term
            t = k * 7 + j
            
            # U vector
            for b_A in range(4): # Block in A
                alpha_val = Alpha[k, b_A]
                if alpha_val == 0:
                    continue
                for r_loc in range(2):
                    for c_loc in range(2):
                        u_val = U_inner[j, r_loc * 2 + c_loc]
                        if u_val == 0:
                            continue
                        idx = get_global_index(n, m, b_A, r_loc, c_loc)
                        U[idx, t] += alpha_val * u_val

            # V vector
            for b_B in range(4): # Block in B
                beta_val = Beta[k, b_B]
                if beta_val == 0:
                    continue
                for r_loc in range(2):
                    for c_loc in range(2):
                        v_val = V_inner[j, r_loc * 2 + c_loc]
                        if v_val == 0:
                            continue
                        # B is 5x5, but we only use 4x4 part
                        # Rows 0-3, Cols 0-3
                        # Block mapping is same
                        idx = get_global_index(m, p, b_B, r_loc, c_loc)
                        V[idx, t] += beta_val * v_val

            # W vector
            for b_C in range(4): # Block in C
                gamma_val = Gamma[b_C, k]
                if gamma_val == 0:
                    continue
                for r_loc in range(2):
                    for c_loc in range(2):
                        w_val = W_inner[j, r_loc * 2 + c_loc]
                        if w_val == 0:
                            continue
                        # C is 4x5 (p x n)
                        # Rows 0-3, Cols 0-3
                        idx = get_global_index(p, n, b_C, r_loc, c_loc)
                        W[idx, t] += gamma_val * w_val

    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
