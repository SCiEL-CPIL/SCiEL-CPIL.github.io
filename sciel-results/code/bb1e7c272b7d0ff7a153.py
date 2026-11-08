import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use a rank-28 decomposition based on applying Strassen's algorithm
    # to four 2x2 block multiplications.
    # A is 2x4, B is 4x4.
    # A = [A1 A2] where A1, A2 are 2x2.
    # B = [[B11, B12], [B21, B22]] where each is 2x2.
    # C = [A1 B11 + A2 B21, A1 B12 + A2 B22]
    #
    # Each 2x2 * 2x2 product can be done in 7 multiplications (Strassen).
    # Total 4 * 7 = 28 multiplications.
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Strassen terms for X * Y -> Z
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    #
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),  # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),        # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),        # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),        # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),        # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),        # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),        # M7
    ]
    
    def apply_block(a_col_off, b_row_off, b_col_off, c_col_off, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(strassen_terms):
            col = start_col + t_idx
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                gi = r
                gj = a_col_off * 2 + c
                U[u_idx(gi, gj), col] += coeff
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                gj = b_row_off * 2 + r
                gk = b_col_off * 2 + c
                V[v_idx(gj, gk), col] += coeff
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                gi = r
                gk = c_col_off * 2 + c
                W[w_idx(gk, gi), col] += coeff
                
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    apply_block(0, 0, 0, 0, 0)
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    apply_block(1, 1, 0, 0, 7)
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    apply_block(0, 0, 1, 1, 14)
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    apply_block(1, 1, 1, 1, 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
