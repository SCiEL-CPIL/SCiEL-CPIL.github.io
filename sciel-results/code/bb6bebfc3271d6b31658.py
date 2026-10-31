import numpy as np

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    for the rank-49 decomposition of 4x4 matrix multiplication.
    
    Shapes:
    U: (16, 49)
    V: (16, 49)
    W: (16, 49)
    """
    n = 4
    m = 4
    p = 4
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper for W index: W[k*n + i, r] corresponds to C[i, k] coefficient
    def w_idx(i, k):
        return k * n + i

    # Outer Strassen terms for 4x4 split into 2x2 blocks
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)(B00)
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)(B11)
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Inner Strassen terms for 2x2 blocks (scalar multiplication of 2x2 matrices)
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # N0 = (x00+x11)(y00+y11)
    # N1 = (x10+x11)(y00)
    # N2 = x00(y01-y11)
    # N3 = x11(y10-y00)
    # N4 = (x00+x01)(y11)
    # N5 = (x10-x00)(y00+y01)
    # N6 = (x01-x11)(y10+y11)
    
    # Z00 = N0 + N3 - N4 + N6
    # Z01 = N2 + N4
    # Z10 = N1 + N3
    # Z11 = N0 - N1 + N2 + N5
    
    # Definitions
    # Block offsets (row, col) for 2x2 blocks in 4x4 matrix
    # 0: (0,0), 1: (0,2), 2: (2,0), 3: (2,2)
    
    # Outer A coefficients: list of 7 dicts {block_id: coeff}
    # block_id: 0=A00, 1=A01, 2=A10, 3=A11
    outer_A = [
        {0: 1, 3: 1},   # M0: A00+A11
        {2: 1, 3: 1},   # M1: A10+A11
        {0: 1},         # M2: A00
        {3: 1},         # M3: A11
        {0: 1, 1: 1},   # M4: A00+A01
        {2: 1, 0: -1},  # M5: A10-A00
        {1: 1, 3: -1}   # M6: A01-A11
    ]
    
    # Outer B coefficients: list of 7 dicts {block_id: coeff}
    outer_B = [
        {0: 1, 3: 1},   # M0: B00+B11
        {0: 1},         # M1: B00
        {1: 1, 3: -1},  # M2: B01-B11
        {2: 1, 0: -1},  # M3: B10-B00
        {3: 1},         # M4: B11
        {0: 1, 1: 1},   # M5: B00+B01
        {2: 1, 3: 1}    # M6: B10+B11
    ]
    
    # Inner X coefficients: list of 7 dicts {(local_row, local_col): coeff}
    # local indices 0,1
    inner_X = [
        {(0,0): 1, (1,1): 1},   # N0: x00+x11
        {(1,0): 1, (1,1): 1},   # N1: x10+x11
        {(0,0): 1},             # N2: x00
        {(1,1): 1},             # N3: x11
        {(0,0): 1, (0,1): 1},   # N4: x00+x01
        {(1,0): 1, (0,0): -1},  # N5: x10-x00
        {(0,1): 1, (1,1): -1}   # N6: x01-x11
    ]
    
    # Inner Y coefficients: list of 7 dicts {(local_row, local_col): coeff}
    inner_Y = [
        {(0,0): 1, (1,1): 1},   # N0: y00+y11
        {(0,0): 1},             # N1: y00
        {(0,1): 1, (1,1): -1},  # N2: y01-y11
        {(1,0): 1, (0,0): -1},  # N3: y10-y00
        {(1,1): 1},             # N4: y11
        {(0,0): 1, (0,1): 1},   # N5: y00+y01
        {(1,0): 1, (1,1): 1}    # N6: y10+y11
    ]
    
    # Inner Z contributions: list of 7 dicts {(local_row, local_col): coeff}
    inner_Z = [
        {(0,0): 1, (1,1): 1},   # N0 -> Z00(+1), Z11(+1)
        {(1,0): 1, (1,1): -1},  # N1 -> Z10(+1), Z11(-1)
        {(0,1): 1, (1,1): 1},   # N2 -> Z01(+1), Z11(+1)
        {(0,0): 1, (1,0): 1},   # N3 -> Z00(+1), Z10(+1)
        {(0,0): -1, (0,1): 1},  # N4 -> Z00(-1), Z01(+1)
        {(1,1): 1},             # N5 -> Z11(+1)
        {(0,0): 1}              # N6 -> Z00(+1)
    ]
    
    # Outer C contributions: list of 7 dicts {(block_row, block_col): coeff}
    # block indices 0,1
    outer_C = [
        {(0,0): 1, (1,1): 1},   # M0 -> C00(+1), C11(+1)
        {(1,0): 1, (1,1): -1},  # M1 -> C10(+1), C11(-1)
        {(0,1): 1, (1,1): 1},   # M2 -> C01(+1), C11(+1)
        {(0,0): 1, (1,0): 1},   # M3 -> C00(+1), C10(+1)
        {(0,0): -1, (0,1): 1},  # M4 -> C00(-1), C01(+1)
        {(1,1): 1},             # M5 -> C11(+1)
        {(0,0): 1}              # M6 -> C00(+1)
    ]
    
    # Block offsets
    offsets = {0: (0,0), 1: (0,2), 2: (2,0), 3: (2,2)}
    
    col = 0
    for t in range(7):
        a_blocks = outer_A[t]
        b_blocks = outer_B[t]
        
        for k in range(7):
            x_coeffs = inner_X[k]
            y_coeffs = inner_Y[k]
            
            # Fill U
            for (li, lj), cx in x_coeffs.items():
                for bid, cb in a_blocks.items():
                    coeff = cx * cb
                    if coeff == 0: continue
                    ro, co = offsets[bid]
                    gi = ro + li
                    gj = co + lj
                    U[gi * m + gj, col] += coeff
            
            # Fill V
            for (li, lj), cy in y_coeffs.items():
                for bid, cb in b_blocks.items():
                    coeff = cy * cb
                    if coeff == 0: continue
                    ro, co = offsets[bid]
                    gj = ro + li
                    gk = co + lj
                    V[gj * p + gk, col] += coeff
            
            # Fill W
            c_contrib = outer_C[t]
            z_contrib = inner_Z[k]
            
            for (cbr, cbc), cc in c_contrib.items():
                for (zli, zlj), zc in z_contrib.items():
                    coeff = cc * zc
                    if coeff == 0: continue
                    gi = cbr * 2 + zli
                    gk = cbc * 2 + zlj
                    W[w_idx(gi, gk), col] += coeff
            
            col += 1
            
    assert col == 49
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
