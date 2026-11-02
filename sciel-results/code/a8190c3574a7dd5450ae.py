import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # Initialize U, V, W as lists of lists for the interface
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2
    # A factors: 7 linear forms on 4 entries (0,1,2,3)
    # Order: 00, 01, 10, 11
    A_factors = [
        [1, 0, 0, 1],  # 1: 00 + 11
        [0, 0, 1, 1],  # 2: 10 + 11
        [1, 0, 0, 0],  # 3: 00
        [0, 0, 0, 1],  # 4: 11
        [1, 1, 0, 0],  # 5: 00 + 01
        [-1, 0, 1, 0], # 6: 10 - 00
        [0, 1, 0, -1]  # 7: 01 - 11
    ]
    
    B_factors = [
        [1, 0, 0, 1],  # 1: 00 + 11
        [1, 0, 0, 0],  # 2: 00
        [0, 1, 0, -1], # 3: 01 - 11
        [0, 0, 1, -1], # 4: 10 - 00
        [0, 0, 0, 1],  # 5: 11
        [1, 1, 0, 0],  # 6: 00 + 01
        [0, 0, 1, 1]   # 7: 10 + 11
    ]
    
    # Reconstruction coefficients for 2x2 output
    # C[rc] = sum_l W[l, rc] * (L_A[l] * L_B[l])
    # rc order: 00, 01, 10, 11
    W_factors = [
        [1, 0, 0, 0],  # 1: C00
        [0, 0, 1, 0],  # 2: C10
        [0, 1, 0, 0],  # 3: C01
        [0, 0, 0, 1],  # 4: C11
        [0, 1, 0, 0],  # 5: C01
        [0, 0, 0, 1],  # 6: C11
        [1, 0, 0, 0],  # 7: C00
        # Wait, let's check Strassen reconstruction
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So W_factors should be:
        # l=1: C00 += 1, C11 += 1
        # l=2: C10 += 1, C11 -= 1
        # l=3: C01 += 1, C11 += 1
        # l=4: C00 += 1, C10 += 1
        # l=5: C00 -= 1, C01 += 1
        # l=6: C11 += 1
        # l=7: C00 += 1
        
        # Let's define W_factors as a 7x4 matrix where W_factors[l][rc] is the coeff.
        # rc: 0=00, 1=01, 2=10, 3=11
    ]
    
    W_factors = [
        [1, 0, 0, 1],  # l=1: C00=1, C11=1
        [0, 0, 1, -1], # l=2: C10=1, C11=-1
        [0, 1, 0, 1],  # l=3: C01=1, C11=1
        [1, 0, 1, 0],  # l=4: C00=1, C10=1
        [-1, 1, 0, 0], # l=5: C00=-1, C01=1
        [0, 0, 0, 1],  # l=6: C11=1
        [1, 0, 0, 0]   # l=7: C00=1
    ]
    
    # Outer Strassen coefficients
    # S_A[k] = sum_i c_A[k, i] * A_block[i]
    # S_B[k] = sum_j c_B[k, j] * B_block[j]
    # k=0..6 corresponds to M1..M7
    
    c_A = [
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [-1, 0, 1, 0], # M6: A10 - A00
        [0, 1, 0, -1]  # M7: A01 - A11
    ]
    
    c_B = [
        [1, 0, 0, 1],  # M1: B00 + B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [0, 0, 1, -1], # M4: B10 - B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00 + B01
        [0, 0, 1, 1]   # M7: B10 + B11
    ]
    
    # K[IJ, k] coefficients for C_IJ = sum_k K[IJ, k] Q_k
    # IJ: 0=00, 1=01, 2=10, 3=11
    K = [
        [1, 0, 0, 1, -1, 0, 1],  # C00: M1 + M4 - M5 + M7
        [0, 0, 1, 0, 1, 0, 0],   # C01: M3 + M5
        [0, 1, 0, 1, 0, 0, 0],   # C10: M2 + M4
        [1, -1, 1, 0, 0, 1, 0]   # C11: M1 - M2 + M3 + M6
    ]
    
    term_idx = 0
    for k in range(7):
        for i in range(4):
            for j in range(4):
                for l in range(7):
                    coeff_A = c_A[k][i]
                    coeff_B = c_B[k][j]
                    
                    if coeff_A == 0 or coeff_B == 0:
                        continue
                        
                    # Fill U
                    # u_t is non-zero only in block i of A
                    # Block i corresponds to rows i//2 and cols i%2 of the 2x2 block structure
                    # In 4x4 matrix, block i (0,1,2,3) covers:
                    # i=0: rows 0-1, cols 0-1
                    # i=1: rows 0-1, cols 2-3
                    # i=2: rows 2-3, cols 0-1
                    # i=3: rows 2-3, cols 2-3
                    row_base = (i // 2) * 2
                    col_base = (i % 2) * 2
                    
                    for rc in range(4):
                        a_coeff = A_factors[l][rc]
                        if a_coeff != 0:
                            r = row_base + rc // 2
                            c = col_base + rc % 2
                            U[r * m + c][term_idx] = coeff_A * a_coeff
                    
                    # Fill V
                    # v_t is non-zero only in block j of B
                    row_base_B = (j // 2) * 2
                    col_base_B = (j % 2) * 2
                    
                    for rc in range(4):
                        b_coeff = B_factors[l][rc]
                        if b_coeff != 0:
                            r = row_base_B + rc // 2
                            c = col_base_B + rc % 2
                            V[r * p + c][term_idx] = coeff_B * b_coeff
                    
                    # Fill W
                    # w_t contributes to all output blocks IJ
                    for IJ in range(4):
                        k_coeff = K[IJ][k]
                        if k_coeff == 0:
                            continue
                        
                        row_base_C = (IJ // 2) * 2
                        col_base_C = (IJ % 2) * 2
                        
                        for rc in range(4):
                            w_coeff = W_factors[l][rc]
                            if w_coeff != 0:
                                r = row_base_C + rc // 2
                                c = col_base_C + rc % 2
                                W[r * n + c][term_idx] = k_coeff * w_coeff * coeff_A * coeff_B
                    
                    term_idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
