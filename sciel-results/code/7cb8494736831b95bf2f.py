"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 6x6 = 3x3 grid of 2x2 blocks
    # Outer: 3x3 matrix multiply (schoolbook, rank 27)
    # Inner: 2x2 matrix multiply (Strassen, rank 7)
    # Total rank = 27 * 7 = 189
    
    outer_blocks = 3  # 3x3 grid
    inner_size = 2    # each block is 2x2
    dim = n  # 6
    
    # Strassen coefficients for 2x2 multiplication
    # For each of 7 Strassen products:
    # U_coeffs[s]: linear form on 2x2 A block (4 entries: [a00, a01, a10, a11])
    # V_coeffs[s]: linear form on 2x2 B block (4 entries: [b00, b01, b10, b11])
    # W_coeffs[s]: linear form on 2x2 C block (4 entries: [c00, c01, c10, c11])
    
    # Strassen formulas:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U_coeffs[s][li][lj] = coefficient of A_block[li][lj] in Strassen product s
    U_coeffs = [
        [[1, 0], [0, 1]],   # M1: A00 + A11
        [[0, 0], [1, 1]],   # M2: A10 + A11
        [[1, 0], [0, 0]],   # M3: A00
        [[0, 0], [0, 1]],   # M4: A11
        [[1, 1], [0, 0]],   # M5: A00 + A01
        [[-1, 0], [1, 0]],  # M6: -A00 + A10
        [[0, 1], [0, -1]]   # M7: A01 - A11
    ]
    
    # V_coeffs[s][lj][lk] = coefficient of B_block[lj][lk] in Strassen product s
    V_coeffs = [
        [[1, 0], [0, 1]],   # M1: B00 + B11
        [[1, 0], [0, 0]],   # M2: B00
        [[0, 1], [0, -1]],  # M3: B01 - B11
        [[-1, 0], [1, 0]],  # M4: -B00 + B10
        [[0, 0], [0, 1]],   # M5: B11
        [[1, 1], [0, 0]],   # M6: B00 + B01
        [[0, 0], [1, 1]]    # M7: B10 + B11
    ]
    
    # W_coeffs[s][li][lk] = coefficient of C_block[li][lk] from Strassen product s
    W_coeffs = [
        [[1, 0], [0, 1]],   # M1: C00 += 1, C11 += 1
        [[0, 0], [1, -1]],  # M2: C10 += 1, C11 -= 1
        [[0, 1], [0, 1]],   # M3: C01 += 1, C11 += 1
        [[1, 0], [1, 0]],   # M4: C00 += 1, C10 += 1
        [[-1, 1], [0, 0]],  # M5: C00 -= 1, C01 += 1
        [[0, 0], [0, 1]],   # M6: C11 += 1
        [[1, 0], [0, 0]]    # M7: C00 += 1
    ]
    
    total_rank = 27 * 7  # 189
    
    U = [[0] * total_rank for _ in range(dim * dim)]
    V = [[0] * total_rank for _ in range(dim * dim)]
    W = [[0] * total_rank for _ in range(dim * dim)]
    
    col = 0
    for i in range(outer_blocks):
        for j in range(outer_blocks):
            for k in range(outer_blocks):
                for s in range(7):
                    # U: A block at (i, j), inner indices (li, lj)
                    for li in range(inner_size):
                        for lj in range(inner_size):
                            coeff = U_coeffs[s][li][lj]
                            if coeff != 0:
                                row_u = (i * inner_size + li) * dim + (j * inner_size + lj)
                                U[row_u][col] = coeff
                    
                    # V: B block at (j, k), inner indices (lj, lk)
                    for lj in range(inner_size):
                        for lk in range(inner_size):
                            coeff = V_coeffs[s][lj][lk]
                            if coeff != 0:
                                row_v = (j * inner_size + lj) * dim + (k * inner_size + lk)
                                V[row_v][col] = coeff
                                
                    # W: C block at (i, k), inner indices (li, lk)
                    # Trace convention: W[k*n + i, r]
                    for li in range(inner_size):
                        for lk in range(inner_size):
                            coeff = W_coeffs[s][li][lk]
                            if coeff != 0:
                                # Global row of C = i * inner_size + li
                                # Global col of C = k * inner_size + lk
                                # W index = col * n + row = (k*inner_size + lk) * 6 + (i*inner_size + li)
                                row_w = (k * inner_size + lk) * dim + (i * inner_size + li)
                                W[row_w][col] = coeff
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
