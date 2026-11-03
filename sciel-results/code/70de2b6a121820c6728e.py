"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Strassen-based decomposition for 2x4 * 4x8 matrix multiplication.
    Rank: 56 (vs schoolbook 64)
    """
    import time
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Strassen's 7 products for 2x2 * 2x2
    # X = [[x0, x1], [x2, x3]], Y = [[y0, y1], [y2, y3]]
    # Z = X*Y
    # M1 = (x0+x3)*(y0+y3)
    # M2 = (x2+x3)*y0
    # M3 = x0*(y1-y3)
    # M4 = x3*(y2-y0)
    # M5 = (x0+x1)*y3
    # M6 = (x2-x0)*(y0+y1)
    # M7 = (x1-x3)*(y2+y3)
    # z00 = M1 + M4 - M5 + M7
    # z01 = M3 + M5
    # z10 = M2 + M4
    # z11 = M1 - M2 + M3 + M6
    
    strassen_products = [
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),   # M1
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, 0)),   # M2
        ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 1)),  # M3
        ((0, 0, 0, 1), (0, 0, 1, -1), (1, 0, 1, 0)),  # M4
        ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)),  # M5
        ((-1, 0, 1, 0), (1, 1, 0, 0), (0, 0, 0, 1)),  # M6
        ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0)),  # M7
    ]
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]   # 8 rows
    V = [[0] * rank for _ in range(m * p)]   # 32 rows
    W = [[0] * rank for _ in range(p * n)]   # 16 rows
    
    column = 0
    
    for a_side in ['left', 'right']:
        for b_side in ['top', 'bot']:
            for col_block in range(4):
                if time.time() > deadline:
                    break
                a_offset = 0 if a_side == 'left' else 2
                b_offset = 0 if b_side == 'top' else 2
                
                for (x_coeffs, y_coeffs, z_coeffs) in strassen_products:
                    for xi in range(2):
                        for xj in range(2):
                            coeff = x_coeffs[xi * 2 + xj]
                            if coeff != 0:
                                U[xi * 4 + (a_offset + xj)][column] = coeff
                    
                    for yi in range(2):
                        for yj in range(2):
                            coeff = y_coeffs[yi * 2 + yj]
                            if coeff != 0:
                                V[(b_offset + yi) * 8 + (col_block * 2 + yj)][column] = coeff
                    
                    for zi in range(2):
                        for zj in range(2):
                            coeff = z_coeffs[zi * 2 + zj]
                            if coeff != 0:
                                W[(col_block * 2 + zj) * 2 + zi][column] = coeff
                    
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
