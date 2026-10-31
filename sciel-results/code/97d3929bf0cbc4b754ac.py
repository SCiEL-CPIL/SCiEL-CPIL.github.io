"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen coefficients for 2x2 matrices
    # M1 = (X00 + X11)(Y00 + Y11)
    # M2 = (X10 + X11)Y00
    # M3 = X00(Y01 - Y11)
    # M4 = X11(Y10 - Y00)
    # M5 = (X00 + X01)Y11
    # M6 = (X10 - X00)(Y00 + Y01)
    # M7 = (X01 - X11)(Y10 + Y11)
    
    # Coefficients for X and Y in each multiplication
    x_coeffs = [
        [[1, 0], [0, 1]],  # M1
        [[0, 0], [1, 1]],  # M2
        [[1, 0], [0, 0]],  # M3
        [[0, 0], [0, 1]],  # M4
        [[1, 1], [0, 0]],  # M5
        [[-1, 0], [1, 0]], # M6
        [[0, 1], [0, -1]]  # M7
    ]
    y_coeffs = [
        [[1, 0], [0, 1]],  # M1
        [[1, 0], [0, 0]],  # M2
        [[0, 1], [0, -1]], # M3
        [[0, -1], [1, 0]], # M4
        [[0, 0], [0, 1]],  # M5
        [[1, 1], [0, 0]],  # M6
        [[0, 1], [1, 1]]   # M7
    ]
    
    # Coefficients for the result Z = XY in terms of M1..M7
    z_coeffs = [
        [1, 0, 0, 1, -1, 0, 1],  # Z00
        [0, 0, 1, 0, 1, 0, 0],   # Z01
        [0, 1, 0, 1, 0, 0, 0],   # Z10
        [1, -1, 1, 0, 0, 1, 0]   # Z11
    ]
    
    # Outer Strassen block coefficients
    block_x = [[[[[0]*4 for _ in range(4)] for _ in range(2)] for _ in range(2)] for _ in range(7)]
    block_y = [[[[[0]*4 for _ in range(4)] for _ in range(2)] for _ in range(2)] for _ in range(7)]
    
    def add_block(coeff, block_idx, sign):
        if block_idx == 0:
            for i in range(2):
                for j in range(2):
                    coeff[i][j] += sign
        elif block_idx == 1:
            for i in range(2):
                for j in range(2, 4):
                    coeff[i][j] += sign
        elif block_idx == 2:
            for i in range(2, 4):
                for j in range(2):
                    coeff[i][j] += sign
        elif block_idx == 3:
            for i in range(2, 4):
                for j in range(2, 4):
                    coeff[i][j] += sign

    # s=0: M1 = (A00 + A11)(B00 + B11)
    add_block(block_x[0][0][0], 0, 1)
    add_block(block_x[0][0][0], 3, 1)
    add_block(block_y[0][0][0], 0, 1)
    add_block(block_y[0][0][0], 3, 1)
    
    # s=1: M2 = (A10 + A11)B00
    add_block(block_x[1][1][0], 2, 1)
    add_block(block_x[1][1][0], 3, 1)
    add_block(block_y[1][0][0], 0, 1)
    
    # s=2: M3 = A00(B01 - B11)
    add_block(block_x[2][0][0], 0, 1)
    add_block(block_y[2][0][1], 1, 1)
    add_block(block_y[2][1][1], 3, -1)
    
    # s=3: M4 = A11(B10 - B00)
    add_block(block_x[3][1][1], 3, 1)
    add_block(block_y[3][1][0], 2, 1)
    add_block(block_y[3][0][0], 0, -1)
    
    # s=4: M5 = (A00 + A01)B11
    add_block(block_x[4][0][0], 0, 1)
    add_block(block_x[4][0][1], 1, 1)
    add_block(block_y[4][1][1], 3, 1)
    
    # s=5: M6 = (A10 - A00)(B00 + B01)
    add_block(block_x[5][1][0], 2, 1)
    add_block(block_x[5][0][0], 0, -1)
    add_block(block_y[5][0][0], 0, 1)
    add_block(block_y[5][0][1], 1, 1)
    
    # s=6: M7 = (A01 - A11)(B10 + B11)
    add_block(block_x[6][0][1], 1, 1)
    add_block(block_x[6][1][1], 3, -1)
    add_block(block_y[6][1][0], 2, 1)
    add_block(block_y[6][1][1], 3, 1)
    
    # Precompute w_outer for each p, q, s
    w_outer = [[[0]*7 for _ in range(2)] for _ in range(2)]
    w_outer[0][0][0] = 1
    w_outer[0][0][3] = 1
    w_outer[0][0][4] = -1
    w_outer[0][0][6] = 1
    
    w_outer[0][1][2] = 1
    w_outer[0][1][4] = 1
    
    w_outer[1][0][1] = 1
    w_outer[1][0][3] = 1
    
    w_outer[1][1][0] = 1
    w_outer[1][1][1] = -1
    w_outer[1][1][2] = 1
    w_outer[1][1][5] = 1
    
    for s in range(7):
        for t in range(7):
            r = 7 * s + t
            for i in range(4):
                for j in range(4):
                    val = 0
                    for u in range(2):
                        for v in range(2):
                            val += x_coeffs[t][u][v] * block_x[s][u][v][i][j]
                    U[i * 4 + j][r] = val
            for k in range(4):
                for l in range(4):
                    val = 0
                    for w in range(2):
                        for z in range(2):
                            val += y_coeffs[t][w][z] * block_y[s][w][z][k][l]
                    V[k * 4 + l][r] = val
            for k in range(4):
                for i in range(4):
                    p = k // 2
                    q = i // 2
                    W[k * 4 + i][r] = z_coeffs[p][q][t] * w_outer[p][q][s]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
