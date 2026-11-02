"""Strassen-based rank-14 decomposition for 2x2 * 2x4 matrix multiplication."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 4
    # Block 1: 2x2 * 2x2, rank 7
    # Block 2: 2x2 * 2x2, rank 7
    # Total rank 14
    rank = 14
    U = [[0] * rank for _ in range(n * m)]   # 4 x 14
    V = [[0] * rank for _ in range(m * p)]   # 8 x 14
    W = [[0] * rank for _ in range(p * n)]   # 8 x 14

    # Strassen's 7 products for a 2x2 * 2x2 block
    # A = [[a00, a01], [a10, a11]]  (2x2)
    # B = [[b00, b01], [b10, b11]]  (2x2)
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a10)(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # U rows: a00=0, a01=1, a10=2, a11=3
    # V rows for block 1: b00=0, b01=1, b10=2, b11=3
    # V rows for block 2: b00=4, b01=5, b10=6, b11=7
    # W rows: C00_block1=0, C01_block1=1, C10_block1=2, C11_block1=3
    #          C00_block2=4, C01_block2=5, C10_block2=6, C11_block2=7

    # Strassen U coefficients (for each of the 7 products):
    # M1: a00=1, a01=0, a10=0, a11=1
    # M2: a00=0, a01=0, a10=1, a11=1
    # M3: a00=1, a01=0, a10=0, a11=0
    # M4: a00=0, a01=0, a10=0, a11=1
    # M5: a00=1, a01=1, a10=0, a11=0
    # M6: a00=-1, a01=0, a10=1, a11=0
    # M7: a00=0, a01=1, a10=-1, a11=0

    strassen_U = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, -1, 0],  # M7
    ]

    # Strassen V coefficients for block (rows b00,b01,b10,b11):
    # M1: b00=1, b01=0, b10=0, b11=1
    # M2: b00=1, b01=0, b10=0, b11=0
    # M3: b00=0, b01=0, b10=1, b11=-1
    # M4: b00=-1, b01=1, b10=0, b11=0
    # M5: b00=0, b01=0, b10=0, b11=1
    # M6: b00=1, b01=1, b10=0, b11=0
    # M7: b00=0, b01=0, b10=1, b11=1

    strassen_V = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 0, 1, -1],  # M3
        [-1, 1, 0, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ]

    # Strassen W coefficients (rows C00, C01, C10, C11):
    # M1: 1, 0, 0, 1
    # M2: 0, 0, 1, -1
    # M3: 0, 1, 0, 1
    # M4: 1, 0, 1, 0
    # M5: -1, 1, 0, 0
    # M6: 0, 0, 0, 1
    # M7: 1, 0, 0, 0

    strassen_W = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],   # M3
        [1, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0],   # M7
    ]

    # Fill block 1 (columns 0-6)
    for r in range(7):
        # U
        for i in range(4):
            U[i][r] = strassen_U[r][i]
        # V (rows 0-3)
        for j in range(4):
            V[j][r] = strassen_V[r][j]
        # W (rows 0-3)
        for k in range(4):
            W[k][r] = strassen_W[r][k]

    # Fill block 2 (columns 7-13)
    for r in range(7):
        col = 7 + r
        # U (same structure)
        for i in range(4):
            U[i][col] = strassen_U[r][i]
        # V (rows 4-7)
        for j in range(4):
            V[4 + j][col] = strassen_V[r][j]
        # W (rows 4-7)
        for k in range(4):
            W[4 + k][col] = strassen_W[r][k]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
