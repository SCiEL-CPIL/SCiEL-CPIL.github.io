"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank 14 decomposition using two Strassen blocks.
    n, m, p = DIMENSIONS
    rank = 14
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # P1 = (a00+a10)(b00+b10)
    # P2 = a10(b00-b10)
    # P3 = a00(b10+b11)
    # P4 = a11(b00-b01)
    # P5 = (a00+a11)(b00+b11)
    # P6 = (a01-a11)(b10+b11)
    # P7 = (a00-a10)(b00+b01)
    
    # U rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V rows: 0:(0,0), 1:(0,1), 2:(0,2), 3:(0,3), 4:(1,0), 5:(1,1), 6:(1,2), 7:(1,3)
    # W rows: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1), 4:(0,2), 5:(1,2), 6:(0,3), 7:(1,3)
    
    # Block 1: Cols 0,1 of B -> Cols 0,1 of C (W rows 0,1,2,3)
    # U coefficients for a
    u_block = [
        [1, 0, 1, 0, 1, 0, 1], # a00
        [0, 0, 0, 0, 0, 1, 0], # a01
        [1, 1, 0, 0, 0, 0, -1],# a10
        [0, 0, 0, 1, 1, -1, 0] # a11
    ]
    
    # V coefficients for b (only b00, b01, b10, b11 used)
    v_block = [
        [1, 1, 0, 1, 1, 0, 1], # b00
        [0, 0, 0, -1, 0, 0, 1],# b01
        [0, 0, 0, 0, 0, 0, 0], # b02
        [0, 0, 0, 0, 0, 0, 0], # b03
        [1, -1, 1, 0, 0, 1, 0],# b10
        [0, 0, 1, 0, 1, 1, 0], # b11
        [0, 0, 0, 0, 0, 0, 0], # b12
        [0, 0, 0, 0, 0, 0, 0]  # b13
    ]
    
    # W coefficients for C (only C00, C10, C01, C11 used)
    # C00 = P5 + P4 - P3 - P7
    # C10 = P2 + P5
    # C01 = P1 + P7
    # C11 = P3 + P5 - P1 - P6
    w_block = [
        [0, 0, -1, 1, 1, 0, -1], # C00 (W row 0)
        [0, 1, 0, 0, 1, 0, 0],   # C10 (W row 1)
        [1, 0, 0, 0, 0, 0, 1],   # C01 (W row 2)
        [-1, 0, 1, 0, 1, -1, 0], # C11 (W row 3)
        [0, 0, 0, 0, 0, 0, 0],   # C02 (W row 4)
        [0, 0, 0, 0, 0, 0, 0],   # C12 (W row 5)
        [0, 0, 0, 0, 0, 0, 0],   # C03 (W row 6)
        [0, 0, 0, 0, 0, 0, 0]    # C13 (W row 7)
    ]
    
    # Fill U, V, W for block 1 (cols 0-6)
    for i in range(4):
        for c in range(7):
            U[i][c] = u_block[i][c]
    for i in range(8):
        for c in range(7):
            V[i][c] = v_block[i][c]
    for i in range(8):
        for c in range(7):
            W[i][c] = w_block[i][c]
            
    # Block 2: Cols 2,3 of B -> Cols 2,3 of C (W rows 4,5,6,7)
    # U is the same
    for i in range(4):
        for c in range(7):
            U[i][c + 7] = u_block[i][c]
            
    # V coefficients for b (only b02, b03, b12, b13 used)
    v_block2 = [
        [0, 0, 0, 0, 0, 0, 0],   # b00
        [0, 0, 0, 0, 0, 0, 0],   # b01
        [1, 1, 0, 1, 1, 0, 1],   # b02
        [0, 0, 0, -1, 0, 0, 1],  # b03
        [0, 0, 0, 0, 0, 0, 0],   # b10
        [0, 0, 0, 0, 0, 0, 0],   # b11
        [1, -1, 1, 0, 0, 1, 0],  # b12
        [0, 0, 1, 0, 1, 1, 0]    # b13
    ]
    
    for i in range(8):
        for c in range(7):
            V[i][c + 7] = v_block2[i][c]
            
    # W coefficients for C (only C02, C12, C03, C13 used)
    # C02 = P5 + P4 - P3 - P7
    # C12 = P2 + P5
    # C03 = P1 + P7
    # C13 = P3 + P5 - P1 - P6
    w_block2 = [
        [0, 0, 0, 0, 0, 0, 0],   # C00 (W row 0)
        [0, 0, 0, 0, 0, 0, 0],   # C10 (W row 1)
        [0, 0, 0, 0, 0, 0, 0],   # C01 (W row 2)
        [0, 0, 0, 0, 0, 0, 0],   # C11 (W row 3)
        [0, 0, -1, 1, 1, 0, -1], # C02 (W row 4)
        [0, 1, 0, 0, 1, 0, 0],   # C12 (W row 5)
        [1, 0, 0, 0, 0, 0, 1],   # C03 (W row 6)
        [-1, 0, 1, 0, 1, -1, 0]  # C13 (W row 7)
    ]
    
    for i in range(8):
        for c in range(7):
            W[i][c + 7] = w_block2[i][c]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
