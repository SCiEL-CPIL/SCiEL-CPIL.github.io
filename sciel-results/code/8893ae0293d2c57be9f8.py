"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 12
    num_blocks = p // 2  # 6
    rank = 7 * num_blocks  # 42
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    A_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    
    B_strassen = [
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    
    C_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    
    for t in range(7):
        for block in range(num_blocks):
            col = block * 7 + t
            
            for ai in range(4):
                U[ai][col] = A_strassen[t][ai]
            
            V[2 * block][col] = B_strassen[t][0]
            V[2 * block + 1][col] = B_strassen[t][1]
            V[12 + 2 * block][col] = B_strassen[t][2]
            V[12 + 2 * block + 1][col] = B_strassen[t][3]
            
            W[4 * block][col] = C_strassen[t][0]
            W[4 * block + 1][col] = C_strassen[t][1]
            W[4 * block + 2][col] = C_strassen[t][2]
            W[4 * block + 3][col] = C_strassen[t][3]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
