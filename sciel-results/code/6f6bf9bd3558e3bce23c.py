"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 7 decomposition for 2x2x2 (Strassen)
    rank7_U = [
        [1, 0, 0, 1, 1, 0, 1],
        [1, 0, 0, 1, 0, 1, 0],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 1, -1, 1, -1, 0, -1]
    ]
    rank7_V = [
        [1, 1, 0, 0, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, -1],
        [0, 0, 1, 1, 0, 1, 0],
        [0, 0, 1, 1, 1, 0, 1]
    ]
    rank7_W = [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0]
    ]
    
    # Build 4x4x4 decomposition (Rank 49)
    rank444 = 49
    U444 = [[0] * rank444 for _ in range(16)]
    V444 = [[0] * rank444 for _ in range(16)]
    W444 = [[0] * rank444 for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U444[i*4+j][r] = rank7_U[i1*2+j1][r1] * rank7_U[i2*2+j2][r2]
                            for k1 in range(2):
                                for k2 in range(2):
                                    k = k1 * 2 + k2
                                    V444[j*4+k][r] = rank7_V[j1*2+k1][r1] * rank7_V[j2*2+k2][r2]
                                    W444[k*4+i][r] = rank7_W[k1*2+i1][r1] * rank7_W[k2*2+i2][r2]
    
    # Total rank for 4x4x5 is 49 (for first 4 cols) + 16 (for 5th col) = 65
    total_rank = 65
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Copy 4x4x4 part
    for row in range(16):
        for r in range(rank444):
            U[row][r] = U444[row][r]
            V[row][r] = V444[row][r]
            W[row][r] = W444[row][r]
            
    # Add schoolbook for k=4
    col = rank444
    for i in range(n):
        for j in range(m):
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
