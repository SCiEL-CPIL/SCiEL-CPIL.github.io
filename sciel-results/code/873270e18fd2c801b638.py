"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    rank = 66
    U = [[0] * rank for _ in range(n * m)]  # (12, 66)
    V = [[0] * rank for _ in range(m * p)]  # (24, 66)
    W = [[0] * rank for _ in range(p * n)]  # (18, 66)
    
    # Strassen 2x2x2 matrices
    strassen_U = [
        [1, 0, 1, 0, 1, -1, 0],  # (0,0)
        [0, 0, 0, 0, 1, 0, 1],   # (0,1)
        [0, 1, 0, 0, 0, 1, 0],   # (1,0)
        [1, 1, 0, 1, 0, 0, -1],  # (1,1)
    ]
    strassen_V = [
        [1, 1, 0, -1, 0, 1, 0],  # (0,0)
        [1, 0, 1, 0, 0, 1, 0],   # (0,1)
        [0, 0, 0, 1, 0, 0, 1],   # (1,0)
        [0, 0, -1, 0, 1, 0, 1],  # (1,1)
    ]
    strassen_W = [
        [1, 0, 0, -1, 1, 0, 1],  # (0,0)
        [1, 1, 0, 0, 0, 0, 0],   # (0,1)
        [0, 0, 1, 1, 0, 0, 0],   # (1,0)
        [1, 0, -1, 0, 0, 1, 1],  # (1,1)
    ]
    
    col = 0
    
    # Split m=4 into two groups: j in {0,1} and j in {2,3}
    for j_group in range(2):
        j_offset = j_group * 2  # 0 or 2
        
        # Part 1: i=0, 1x2x6, schoolbook rank 12
        for j_local in range(2):
            j = j_offset + j_local
            for k in range(6):
                U[0 * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + 0][col] = 1
                col += 1
        
        # Part 2: i in {1,2}, 2x2x6, Strassen rank 21
        for k_group in range(3):
            k_offset = k_group * 2
            for c_local in range(7):
                c = col + c_local
                for i_local in range(2):
                    i = 1 + i_local
                    for j_local in range(2):
                        j = j_offset + j_local
                        U[i * m + j][c] = strassen_U[i_local * 2 + j_local][c_local]
                for j_local in range(2):
                    j = j_offset + j_local
                    for k_local in range(2):
                        k = k_offset + k_local
                        V[j * p + k][c] = strassen_V[j_local * 2 + k_local][c_local]
                for k_local in range(2):
                    k = k_offset + k_local
                    for i_local in range(2):
                        i = 1 + i_local
                        W[k * n + i][c] = strassen_W[k_local * 2 + i_local][c_local]
            col += 7
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
