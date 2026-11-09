"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    # Strassen U, V, W matrices for 2x2x2 (local indices)
    U_strassen = [
        [[1, 0], [0, 1]],   # M1
        [[0, 0], [1, 1]],   # M2
        [[1, 0], [0, 0]],   # M3
        [[0, 0], [0, 1]],   # M4
        [[1, 1], [0, 0]],   # M5
        [[-1, 0], [1, 0]],  # M6
        [[0, 1], [0, -1]],  # M7
    ]
    V_strassen = [
        [[1, 0], [0, 1]],   # M1
        [[1, 0], [0, 0]],   # M2
        [[0, 1], [0, -1]],  # M3
        [[0, -1], [1, 0]],  # M4
        [[0, 0], [0, 1]],   # M5
        [[1, 1], [0, 0]],   # M6
        [[0, 0], [1, 1]],   # M7
    ]
    W_strassen = [
        [[1, 0], [0, 0]],   # M1
        [[1, 0], [0, 0]],   # M2
        [[0, 0], [2, 0]],   # M3
        [[0, 2], [0, 0]],   # M4
        [[0, 0], [0, 2]],   # M5
        [[0, 0], [1, 0]],   # M6
        [[0, 1], [0, -1]],  # M7
    ]
    
    # Strassen blocks: (j_offset, k_offset)
    strassen_blocks = [
        (0, 0),  # j in {0,1}, k in {0,1}
        (0, 2),  # j in {0,1}, k in {2,3}
        (2, 0),  # j in {2,3}, k in {0,1}
        (2, 2),  # j in {2,3}, k in {2,3}
    ]
    
    col = 0
    for j_off, k_off in strassen_blocks:
        for t in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * m + (j_off + j)][col] = U_strassen[t][i][j]
            for j in range(2):
                for k in range(2):
                    V[(j_off + j) * p + (k_off + k)][col] = V_strassen[t][j][k]
            for k in range(2):
                for i in range(2):
                    W[(k_off + k) * n + i][col] = W_strassen[t][k][i]
            col += 1
    
    # Schoolbook blocks for k=4, j in {0,1} and j in {2,3}
    for j_off in [0, 2]:
        for j in range(2):
            for i in range(2):
                U[i * m + (j_off + j)][col] = 1
                V[(j_off + j) * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
