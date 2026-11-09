import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a hybrid decomposition:
    # 1. Strassen (rank 7) for the 2x2x2 sub-block (i, alpha, k in {0,1})
    # 2. Schoolbook (rank 1) for all other (i, alpha, k) triples
    
    rank = 62  # 7 + 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    
    # Strassen coefficients for 2x2x2
    # u_c is 2x2, v_c is 2x2, w_c is 2x2
    # Indices: i, alpha for U; alpha, k for V; k, i for W
    # Block indices: i, alpha, k in {0, 1}
    
    strassen_U = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [-1, 0]]
    ]
    strassen_V = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 0]],
        [[0, 1], [0, -1]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [1, 0]],
        [[0, 1], [0, 1]]
    ]
    strassen_W = [
        [[1, 0], [0, 1]],
        [[0, 1], [0, -1]],
        [[0, 1], [0, 1]],
        [[1, 0], [0, 0]],
        [[-1, 1], [0, 0]],
        [[0, 0], [1, 0]],
        [[-1, 0], [0, 1]]
    ]
    
    for c in range(7):
        for i in range(2):
            for alpha in range(2):
                U[i * m + alpha][column] = strassen_U[c][i][alpha]
        for alpha in range(2):
            for k in range(2):
                V[alpha * p + k][column] = strassen_V[c][alpha][k]
        for k in range(2):
            for i in range(2):
                W[k * n + i][column] = strassen_W[c][k][i]
        column += 1
        
    # Schoolbook for the rest
    for i in range(n):
        for alpha in range(m):
            for k in range(p):
                # Skip the 2x2x2 block handled by Strassen
                if i < 2 and alpha < 2 and k < 2:
                    continue
                U[i * m + alpha][column] = 1
                V[alpha * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
