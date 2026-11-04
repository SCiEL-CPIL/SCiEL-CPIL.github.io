import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 33 decomposition: 3 blocks of (Strassen 2x2x2 + Standard 2x1x2)
    rank = 33
    
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    # Strassen 2x2x2 factors
    S_U = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]]
    ]
    S_V = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]]
    ]
    S_W = [
        [[1, 0], [0, 1]],
        [[0, 1], [0, -1]],
        [[0, 0], [1, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]]
    ]
    
    # Standard 2x1x2 factors
    Std_U = [
        [[1], [0]],
        [[1], [0]],
        [[0], [1]],
        [[0], [1]]
    ]
    Std_V = [
        [[1, 0]],
        [[0, 1]],
        [[1, 0]],
        [[0, 1]]
    ]
    Std_W = [
        [[1, 0], [0, 0]],
        [[0, 0], [1, 0]],
        [[0, 1], [0, 0]],
        [[0, 0], [0, 1]]
    ]
    
    col = 0
    for b in range(3):
        # Strassen part
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * m + j, col] = S_U[r][i][j]
            for j in range(2):
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[j * p + k_global, col] = S_V[r][j][kp]
            for kp in range(2):
                for i in range(2):
                    k_global = 2 * b + kp
                    W[k_global * n + i, col] = S_W[r][kp][i]
            col += 1
            
        # Standard part
        for r in range(4):
            for i in range(2):
                U[i * m + 2, col] = Std_U[r][i][0]
            for kp in range(2):
                k_global = 2 * b + kp
                V[2 * p + k_global, col] = Std_V[r][0][kp]
            for kp in range(2):
                for i in range(2):
                    k_global = 2 * b + kp
                    W[k_global * n + i, col] = Std_W[r][kp][i]
            col += 1
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
