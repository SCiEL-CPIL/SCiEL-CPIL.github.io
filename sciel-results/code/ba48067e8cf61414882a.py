import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen coefficients for 2x2 block
    # U_t is 2x2, V_t is 2x2, W_t is 2x2
    # Indices: U[i,j], V[j,k], W[k,i]
    
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
        [[1, 0], [0, 0]],
        [[0, 1], [-1, 0]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]]
    ]
    strassen_W = [
        [[1, 0], [0, 1]],
        [[0, 1], [0, -1]],
        [[0, 0], [1, 1]],
        [[-1, 1], [0, 0]],
        [[1, 0], [0, 1]],
        [[0, 0], [0, 1]],
        [[-1, 0], [1, 0]]
    ]
    
    col = 0
    # 6 blocks of 2x2
    for b in range(6):
        for t in range(7):
            # Add Strassen term t for block b
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] += strassen_U[t][i][j]
            
            for j in range(m):
                for k in range(2):
                    # k in block is j_block * 2 + k_local? No.
                    # Block b covers columns 2*b and 2*b+1.
                    # V index is j*p + k_global.
                    k_global = 2 * b + k
                    V[j * p + k_global][col] += strassen_V[t][j][k]
            
            for k in range(2):
                k_global = 2 * b + k
                for i in range(n):
                    W[k_global * n + i][col] += strassen_W[t][k][i]
            
            col += 1
            
    # Last column k=12, trivial 4 terms
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 12][col] = 1
            W[12 * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
