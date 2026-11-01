import time

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(p):
        # 4 terms for column k
        terms = [
            (0, 0, 0),  # (i, j, k_out): A[0,0]*B[0,k] -> C[0,k]
            (0, 1, 0),  # A[0,1]*B[1,k] -> C[0,k]
            (1, 0, 1),  # A[1,0]*B[0,k] -> C[1,k]
            (1, 1, 1),  # A[1,1]*B[1,k] -> C[1,k]
        ]
        for (i, j, i_out) in terms:
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i_out][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
