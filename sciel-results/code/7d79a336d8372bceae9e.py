"""Direct construction: rank-42 schoolbook tensor decomposition for 2x3 x 3x7 matrix mult."""
import time

DIMENSIONS = [2, 3, 7]
DEADLINE = 150.0
START_TIME = time.time()

def construct():
    # Schoolbook decomposition of rank n*m*p = 2*3*7 = 42
    # The index for W follows the convention W[k*n + i, r]
    # where k is the column index of the output C,
    # and i is the row index of the output C.
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A_{i,j} * B_{j,k} contributes to C_{i,k}
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1.0
                V[v_idx][col] = 1.0
                W[w_idx][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
