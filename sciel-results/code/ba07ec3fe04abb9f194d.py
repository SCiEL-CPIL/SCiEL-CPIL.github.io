"""Matrix multiplication 4x5x5 tensor decomposition."""
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns an exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    
    Uses the schoolbook (trivial) decomposition with rank = n*m*p = 100.
    The tensor identity: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    for all i in [0,4), j in [0,5), k in [0,5).
    """
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 100
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        if time.time() - start_time > deadline:
            break
        for j in range(m):
            if time.time() - start_time > deadline:
                break
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                if time.time() - start_time > deadline:
                    break
            if time.time() - start_time > deadline:
                break
        if time.time() - start_time > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
