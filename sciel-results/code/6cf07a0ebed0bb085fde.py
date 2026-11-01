import time

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Constructs the tensor decomposition for matrix multiplication 2x3 by 3x10.
    Uses the schoolbook decomposition which has rank n*m*p.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook Rank
    rank = n * m * p
    
    # Initialize matrices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][col] = 1
                # V index: j*p + k
                V[j * p + k][col] = 1
                # W index: k*n + i (Official trace convention W[k*n+i, r])
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
