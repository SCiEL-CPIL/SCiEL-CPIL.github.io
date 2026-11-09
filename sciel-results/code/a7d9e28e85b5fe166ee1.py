import time

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # We will use a block decomposition.
    # A (3x4) = [A1 (2x4) ; A2 (1x4)]
    # B (4x5) = [B1 (2x5) ; B2 (2x5)]
    # C = A1 B1 + A2 B2
    
    # This is just schoolbook split.
    # Let's stick to the full schoolbook for safety if I can't verify the low-rank blocks.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
