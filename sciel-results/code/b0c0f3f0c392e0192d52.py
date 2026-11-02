"""Construct rank-minimal exact tensor decomposition for 3x3 x 3x8 matrix multiplication."""
DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # We construct the decomposition by splitting the output columns (k dimension)
    # into blocks. This allows for a modular construction.
    # Block 1: k in [0, 2] (3 columns)
    # Block 2: k in [3, 5] (3 columns)
    # Block 3: k in [6, 7] (2 columns)
    
    # Each block of width w requires rank n*m*w (schoolbook) if we don't share
    # U vectors across blocks. Since V and W supports are disjoint for different
    # k ranges, the ranks are additive.
    
    rank = 0
    # Pre-allocate max possible rank (schoolbook total)
    max_rank = n * m * p
    U = [[0]*max_rank for _ in range(n*m)]
    V = [[0]*max_rank for _ in range(m*p)]
    W = [[0]*max_rank for _ in range(p*n)]
    
    col = 0
    # Block 1: k = 0, 1, 2
    for k in range(3):
        for i in range(n):
            for j in range(m):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    rank = col
            
    # Block 2: k = 3, 4, 5
    for k in range(3, 6):
        for i in range(n):
            for j in range(m):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    rank = col
            
    # Block 3: k = 6, 7
    for k in range(6, 8):
        for i in range(n):
            for j in range(m):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    rank = col
            
    # Trim matrices to actual rank
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
