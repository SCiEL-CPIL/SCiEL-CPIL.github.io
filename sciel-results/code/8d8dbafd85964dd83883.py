import time

DIMENSIONS = [2, 2, 3]

def construct():
    start_time = time.time()
    deadline = 110  # Safety margin
    
    n, m, p = DIMENSIONS
    # Rank 11 decomposition
    rank = 11
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part for k in {0, 1}
    # U_strassen (4x7)
    u_s = [
        [1, 0, 1, 0, 1, -1, 0], # i=0, j=0
        [0, 0, 0, 0, 1, 0, 1],  # i=0, j=1
        [0, 1, 0, 0, 0, 1, 0],  # i=1, j=0
        [1, 1, 0, 1, 0, 0, -1]  # i=1, j=1
    ]
    
    # V_strassen (4x7)
    v_s = [
        [1, 1, 0, -1, 0, 1, 0], # j=0, k=0
        [0, 0, 0, 1, 0, 0, 1],  # j=0, k=1
        [0, 0, 1, 0, 0, 1, 0],  # j=1, k=0
        [1, 0, -1, 0, 1, 0, 1]  # j=1, k=1
    ]
    
    # W_strassen (4x7)
    w_s = [
        [1, 0, 0, 1, -1, 0, 1], # k=0, i=0
        [0, 0, 1, 0, 1, 0, 0],  # k=0, i=1
        [0, 1, 0, 1, 0, 0, 0],  # k=1, i=0
        [1, -1, 1, 0, 0, 1, 0]  # k=1, i=1
    ]
    
    # Fill U, V, W with Strassen factors
    for r in range(7):
        for i in range(n):
            for j in range(m):
                U[i * m + j][r] = u_s[i * m + j][r]
        
        for j in range(m):
            for k in range(2): # Only k=0,1
                V[j * p + k][r] = v_s[j * 2 + k][r]
            # k=2 remains 0
        
        for k in range(2): # Only k=0,1
            for i in range(n):
                W[k * n + i][r] = w_s[k * n + i][r]
        # k=2 remains 0
        
    # Schoolbook part for k=2
    # Terms:
    # r=7: i=0, j=0
    # r=8: i=0, j=1
    # r=9: i=1, j=0
    # r=10: i=1, j=1
    
    # U
    U[0 * m + 0][7] = 1
    U[0 * m + 1][8] = 1
    U[1 * m + 0][9] = 1
    U[1 * m + 1][10] = 1
    
    # V
    # V[j*p + k, r]
    # k=2
    V[0 * p + 2][7] = 1
    V[1 * p + 2][8] = 1
    V[0 * p + 2][9] = 1
    V[1 * p + 2][10] = 1
    
    # W
    # W[k*n + i, r]
    # k=2
    W[2 * n + 0][7] = 1
    W[2 * n + 0][8] = 1
    W[2 * n + 1][9] = 1
    W[2 * n + 1][10] = 1
    
    return {"U": U, "V": V, "W": W}
