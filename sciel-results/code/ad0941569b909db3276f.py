import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Rank 55 decomposition
    # Part 1: 5 Strassen blocks for 2x2x2 (rank 7 each) -> 35
    # Part 2: 20 schoolbook terms for 2x1x10 (rank 1 each) -> 20
    rank = 55
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients
    LA = [
        [1, 0, 0, 1, 0, 0], # a+d
        [0, 0, 1, 1, 0, 0], # c+d
        [1, 0, 0, 0, 0, 0], # a
        [0, 0, 0, 1, 0, 0], # d
        [1, 1, 0, 0, 0, 0], # a+b
        [0, 0, 1, -1, 0, 0],# c-a
        [0, 1, 0, -1, 0, 0] # b-d
    ]
    
    LB = [
        [1, 1, 0, 0], # e+h
        [1, 0, 0, 0], # e
        [0, 1, 0, -1],# f-h
        [0, -1, 1, 0],# g-e
        [0, 0, 0, 1], # h
        [1, 1, 0, 0], # e+f
        [0, 0, 1, 1]  # g+h
    ]
    
    W_strassen = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, -1]],
        [[0, 1], [0, 1]],
        [[1, 0], [1, 0]],
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]]
    ]
    
    for s in range(5):
        k0 = 2 * s
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * 2 + j][col] = LA[r][i * 2 + j]
            
            for j in range(2):
                for k_local in range(2):
                    k_global = k0 + k_local
                    V[j * 10 + k_global][col] = LB[r][j * 2 + k_local]
            
            for i in range(2):
                for k_local in range(2):
                    k_global = k0 + k_local
                    W[k_global * 2 + i][col] = W_strassen[r][i][k_local]
            
            col += 1
            
    for k in range(10):
        for i in range(2):
            U[i * 3 + 2][col] = 1
            V[2 * 10 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
