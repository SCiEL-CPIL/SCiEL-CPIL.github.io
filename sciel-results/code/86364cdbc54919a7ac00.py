import time

DIMENSIONS = [2, 2, 6]

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    rank = 21
    
    U_strassen = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 1], [1, -1]], # A10, A00
        [[0, 1], [-1, 0]]  # A01, A11
    ]
    
    V_strassen = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [-1, 0]], # B01, B11
        [[1, 0], [0, -1]], # B10, B00
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 1], [0, 1]]
    ]
    
    W_strassen = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 0]],
        [[0, 0], [1, 0]],
        [[0, 0], [1, 0]],
        [[0, 0], [1, 0]],
        [[-1, -1], [1, 1]],
        [[1, 0], [-1, -1]]
    ]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(7):
        for b in range(3):
            idx = r * 3 + b
            
            # U: 2x2 block repeated 3 times
            for i in range(2):
                for j in range(2):
                    U[i * 2 + j][idx] = U_strassen[r][i][j]
            
            # V: 2x2 block in slice b
            # V rows: j*2 + k. j in 0..1, k in 0..5.
            # Block b uses k in 2b..2b+1.
            for j in range(2):
                for k in range(2):
                    V[j * 6 + (2 * b + k)][idx] = V_strassen[r][j][k]
            
            # W: 2x2 block in slice b
            # W rows: k*2 + i. k in 0..5, i in 0..1.
            # Block b uses k in 2b..2b+1.
            for k in range(2):
                for i in range(2):
                    W[(2 * b + k) * 2 + i][idx] = W_strassen[r][k][i]
                    
    return {"U": U, "V": V, "W": W}
