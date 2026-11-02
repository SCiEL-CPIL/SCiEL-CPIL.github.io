import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS
    # Strassen decomposition for 2x2x2
    U_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    V_strassen = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0]
    ]
    
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(5):
        for r in range(7):
            # U is the same for all blocks
            for i in range(4):
                U[i][col] = U_strassen[r][i]
            
            # V is local to block t
            # V index: j * p + k. k is 2t or 2t+1.
            # V_strassen[r] has indices 0..3 corresponding to (0,0), (0,1), (1,0), (1,1) of local B.
            # Local (j, s) -> Global (j, 2t+s) -> Index j*10 + 2t + s.
            for j in range(2):
                for s in range(2):
                    local_idx = j * 2 + s
                    global_idx = j * p + (2 * t + s)
                    V[global_idx][col] = V_strassen[r][local_idx]
                    
            # W is local to block t
            # W index: k * n + i. k is 2t or 2t+1.
            # W_strassen[r] has indices 0..3 corresponding to (0,0), (0,1), (1,0), (1,1) of local C.
            # Local (s, i) -> Global (2t+s, i) -> Index (2t+s)*2 + i.
            for s in range(2):
                for i in range(2):
                    local_idx = s * 2 + i
                    global_idx = (2 * t + s) * n + i
                    W[global_idx][col] = W_strassen[r][local_idx]
                    
            col += 1
            
    return {"U": U, "V": V, "W": W}
