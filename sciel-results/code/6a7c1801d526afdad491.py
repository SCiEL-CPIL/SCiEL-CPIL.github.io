import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n, m, p = DIMENSIONS
    rank = 189
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients
    alpha = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    beta = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    gamma = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    def get_idx_A(r, c, b):
        if b == 0: return r * 6 + c
        if b == 1: return r * 6 + (c + 3)
        if b == 2: return (r + 3) * 6 + c
        if b == 3: return (r + 3) * 6 + (c + 3)
        
    def get_idx_B(r, c, b):
        if b == 0: return r * 6 + c
        if b == 1: return r * 6 + (c + 3)
        if b == 2: return (r + 3) * 6 + c
        if b == 3: return (r + 3) * 6 + (c + 3)
        
    def get_idx_C(r, c, b):
        if b == 0: return r * 6 + c
        if b == 1: return r * 6 + (c + 3)
        if b == 2: return (r + 3) * 6 + c
        if b == 3: return (r + 3) * 6 + (c + 3)

    for i in range(7):
        if time.time() > deadline:
            break
        for r in range(3):
            for i_sub in range(3):
                for s in range(3):
                    k = r * 9 + i_sub * 3 + s
                    c = i * 27 + k
                    
                    for b in range(4):
                        if alpha[i][b] != 0:
                            idx = get_idx_A(r, i_sub, b)
                            U[idx, c] = alpha[i][b]
                            
                    for b in range(4):
                        if beta[i][b] != 0:
                            idx = get_idx_B(i_sub, s, b)
                            V[idx, c] = beta[i][b]
                            
                    for b in range(4):
                        if gamma[i][b] != 0:
                            idx = get_idx_C(r, s, b)
                            W[idx, c] = gamma[i][b]
                            
    return {"U": U, "V": V, "W": W}
