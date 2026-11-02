    import numpy as np
    import time
    
    DIMENSIONS = [3, 3, 8]
    START_TIME = time.time()
    DEADLINE = 120  # seconds
    
    def construct():
        n, m, p = DIMENSIONS
        rank = n * m * p
        
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    
                    U[a_idx, r] = 1
                    V[b_idx, r] = 1
                    W[c_idx, r] = 1
                    r += 1
                    
        return {"U": U, "V": V, "W": W}
    