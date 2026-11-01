import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # 1. Construct the schoolbook solution (Rank = n*m*p)
    def schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R), dtype=int)
        V = np.zeros((m * p, R), dtype=int)
        W = np.zeros((p * n, R), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, R

    best_U, best_V, best_W, best_R = schoolbook()
    
    # 2. Attempt to find a lower rank
    # We try a few ranks below the schoolbook rank.
    # Given the time constraint, we limit the search significantly.
    
    ranks_to_try = [best_R - 1, best_R - 2, best_R - 3]
    
    for R in ranks_to_try:
        if time.time() > deadline - 20:
            break
            
        # Try a few random restarts
        for attempt in range(3):
            if time.time() > deadline - 20:
                break
                
            # Random initialization with small integers/half-integers
            rng = np.random.default_rng(attempt * 100 + R)
            # Values: -1, -0.5, 0, 0.5, 1
            vals = [-1, -0.5, 0, 0.5, 1]
            
            U = np.array([[rng.choice(vals) for _ in range(R)] for _ in range(n * m)])
            V = np.array([[rng.choice(vals) for _ in range(R)] for _ in range(m * p)])
            W = np.array([[rng.choice(vals) for _ in range(R)] for _ in range(p * n)])
            
            # Check validity
            # For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
            valid = True
            for i in range(n):
                if not valid: break
                for j in range(m):
                    if not valid: break
                    for k in range(p):
                        s = 0
                        for r in range(R):
                            s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        if abs(s - 1) > 1e-9:
                            valid = False
                            break
            
            if valid:
                best_U, best_V, best_W, best_R = U, V, W, R
                break # Found a lower rank, stop searching this R
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
