import numpy as np
import time

def construct():
    n, m, p = 3, 4, 6
    rank_max = n * m * p  # 72
    deadline = time.time() + 110  # 10 second safety margin
    
    # Try to find a low-rank decomposition
    best_rank = rank_max
    best_result = None
    
    # First, set up the schoolbook as fallback
    U_fb = [[0] * rank_max for _ in range(n * m)]
    V_fb = [[0] * rank_max for _ in range(m * p)]
    W_fb = [[0] * rank_max for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Try random search for lower rank
    for r in range(20, rank_max):
        if time.time() > deadline:
            break
        for attempt in range(100):
            if time.time() > deadline:
                break
            # Random initialization with small half-integer values
            U = np.random.choice([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], size=(n*m, r))
            V = np.random.choice([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], size=(m*p, r))
            W = np.random.choice([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], size=(p*n, r))
            
            # Check if this is an exact decomposition
            ok = True
            for i in range(n):
                if not ok:
                    break
                for j in range(m):
                    if not ok:
                        break
                    for k in range(p):
                        val = 0
                        for c in range(r):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if abs(val - 1) > 1e-9:
                            ok = False
                            break
            
            if ok:
                best_rank = r
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
        if best_rank < rank_max:
            break
    
    return best_result
