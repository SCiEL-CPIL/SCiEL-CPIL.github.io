"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank decompositions
    np.random.seed(42)
    
    for trial_rank in range(40, 35, -1):
        if time.time() > deadline - 30:
            break
        
        for attempt in range(100):
            if time.time() > deadline - 30:
                break
            
            U_try = np.zeros((n * m, trial_rank))
            V_try = np.zeros((m * p, trial_rank))
            W_try = np.zeros((p * n, trial_rank))
            
            for c in range(trial_rank):
                U_try[:, c] = np.random.choice([-1, 0, 1], size=n * m)
                V_try[:, c] = np.random.choice([-1, 0, 1], size=m * p)
                W_try[:, c] = np.random.choice([-1, 0, 1], size=p * n)
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_try[i*m+j] * V_try[j*p+k] * W_try[k*n+i])
                        if abs(val - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = trial_rank
                best_U = U_try.tolist()
                best_V = V_try.tolist()
                best_W = W_try.tolist()
                break
        
        if best_rank == trial_rank:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
