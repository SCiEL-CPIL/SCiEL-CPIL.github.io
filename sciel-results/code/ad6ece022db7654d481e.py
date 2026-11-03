"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    def make_schoolbook():
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
        return {"U": U, "V": V, "W": W}
    
    best_result = make_schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank using random search with local optimization
    # We'll try ranks from 50 up to best_rank
    
    def compute_error(U, V, W, r):
        """Compute the maximum absolute error over all (i,j,k)."""
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err = abs(s - 1)
                    if err > max_err:
                        max_err = err
        return max_err
    
    # Try a few candidate ranks with random initialization
    for r in range(50, min(best_rank, 80)):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(20):
            if time.time() > deadline:
                break
            
            np.random.seed(trial * 1000 + r)
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # Simple iterative refinement: for each term, adjust to reduce error
            # This is a very rough approach
            # A better approach would use gradient descent on the non-convex problem
            
            # Check initial error
            err = compute_error(U, V, W, r)
            if err < 1e-6:
                # Found a valid decomposition
                best_rank = r
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
        else:
            continue
        break
    
    if time.time() > deadline:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
