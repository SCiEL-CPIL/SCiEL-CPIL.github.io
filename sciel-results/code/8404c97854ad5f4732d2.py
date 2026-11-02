"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # For each (i,j,k), check the sum
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _try_rank(r, n, m, p, max_iters=5000, seed=42):
    """Try to find a rank-r decomposition via random init + gradient descent."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n*m, r))
    V = rng.standard_normal((m*p, r))
    W = rng.standard_normal((p*n, r))
    
    lr = 0.01
    for it in range(max_iters):
        # Compute gradient
        gradU = np.zeros_like(U)
        gradV = np.zeros_like(V)
        gradW = np.zeros_like(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # d/dc of (U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)
                    for c in range(r):
                        val = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        # gradient of (val - 1) w.r.t. U[i*m+j,c]
                        gradU[i*m+j, c] += (val - 1.0) * V[j*p+k, c] * W[k*n+i, c]
                        gradV[j*p+k, c] += (val - 1.0) * U[i*m+j, c] * W[k*n+i, c]
                        gradW[k*n+i, c] += (val - 1.0) * U[i*m+j, c] * V[j*p+k, c]
        
        U -= lr * gradU
        V -= lr * gradV
        W -= lr * gradW
        
        # Check convergence
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j, c]*V[j*p+k, c]*W[k*n+i, c] for c in range(r))
                    max_err = max(max_err, abs(s - 1.0))
        if max_err < 1e-8:
            return U, V, W
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    # Best known result so far
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Based on known results, try ranks from 45 upward
    for r in range(45, min(best_rank, 55)):
        if time.time() > deadline - 10:
            break
        # Try multiple random seeds
        found = False
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            result = _try_rank(r, n, m, p, max_iters=2000, seed=seed + r*100)
            if result is not None:
                U, V, W = result
                # Verify
                if _verify(U, V, W, n, m, p):
                    # Convert to list of lists
                    U_list = [[float(U[i,j]) for j in range(r)] for i in range(n*m)]
                    V_list = [[float(V[i,j]) for j in range(r)] for i in range(m*p)]
                    W_list = [[float(W[i,j]) for j in range(r)] for i in range(p*n)]
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = r
                    found = True
                    break
        if found:
            # Found at this rank, try to go lower
            continue
    
    return best
# EVOLVE-BLOCK-END
