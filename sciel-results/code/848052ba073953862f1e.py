"""Search for low-rank tensor decomposition of 5x5x5 matrix multiplication."""
import numpy as np
import time
import random

DIMENSIONS = [5, 5, 5]
DEADLINE = time.time() + 140

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    best_rank = n * m * p
    best_decomp = _schoolbook(n, m, p)
    
    rng = np.random.default_rng(42)
    random.seed(42)
    
    # Try to find decomposition with rank < 125
    for target_rank in range(49, best_rank):
        if time.time() > DEADLINE:
            break
        result = _try_random_decomposition(n, m, p, target_rank, rng)
        if result is not None:
            best_rank = target_rank
            best_decomp = result
            break
    
    return best_decomp

def _schoolbook(n, m, p):
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

def _try_random_decomposition(n, m, p, target_rank, rng):
    """Try random sparse init + alternating minimization for exact rank-r decomposition."""
    U = np.zeros((n * m, target_rank), dtype=float)
    V = np.zeros((m * p, target_rank), dtype=float)
    W = np.zeros((p * n, target_rank), dtype=float)
    
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() > DEADLINE:
            return None
        
        # Random sparse init with ±1 entries
        for c in range(target_rank):
            # Each column has sparse nonzero patterns
            for i in range(n):
                for j in range(m):
                    if rng.random() < 0.3:
                        U[i*m+j, c] = rng.choice([-1.0, 1.0])
            for j in range(m):
                for k in range(p):
                    if rng.random() < 0.3:
                        V[j*p+k, c] = rng.choice([-1.0, 1.0])
            for k in range(p):
                for i in range(n):
                    if rng.random() < 0.3:
                        W[k*n+i, c] = rng.choice([-1.0, 1.0])
        
        # Alternating minimization
        for step in range(100):
            if time.time() > DEADLINE:
                return None
            U, V, W = _alt_min_step(U, V, W, n, m, p, target_rank)
        
        if _check_exact(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _alt_min_step(U, V, W, n, m, p, r):
    """One alternating minimization step."""
    U_new = U.copy()
    V_new = V.copy()
    W_new = W.copy()
    
    for c in range(r):
        # Update U[:, c] given V and W
        for i in range(n):
            for j in range(m):
                num = 0.0
                den = 0.0
                for k in range(p):
                    num += V[j*p+k, c] * W[k*n+i, c]
                    den += (V[j*p+k, c] * W[k*n+i, c]) ** 2
                if abs(den) > 1e-12:
                    U_new[i*m+j, c] = num / den
                # else keep old value
        
        # Update V[:, c] given U and W
        for j in range(m):
            for k in range(p):
                num = 0.0
                den = 0.0
                for i in range(n):
                    num += U[i*m+j, c] * W[k*n+i, c]
                    den += (U[i*m+j, c] * W[k*n+i, c]) ** 2
                if abs(den) > 1e-12:
                    V_new[j*p+k, c] = num / den
        
        # Update W[:, c] given U and V
        for k in range(p):
            for i in range(n):
                num = 0.0
                den = 0.0
                for j in range(m):
                    num += U[i*m+j, c] * V[j*p+k, c]
                    den += (U[i*m+j, c] * V[j*p+k, c]) ** 2
                if abs(den) > 1e-12:
                    W_new[k*n+i, c] = num / den
    
    return U_new, V_new, W_new

def _check_exact(U, V, W, n, m, p):
    """Check if decomposition exactly reproduces T[i,j,k]=1 for all i,j,k."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True
# EVOLVE-BLOCK-END
