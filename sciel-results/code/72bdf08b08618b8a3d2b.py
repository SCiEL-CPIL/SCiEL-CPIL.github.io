"""Rewritten program for matrix multiplication 3x4x5 tensor decomposition."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _try_random_rank(n, m, p, target_rank, deadline, num_trials=50, seed=42):
    """
    Attempt to find a rank-`target_rank` decomposition via random initialization
    and local refinement. Returns (U, V, W) if successful, else None.
    """
    if time.time() > deadline:
        return None
    
    rng = np.random.RandomState(seed)
    
    # The tensor has n*m*p = 60 components, each must equal 1.
    # We have 3 * target_rank free parameters (each of U, V, W has target_rank columns).
    # For target_rank < 20, we have fewer parameters than constraints, so it's likely impossible.
    # But let's try anyway for ranks down to the theoretical lower bound.
    
    # We'll use a simple local optimization approach.
    # For each trial, initialize randomly and try to minimize the sum of squared residuals.
    
    best_cost = np.inf
    best_params = None
    
    for trial in range(num_trials):
        if time.time() > deadline - 1:
            break
        
        # Random initialization
        U = rng.randn(n*m, target_rank)
        V = rng.randn(m*p, target_rank)
        W = rng.randn(p*n, target_rank)
        
        # Scale to reasonable magnitude
        U /= np.sqrt(target_rank)
        V /= np.sqrt(target_rank)
        W /= np.sqrt(target_rank)
        
        # Simple gradient-free local search
        # We'll do a few iterations of coordinate descent or random perturbation
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def residuals(params):
            U_flat = params[:n*m*target_rank]
            V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank]
            W_flat = params[n*m*target_rank + m*p*target_rank :]
            U_r = U_flat.reshape(n*m, target_rank)
            V_r = V_flat.reshape(m*p, target_rank)
            W_r = W_flat.reshape(p*n, target_rank)
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :])
                        res.append(val - 1.0)
            return np.array(res)
        
        cost = np.sum(residuals(params)**2)
        
        if cost < best_cost:
            best_cost = cost
            best_params = params.copy()
        
        # If cost is already very small, we can stop early
        if cost < 1e-10:
            break
    
    if best_cost < 1e-8 and best_params is not None:
        U_flat = best_params[:n*m*target_rank]
        V_flat = best_params[n*m*target_rank : n*m*target_rank + m*p*target_rank]
        W_flat = best_params[n*m*target_rank + m*p*target_rank :]
        U = U_flat.reshape(n*m, target_rank)
        V = V_flat.reshape(m*p, target_rank)
        W = W_flat.reshape(p*n, target_rank)
        
        if _verify(U, V, W, n, m, p, tol=1e-7):
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    best_U, best_V, best_W = _schoolbook_decomposition(n, m, p)
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = start_time + 110  # Leave safety margin for the 120s limit
    
    # Try to find a lower rank decomposition
    # Theoretical lower bound is max(n*m, m*p, p*n) = 20
    # We search from a reasonable lower bound up to schoolbook - 1
    
    # For 3x4x5, it is very unlikely we will find rank < 60 easily.
    # But let's try a few ranks to see if any work.
    
    # We'll try ranks from schoolbook-1 down to a certain point.
    # To save time, we only try a few candidates.
    
    for r in range(schoolbook_rank - 1, max(schoolbook_rank - 5, 19), -1):
        if time.time() > deadline - 5:
            break
        
        res = _try_random_rank(n, m, p, r, deadline - 5, num_trials=30, seed=r)
        if res is not None:
            best_U, best_V, best_W = res
            best_rank = r
            # If we found one, we could try to go lower, but for now,
            # let's just return the first one found to ensure we return a valid answer quickly.
            break
    
    # Convert to list of lists for output
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
