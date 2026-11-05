"""Matrix multiplication 3x3x5 rank-minimal exact tensor decomposition."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[(i,j),(j,k),(i,k)] = 1."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W, T):
    """Check if U,V,W exactly decompose T (within tolerance)."""
    rank = U.shape[1]
    approx = np.zeros_like(T)
    for r in range(rank):
        approx += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(T, approx, atol=1e-10)

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition, rank = n*m*p."""
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

def _round_to_halfint(x):
    """Round array to nearest half-integer."""
    return np.round(x * 2) / 2

def _try_rank(n, m, p, T, target_rank, deadline, seed=0):
    """
    Attempt to find a rank-target_rank decomposition.
    Uses structured + random initializations with L-BFGS-B.
    Returns (U, V, W) if found, else None.
    """
    rng = np.random.RandomState(seed)
    max_restarts = 8
    
    for restart in range(max_restarts):
        if time.time() > deadline - 3:
            return None
        
        if restart == 0:
            # Structured: try to exploit the Kronecker product structure
            # For 3x3x3, rank 23 is known (Strassen-like). For 3x3x5, try structured init.
            U_init = np.zeros((n*m, target_rank))
            V_init = np.zeros((m*p, target_rank))
            W_init = np.zeros((p*n, target_rank))
            # Use a subset of schoolbook columns as starting point
            sch_U, sch_V, sch_W = _schoolbook(n, m, p)
            cols_to_use = min(target_rank, n*m*p)
            # Pick evenly spaced columns from schoolbook
            idx = np.linspace(0, n*m*p-1, cols_to_use).astype(int)
            U_init[:, :cols_to_use] = sch_U[:, idx]
            V_init[:, :cols_to_use] = sch_V[:, idx]
            W_init[:, :cols_to_use] = sch_W[:, idx]
        elif restart == 1:
            # Binary-like initialization
            U_init = rng.choice([0, 1, -1], size=(n*m, target_rank))
            V_init = rng.choice([0, 1, -1], size=(m*p, target_rank))
            W_init = rng.choice([0, 1, -1], size=(p*n, target_rank))
        else:
            # Random continuous
            scale = 0.5
            U_init = rng.randn(n*m, target_rank) * scale
            V_init = rng.randn(m*p, target_rank) * scale
            W_init = rng.randn(p*n, target_rank) * scale
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def objective(params):
            U_r = params[:n*m*target_rank].reshape(n*m, target_rank)
            V_r = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_r = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            approx = np.zeros_like(T)
            for r in range(target_rank):
                approx += np.einsum('a,b,c->abc', U_r[:,r], V_r[:,r], W_r[:,r])
            diff = T - approx
            return np.sum(diff * diff)
        
        try:
            result = minimize(objective, params_init, method='L-BFGS-B',
                             options={'maxiter': 300, 'maxfun': 800, 'ftol': 1e-12})
            
            if result.fun < 1e-8:
                U_r = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_r = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_r = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                # Round to half-integers
                U_round = _round_to_halfint(U_r)
                V_round = _round_to_halfint(V_r)
                W_round = _round_to_halfint(W_r)
                
                if _verify(U_round, V_round, W_round, T):
                    return U_round, V_round, W_round
                
                # Also try integer rounding
                U_int = np.round(U_r)
                V_int = np.round(V_r)
                W_int = np.round(W_r)
                if _verify(U_int, V_int, W_int, T):
                    return U_int, V_int, W_int
        except Exception:
            pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    T = _build_tensor(n, m, p)
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as guaranteed fallback
    U_sch, V_sch, W_sch = _schoolbook(n, m, p)
    best_result = (U_sch, V_sch, W_sch)
    best_rank = schoolbook_rank
    
    # Lower bounds for matrix multiplication rank
    # For multiplying (n x m) by (m x p):
    #   rank >= n*m (from first mode)
    #   rank >= m*p (from second mode)  
    #   rank >= p*n (from third mode)
    # Also: rank >= n*p (from output matrix dimension)
    # And various other lower bounds
    lower_bound = max(n*m, m*p, p*n, n*p)
    
    # For 3x3x5: lower_bound = max(9, 15, 15, 15) = 15
    # Known results: 3x3x3 has rank 23 (or 22? actually 23 is best known upper, 19 is lower)
    # For 3x3x5, we expect rank somewhere between 15 and 45
    
    # Try ranks from lower_bound upward, but focus on promising range
    # Use a smarter search: try specific candidate ranks
    candidate_ranks = list(range(lower_bound, min(schoolbook_rank, 40)))
    
    # Also add some "known" or "plausible" ranks
    # For 3x3x3: 23 is a known upper bound
    # For 3x3x5: try around 25-35
    candidate_ranks = sorted(set(candidate_ranks))
    
    # Try each candidate rank with multiple seeds
    for target_rank in candidate_ranks:
        if time.time() > deadline - 15:
            break
        
        found = False
        for seed in range(12):
            if time.time() > deadline - 5:
                break
            
            result = _try_rank(n, m, p, T, target_rank, deadline, seed=seed + target_rank)
            if result is not None:
                U, V, W = result
                best_result = (U, V, W)
                best_rank = target_rank
                found = True
                break
        
        if found:
            # Found a decomposition at this rank, try to go lower
            continue
        else:
            # Could not find at this rank; continue to next
            continue
    
    # Final verification of best result
    U_final, V_final, W_final = best_result
    
    if not _verify(U_final, V_final, W_final, T):
        # Fallback to schoolbook if something went wrong
        U_final, V_final, W_final = _schoolbook(n, m, p)
    
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }

# EVOLVE-BLOCK-END
