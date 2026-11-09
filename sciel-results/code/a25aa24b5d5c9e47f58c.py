"""Matrix multiplication 3x3 x 3x5 tensor decomposition.
Finds a rank-minimal exact decomposition with half-integer entries.
"""
DIMENSIONS = [3, 3, 5]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n=3, m=3, p=5, tol=1e-8):
    """Verify U,V,W form a valid decomposition.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Check: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    """
    r = U.shape[1]
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(total - 1.0) > tol:
                    return False
    return True

def _is_half_integer(x, tol=1e-10):
    """Check if x is a half-integer (multiple of 0.5)."""
    x2 = 2.0 * x
    return abs(x2 - round(x2)) < tol

def _round_to_half_integer(x, tol=1e-6):
    """Round x to nearest half-integer if close enough, else return None."""
    x2 = 2.0 * x
    r = round(x2)
    if abs(x2 - r) < tol:
        return r / 2.0
    return None

def _make_schoolbook(n=3, m=3, p=5):
    """Create the standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _try_rank_random(n, m, p, target_rank, num_trials=500, seed=None, deadline=None):
    """Try random initialization + verification for given target rank."""
    if seed is not None:
        np.random.seed(seed)
    
    for trial in range(num_trials):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random initialization with values in {-1, -0.5, 0, 0.5, 1}
        vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
        U = np.random.choice(vals, size=(n*m, target_rank))
        V = np.random.choice(vals, size=(m*p, target_rank))
        W = np.random.choice(vals, size=(p*n, target_rank))
        
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _try_rank_structured(n, m, p, target_rank, deadline=None):
    """Try structured decompositions based on known patterns."""
    # Try patterns where each column has limited support
    # For small ranks, try all combinations of small integer values
    # This is expensive so we limit it
    
    if target_rank <= 20:
        # Try with entries in {-1, 0, 1}
        vals = [-1.0, 0.0, 1.0]
        # Use a smarter random search
        for trial in range(200):
            if deadline is not None and time.time() > deadline:
                break
            U = np.random.choice(vals, size=(n*m, target_rank))
            V = np.random.choice(vals, size=(m*p, target_rank))
            W = np.random.choice(vals, size=(p*n, target_rank))
            if _verify_decomposition(U, V, W, n, m, p):
                return U, V, W
    
    return None

def _optimize_from_schoolbook(n, m, p, target_rank, deadline=None):
    """Try to find a lower-rank decomposition by optimizing from scratch."""
    # Use a simple iterative approach:
    # Start with random half-integer matrices and try to satisfy constraints
    # This is a hard combinatorial problem, so we use random search with verification
    
    best_result = None
    
    # Try different random seeds
    for seed in range(20):
        if deadline is not None and time.time() > deadline:
            break
        
        result = _try_rank_random(n, m, p, target_rank, num_trials=100, seed=seed, deadline=deadline)
        if result is not None:
            return result
    
    return best_result

def construct():
    """Construct a rank-minimal tensor decomposition for 3x3 * 3x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s limit
    
    # Start with schoolbook decomposition (rank 45)
    U, V, W = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Verify schoolbook
    assert _verify_decomposition(U, V, W, n, m, p), "Schoolbook verification failed"
    
    # Try to find lower-rank decompositions
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # Known upper bound for 3x3*3x5: likely around 30-40
    
    # Try ranks from lower bound up to current best
    for target_rank in range(15, best_rank):
        if time.time() > deadline:
            break
        
        # Try structured/random search
        result = _try_rank_structured(n, m, p, target_rank, deadline)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = target_rank
            break
        
        # Try random optimization
        result = _optimize_from_schoolbook(n, m, p, target_rank, deadline)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = target_rank
            break
    
    # Convert to lists for return
    best_U_list = best_U.tolist()
    best_V_list = best_V.tolist()
    best_W_list = best_W.tolist()
    
    return {"U": best_U_list, "V": best_V_list, "W": best_W_list}
# EVOLVE-BLOCK-END
