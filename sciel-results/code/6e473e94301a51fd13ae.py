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

def _construct_via_optimization(n, m, p, deadline=None):
    """
    Try to find a lower-rank decomposition using numerical optimization.
    We minimize the Frobenius norm of the residual tensor while constraining
    entries to be close to half-integers via a penalty or by rounding after.
    """
    # Theoretical lower bound is 15. Known results suggest rank can be lower than 45.
    # Let's try ranks from 15 upwards.
    
    for rank in range(15, 45):
        if deadline is not None and time.time() > deadline:
            break
            
        # We use a simple iterative refinement approach.
        # Initialize with random half-integers.
        np.random.seed(42 + rank)
        vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        # We will try to solve for U, V, W such that the tensor matches.
        # This is a non-convex problem. We use a simple gradient-free approach:
        # Random restarts with local search on the discrete space.
        
        # However, for 3x3x5, the space is large. 
        # Let's try a smarter heuristic:
        # Fix U and V, solve for W? No, it's bilinear.
        
        # Let's stick to random search but with more trials for lower ranks,
        # and perhaps a structured search for specific patterns.
        
        # For now, we rely on the random search in the main loop.
        # This function is a placeholder for more advanced methods if needed.
        pass

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
    
    # We will search from the lower bound up to the current best rank.
    # For each rank, we perform a limited number of random trials.
    
    # To improve efficiency, we can reduce the number of trials for higher ranks
    # and increase for lower ranks, or use a time-based budget.
    
    # Let's allocate time per rank.
    # Total time available: ~120s.
    # Ranks to try: 15 to 44 (30 ranks).
    # Time per rank: 120 / 30 = 4s.
    
    time_per_rank = 3.0  # seconds
    
    for target_rank in range(15, best_rank):
        if time.time() > deadline:
            break
        
        rank_start = time.time()
        rank_deadline = rank_start + time_per_rank
        
        # Try random search with a budget for this rank
        # Number of trials depends on time available
        # Each trial is fast, so we can do many.
        
        # Use a fixed number of trials based on rank
        # Lower ranks are harder, so maybe more trials?
        # But they are less likely to exist.
        # Let's use a constant number of trials per rank, adjusted by time.
        
        num_trials = 1000  # Adjust this based on performance
        
        result = _try_rank_random(n, m, p, target_rank, num_trials=num_trials, 
                                  seed=target_rank, deadline=rank_deadline)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = target_rank
            # If we find a decomposition, we can stop if we hit the lower bound
            if target_rank == 15:
                break
            # Otherwise, continue to see if we can go lower?
            # No, we are iterating from low to high.
            # So if we find one at rank R, we should check if we missed lower ones?
            # No, we iterate 15, 16, 17...
            # So the first one found is the minimal rank in this search.
            break
    
    # Convert to lists for return
    best_U_list = best_U.tolist()
    best_V_list = best_V.tolist()
    best_W_list = best_W.tolist()
    
    return {"U": best_U_list, "V": best_V_list, "W": best_W_list}
# EVOLVE-BLOCK-END
