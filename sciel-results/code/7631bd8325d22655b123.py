"""Rewritten program: optimize signed step function to minimize autoconvolution upper bound."""
import numpy as np
import time

N = 400
DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _compute_c3_bound(heights):
    """
    Compute the upper bound on C3 for a signed step function.
    
    The autoconvolution at lag k is c[k] = sum_i h[i] * h[i+k].
    The C3 upper bound is taken as the maximum absolute value over non-zero lags,
    which is the key quantity in the third autocorrelation inequality.
    """
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
    # Compute autoconvolution for lags 0..n-1
    # c[k] = sum_{i=0}^{n-1-k} h[i] * h[i+k]
    c = np.correlate(h, h, mode='full')
    # c has length 2n-1, centered at index n-1
    # c[n-1+k] = sum_i h[i]*h[i+k] for k>=0
    # c[n-1-k] = sum_i h[i]*h[i-k] for k>=0
    # For our purpose, we want c[k] for k=0,1,...,n-1
    c_lags = c[n-1:n-1+n]  # lags 0 to n-1
    
    # The C3 bound: max absolute value over non-zero lags
    # Exclude lag 0 as it's the trivial self-correlation
    if len(c_lags) > 1:
        return float(np.max(np.abs(c_lags[1:])))
    else:
        return 0.0


def _try_random(seed):
    """Try a random signed step function."""
    rng = np.random.RandomState(seed)
    h = rng.choice([-1.0, 1.0], size=N)
    return h, _compute_c3_bound(h)


def _local_search(h, rng, max_iter=500):
    """Local search: flip random elements to improve the bound."""
    h = h.copy()
    best_bound = _compute_c3_bound(h)
    
    for _ in range(max_iter):
        idx = rng.randint(N)
        h[idx] *= -1.0  # flip sign
        new_bound = _compute_c3_bound(h)
        if new_bound < best_bound:
            best_bound = new_bound
        else:
            h[idx] *= -1.0  # revert
    
    return h, best_bound


def construct():
    """
    Construct a signed step function that minimizes the C3 autoconvolution upper bound.
    
    Returns:
        np.array: Array of 400 signed step heights (±1)
    """
    start_time = time.time()
    
    # Track best solution found
    best_h = np.ones(N, dtype=np.float64)
    best_bound = _compute_c3_bound(best_h)
    
    # Phase 1: Random restarts
    seed = 42
    rng = np.random.RandomState(seed)
    
    while time.time() - start_time < DEADLINE - 20:
        # Generate random signed function
        h = rng.choice([-1.0, 1.0], size=N)
        bound = _compute_c3_bound(h)
        
        if bound < best_bound:
            best_h = h.copy()
            best_bound = bound
        
        # Local search from this random start
        h_ls, bound_ls = _local_search(h, rng, max_iter=200)
        if bound_ls < best_bound:
            best_h = h_ls.copy()
            best_bound = bound_ls
        
        seed += 1
    
    # Phase 2: More intensive local search from best found
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 5:
        rng2 = np.random.RandomState(123)
        h_final, bound_final = _local_search(best_h, rng2, max_iter=2000)
        if bound_final < best_bound:
            best_h = h_final.copy()
            best_bound = bound_final
    
    return best_h

# EVOLVE-BLOCK-END

