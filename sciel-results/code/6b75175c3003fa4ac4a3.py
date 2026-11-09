"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _overlap_max(h):
    """Compute max overlap of h with shifted (1-h)."""
    n = len(h)
    max_ov = 0.0
    for k in range(n):
        shifted = np.roll(h, k)
        ov = (2.0 / n) * np.dot(h, (1.0 - shifted))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _autocorr_profile(h):
    """Compute autocorrelation for all lags."""
    n = len(h)
    ac = np.zeros(n)
    for k in range(n):
        ac[k] = np.dot(h, np.roll(h, k))
    return ac

def _min_ac_excl_zero(ac):
    """Minimum autocorrelation excluding lag 0."""
    return np.min(ac[1:])

def _construct_pattern():
    """Construct a binary sequence with good autocorrelation properties."""
    # Use a pattern based on evenly spaced ones with perturbation
    h = np.zeros(N)
    num_ones = N // 2
    
    # Start with evenly spaced
    spacing = N // num_ones
    for i in range(num_ones):
        h[i * spacing] = 1.0
    
    return h

def _local_search_binary(h, deadline):
    """Local search: swap 1s and 0s to maximize min autocorrelation."""
    n = len(h)
    h = h.copy()
    
    # Compute initial autocorrelation
    ac = _autocorr_profile(h)
    best_min_ac = _min_ac_excl_zero(ac)
    best_h = h.copy()
    
    start_time = time.time()
    iters = 0
    
    while time.time() < deadline:
        iters += 1
        
        # Random swap
        ones = np.where(h == 1)[0]
        zeros = np.where(h == 0)[0]
        
        if len(ones) == 0 or len(zeros) == 0:
            break
        
        i = np.random.randint(len(ones))
        j = np.random.randint(len(zeros))
        pi = ones[i]
        pj = zeros[j]
        
        if pi == pj:
            continue
        
        # Perform swap
        h[pi] = 0
        h[pj] = 1
        
        # Recompute autocorrelation (O(N^2) but N=100 so it's 10000 ops)
        ac = _autocorr_profile(h)
        min_ac = _min_ac_excl_zero(ac)
        
        if min_ac > best_min_ac:
            best_min_ac = min_ac
            best_h = h.copy()
        else:
            # Revert
            h[pi] = 1
            h[pj] = 0
    
    return best_h

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Construct initial pattern
    h = _construct_pattern()
    
    # Verify initial constraints
    assert len(h) == N
    assert np.all(h >= 0) and np.all(h <= 1)
    assert abs(np.sum(h) - N/2) < 1e-9
    
    # Local search to improve
    best_h = _local_search_binary(h, deadline)
    
    # Final verification
    assert len(best_h) == N
    assert np.all(best_h >= 0) and np.all(best_h <= 1)
    assert abs(np.sum(best_h) - N/2) < 1e-9
    
    return best_h
# EVOLVE-BLOCK-END

