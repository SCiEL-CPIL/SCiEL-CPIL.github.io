"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 16

# EVOLVE-BLOCK-START
def _merge(k):
    """
    Generates comparators to merge two sorted lists of length k.
    The lists are assumed to be at indices 0..k-1 and k..2k-1.
    """
    if k == 1:
        return [(0, 1)]
    
    k_half = k // 2
    
    # 1. Merge even-indexed elements
    even_comps = []
    for i, j in _merge(k_half):
        mi = 2 * i if i < k_half else 2 * (i - k_half) + k
        mj = 2 * j if j < k_half else 2 * (j - k_half) + k
        even_comps.append((mi, mj))
        
    # 2. Merge odd-indexed elements
    odd_comps = []
    for i, j in _merge(k_half):
        mi = 2 * i + 1 if i < k_half else 2 * (i - k_half) + k + 1
        mj = 2 * j + 1 if j < k_half else 2 * (j - k_half) + k + 1
        odd_comps.append((mi, mj))
        
    # 3. Final comparisons between adjacent pairs (2i+1, 2i+2)
    final_comps = [(2 * i + 1, 2 * i + 2) for i in range(k - 1)]
    
    return even_comps + odd_comps + final_comps

def _batcher(n):
    """
    Generates Batcher's Odd-Even Merge Sort network for n wires.
    """
    if n == 1:
        return []
    
    k = n // 2
    left = _batcher(k)
    right = [(i + k, j + k) for i, j in _batcher(k)]
    merge = _merge(k)
    
    return left + right + merge

def _precompute_inputs(n=N):
    """Precompute all 2^n Boolean inputs as a uint8 matrix for fast verification."""
    return (np.arange(1 << n, dtype=np.int32)[:, None] >> np.arange(n, dtype=np.int32)[None, :]) & 1

def _verify_network(comparators, n=N, inputs=None):
    """Verify network sorts all 2^n Boolean inputs using zero-one principle with NumPy."""
    if inputs is None:
        inputs = _precompute_inputs(n)
    
    # Work on a copy
    state = inputs.copy()
    
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
        
    # Check if sorted: each row should be non-decreasing
    diffs = np.diff(state, axis=1)
    return bool(np.all(diffs >= 0))

def _verify_batch_remove(comparators, n=N, inputs=None):
    """
    Verify that removing a set of comparators still yields a valid sorting network.
    Uses a single pass through the remaining comparators.
    """
    if inputs is None:
        inputs = _precompute_inputs(n)
    
    state = inputs.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
        
    diffs = np.diff(state, axis=1)
    return bool(np.all(diffs >= 0))

def _remove_redundant(comparators, n=N, time_limit=55):
    """
    Aggressively remove redundant comparators.
    Strategy:
    1. Precompute inputs once.
    2. Repeatedly scan and remove any single comparator that can be dropped.
    3. Use a priority order: try comparators in the middle of the network first
       (they're more likely to be redundant), then edges.
    4. After no single removal works, try pairs of adjacent removals.
    """
    start_time = time.time()
    current = list(comparators)
    inputs = _precompute_inputs(n)
    
    def elapsed():
        return time.time() - start_time
    
    # Phase 1: Greedy single-removal, scanning from middle outward
    while True:
        if elapsed() > time_limit:
            break
        
        removed = False
        n_comps = len(current)
        
        # Build order: middle indices first, then edges
        if n_comps > 0:
            mid = n_comps // 2
            order = list(range(mid, n_comps)) + list(range(mid - 1, -1, -1))
        else:
            break
        
        for idx in order:
            if elapsed() > time_limit:
                break
            
            candidate = current[:idx] + current[idx+1:]
            if _verify_batch_remove(candidate, n, inputs):
                current = candidate
                removed = True
                break
        
        if not removed:
            break
    
    # Phase 2: Try removing pairs of adjacent comparators
    while True:
        if elapsed() > time_limit:
            break
        
        removed = False
        n_comps = len(current)
        
        # Try removing pairs (i, i+1) for all i
        for i in range(n_comps - 1):
            if elapsed() > time_limit:
                break
            
            candidate = current[:i] + current[i+2:]
            if _verify_batch_remove(candidate, n, inputs):
                current = candidate
                removed = True
                break
        
        if not removed:
            break
    
    # Phase 3: Final single-removal pass (in case phase 2 opened up new singles)
    while True:
        if elapsed() > time_limit:
            break
        
        removed = False
        n_comps = len(current)
        
        for idx in range(n_comps):
            if elapsed() > time_limit:
                break
            
            candidate = current[:idx] + current[idx+1:]
            if _verify_batch_remove(candidate, n, inputs):
                current = candidate
                removed = True
                break
        
        if not removed:
            break
    
    return current

def construct():
    # Generate Batcher's network
    network = _batcher(N)
    
    # Try to remove redundant comparators
    network = _remove_redundant(network, N, time_limit=55)
    
    return network
# EVOLVE-BLOCK-END
