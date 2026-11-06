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
    """Precompute all 2^n Boolean inputs as a numpy array."""
    return (np.arange(1 << n)[:, None] >> np.arange(n)[None, :]) & 1

def _apply_comparators(inputs, comparators):
    """Apply a list of comparators to the inputs matrix."""
    for i, j in comparators:
        min_val = np.minimum(inputs[:, i], inputs[:, j])
        max_val = np.maximum(inputs[:, i], inputs[:, j])
        inputs[:, i] = min_val
        inputs[:, j] = max_val
    return inputs

def _verify_network(comparators, n=N, all_inputs=None):
    """Verify network sorts all 2^n Boolean inputs using zero-one principle with NumPy."""
    if all_inputs is None:
        all_inputs = _precompute_inputs(n)
    
    inputs = all_inputs.copy()
    _apply_comparators(inputs, comparators)
    
    # Check if sorted: for each row, check if non-decreasing
    diffs = np.diff(inputs, axis=1)
    if not np.all(diffs >= 0):
        return False
    return True

def _verify_batch_remove(comparators, indices_to_remove, n=N, all_inputs=None):
    """Verify that removing the specified indices still gives a valid network."""
    if all_inputs is None:
        all_inputs = _precompute_inputs(n)
    
    # Build the candidate network without the removed indices
    remove_set = set(indices_to_remove)
    candidate = [comp for idx, comp in enumerate(comparators) if idx not in remove_set]
    
    inputs = all_inputs.copy()
    _apply_comparators(inputs, candidate)
    
    diffs = np.diff(inputs, axis=1)
    if not np.all(diffs >= 0):
        return False
    return True

def _remove_redundant(comparators, n=N, time_limit=55):
    """Try to remove redundant comparators with deadline guard."""
    start_time = time.time()
    deadline = start_time + time_limit
    current = list(comparators)
    all_inputs = _precompute_inputs(n)
    
    # First pass: greedy removal, one at a time
    while True:
        if time.time() >= deadline - 2:
            break
        
        removed = False
        for idx in range(len(current)):
            if time.time() >= deadline - 2:
                break
            
            candidate = current[:idx] + current[idx+1:]
            if _verify_network(candidate, n, all_inputs):
                current = candidate
                removed = True
                break
        
        if not removed:
            break
    
    # Second pass: try to remove pairs of non-adjacent comparators
    while True:
        if time.time() >= deadline - 2:
            break
        
        found_pair = False
        n_comps = len(current)
        
        for i in range(n_comps):
            if time.time() >= deadline - 2:
                break
            for j in range(i + 2, n_comps):
                if time.time() >= deadline - 2:
                    break
                
                if _verify_batch_remove(current, [i, j], n, all_inputs):
                    remove_set = {i, j}
                    current = [comp for idx, comp in enumerate(current) if idx not in remove_set]
                    found_pair = True
                    break
            if found_pair:
                break
        
        if not found_pair:
            break
    
    # Third pass: single removals again after pair removals
    while True:
        if time.time() >= deadline - 2:
            break
        
        removed = False
        for idx in range(len(current)):
            if time.time() >= deadline - 2:
                break
            
            candidate = current[:idx] + current[idx+1:]
            if _verify_network(candidate, n, all_inputs):
                current = candidate
                removed = True
                break
        
        if not removed:
            break
    
    return current

def construct():
    # Generate Batcher's network
    network = _batcher(N)
    
    # Use most of the time budget for optimization
    network = _remove_redundant(network, N, time_limit=85)
    
    return network
# EVOLVE-BLOCK-END
