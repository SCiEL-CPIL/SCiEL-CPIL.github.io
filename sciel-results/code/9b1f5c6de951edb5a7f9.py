import numpy as np
import time
import random

N = 18

def _fitness(comps):
    """
    Returns (is_sorted, comparator_count)
    """
    if not _is_sorted_network(comps, N):
        return False, len(comps)
    return True, len(comps)

def _is_sorted_network(comps, N):
    # Same as before
    inputs = np.arange(2**N, dtype=np.uint32)
    bits = 1 << np.arange(N)
    matrix = (inputs[:, None] & bits[None, :]) > 0
    for i, j in comps:
        min_val = np.minimum(matrix[:, i], matrix[:, j])
        max_val = np.maximum(matrix[:, i], matrix[:, j])
        matrix[:, i] = min_val
        matrix[:, j] = max_val
    diff = np.diff(matrix, axis=1)
    return np.all(diff >= 0)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Start with Odd-Even Transposition
    best_comps = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    best_comps = list(set(best_comps))
    
    # Prune first
    i = 0
    while i < len(best_comps) and time.time() < deadline:
        test_comps = best_comps[:i] + best_comps[i+1:]
        if _is_sorted_network(test_comps, N):
            best_comps = test_comps
        else:
            i += 1
            
    # Now try to improve further with random removals (might have missed some due to order)
    # Or try to add comparators and then prune more?
    # Actually, if we pruned greedily, we might be stuck in a local minimum.
    # Let's do a few random restarts of the pruning order.
    
    best_size = len(best_comps)
    
    for restart in range(10):
        if time.time() > deadline:
            break
            
        # Shuffle the current best
        current = list(best_comps)
        random.shuffle(current)
        
        # Prune
        i = 0
        while i < len(current) and time.time() < deadline:
            test_comps = current[:i] + current[i+1:]
            if _is_sorted_network(test_comps, N):
                current = test_comps
            else:
                i += 1
                
        if len(current) < best_size:
            best_comps = current
            best_size = len(current)
            
    return best_comps
