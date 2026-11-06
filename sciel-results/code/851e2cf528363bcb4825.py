import time
import numpy as np

N = 18

def _odd_even_transposition(n):
    """Generate odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(n)
            for i in range(phase % 2, n - 1, 2)]

def _verify_network_fast(comparators, n=N):
    """Verify network sorts all 2^n inputs using vectorized operations."""
    num_inputs = 1 << n
    batch_size = 4096
    
    for start in range(0, num_inputs, batch_size):
        end = min(start + batch_size, num_inputs)
        count = end - start
        
        masks = np.arange(start, end, dtype=np.uint32)
        wires = np.zeros((count, n), dtype=np.int8)
        for i in range(n):
            wires[:, i] = ((masks >> i) & 1).astype(np.int8)
        
        for (i, j) in comparators:
            swap = wires[:, i] > wires[:, j]
            if np.any(swap):
                wires[swap, i] = 0
                wires[swap, j] = 1
        
        diffs = np.diff(wires, axis=1)
        if np.any(diffs < 0):
            return False
    
    return True

def _prune_network(comparators, n=N, deadline=None):
    """Prune redundant comparators."""
    if deadline is None:
        deadline = time.time() + 80
    
    comparators = list(comparators)
    
    while time.time() < deadline - 5:
        removed = False
        for idx in range(len(comparators)):
            if time.time() > deadline - 5:
                break
            
            trial = comparators[:idx] + comparators[idx+1:]
            if _verify_network_fast(trial, n):
                comparators = trial
                removed = True
                break
        if not removed:
            break
    
    return comparators

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    comps = _odd_even_transposition(N)
    comps = _prune_network(comps, N, deadline=deadline)
    
    # Final safety check
    if not _verify_network_fast(comps, N):
        # Fallback to unpruned
        comps = _odd_even_transposition(N)
    
    return comps
