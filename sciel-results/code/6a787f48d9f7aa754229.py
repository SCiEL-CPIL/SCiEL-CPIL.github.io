"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 16

# EVOLVE-BLOCK-START
def _batcher(n):
    """
    Generates Batcher's Odd-Even Merge Sort network for n wires.
    """
    if n == 1:
        return []
    
    k = n // 2
    left = _batcher(k)
    right = [(i + k, j + k) for i, j in _batcher(k)]
    
    # Merge step
    merge = []
    # Odd-even merge
    def _merge(k):
        if k == 1:
            return [(0, 1)]
        k_half = k // 2
        even_comps = []
        for i, j in _merge(k_half):
            mi = 2 * i if i < k_half else 2 * (i - k_half) + k
            mj = 2 * j if j < k_half else 2 * (j - k_half) + k
            even_comps.append((mi, mj))
        
        odd_comps = []
        for i, j in _merge(k_half):
            mi = 2 * i + 1 if i < k_half else 2 * (i - k_half) + k + 1
            mj = 2 * j + 1 if j < k_half else 2 * (j - k_half) + k + 1
            odd_comps.append((mi, mj))
        
        final_comps = [(2 * i + 1, 2 * i + 2) for i in range(k - 1)]
        
        return even_comps + odd_comps + final_comps

    merge = _merge(k)
    return left + right + merge

def _verify_network(comparators, n=N):
    """Verify network sorts all 2^n Boolean inputs using zero-one principle."""
    # Vectorized verification
    masks = np.arange(1 << n)
    wires = np.zeros((1 << n, n), dtype=np.uint8)
    for i in range(n):
        wires[:, i] = (masks >> i) & 1
    
    for i, j in comparators:
        # If wires[:, i] > wires[:, j], swap
        swap_mask = wires[:, i] > wires[:, j]
        wires[swap_mask, i] = 1 - wires[swap_mask, i]
        wires[swap_mask, j] = 1 - wires[swap_mask, j]
    
    # Check if sorted
    # A binary array is sorted if it is non-decreasing
    # i.e., no 1 followed by 0
    diffs = np.diff(wires, axis=1)
    if np.any(diffs == -1):
        return False
    return True

def _remove_redundant(comparators, n=N, deadline=None):
    """Try to remove redundant comparators using multiple strategies."""
    import random
    if deadline is None:
        deadline = time.time() + 80  # Safety margin for 90s limit
    
    start_time = time.time()
    current = list(comparators)
    
    def try_remove(indices_to_try):
        nonlocal current
        for idx in indices_to_try:
            if time.time() > deadline:
                return False
            if idx >= len(current):
                continue
            candidate = current[:idx] + current[idx+1:]
            if _verify_network(candidate, n):
                current = candidate
                return True
        return False

    while True:
        if time.time() > deadline:
            break
        
        progress = False
        L = len(current)
        if L == 0:
            break

        # Strategy 1: Forward pass
        if try_remove(range(L)):
            progress = True
        
        # Strategy 2: Backward pass
        if try_remove(range(L - 1, -1, -1)):
            progress = True
            
        # Strategy 3: Random pass
        indices = list(range(L))
        random.shuffle(indices)
        if try_remove(indices):
            progress = True

        if not progress:
            break
    
    return current

def _fallback_network(n=16):
    """Generate a simple, known-correct network (Odd-Even Transposition)."""
    comps = []
    for i in range(n):
        for j in range(0, n - 1, 2):
            if (i + j) % 2 == 0:
                comps.append((j, j + 1))
            else:
                comps.append((j + 1, j + 2)) if j + 2 < n else None
    # Correct Odd-Even Transposition
    comps = []
    for step in range(n):
        if step % 2 == 0:
            for j in range(0, n - 1, 2):
                comps.append((j, j + 1))
        else:
            for j in range(1, n - 1, 2):
                comps.append((j, j + 1))
    return comps

def construct():
    # Generate Batcher's network
    try:
        network = _batcher(N)
    except Exception:
        network = _fallback_network(N)
    
    # Ensure we have a valid start
    if not network or not _verify_network(network, N):
        network = _fallback_network(N)
    
    # Try to remove redundant comparators
    # Use a deadline that leaves 10s for the evaluator's overhead
    network = _remove_redundant(network, N, deadline=time.time() + 80)
    
    # Final check
    if not network or not _verify_network(network, N):
        network = _fallback_network(N)
        
    return network
# EVOLVE-BLOCK-END
