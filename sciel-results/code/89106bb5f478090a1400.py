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

def _verify_network(comparators, n=N):
    """Verify network sorts all 2^n Boolean inputs using zero-one principle (Vectorized)."""
    x = np.arange(1 << n, dtype=np.uint8)
    shifts = np.arange(n, dtype=np.uint8)
    bits = (x[:, None] >> shifts[None, :]) & 1
    
    for i, j in comparators:
        tmp = bits[:, i].copy()
        bits[:, i] = np.minimum(tmp, bits[:, j])
        bits[:, j] = np.maximum(tmp, bits[:, j])
        
    return bool(np.all(bits[:, 1:] >= bits[:, :-1]))

def _remove_redundant(comparators, n=N, time_limit=55):
    """Try to remove redundant comparators using multiple strategies."""
    start_time = time.time()
    current = list(comparators)
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        L = len(current)
        if L == 0:
            break

        removed = False
        for idx in range(L):
            if time.time() - start_time > time_limit:
                return current
            
            candidate = current[:idx] + current[idx+1:]
            if _verify_network(candidate, n):
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
    network = _remove_redundant(network, N, time_limit=50)
    
    return network
# EVOLVE-BLOCK-END
