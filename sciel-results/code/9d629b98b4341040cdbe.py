import time
import numpy as np

N = 16
MAX_TIME = 58.0

def _generate_batcher_network():
    """
    Generates Batcher's Odd-Even Merge Sort network for N=16.
    Corrected to avoid redundant comparators where possible, 
    though standard Batcher is 63.
    """
    def merge(n, offset):
        if n == 1:
            return [(offset, offset + 1)]
        
        h = n // 2
        step1 = [(offset + i, offset + i + n) for i in range(n)]
        
        def strided_merge(n, offset, stride):
            if n == 1:
                return [(offset, offset + stride)]
            
            h = n // 2
            step1_s = [(offset + i * stride, offset + (n + i) * stride) for i in range(n)]
            
            even = strided_merge(h, offset, 2 * stride)
            odd = strided_merge(h, offset + stride, 2 * stride)
            
            return step1_s + even + odd
        
        even_merge = strided_merge(h, offset, 2)
        odd_merge = strided_merge(h, offset + 1, 2)
        
        return step1 + even_merge + odd_merge

    def sort(n, offset):
        if n <= 1:
            return []
        h = n // 2
        left = sort(h, offset)
        right = sort(h, offset + h)
        m = merge(h, offset)
        return left + right + m

    return sort(N, 0)

def _apply_network_vectorized(comps, inputs):
    vec = inputs.copy()
    for i, j in comps:
        min_val = np.minimum(vec[:, i], vec[:, j])
        max_val = np.maximum(vec[:, i], vec[:, j])
        vec[:, i] = min_val
        vec[:, j] = max_val
    return vec

def _is_sorted_network_vectorized(comps, bits):
    out = _apply_network_vectorized(comps, bits)
    diffs = np.diff(out, axis=1)
    return np.all(diffs >= 0)

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    n_inputs = 1 << N
    indices = np.arange(n_inputs)
    bits = (indices[:, None] >> np.arange(N)[::-1]) & 1
    
    # 1. Generate Initial Network
    initial_comps = _generate_batcher_network()
    
    # Verify
    if not _is_sorted_network_vectorized(initial_comps, bits):
        # Fallback to Odd-Even Transposition (120)
        initial_comps = []
        for i in range(N):
            if i % 2 == 0:
                for j in range(0, N - 1, 2):
                    initial_comps.append((j, j + 1))
            else:
                for j in range(1, N - 1, 2):
                    initial_comps.append((j, j + 1))
    
    best_comps = initial_comps[:]
    best_len = len(best_comps)
    
    # 2. Greedy Reduction Phase
    # Try to remove comparators in a systematic order
    # We iterate through the list and try to remove each one
    # If removal is valid, we keep it removed.
    # This is a forward greedy pass.
    
    # We might need multiple passes or a smarter order.
    # Let's try a random shuffle of indices for the greedy pass to avoid bias.
    rng = np.random.default_rng(42)
    
    # Greedy Pass 1: Random Order
    order = rng.permutation(len(best_comps))
    for idx in order:
        if time.time() > deadline - 5.0:
            break
        if len(best_comps) <= idx:
            continue
        # Try removing comparator at idx
        removed = best_comps.pop(idx)
        if _is_sorted_network_vectorized(best_comps, bits):
            best_len -= 1
        else:
            best_comps.insert(idx, removed)
            
    # 3. Local Search Phase for further reduction
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    while time.time() < deadline - 2.0:
        current_comps = best_comps[:]
        improved = False
        
        # Try random removal
        if len(current_comps) > 1:
            idx = rng.integers(0, len(current_comps))
            removed = current_comps.pop(idx)
            if _is_sorted_network_vectorized(current_comps, bits):
                best_comps = current_comps
                best_len -= 1
                improved = True
            else:
                current_comps.insert(idx, removed)
        
        # Try random replacement
        if not improved and len(current_comps) > 1:
            idx = rng.integers(0, len(current_comps))
            old_comp = current_comps[idx]
            new_comp = all_comps[rng.integers(0, len(all_comps))]
            if new_comp != old_comp:
                current_comps[idx] = new_comp
                if _is_sorted_network_vectorized(current_comps, bits):
                    best_comps = current_comps
                    improved = True
                else:
                    current_comps[idx] = old_comp
        
        # If stuck, break to save time
        if not improved:
             # Optional: try a few more times
             pass

    return best_comps
