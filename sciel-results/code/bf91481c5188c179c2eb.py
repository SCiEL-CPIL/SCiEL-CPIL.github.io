"""
Strategy:
1. Start with Batcher's Odd-Even Merge Sort network for N=16 (63 comparators).
2. Use a structured greedy pruning approach:
   - For each comparator, check if removing it still yields a valid sorting network.
   - If yes, remove it and repeat (greedy removal).
   - Use vectorized validity check for speed.
3. Additionally, try local search with comparator replacement to escape local minima.
4. Return the best valid network found within time limit.
"""

import time
import numpy as np

N = 16
MAX_TIME = 55.0  # Safety margin for 60s limit

def _generate_batcher_network():
    """
    Generates Batcher's Odd-Even Merge Sort network for N=16.
    """
    def strided_merge(n, offset, stride):
        if n == 1:
            return [(offset, offset + stride)]
        
        h = n // 2
        # Compare elements that are 'stride' apart, starting from offset
        step1 = [(offset + i * stride, offset + (n + i) * stride) for i in range(n)]
        even = strided_merge(h, offset, 2 * stride)
        odd = strided_merge(h, offset + stride, 2 * stride)
        return step1 + even + odd
    
    def merge(n, offset):
        if n == 1:
            return [(offset, offset + 1)]
        h = n // 2
        # Step 1: Compare (offset+i, offset+i+n) for i in range(n)
        step1 = [(offset + i, offset + i + n) for i in range(n)]
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
    """
    Apply the network to a batch of inputs.
    """
    vec = inputs.copy()
    for i, j in comps:
        min_val = np.minimum(vec[:, i], vec[:, j])
        max_val = np.maximum(vec[:, i], vec[:, j])
        vec[:, i] = min_val
        vec[:, j] = max_val
    return vec

def _is_sorted_network_vectorized(comps, bits):
    """
    Check if the network sorts all 2^16 boolean inputs.
    bits: pre-computed input matrix of shape (65536, 16)
    """
    out = _apply_network_vectorized(comps, bits)
    diffs = np.diff(out, axis=1)
    return np.all(diffs >= 0)

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # Pre-compute all boolean inputs
    n_inputs = 1 << N
    indices = np.arange(n_inputs)
    bits = (indices[:, None] >> np.arange(N)[::-1]) & 1
    
    # 1. Generate initial network (Batcher's, 63 comparators)
    best_comps = _generate_batcher_network()
    
    # Verify initial network
    if not _is_sorted_network_vectorized(best_comps, bits):
        # Fallback: Odd-Even Transposition (120 comps)
        best_comps = []
        for i in range(N):
            if i % 2 == 0:
                for j in range(0, N - 1, 2):
                    best_comps.append((j, j + 1))
            else:
                for j in range(1, N - 1, 2):
                    best_comps.append((j, j + 1))
        if not _is_sorted_network_vectorized(best_comps, bits):
            return best_comps
            
    best_len = len(best_comps)
    
    # 2. Greedy Pruning: Try to remove comparators one by one
    # We'll do multiple passes since removing one comparator might make another removable
    current_comps = best_comps[:]
    current_len = best_len
    
    # Greedy pruning with multiple passes
    max_passes = 20
    for p in range(max_passes):
        if time.time() > deadline - 5.0:
            break
        removed_any = False
        new_comps = current_comps[:]
        # Try removing each comparator
        for i in range(len(new_comps)):
            if time.time() > deadline - 5.0:
                break
            # Temporarily remove comparator i
            candidate = new_comps[:i] + new_comps[i+1:]
            if _is_sorted_network_vectorized(candidate, bits):
                # Remove it
                new_comps = candidate
                removed_any = True
                # Restart the loop since indices changed
                break
        if not removed_any:
            # No more removals possible in this pass
            break
        current_comps = new_comps
        current_len = len(current_comps)
        if current_len < best_len:
            best_comps = current_comps[:]
            best_len = current_len
    
    # 3. Local Search: Try replacing comparators to find better structures
    # This can help escape local minima in the pruning process
    rng = np.random.default_rng(42)
    
    # Number of local search iterations
    max_iterations = 5000
    
    for it in range(max_iterations):
        if time.time() > deadline - 2.0:
            break
        
        # Try a random replacement
        if len(current_comps) > 0:
            idx = rng.integers(0, len(current_comps))
            i = rng.integers(0, N)
            j = rng.integers(0, N)
            if i == j:
                continue
            if i > j:
                i, j = j, i
                
            candidate = current_comps[:]
            candidate[idx] = (i, j)
            
            if _is_sorted_network_vectorized(candidate, bits):
                # Check if we can now remove more comparators
                test_comps = candidate
                removed_any = False
                for ri in range(len(test_comps)):
                    if time.time() > deadline - 2.0:
                        break
                    test_candidate = test_comps[:ri] + test_comps[ri+1:]
                    if _is_sorted_network_vectorized(test_candidate, bits):
                        test_comps = test_candidate
                        removed_any = True
                        break
                
                if len(test_comps) < len(current_comps):
                    # Found a better structure
                    current_comps = test_comps
                    current_len = len(current_comps)
                    if current_len < best_len:
                        best_comps = current_comps[:]
                        best_len = current_len
                elif len(test_comps) == len(current_comps):
                    # Same length, but different structure - accept with some probability
                    if rng.random() < 0.1:  # 10% chance to accept for diversity
                        current_comps = candidate
                        current_len = len(candidate)
    
    return best_comps

# EVOLVE-BLOCK-END
