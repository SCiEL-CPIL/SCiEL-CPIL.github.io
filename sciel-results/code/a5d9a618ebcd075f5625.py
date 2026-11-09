import numpy as np
import time
import math
import random

N = 18

def _batcher_merge(N1, N2):
    if N1 == 0 or N2 == 0:
        return []
    depth = math.ceil(math.log2(N1 + N2))
    comps = []
    for j in range(depth):
        step = 1 << j
        for i in range(N1 + N2 - step):
            if j == 0:
                comps.append((i, i + step))
            else:
                if (i % step) < (step // 2):
                    comps.append((i, i + step))
    return comps

def _batcher_sort(N):
    if N == 1:
        return []
    N1 = N // 2
    N2 = N - N1
    comps = []
    comps.extend(_batcher_sort(N1))
    half2_comps = _batcher_sort(N2)
    comps.extend([(i + N1, j + N1) for i, j in half2_comps])
    comps.extend(_batcher_merge(N1, N2))
    return comps

def _is_sorted_network(comps, N):
    """
    Check if the network sorts all 0-1 inputs using NumPy.
    """
    if N <= 1:
        return True
    
    # Generate all 0-1 inputs
    # For N=18, 2^18 = 262144. This is manageable.
    inputs = np.arange(2**N, dtype=np.uint32)
    
    # Convert to binary matrix
    # matrix[input_idx, wire_idx]
    bits = 1 << np.arange(N, dtype=np.uint32)
    matrix = (inputs[:, None] & bits[None, :]) > 0
    
    # Apply network
    for i, j in comps:
        # min to i, max to j
        # Since values are boolean, min is AND, max is OR
        # But standard min/max works fine too.
        min_val = np.minimum(matrix[:, i], matrix[:, j])
        max_val = np.maximum(matrix[:, i], matrix[:, j])
        matrix[:, i] = min_val
        matrix[:, j] = max_val
        
    # Check if sorted: each row should be non-decreasing
    # diff >= 0 for all adjacent pairs
    diff = np.diff(matrix, axis=1)
    return np.all(diff >= 0)

def _verify_sorts_bitmask(nets, n_wires):
    """
    Alternative verification using bitmask simulation, potentially faster for small N.
    """
    total = 1 << n_wires
    mask_all = (1 << n_wires) - 1
    
    for x in range(total):
        for (i, j) in nets:
            bi = (x >> i) & 1
            bj = (x >> j) & 1
            if bi > bj:
                # Swap: clear bit i, set bit j
                x &= ~(1 << i)
                x |= (1 << j)
        
        # Check if sorted: x must be of the form 0...0 1...1
        # i.e., y = ~x & mask_all must be of form 1...1 0...0 ? No.
        # Sorted means lower wires have 0, upper have 1.
        # So x has 0s in lower bits, 1s in upper bits.
        # x = sum_{k=m}^{n-1} 2^k = 2^n - 2^m.
        # Let y = mask_all ^ x. y has 1s in lower bits, 0s in upper bits.
        # y = sum_{k=0}^{m-1} 2^k = 2^m - 1.
        # So y & (y+1) == 0.
        y = mask_all ^ x
        if y & (y + 1) != 0:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill
    
    # 1. Generate Batcher network
    try:
        comps = _batcher_sort(N)
        # Deduplicate and preserve order? Batcher doesn't have duplicates usually.
        # But let's be safe.
        seen = set()
        unique_comps = []
        for c in comps:
            if c not in seen:
                seen.add(c)
                unique_comps.append(c)
        comps = unique_comps
    except Exception as e:
        # Fallback to Odd-Even Transposition Sort
        comps = []
        for k in range(N):
            if k % 2 == 0:
                for i in range(0, N-1, 2):
                    comps.append((i, i+1))
            else:
                for i in range(1, N-1, 2):
                    comps.append((i, i+1))
            
    best_comps = comps
    best_count = len(comps)
    
    # 2. Prune redundant comparators
    # We iterate through the list and try to remove each comparator.
    # If removal keeps the network valid, we remove it.
    # We do this in a loop, potentially multiple passes if time permits,
    # but a single pass is often effective for Batcher.
    
    i = 0
    # Use a faster verification if possible? 
    # _is_sorted_network uses NumPy which is vectorized over all inputs.
    # For N=18, 2^18 inputs. 
    # Each check involves iterating over ~100 comparators and doing vector ops on 2^18 x 18 arrays.
    # This should be fast enough (milliseconds).
    
    try:
        while i < len(comps) and time.time() < deadline:
            # Try to remove comps[i]
            test_comps = comps[:i] + comps[i+1:]
            
            if _is_sorted_network(test_comps, N):
                # Remove it
                comps = test_comps
                best_comps = comps
                best_count = len(comps)
                # Do not increment i, check the new element at i
            else:
                i += 1
    except Exception as e:
        # If pruning crashes, keep the best found so far
        pass
    
    # 3. Additional optimization: Randomized local search?
    # Given the time budget, simple pruning is the most reliable improvement.
    # The Batcher network for N=18 is likely around 100-150 comparators.
    # Pruning can reduce this significantly.
    
    # Ensure we return a valid list
    if not best_comps:
        # Fallback
        comps = []
        for k in range(N):
            if k % 2 == 0:
                for i in range(0, N-1, 2):
                    comps.append((i, i+1))
            else:
                for i in range(1, N-1, 2):
                    comps.append((i, i+1))
        best_comps = comps

    return best_comps
