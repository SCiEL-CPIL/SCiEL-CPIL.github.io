import time
import numpy as np

N = 16
MAX_TIME = 57.0

def _generate_batcher_network():
    """
    Generates Batcher's Bitonic Sort network for N=16.
    This network has 63 comparators.
    """
    comps = []
    def bitonic_sort(n, offset, stride, direction):
        # direction: 0 for ascending, 1 for descending
        if n <= 1:
            return
        h = n // 2
        bitonic_sort(h, offset, stride, 0)
        bitonic_sort(h, offset + h * stride, stride, 1)
        # Merge step
        for i in range(h):
            if direction == 0:
                comps.append((offset + i * stride, offset + (h + i) * stride))
            else:
                comps.append((offset + (h + i) * stride, offset + i * stride))

    def bitonic_merge(n, offset, stride, direction):
        if n <= 1:
            return
        h = n // 2
        # This is the standard bitonic merge step
        for i in range(h):
            if direction == 0:
                # Standard bitonic merge assumes the second half is reverse sorted relative to the first?
                # Actually, Batcher's bitonic sort:
                # 1. Split into two halves.
                # 2. Recursively sort first half ascending, second half descending.
                # 3. Compare elements (i, h+i) and put min in i, max in h+i.
                # This step 3 is what we are doing here?
                # Wait, the standard bitonic sort network is:
                # bitonic_sort(n) = bitonic_sort(n/2) on left, bitonic_sort(n/2) on right (one asc, one desc)
                # then bitonic_merge(n).
                
                # Let's use the explicit recursive definition for correctness.
                # My previous implementation was slightly off or incomplete for the merge.
                # Let's stick to a known correct implementation.
                
                # Correct Batcher Bitonic Sort:
                # def bitonic_sort(n, start, stride, direction)
                # if n > 1:
                #   h = n // 2
                #   bitonic_sort(h, start, stride, 0)
                #   bitonic_sort(h, start + h*stride, stride, 1)
                #   for i in range(h):
                #      if direction == 0:
                #         comp(start + i*stride, start + (h+i)*stride)
                #      else:
                #         comp(start + (h+i)*stride, start + i*stride)
                
                pass

    # Let's implement the standard Batcher's bitonic sort correctly.
    # Reference: Knuth, TAOCP Vol 3.
    
    comps = []
    def bitonic_sort(n, offset, stride, direction):
        if n <= 1:
            return
        h = n // 2
        bitonic_sort(h, offset, stride, 0)
        bitonic_sort(h, offset + h * stride, stride, 1)
        for i in range(h):
            a = offset + i * stride
            b = offset + (h + i) * stride
            if direction == 0:
                comps.append((a, b))
            else:
                comps.append((b, a))

    def bitonic_merge(n, offset, stride, direction):
        # This is actually part of the sort in the above recursion?
        # No, the above `bitonic_sort` produces a bitonic sequence?
        # No, `bitonic_sort` produces a sorted sequence if called with direction.
        # Wait, the standard algorithm is:
        # bitonic_sort(n, start, stride, direction):
        #   if n > 1:
        #     h = n // 2
        #     bitonic_sort(h, start, stride, 0)
        #     bitonic_sort(h, start + h*stride, stride, 1)
        #     for i in range(h):
        #       if direction == 0:
        #         comp(start + i*stride, start + (h+i)*stride)
        #       else:
        #         comp(start + (h+i)*stride, start + i*stride)
        
        # This recursive definition sorts the n elements.
        bitonic_sort(n, offset, stride, direction)

    bitonic_sort(N, 0, 1, 0)
    return comps

def _generate_odd_even_transposition():
    """
    Generates the Odd-Even Transposition network.
    This has 120 comparators.
    """
    comps = []
    for k in range(N):
        if k % 2 == 0:
            for i in range(0, N - 1, 2):
                comps.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                comps.append((i, i + 1))
    return comps

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
    
    # Generate initial networks
    batcher_comps = _generate_batcher_network()
    if not _is_sorted_network_vectorized(batcher_comps, bits):
        batcher_comps = _generate_odd_even_transposition()
        
    # We will perform a local search starting from Batcher (63 comps).
    # Strategy:
    # 1. Try to remove comparators one by one.
    # 2. If removal fails, try swapping a comparator with another.
    # 3. Use a time budget to run this as long as possible.
    # 4. Keep the best (shortest) valid network found.
    
    best_comps = batcher_comps[:]
    best_len = len(best_comps)
    
    rng = np.random.default_rng(42)
    
    # Pre-compute all possible comparators
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # To speed up removal attempts, we can try them in a specific order or randomly.
    # Random is usually fine.
    
    # We will maintain a "current" network which is the best so far.
    # We will attempt to improve it.
    
    # Optimization: Instead of checking all 2^16 inputs every time (which is fast but not instant),
    # we can try to be smart. But 65536 * 16 * 60 ops is ~63M ops, which is fast in numpy.
    # Let's estimate: 65536 inputs * 16 wires = 1M floats. 60 comparators * 1M ops = 6M ops.
    # Numpy can do ~100M ops/s easily. So one check takes ~0.1ms?
    # Wait, 6M ops is very fast. Let's say 1ms per check.
    # In 57 seconds, we can do ~57,000 checks.
    # That's plenty for local search.
    
    current_comps = best_comps[:]
    
    # Phase 1: Greedy Removal
    # Try to remove each comparator. If valid, remove it.
    # We do this in random order to avoid bias.
    removal_order = rng.permutation(len(current_comps))
    
    for idx in removal_order:
        if time.time() > deadline - 5.0:
            break
        if len(current_comps) <= idx:
            continue
            
        # Skip if already removed? 
        # Since we are modifying the list, indices shift.
        # It's safer to copy and test, or use a set.
        # Let's use a simpler approach:
        # Pick a random index, remove it, check.
        
    # Reset to a cleaner loop structure for the main search
    current_comps = best_comps[:]
    
    # Main Local Search Loop
    # Moves:
    # 1. Remove a random comparator.
    # 2. Replace a random comparator with a different one.
    # 3. Insert a random comparator (if we want to escape local minima, but we want to minimize size, so this is counter-intuitive unless it enables larger deletions).
    
    # We will focus on Removal and Replacement.
    
    # To make it faster, we can try to identify "critical" comparators.
    # But for now, random sampling is robust.
    
    while time.time() < deadline - 2.0:
        # 1. Try Removal
        if len(current_comps) > 0:
            idx = rng.integers(0, len(current_comps))
            removed_comp = current_comps[idx]
            current_comps.pop(idx)
            
            if _is_sorted_network_vectorized(current_comps, bits):
                # Improvement!
                best_comps = current_comps[:]
                best_len = len(best_comps)
                # Continue from this new state
                continue
            else:
                # Revert
                current_comps.insert(idx, removed_comp)
        
        # 2. Try Replacement
        # Only if we have at least 1 comparator
        if len(current_comps) > 0:
            idx = rng.integers(0, len(current_comps))
            old_comp = current_comps[idx]
            new_comp = all_comps[rng.integers(0, len(all_comps))]
            
            if new_comp != old_comp:
                current_comps[idx] = new_comp
                if _is_sorted_network_vectorized(current_comps, bits):
                    # Accept the replacement (size remains same)
                    # This might enable future removals
                    # Update best? No, size is same. But it might be a better "hub".
                    # We don't update best_comps yet, as we want to reduce size.
                    # But we keep current_comps at this new state.
                    pass
                else:
                    current_comps[idx] = old_comp
        
        # 3. Occasionally try a "jump" to escape local minima
        # This involves inserting a comparator, then trying to remove others.
        # But since we are minimizing, we should be careful.
        # A better escape is to restart from best_comps with a different seed/order?
        # Or just continue random search.
        
        # To avoid getting stuck in a "dead end" where no removal works,
        # we can occasionally reset current_comps to best_comps and start a new random walk.
        if rng.random() < 0.01:
            current_comps = best_comps[:]
            
    return best_comps

# EVOLVE-BLOCK-END
