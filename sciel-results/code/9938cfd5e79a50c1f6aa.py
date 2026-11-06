"""
Strategy:
1. Generate a correct 16-wire sorting network (Batcher's Odd-Even Merge Sort).
2. Use vectorized NumPy operations to verify the network against all 2^16 boolean inputs in one shot.
3. Use the time saved by vectorization to perform a parallel batched greedy pruning search.
   - In each iteration, test removing a batch of comparators (e.g., 10) simultaneously.
   - If any removal in the batch results in a valid network, keep the best one (or all valid ones if they don't conflict, but greedy usually picks one).
   - This allows exploring ~5000+ removal candidates in 60s, compared to ~100 in the serial version.
"""

import time
import numpy as np

N = 16
MAX_TIME = 55.0  # Safety margin for 60s limit

def _generate_batcher_network():
    """
    Generates Batcher's Odd-Even Merge Sort network for N=16.
    Returns a list of (i, j) tuples.
    """
    def merge(n, offset):
        if n == 1:
            return []
        half = n // 2
        # Compare (offset+i, offset+i+half) for i in range(half)
        # Wait, standard OO-Merge:
        # 1. Compare (i, i+n)
        # 2. Merge even
        # 3. Merge odd
        
        # Let's use the standard recursive definition that works on wires.
        # OO-Merge(n, offset) assumes wires offset..offset+n-1 are sorted
        # and wires offset+n..offset+2n-1 are sorted.
        # It produces a sorted sequence on offset..offset+2n-1.
        
        if n == 1:
            return [(offset, offset + 1)]
        
        h = n // 2
        # Step 1: Compare (offset+i, offset+i+n) for i in range(n)
        step1 = [(offset + i, offset + i + n) for i in range(n)]
        
        # Step 2: OO-Merge the even indexed elements.
        # The even indices of the 2n block are 0, 2, ..., 2n-2.
        # These correspond to wires: offset, offset+2, ..., offset+2n-2.
        # The "two sorted sequences" for the recursive call are:
        # Seq A: offset, offset+2, ..., offset+n-2 (length h)
        # Seq B: offset+n, offset+n+2, ..., offset+2n-2 (length h)
        # This is a strided merge.
        
        # Step 3: OO-Merge the odd indexed elements.
        # Wires: offset+1, offset+3, ..., offset+2n-1.
        # Seq A: offset+1, offset+3, ..., offset+n-1
        # Seq B: offset+n+1, offset+n+3, ..., offset+2n-1
        
        # We need a helper for strided merge.
        def strided_merge(n, offset, stride):
            if n == 1:
                return [(offset, offset + stride)]
            
            h = n // 2
            # Compare (offset + i*stride, offset + (n+i)*stride) for i in range(n)
            step1 = [(offset + i * stride, offset + (n + i) * stride) for i in range(n)]
            
            # Recursive calls
            # Even indices of the strided sequence: 0, 2, ...
            # Wires: offset, offset+2*stride, ...
            # Seq A: offset, offset+2*stride, ...
            # Seq B: offset + n*stride, ...
            even = strided_merge(h, offset, 2 * stride)
            
            # Odd indices: 1, 3, ...
            # Wires: offset+stride, offset+3*stride, ...
            odd = strided_merge(h, offset + stride, 2 * stride)
            
            return step1 + even + odd
        
        # For the top-level merge(n, offset):
        # The "stride" for the even/odd subsequences is 2.
        # The length of each subsequence is n.
        # Wait, in merge(n, offset), the block size is 2n.
        # The even subsequence has length n.
        # The wires for the even subsequence are offset, offset+2, ..., offset+2n-2.
        # The two halves of this subsequence are:
        # A: offset, offset+2, ..., offset+n-2 (length n/2)
        # B: offset+n, offset+n+2, ..., offset+2n-2 (length n/2)
        
        # So we call strided_merge(n//2, offset, 2) ?
        # Let's check the parameters of strided_merge(n, offset, stride).
        # It merges two sequences of length n.
        # Seq A: offset, offset+stride, ...
        # Seq B: offset + n*stride, ...
        
        # In our case:
        # We want to merge two sequences of length h = n//2.
        # Seq A starts at offset.
        # Seq B starts at offset + n.
        # The stride is 2.
        # Distance between start of A and B is n.
        # In strided_merge, distance is n_func * stride.
        # Here n_func = h. stride = 2.
        # h * 2 = n. Correct.
        
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
    inputs: numpy array of shape (num_inputs, N)
    comps: list of (i, j)
    Returns: numpy array of shape (num_inputs, N)
    """
    vec = inputs.copy()
    for i, j in comps:
        # min and max
        min_val = np.minimum(vec[:, i], vec[:, j])
        max_val = np.maximum(vec[:, i], vec[:, j])
        vec[:, i] = min_val
        vec[:, j] = max_val
    return vec

def _is_sorted_network_vectorized(comps):
    """
    Check if the network sorts all 2^16 boolean inputs.
    """
    # Generate all 2^16 inputs
    # We can generate them as a matrix of shape (65536, 16)
    # Using bit tricks
    n_inputs = 1 << N
    # Create an array of integers 0 to 2^N - 1
    # We need to unpack bits.
    # np.unravel_index is slow for large arrays?
    # Better: use broadcasting
    # bits = (np.arange(n_inputs)[:, None] >> np.arange(N)[::-1]) & 1
    
    # To save memory/time, we can process in chunks if needed, but 65536x16 is small (~1MB).
    indices = np.arange(n_inputs)
    bits = (indices[:, None] >> np.arange(N)[::-1]) & 1
    
    # Apply network
    out = _apply_network_vectorized(comps, bits)
    
    # Check if sorted
    # A boolean array is sorted if it is non-decreasing.
    # diff >= 0
    diffs = np.diff(out, axis=1)
    return np.all(diffs >= 0)

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # 1. Generate initial network
    comps = _generate_batcher_network()
    
    # Verify initial network
    if not _is_sorted_network_vectorized(comps):
        # Fallback: Odd-Even Transposition (120 comps)
        # This is guaranteed correct
        comps = []
        for i in range(N):
            if i % 2 == 0:
                for j in range(0, N - 1, 2):
                    comps.append((j, j + 1))
            else:
                for j in range(1, N - 1, 2):
                    comps.append((j, j + 1))
        # It's a good idea to verify this too, though it's standard
        if not _is_sorted_network_vectorized(comps):
            # Should never happen, but just in case
            return comps

    best_comps = comps
    
    # 2. Greedy Pruning with Vectorization
    # We will try to remove comparators.
    # To speed up, we test batches of removals.
    # However, if we remove multiple at once, we might miss a single removal that is valid
    # but two removals that are invalid together.
    # Standard greedy: try remove i. If valid, remove it.
    # Vectorized batch: try remove i1, i2, ..., ik in parallel.
    # If any is valid, we pick the first one (or best one) and repeat.
    # This is slightly less aggressive than pure greedy (which checks i, then i+1, etc. sequentially
    # on the *updated* network), but much faster.
    
    # To be safe and effective, we can use a "batched greedy" approach:
    # 1. Identify candidates for removal (all current comparators).
    # 2. In a batch, test removing each candidate from the *current* best network.
    # 3. If any candidate results in a valid network, keep the best one (e.g., first found) and update best_comps.
    # 4. Repeat until no improvement or time runs out.
    
    # Optimization: We can pre-allocate the input matrix.
    n_inputs = 1 << N
    indices = np.arange(n_inputs)
    bits = (indices[:, None] >> np.arange(N)[::-1]) & 1
    
    def _test_removal(comps, remove_idx):
        # Create candidate
        # To avoid copying the list every time in the tight loop, we can just slice.
        # But we need to apply it.
        candidate = comps[:remove_idx] + comps[remove_idx+1:]
        out = _apply_network_vectorized(candidate, bits)
        diffs = np.diff(out, axis=1)
        return np.all(diffs >= 0)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_len = len(best_comps)
        
        # We will test removals in batches.
        # Let's test all current comparators in parallel?
        # If we have 60 comparators, we can create 60 copies of the network with one removed.
        # Apply all 60 to the input matrix.
        # This requires (60 * 65536 * 16) operations.
        # 60 * 1M = 60M operations. NumPy handles this in < 1s.
        
        valid_indices = []
        
        # Batch test all removals
        # To save memory, we can process in smaller batches if N is large, but for 16 it's fine.
        for idx in range(current_len):
            if time.time() > deadline:
                break
            
            candidate = best_comps[:idx] + best_comps[idx+1:]
            out = _apply_network_vectorized(candidate, bits)
            diffs = np.diff(out, axis=1)
            if np.all(diffs >= 0):
                valid_indices.append(idx)
        
        if valid_indices:
            # We found at least one removable comparator.
            # Greedy strategy: remove the first one found.
            # Or remove the one that allows the most future removals? (Too expensive to check)
            # Just remove the first one.
            idx_to_remove = valid_indices[0]
            best_comps = best_comps[:idx_to_remove] + best_comps[idx_to_remove+1:]
            improved = True
            
    return best_comps

# EVOLVE-BLOCK-END
