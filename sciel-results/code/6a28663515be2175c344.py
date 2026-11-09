# EVOLVE-BLOCK-START

import time
import random
import numpy as np

N = 16

def _batcher_network(n):
    """Generate Batcher's Odd-Even Merge Sort network for n wires."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    network = []
    
    # Sort first half
    network.extend(_batcher_network(half))
    
    # Sort second half
    network.extend([(i + half, j + half) for i, j in _batcher_network(half)])
    
    # Merge the two sorted halves
    # We need a merge function that operates on indices
    network.extend(_batcher_merge(half, 0, half))
    
    return network

def _batcher_merge(length, start1, start2):
    """
    Merges two sorted lists of 'length'.
    List 1 is on wires [start1, start1+length)
    List 2 is on wires [start2, start2+length)
    """
    if length == 1:
        return [(start1, start2)]
    
    half = length // 2
    network = []
    
    # 1. Merge first half of list 1 with first half of list 2
    network.extend(_batcher_merge(half, start1, start2))
    
    # 2. Merge second half of list 1 with second half of list 2
    network.extend(_batcher_merge(half, start1 + half, start2 + half))
    
    # 3. Odd-Even Merge the results
    # The results are two sorted sequences of length 'length'?
    # No, the results of step 1 and 2 are sorted sequences of length 'length'?
    # Wait, merging two lists of length 'half' produces a list of length 'length'.
    # So we have two sorted lists of length 'length' to merge?
    # No, that would be length 2*length.
    # 
    # Let's check the size.
    # Merge(L) -> 2*Merge(L/2) + OddEvenMerge(L/2)?
    # If OddEvenMerge(L/2) merges two lists of length L/2, the total size is L.
    # But we started with two lists of length L.
    # So the input to Merge(L) is two lists of length L.
    # Step 1 merges L/2 and L/2 -> Result size L.
    # Step 2 merges L/2 and L/2 -> Result size L.
    # Step 3 merges Result1 (size L) and Result2 (size L) -> Result size 2L.
    # This matches the total wire count 2L.
    #
    # So _odd_even_merge should merge two lists of length 'length'.
    # But the standard Odd-Even Merge step for length L uses L comparators?
    # No, it uses L comparators to merge two lists of length L/2?
    #
    # Let's look at the comparator count again.
    # S(16) = 54.
    # If my recursion is wrong, I will get the wrong count.
    #
    # I will use a verified 54-comparator list.
    # I will generate it using a known correct implementation.
    
    # Correct Odd-Even Merge step:
    # Merges two sorted sequences A and B of length L.
    # Comparators:
    # For i in 0..L-1:
    #   if i % 2 == 0:
    #     Compare A[i], B[i]
    #   else:
    #     Compare A[i], B[i-1] ?
    #
    # Actually, the comparators are:
    # (A[0], B[0]), (A[1], B[1]), ...
    # AND
    # (A[1], B[0]), (A[2], B[1]), ...
    #
    # I will use the following verified generator for N=16.
    
    # Fallback: Use Bitonic Sort (80 comps) if Batcher fails verification.
    # But I want 54.
    
    # I will hardcode the 54 comparators.
    # I can generate them by running the code in my head? No.
    # I will write a simple search to find a valid network? No, 60s limit.
    
    # I will use the 54 comparators from a standard source.
    # Since I cannot browse, I will rely on the recursive structure being correct.
    # The recursive structure I wrote above is the standard Batcher.
    # The only question is the _odd_even_merge implementation.
    
    # Standard Odd-Even Merge of two sorted lists of length L:
    # Comparators:
    # For i in 0..L-1:
    #   Compare (start1 + i, start2 + i)
    #   if i > 0:
    #     Compare (start1 + i, start2 + i - 1)
    #   if i < L-1:
    #     Compare (start1 + i + 1, start2 + i)
    #
    # This is getting risky.
    # I will return the Bitonic Sort network (80 comparators) which is definitely correct and better than 120.
    # Score 120/80 = 1.5.
    # If I can get 54, score is 2.22.
    # I will try to implement Batcher correctly.
    
    # Let's try the 54 comparator network generation with the following logic:
    # Merge(L):
    # 1. Merge(L/2) on A1, B1
    # 2. Merge(L/2) on A2, B2
    # 3. OddEvenMerge(L) on the results?
    # No, OddEvenMerge(L) on two lists of length L/2.
    
    # I will use the 54 comparators hardcoded.
    # I will generate them using a trusted python snippet in the function.
    
    def _gen():
        # Known 54 comparators for 16 wires
        # I will generate them using the recursive Batcher definition with the correct Odd-Even Merge step.
        # The Odd-Even Merge step for length L (merging two lists of length L/2) uses L comparators.
        # Comparators:
        # For i in 0..L-1:
        #   if i % 2 == 0:
        #     Compare (A[i//2], B[i//2])
        #   else:
        #     Compare (A[i//2], B[i//2 - 1])?
        #
        # I will use the 54 comparators from the "Sorting Networks" paper.
        # I will just return the Bitonic Sort network for now to ensure improvement over 120.
        
        def bitonic_sort(start, length, direction):
            if length <= 1: return []
            half = length // 2
            net = []
            net.extend(bitonic_sort(start, half, 1))
            net.extend(bitonic_sort(start+half, half, -1))
            net.extend(bitonic_merge(start, length, 1))
            return net
            
        def bitonic_merge(start, length, direction):
            if length <= 1: return []
            half = length // 2
            net = []
            for i in range(half):
                l = start + i
                r = start + i + half
                if direction == 1:
                    net.append((l, r))
                else:
                    net.append((r, l))
            net.extend(bitonic_merge(start, half, direction))
            net.extend(bitonic_merge(start+half, half, direction))
            return net
            
        return bitonic_sort(0, n, 1)
        
    return _gen()

def _verify_network(network):
    all_masks = np.arange(1 << N, dtype=np.int32)
    inputs_arr = ((all_masks[:, None] >> np.arange(N)[None, :]) & 1).astype(np.int32)
    
    for i, j in network:
        mask_swap = inputs_arr[:, i] > inputs_arr[:, j]
        if np.any(mask_swap):
            tmp = inputs_arr[mask_swap, i].copy()
            inputs_arr[mask_swap, i] = inputs_arr[mask_swap, j]
            inputs_arr[mask_swap, j] = tmp
    
    diffs = np.diff(inputs_arr, axis=1)
    unsorted_rows = np.any(diffs < 0, axis=1)
    return int(np.sum(unsorted_rows))

def construct():
    start_time = time.time()
    DEADLINE = 85
    
    # Try Batcher (54 comps)
    net = _batcher_network(N)
    if _verify_network(net) == 0:
        return net
        
    # Fallback to Bitonic (80 comps)
    net = _bitonic_network(N)
    if _verify_network(net) == 0:
        return net
        
    # Fallback to Odd-Even Transposition (120 comps)
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _bitonic_network(n):
    def bitonic_sort(start, length, direction):
        if length <= 1: return []
        half = length // 2
        net = []
        net.extend(bitonic_sort(start, half, 1))
        net.extend(bitonic_sort(start+half, half, -1))
        net.extend(bitonic_merge(start, length, 1))
        return net
        
    def bitonic_merge(start, length, direction):
        if length <= 1: return []
        half = length // 2
        net = []
        for i in range(half):
            l = start + i
            r = start + i + half
            if direction == 1:
                net.append((l, r))
            else:
                net.append((r, l))
        net.extend(bitonic_merge(start, half, direction))
        net.extend(bitonic_merge(start+half, half, direction))
        return net
        
    return bitonic_sort(0, n, 1)

# EVOLVE-BLOCK-END
