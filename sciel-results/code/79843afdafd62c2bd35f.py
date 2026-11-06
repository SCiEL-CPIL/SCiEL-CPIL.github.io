import time
import numpy as np

def _bitonic_sort_network(n):
    """
    Generates a Batcher's Bitonic Sort network for n wires.
    Returns a list of (i, j) tuples.
    """
    if n <= 1:
        return []
    
    # Bitonic sort is typically defined for powers of 2.
    # We will implement the recursive definition.
    # sort(n):
    #   if n == 1: return []
    #   return sort(n/2) on evens + sort(n/2) on odds + merge(n)
    
    # merge(n) assumes inputs are bitonic (increasing then decreasing, or vice versa)
    # Actually, standard bitonic sort generates a bitonic sequence then merges.
    # Let's use the standard recursive construction for Bitonic Sort.
    
    def generate(n, offset=0, reverse=False):
        # This function generates comparators for a bitonic sort of size n, 
        # operating on wires offset..offset+n-1.
        # It produces a sorted output.
        
        if n == 1:
            return []
        
        # Bitonic sort consists of:
        # 1. Sort the first half and second half in opposite directions to create a bitonic sequence.
        # 2. Merge the bitonic sequence.
        
        # However, the standard recursive bitonic sort is:
        # BitonicSort(n):
        #   if n == 1: return
        #   BitonicSort(n/2)
        #   BitonicSort(n/2, reverse=True)
        #   BitonicMerge(n)
        
        # Let's implement BitonicMerge(n)
        def merge(n, offset=0, reverse=False):
            if n == 1:
                return []
            # Merge two bitonic sequences of size n/2
            # The comparators are (offset+i, offset+i+n/2) for i in 0..n/2-1
            # Then recursively merge the two halves.
            
            half = n // 2
            comps = []
            
            # First level of merge: compare i with i + half
            for i in range(half):
                idx1 = offset + i
                idx2 = offset + i + half
                if idx1 > idx2: idx1, idx2 = idx2, idx1
                comps.append((idx1, idx2))
            
            # Recursively merge the two halves
            # The first half is now sorted ascending?
            # The second half is now sorted descending?
            # In standard Bitonic Merge, after the first level, the top half contains the larger elements and is sorted ascending, 
            # and the bottom half contains the smaller elements and is sorted descending?
            # Or vice versa?
            # Let's check N=4 again.
            # Inputs: A, B (asc), C, D (asc).
            # (0,2): min(A,C) at 0, max(A,C) at 2.
            # (1,3): min(B,D) at 1, max(B,D) at 3.
            # Now wire 0,1 have mins, wire 2,3 have maxs.
            # Is 0,1 sorted? Not necessarily.
            # Is 2,3 sorted? Not necessarily.
            # But the standard merge recursion says:
            # Merge(0, half) and Merge(half, n).
            # Merge(0, 2) on wires 0,1.
            # Merge(2, 4) on wires 2,3.
            # This matches my N=4 example: (0,1) and (2,3) were the recursive merges?
            # Wait, in my N=4 trace, I used (0,2), (1,3) then (1,2).
            # The recursive merge of (0,1) would be a single comparator (0,1).
            # The recursive merge of (2,3) would be a single comparator (2,3).
            # So the full set is (0,2), (1,3), (0,1), (2,3).
            # But I used (1,2) instead of (0,1) and (2,3)?
            # Let's re-verify the N=4 Bitonic Merge.
            # Standard Bitonic Merge for 4:
            # (0,2), (1,3)
            # (0,1), (2,3)
            # Total 4 comparators?
            # Let's test 5, 6, 1, 2 with (0,2), (1,3), (0,1), (2,3).
            # (0,2): 1, 5 -> 0:1, 2:5
            # (1,3): 2, 6 -> 1:2, 3:6
            # (0,1): 1, 2 -> 0:1, 1:2
            # (2,3): 5, 6 -> 2:5, 3:6
            # Result: 1, 2, 5, 6. Correct.
            
            # So the recursive merge is:
            # Merge(n) = [(i, i+n/2) for i in range(n/2)] + Merge(n/2, offset) + Merge(n/2, offset+n/2)
            
            comps += merge(half, offset, reverse)
            comps += merge(half, offset + half, reverse)
            
            return comps
        
        # Now, how to generate the full Sort(N)?
        # If I just call Merge(N) on unsorted input, it won't work.
        # I need to generate a Bitonic sequence first.
        # Standard:
        # Sort(N) = Sort(N/2) + Sort(N/2, Reverse) + Merge(N)
        
        # Since I can't do Reverse easily, let's try the "Two Ascending" hypothesis.
        # Hypothesis: Sort(N) = Sort(N/2) + Sort(N/2) + Merge(N) works.
        # Let's test N=4.
        # Sort(2) on [0,1]: (0,1)
        # Sort(2) on [2,3]: (2,3)
        # Merge(4) on [0..3]: (0,2), (1,3), (0,1), (2,3)
        # Total: (0,1), (2,3), (0,2), (1,3), (0,1), (2,3).
        # Duplicates: (0,1) and (2,3) appear twice.
        # Can we remove duplicates?
        # (0,1), (2,3), (0,2), (1,3).
        # Let's test this 4-comparator network.
        # Input: 5, 1, 4, 2.
        # (0,1): 1, 5 -> 0:1, 1:5
        # (2,3): 2, 4 -> 2:2, 3:4
        # (0,2): 1, 2 -> 0:1, 2:2
        # (1,3): 5, 4 -> 1:4, 3:5
        # Result: 1, 4, 2, 5. NOT SORTED.
        
        # So the "Two Ascending" hypothesis is FALSE.
        # We MUST sort one half in descending order.
        
        # How to sort descending with (i, j) min/max?
        # If we want wire i to have the max, we need to compare (j, i) where j < i?
        # But the interface requires i < j.
        # So we can only put min at the smaller index.
        # This means we can ONLY sort in ascending order with the standard wire layout.
        # UNLESS... we permute the wires?
        # No, the output must be sorted on wires 0..N-1.
        
        # Conclusion: The standard Bitonic Sort algorithm CANNOT be directly implemented with fixed (i, j) min/max comparators for the recursive descending steps, UNLESS we change the wire mapping.
        # But we are constructing a network on fixed wires 0..N-1.
        
        # Alternative: Use a different sorting network that works with only ascending comparators.
        # Odd-Even Transposition Sort works with only ascending comparators.
        # It uses N*(N-1)/2 comparators. For N=16, 16*15/2 = 120.
        # The task mentions "initial odd-even transposition network uses 120 comparators".
        # The reference is 60.
        
        # Can we do better than 120 with fixed ascending comparators?
        # Yes, there are better networks.
        # Batcher's Batcher Sort (not Bitonic) uses 2N log N - ... comparators?
        # Actually, Batcher's original "Sorting Networks and Applications" paper presents a network with O(N log^2 N) comparators.
        # For N=16, Batcher's network has 119 comparators?
        # Wait, Bitonic Sort IS Batcher's network.
        # Why did I think Bitonic needs descending?
        # In hardware, comparators are bidirectional.
        # In this software model, comparators are unidirectional (min to low index).
        # Does this constraint increase the comparator count?
        # Yes, potentially.
        
        # However, we can simulate a descending sort by using a "swap" network at the end?
        # No.
        
        # Let's check if we can simulate a descending sort by using a "swap" network at the end?
        # No.
        
        # Let's look at the "60 comparator" reference.
        # If a 60 comparator network exists, it must use the fixed comparator direction.
        # This implies that efficient sorting networks DO exist for this constraint.
        
        # I will implement the Odd-Even Transposition Sort (120 comparators) as a safe baseline,
        # and then try to prune it.
        # Odd-Even Transposition:
        # for step in range(N):
        #   if step % 2 == 0:
        #     for i in range(0, N-1, 2):
        #       comps.append((i, i+1))
        #   else:
        #     for i in range(1, N-1, 2):
        #       comps.append((i, i+1))
        
        # This is simple and correct.
        
        return generate(n)

    # Since I failed to derive the correct recursive Bitonic for this constraint, 
    # and Odd-Even is guaranteed to work, I will use Odd-Even as the base.
    # Then I will prune.
    
    comps = []
    for step in range(n):
        if step % 2 == 0:
            for i in range(0, n - 1, 2):
                comps.append((i, i + 1))
        else:
            for i in range(1, n - 1, 2):
                comps.append((i, i + 1))
                
    return comps

def _verify_network(comparators, n):
    """
    Verifies if the network sorts all 2^n inputs using NumPy.
    """
    # Generate all 2^n inputs
    # n=16 -> 65536 inputs
    # Use bit manipulation to generate all combinations
    num_inputs = 1 << n
    # Create array of shape (num_inputs, n)
    # i-th bit of k determines wire i
    indices = np.arange(num_inputs)
    # Bit j of index k is (k >> j) & 1
    # We want matrix M where M[k, j] = (k >> j) & 1
    # This is equivalent to taking the transpose of the bit matrix
    # Let's construct it efficiently
    # powers of 2
    bits = (indices[:, None] >> np.arange(n)) & 1
    state = bits.astype(np.float64)
    
    for i, j in comparators:
        # min at i, max at j
        min_ij = np.minimum(state[:, i], state[:, j])
        max_ij = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_ij
        state[:, j] = max_ij
        
    # Check if last column is sorted? No, check if each row is sorted.
    # A row is sorted if state[:, 0] <= state[:, 1] <= ... <= state[:, n-1]
    # We can check if np.all(state[:, :-1] <= state[:, 1:])
    
    diffs = np.diff(state, axis=1)
    if np.any(diffs < 0):
        return False
    return True

def construct():
    n = 16
    start_time = time.time()
    deadline = 55.0 # 55s to be safe
    
    # Base network: Odd-Even Transposition
    network = _bitonic_sort_network(n) # Misnamed, actually Odd-Even
    
    # Verify base network (should be fast)
    if not _verify_network(network, n):
        # Fallback: return empty? No, return a known valid one.
        # But Odd-Even is valid.
        pass
        
    # Greedy Removal
    # Try to remove comparators from the end first, then random?
    # Reverse order is often good because later comparators are more likely redundant if earlier ones did the work.
    # But for Odd-Even, they are all necessary?
    # Actually, Odd-Even is minimal for that specific structure, but not globally minimal.
    
    # To allow for a "jump", I will try a random shuffle and removal.
    # But first, let's just return the 120 comparator network.
    # The score is seed/value. 120 is the seed.
    # If I can prune to 119, score > 1.0.
    
    # Let's try to prune.
    # We will iterate through the list and try to remove each comparator.
    # To speed up, we can skip verification if the network size is already small? No.
    
    # Optimization: We can verify only a subset of inputs?
    # No, must be correct for all.
    
    # Let's try removing from the end.
    i = len(network) - 1
    while i >= 0:
        if time.time() > deadline:
            break
            
        # Try removing network[i]
        candidate = network[:i] + network[i+1:]
        
        # Verify
        if _verify_network(candidate, n):
            network = candidate
            # Do not decrement i, because the next element is now at index i
            # So we check the same index again (which is now the next element)
            # But wait, if we removed i, the list shrank.
            # The element that was at i+1 is now at i.
            # So we should check index i again.
        else:
            i -= 1
            
    return network
