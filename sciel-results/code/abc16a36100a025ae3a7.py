import numpy as np
import time

N = 16

def _apply_comparators(inputs, comparators):
    """
    inputs: shape (num_inputs, N) array of values
    comparators: list of (i, j) pairs
    Returns: shape (num_inputs, N) array after applying all comparators
    """
    state = inputs.copy()
    for i, j in comparators:
        min_vals = np.minimum(state[:, i], state[:, j])
        max_vals = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_vals
        state[:, j] = max_vals
    return state

def _is_sorted_network(comparators):
    """
    Verify if the network sorts all 2^16 binary inputs.
    Returns True if sorted, False otherwise.
    """
    # Generate all 2^16 binary inputs
    inputs = np.array([list(map(int, format(i, '016b'))) for i in range(65536)])
    
    # Apply network
    sorted_inputs = _apply_comparators(inputs, comparators)
    
    # Check if all are sorted (each row should be non-decreasing)
    # A row is sorted if for all k in 0..14, row[k] <= row[k+1]
    # Since values are 0/1, sorted means all 0s followed by all 1s.
    # Equivalent to checking if any adjacent pair is (1, 0).
    for k in range(N - 1):
        if np.any((sorted_inputs[:, k] == 1) & (sorted_inputs[:, k+1] == 0)):
            return False
    return True

def _generate_bitonic_network():
    """
    Generates a correct Bitonic sorting network for N wires.
    Uses a recursive structure.
    Returns a list of (i, j) comparators.
    """
    if N <= 1:
        return []
    
    comparators = []
    
    # Bitonic sort recursive structure:
    # To sort N elements (N power of 2):
    # 1. Recursively sort first N/2 ascending.
    # 2. Recursively sort second N/2 descending.
    # 3. Merge the two halves (ascending).
    
    # We implement a helper that generates the network for a specific range and direction.
    # However, "descending" sort with min-to-low comparators is tricky.
    # Standard Bitonic sort generates a bitonic sequence (asc then desc) and then merges.
    # Actually, the standard recursive Bitonic sort algorithm:
    # BitonicSort(start, length, direction):
    #   if length <= 1: return
    #   half = length // 2
    #   BitonicSort(start, half, direction)
    #   BitonicSort(start + half, half, -direction)
    #   Merge(start, length, direction)
    
    # The issue is implementing Merge and BitonicSort with the constraint that comparators are (i, j) with i < j -> min to i, max to j.
    # If direction is 'descending', we want the sequence to be high to low.
    # With min-to-low, we can only put small values at low indices.
    # This implies we CANNOT generate a descending sorted sequence directly at the same indices using standard min-to-low comparators?
    # Wait. If I have values [3, 1] at indices [0, 1].
    # Comparator (0, 1) -> min(3,1)=1 at 0, max(3,1)=3 at 1. Result [1, 3]. Ascending.
    # To get [3, 1] (Descending), I would need 3 at 0 and 1 at 1.
    # A single comparator (0, 1) cannot do this.
    # So, standard Bitonic Sort which relies on creating a descending sub-sequence to merge into an ascending one,
    # might not be directly implementable with ONLY min-to-low comparators on fixed wires.
    
    # However, Batcher's Odd-Merge Sort IS implementable with min-to-low comparators because it always merges into ascending order.
    # Batcher's Odd-Merge network for N=16 has 63 comparators.
    # The reference is 60. So Batcher is better than Transposition (120) but worse than Reference (60).
    # Can we find a network with <= 60?
    
    # Let's try to generate Batcher's Odd-Merge correctly.
    
    def batcher(start, end):
        length = end - start
        if length <= 1:
            return []
        
        mid = start + length // 2
        comps = []
        comps.extend(batcher(start, mid))
        comps.extend(batcher(mid, end))
        
        # Merge the two sorted halves
        # The merge of two sorted sequences of length k (where k = length/2)
        # Uses the Odd-Even Merge network.
        # Odd-Even Merge(M, N) merges two sorted sequences of length M and N.
        # Here M=N=k.
        # The merge comparators are:
        # 1. If k == 1: return [(start, mid)]
        # 2. Recursively merge the "odd" elements and "even" elements.
        #    The odd elements are at indices start+1, start+3... and mid+1, mid+3...
        #    The even elements are at indices start, start+2... and mid, mid+2...
        #    This recursive structure is complex to map to contiguous indices for a fixed network.
        
        # Alternative: Hardcode the 63 comparators for Batcher N=16?
        # Or use a simpler network?
        
        # Let's check if we can use a different approach:
        # "Zero-one principle" allows us to verify.
        # We can start with Batcher (63) and try to prune it.
        # Or start with a known good network and prune.
        
        # Let's generate the comparators for the merge step correctly.
        # The merge of two sorted lists of size k:
        # The comparators are:
        # for i in range(k):
        #   compare (start + i, mid + i)  # This is NOT the full merge.
        
        # Actually, the Batcher Odd-Merge comparators for merging two sorted lists of size k (indices s1..s1+k-1 and s2..s2+k-1):
        # Let k be the size.
        # The merge network M(k) is defined recursively.
        # M(1) = [(s1, s2)]
        # M(k) = M(k/2) on odd indices + M(k/2) on even indices + comparators (s1+i, s2+i) for i in 0..k-1?
        # No, the final step is comparing (s1+i, s2+k-1-i)?
        
        # This is getting complicated. Let's use a known list of comparators for a small N and scale up, or just use a heuristic search.
        
        # Since we have 90 seconds, we can afford a heuristic search.
        # 1. Start with Batcher's network (we need to generate it).
        # 2. Try to remove comparators one by one (greedy pruning) and check correctness.
        
        # To get Batcher's network, I will implement a known correct generator.
        # Here is a Python implementation of Batcher's Odd-Merge network generation.
        
        def _merge(s1, s2, k):
            # s1, s2 are start indices, k is length of each sequence
            if k == 1:
                return [(s1, s2)]
            
            half = k // 2
            comps = []
            
            # Recursively merge odd and even elements
            # Odd elements: s1+1, s1+3... and s2+1, s2+3...
            # Even elements: s1, s1+2... and s2, s2+2...
            # This recursive call on non-contiguous indices is hard to represent as a simple "start, end" range.
            # However, we can generate the comparators directly.
            
            # The Odd-Even Merge network comparators are:
            # 1. M(k/2) applied to the odd positions of the two sequences.
            # 2. M(k/2) applied to the even positions of the two sequences.
            # 3. Comparators (s1 + i, s2 + i) for i in 0..k-1? No.
            
            # Let's look up the exact structure.
            # Batcher's merge of two sorted lists of size N:
            # Let the lists be L and R.
            # The merge network consists of:
            # 1. Merge the odd-indexed elements of L and R (L[1], L[3]... and R[1], R[3]...) recursively.
            # 2. Merge the even-indexed elements of L and R (L[0], L[2]... and R[0], R[2]...) recursively.
            # 3. Compare L[i] and R[i] for i = 0..N-1? No.
            # 3. Compare L[i] and R[i] for i = 0..N-1 is not enough.
            # The final step is:
            # For i in 0..N-2:
            #   compare (L[i], R[i])?
            
            # Actually, the standard definition:
            # Merge(M, N):
            #   if M == 1: return [(0, N)] # relative indices?
            
            # I will use a hardcoded list for N=16 Batcher network if I can recall it, or generate it via a simpler method.
            # Since I can't be sure, I will use the **Bitonic** network but modify it to work with min-to-low?
            # No, Bitonic doesn't work directly.
            
            # Let's try **Shear Sort**? No.
            
            # I will use the **Odd-Even Transposition** (120 comparators) as a fallback, and then **Prune** it.
            # Pruning 120 to 60 is possible with a greedy approach?
            # 120 -> 60 is a 50% reduction. Greedy pruning might get there.
            
            return []

    # I will implement the Odd-Even Transposition network (120 comparators) and then prune it.
    # This is a safe baseline.
    
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
            
    return network

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s safety margin
    
    # Start with Odd-Even Transposition (120 comparators)
    network = _generate_bitonic_network() # This actually returns Transposition in my current impl above? No, I named it bitonic but implemented transposition logic?
    # Let's explicitly generate Transposition.
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Verify it's correct (it should be)
    # _is_sorted_network(network)
    
    # Try to prune the network
    # We will try to remove comparators in a specific order (e.g., last to first, or random) and check correctness.
    # To be efficient, we can check correctness in batches or use a faster verification.
    # 65,536 inputs * 120 comparators * 16 wires = ~126 million operations. NumPy is fast, ~0.1-0.5s per check?
    # If each check is 0.1s, we can do ~850 checks in 85s.
    # Removing 60 comparators requires at least 60 successful checks.
    # So this is feasible.
    
    # Heuristic: Remove comparators from the "middle" of the network first, or in reverse order?
    # Comparators in the middle are more likely to be redundant if the network is over-designed.
    # Let's try removing from the end backwards.
    
    i = len(network) - 1
    while i >= 0:
        if time.time() - start_time > deadline - 5.0:
            break
        
        # Try removing network[i]
        candidate = network[:i] + network[i+1:]
        
        if _is_sorted_network(candidate):
            network = candidate
            # Do not decrement i, because the next element to check is now at index i
            # (since we removed the element at i, the element that was at i+1 is now at i)
            # Actually, if we remove index i, the list shrinks.
            # The next candidate to test for removal is the one at the new index i (which was i+1 before).
            # So we keep i the same.
        else:
            i -= 1
    
    return network
