import time
import numpy as np

N = 16
MAX_COMPARATORS = 4096
TIME_LIMIT = 60  # seconds, with safety margin

# Precompute all 2^16 boolean inputs as a NumPy array for fast verification
# Shape: (65536, 16)
_ALL_INPUTS = np.array([(mask >> i) & 1 for mask in range(1 << N) for i in range(N)]).reshape(1 << N, N).T
# _ALL_INPUTS is now (16, 65536) to allow easier broadcasting?
# Let's keep it (65536, 16) for clarity, or (16, 65536) for column-wise ops.
# Comparators operate on wires i and j.
# If we store inputs as (N, M), then wire i is row i.
# Comparator (i, j): min(row_i, row_j) -> row_i, max(row_i, row_j) -> row_j.
# This is vectorizable.

def _batcher_oes(n):
    """
    Generate Batcher's Odd-Even-Merge Sort network for n wires.
    Returns a list of (i, j) comparators.
    """
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Recursive step
    # Split into two halves
    half = n // 2
    # 1. Recursively sort the two halves
    network = []
    network.extend([(i, j) for i, j in _batcher_oes(half)])
    network.extend([(i + half, j + half) for i, j in _batcher_oes(half)])
    
    # 2. Odd-Even Merge
    # The merge step for size n consists of merging the two sorted halves.
    # The structure is:
    # For k in range(half - 1, 0, -1):
    #   for i in range(k):
    #     comparator (i, i + k)  # Odd-even merge logic
    #     comparator (i + k, i + 2*k) # Wait, the standard OES merge is specific.
    
    # Let's use the standard OES merge construction.
    # Merge of two sorted lists of size m.
    # The merge network for size 2m is:
    # 1. Comparator (0, m)
    # 2. Merge(0, m-1) and Merge(m, 2m-1) recursively? No.
    # 
    # Standard OES Merge for size N (where N is power of 2):
    # def merge(n):
    #   if n == 2: return [(0,1)]
    #   half = n // 2
    #   # Recursively merge the first half and second half? No.
    #   # The OES merge is:
    #   # 1. Comparator (0, half)
    #   # 2. Merge(half) on wires 0..half-1 and wires half..2half-1?
    #   # Actually, the recursive structure is:
    #   # merge(n) = merge(n/2) on (0, n/2-1) and (n/2, n-1) 
    #   #            + comparators (i, i+n/2) for i in 0..n/2-1 ?
    #   # No, that's just a bitonic merge.
    #
    # Let's use the known OES comparator count: 78 for n=16.
    # I will implement the OES network generation carefully.
    
    def _oes_merge(n):
        """Generates comparators for merging two sorted halves of size n/2 into size n."""
        if n == 2:
            return [(0, 1)]
        
        half = n // 2
        comps = []
        # 1. Compare (0, half)
        comps.append((0, half))
        
        # 2. Recursively merge the lower half and upper half
        # The lower half (0..half-1) and upper half (half..2half-1) are not independent.
        # The OES merge structure:
        # After (0, half), the elements at 0 and half are min/max.
        # The remaining elements need to be merged.
        # Specifically, we merge:
        #   - The sub-network on wires 1..half-1 and half..2half-2? 
        #   - And a cross comparator?
        
        # Correct OES Merge Recursion:
        # merge(n):
        #   if n == 2: return [(0,1)]
        #   half = n // 2
        #   # 1. (0, half)
        #   # 2. merge(half) on wires 0..half-1
        #   # 3. merge(half) on wires half..2half-1
        #   # 4. For i in 1..half-1: (i, i+half)
        
        # Wait, the standard OES merge is:
        # M(n) = M(n/2) on first half + M(n/2) on second half + (0, n/2) + { (i, i+n/2) for i=1..n/2-1 }
        # Let's verify the count.
        # C(n) = 2*C(n/2) + 1 + (n/2 - 1) = 2*C(n/2) + n/2
        # C(2) = 1
        # C(4) = 2*1 + 2 = 4
        # C(8) = 2*4 + 4 = 12
        # C(16) = 2*12 + 8 = 32
        # Total OES for 16:
        # Sort(16) = 2*Sort(8) + Merge(16)
        # S(2) = 1
        # S(4) = 2*1 + 4 = 6
        # S(8) = 2*6 + 12 = 24
        # S(16) = 2*24 + 32 = 80?
        # Literature says 78. My recurrence might be slightly off or I'm misremembering the base case or merge structure.
        # Let's check Batcher's OES count for 16:
        # n=2: 1
        # n=4: 6
        # n=8: 24
        # n=16: 78? 
        # 2*24 + 30 = 78. So Merge(16) is 30 comparators.
        # My formula gave 32.
        # The difference is in the merge step.
        # The merge step for OES is not simply "two merges + cross".
        
        # Let's use a known implementation or a simpler, valid network.
        # Bitonic Sort for 16 uses 60 comparators?
        # Bitonic Sort depth: log2(n)(log2(n)+1)/2.
        # Bitonic Sort comparators: n/2 * log2(n)(log2(n)+1)/2 ?
        # For n=16:
        # Stages: 1, 2, 3, 4
        # Stage 1: 8 comparators
        # Stage 2: 16 comparators
        # Stage 3: 24 comparators
        # Stage 4: 32 comparators
        # Total: 80.
        
        # So Bitonic is 80. OES is 78.
        # Optimal is 60.
        
        # I will implement a generic Bitonic Sort network as a safe fallback (80 comps), 
        # and then try to reduce it.
        
        pass

    # Let's just generate Bitonic Sort, which is easy to code correctly.
    def _bitonic_sort(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        half = n // 2
        comps = []
        # 1. Sort first half ascending, second half descending
        comps.extend([(i, j) for i, j in _bitonic_sort(half)])
        # For the second half, we want descending. 
        # Bitonic sort usually sorts both halves in opposite directions.
        # We can generate the network for the second half and then reverse the direction of comparators?
        # Or just generate it and swap i, j?
        # Comparator (i, j) sorts min to i, max to j.
        # To sort descending, we want max to i, min to j.
        # This is equivalent to swapping the outputs, or using a comparator (j, i) if allowed? 
        # No, comparator (i, j) always puts min in i.
        # So to sort a block descending, we generate the ascending network for that block, 
        # but the "wires" are mapped such that the output order is reversed?
        # Actually, it's easier to just generate the network for the second half normally,
        # and then in the merge phase, we handle the bitonic sequence.
        
        # Standard Bitonic Sort:
        # 1. Recursively sort first half ascending.
        # 2. Recursively sort second half descending.
        # 3. Merge the two halves.
        
        # How to generate descending sort network?
        # If we have a network that sorts ascending, we can make it sort descending 
        # by swapping the indices of the wires in the output? 
        # No, the network is a set of comparators.
        # If we apply the same comparators, we get ascending.
        # To get descending, we can just use the same comparators but interpret the result as descending?
        # No, the physical wires are fixed.
        
        # Trick: To sort wires [half, n) in descending order, 
        # we can sort them in ascending order using a network generated for indices [0, half), 
        # but mapped to [half, n). 
        # Then, to make the *sequence* descending, we need the value at wire `half` to be the max, 
        # and wire `n-1` to be the min.
        # An ascending sort puts min at `half` and max at `n-1`.
        # So if we just sort ascending, the block is ascending. 
        # Bitonic merge requires one block ascending and the other descending.
        
        # So, for the second half, we generate the comparators for sorting [0, half) ascending,
        # but we map index `k` in the sub-network to `half + (half - 1 - k)` in the main network?
        # This reverses the wire order.
        # If we reverse the wires, an ascending sort on the reversed wires becomes a descending sort on the original wires.
        
        comps_second = []
        for i, j in _bitonic_sort(half):
            # Map i, j to the second half, reversed
            # i -> half + (half - 1 - i)
            # j -> half + (half - 1 - j)
            ni = half + (half - 1 - i)
            nj = half + (half - 1 - j)
            if ni > nj:
                ni, nj = nj, ni
            comps_second.append((ni, nj))
        comps.extend(comps_second)
        
        # 3. Merge
        # The merge phase of Bitonic Sort for size n:
        # For k in range(half, 0, -1):
        #   for i in range(0, n - k):
        #     comparator (i, i + k)
        
        for k in range(half, 0, -1):
            for i in range(0, n - k):
                comps.append((i, i + k))
                
        return comps

    return _bitonic_sort(n)

def _verify_network_numpy(network):
    """
    Verify network using NumPy on all 2^16 inputs.
    Returns True if sorted.
    """
    # _ALL_INPUTS is (N, 65536)
    data = _ALL_INPUTS.copy()
    for i, j in network:
        # Get rows i and j
        row_i = data[i]
        row_j = data[j]
        
        # min and max
        min_ij = np.minimum(row_i, row_j)
        max_ij = np.maximum(row_i, row_j)
        
        # Update
        data[i] = min_ij
        data[j] = max_ij
        
    # Check if sorted: for all columns, row 0 <= row 1 <= ... <= row N-1
    # Check if data[i] <= data[i+1] for all i, for all columns
    for i in range(N - 1):
        if not np.all(data[i] <= data[i + 1]):
            return False
    return True

def _construct_improved():
    global _best_network, _network_computed
    
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 5  # 5 second safety margin
    
    # Start with Bitonic Sort (80 comparators)
    current_network = _batcher_oes(N)
    current_count = len(current_network)
    
    # Verify it's valid
    if not _verify_network_numpy(current_network):
        # Fallback to Odd-Even Transposition if something went wrong
        current_network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        current_count = len(current_network)
        
    _best_network = current_network[:]
    best_count = current_count
    
    # Greedy removal
    # Try to remove comparators from the end of the list (they are often redundant in structured networks)
    # Or random order.
    # We will iterate through indices and try to remove each one.
    # To speed up, we can try removing from the end first, as merge steps often have redundancy.
    
    indices_to_try = list(range(len(current_network) - 1, -1, -1))
    
    # We will do multiple passes until no more can be removed
    removed_any = True
    while removed_any and time.time() < deadline:
        removed_any = False
        # Shuffle indices to avoid bias? Or keep reverse order?
        # Reverse order is good for structured networks.
        for idx in indices_to_try:
            if time.time() > deadline:
                break
            if idx >= len(current_network):
                continue
                
            # Skip if already removed
            # We are modifying the list in place? 
            # It's better to work with a copy or track removals.
            # Let's just try to remove current_network[idx]
            
            temp_network = current_network[:idx] + current_network[idx+1:]
            
            # Verify
            if _verify_network_numpy(temp_network):
                current_network = temp_network
                best_count = len(current_network)
                _best_network = current_network[:]
                removed_any = True
                # Adjust indices_to_try? 
                # Since we removed an element, all indices > idx are shifted by 1.
                # It's easier to just restart the loop or rebuild the list.
                # Given the small size, rebuilding is fine.
                indices_to_try = list(range(len(current_network) - 1, -1, -1))
                break # Restart the while loop iteration with the new network
        
    _network_computed = True
    return _best_network

def construct():
    """
    Return a fixed compare-exchange network that sorts all 16-input sequences.
    Each comparator is a tuple (i, j) with 0 <= i < j < N.
    """
    global _best_network, _network_computed
    
    if _network_computed:
        return _best_network
    
    # Try to find an improved network
    try:
        network = _construct_improved()
    except Exception:
        # Fallback
        network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    _best_network = network
    _network_computed = True
    return network

# Precompute inputs at module load to save time in construct
_ALL_INPUTS = np.array([(mask >> i) & 1 for mask in range(1 << N) for i in range(N)]).reshape(1 << N, N).T
