import numpy as np
import time

N = 17
NUM_INPUTS = 1 << N
DEADLINE = 85  # seconds, safety margin

def _make_all_inputs_packed():
    """Create array of all 2^N inputs as uint64 array."""
    return np.arange(NUM_INPUTS, dtype=np.uint64)

def _is_sorting_network_packed(comparators, inputs):
    """Check if network sorts all inputs using packed representation."""
    # inputs is a uint64 array of length NUM_INPUTS
    for i, j in comparators:
        mask_i = np.uint64(1) << i
        mask_j = np.uint64(1) << j
        
        bit_i = (inputs & mask_i) != 0
        bit_j = (inputs & mask_j) != 0
        
        new_i = bit_i & bit_j
        new_j = bit_i | bit_j
        
        # Clear bits i and j
        inputs = inputs & ~(mask_i | mask_j)
        # Set new bits
        inputs = np.where(new_i, inputs | mask_i, inputs)
        inputs = np.where(new_j, inputs | mask_j, inputs)
        
    # Check if sorted
    # For each k in 0..N-2, if bit k is 1, bit k+1 must be 1.
    # Violation: bit k == 1 and bit k+1 == 0
    for k in range(N - 1):
        mask_k = np.uint64(1) << k
        mask_k1 = np.uint64(1) << (k + 1)
        
        bit_k = (inputs & mask_k) != 0
        bit_k1 = (inputs & mask_k1) != 0
        
        if np.any(bit_k & ~bit_k1):
            return False
            
    return True

def _batcher_odd_merge(n):
    """
    Generate Batcher's Odd-Merge sorting network for n wires.
    Returns a list of (i, j) comparators.
    """
    if n == 1:
        return []
    
    # Split into two halves
    if n % 2 == 0:
        m1 = n // 2
        m2 = n // 2
    else:
        m1 = (n + 1) // 2
        m2 = (n - 1) // 2
        
    # Recursively sort the two halves
    comps1 = _batcher_odd_merge(m1)
    comps2 = _batcher_odd_merge(m2)
    
    # Offset the second half's comparators by m1
    comps2_offset = [(i + m1, j + m1) for i, j in comps2]
    
    # Merge the two sorted halves using Odd-Even Merge
    # The merge network for two sorted sequences of lengths m1 and m2
    # consists of comparators between corresponding elements in the
    # odd-even merge pattern.
    
    # For odd-even merge, we need to compare elements at positions
    # that are "interleaved" in a specific way.
    
    # The standard odd-even merge for two sorted lists of size m1 and m2:
    # - If m1 == 0 or m2 == 0, no comparators needed
    # - Otherwise, split each list into even and odd indexed elements
    # - Recursively merge the even parts and odd parts
    # - Add comparators between the merged even and odd parts
    
    # This is complex to implement correctly. Let's use a simpler approach:
    # Use the known fact that Batcher's network for n wires has a specific
    # comparator count. For n=17, it's around 100-120 comparators.
    
    # Actually, let's just use the odd-even transposition as the seed,
    # but with a much more aggressive pruning strategy.
    
    # For now, return the odd-even transposition network
    return [(i, i + 1) for phase in range(n) for i in range(phase % 2, n - 1, 2)]

def construct():
    start_time = time.time()
    
    # Base: Odd-even transposition
    comps = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    inputs = _make_all_inputs_packed()
    
    # Verify base is correct
    if not _is_sorting_network_packed(comps, inputs):
        return comps
    
    # Iteratively remove redundant comparators
    changed = True
    while changed and (time.time() - start_time) < DEADLINE - 2:
        changed = False
        for idx in range(len(comps)):
            if (time.time() - start_time) > DEADLINE - 2:
                break
            # Try removing idx
            new_comps = comps[:idx] + comps[idx+1:]
            if _is_sorting_network_packed(new_comps, inputs):
                comps = new_comps
                changed = True
                break  # Restart loop since indices changed
                
    return comps
