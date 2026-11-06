"""
Size-efficient sorting networks on 17 wires.
Strategy: Packed uint64 representation for fast vectorized verification, 
combined with iterative greedy removal (forward and backward) from the 
odd-even transposition base.
"""
import numpy as np
import time

N = 17
NUM_INPUTS = 1 << N  # 131072
DEADLINE = 50  # seconds, safety margin before 60s kill

# EVOLVE-BLOCK-START

def _make_all_inputs_packed():
    """
    Create an array of all 2^N Boolean inputs as a uint64 array.
    Each element is a packed N-bit value representing one input sequence.
    Bit k corresponds to wire k.
    """
    return np.arange(NUM_INPUTS, dtype=np.uint64)

def _apply_comparator_packed(inputs, i, j):
    """
    Apply comparator (i, j) to the packed inputs array.
    Comparator writes min to wire i and max to wire j.
    For bits: 
    - new_i = bit_i AND bit_j
    - new_j = bit_i OR bit_j
    """
    mask_i = np.uint64(1) << i
    mask_j = np.uint64(1) << j
    
    # Extract bits
    bit_i = (inputs & mask_i) != 0
    bit_j = (inputs & mask_j) != 0
    
    # Compute new values
    new_i = bit_i & bit_j
    new_j = bit_i | bit_j
    
    # Clear the bits first
    clear_mask = ~(mask_i | mask_j)
    inputs = inputs & clear_mask
    
    # Set the new bits
    # We need to add back the bits where new value is 1
    # new_i is a boolean array, so we can use np.where or bitwise OR with masked values
    # To be efficient:
    # val_i = np.where(new_i, mask_i, 0)
    # val_j = np.where(new_j, mask_j, 0)
    # But np.where on uint64 might be slower than bitwise tricks?
    # Actually, (new_i.astype(np.uint64) * mask_i) is slow.
    # Let's stick to simple bitwise:
    # If new_i is True, we want mask_i. If False, 0.
    # We can create an array of masks? No, mask is scalar.
    # inputs = np.where(new_i, inputs | mask_i, inputs)
    # inputs = np.where(new_j, inputs | mask_j, inputs)
    
    # Optimization: 
    # Since we cleared the bits, we can just OR in the new values.
    # But we need to convert boolean to uint64 mask.
    # new_i_uint64 = new_i.astype(np.uint64) * mask_i -> This allocates a new array.
    # Alternative:
    # We know that if new_i is True, the bit was 1 in both.
    # Let's just use np.where, it's generally well optimized in NumPy.
    
    inputs = np.where(new_i, inputs | mask_i, inputs)
    inputs = np.where(new_j, inputs | mask_j, inputs)
    
    return inputs

def _is_sorting_network_packed(comparators, inputs):
    """
    Check if the network sorts all inputs.
    Returns True if all inputs are sorted (0s followed by 1s).
    """
    # Apply all comparators
    for i, j in comparators:
        inputs = _apply_comparator_packed(inputs, i, j)
        
    # Check if each row is sorted.
    # A row is sorted if for all k from 0 to N-2, we do NOT have (bit_k == 1 AND bit_{k+1} == 0).
    # Violation condition: bit_k is 1 AND bit_{k+1} is 0.
    # We can check this for all k in parallel for all inputs?
    # For a specific k:
    # mask_k = 1 << k
    # mask_k1 = 1 << (k + 1)
    # bit_k = (inputs & mask_k) != 0
    # bit_k1 = (inputs & mask_k1) != 0
    # violation = bit_k & ~bit_k1
    # if np.any(violation): return False
    
    # To be faster, we can check all k at once?
    # It's hard to vectorize across k easily without unrolling.
    # But N=17 is small. A loop of 16 iterations with vectorized checks on 131k elements is fast.
    
    for k in range(N - 1):
        mask_k = np.uint64(1) << k
        mask_k1 = np.uint64(1) << (k + 1)
        
        # Check if bit k is 1
        has_k = (inputs & mask_k) != 0
        # Check if bit k+1 is 0
        not_k1 = (inputs & mask_k1) == 0
        
        # Violation if both are true
        if np.any(has_k & not_k1):
            return False
            
    return True

def construct():
    start_time = time.time()
    
    # Base: Odd-even transposition network
    # This is a valid sorting network.
    # For N wires, there are N phases.
    # Phase p: comparators (i, i+1) where i has parity p % 2.
    comps = [(i, i + 1) for phase in range(N)
             for i in range(phase % 2, N - 1, 2)]
    
    # Initial number of comparators is N * (N-1)/2 ? No.
    # For N=17:
    # Phase 0: 0,1; 2,3; ... 14,15 (8 comps)
    # Phase 1: 1,2; 3,4; ... 15,16 (8 comps)
    # ...
    # Total is N * (N-1)/2 ? No, it's N * floor(N/2) roughly?
    # Actually, for N=17, it's 17 * 8 = 136 comparators.
    
    inputs = _make_all_inputs_packed()
    
    # Verify base is correct
    if not _is_sorting_network_packed(comps, inputs):
        # Fallback: should not happen
        return comps
    
    # Iteratively remove redundant comparators
    # We use a greedy approach: 
    # 1. Scan forward, try to remove each. If successful, remove and restart scan from 0? 
    #    Or continue? Restarting is safer to find new redundancies exposed by removal.
    #    However, restarting is O(N^2). 
    #    Better: Scan forward, remove if possible. When we reach the end, scan backward.
    #    Repeat until no changes.
    
    changed = True
    direction = 1 # 1 for forward, -1 for backward
    
    while changed and (time.time() - start_time) < DEADLINE - 2:
        changed = False
        
        if direction == 1:
            indices = range(len(comps))
        else:
            indices = range(len(comps) - 1, -1, -1)
        
        for idx in indices:
            if (time.time() - start_time) > DEADLINE - 2:
                break
                
            # Try removing comparator at idx
            new_comps = comps[:idx] + comps[idx+1:]
            
            if _is_sorting_network_packed(new_comps, inputs):
                comps = new_comps
                changed = True
                break
        
        direction *= -1
    
    return comps

# EVOLVE-BLOCK-END
