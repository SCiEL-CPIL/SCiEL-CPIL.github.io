"""Size-efficient sorting networks on 18 wires.
Uses a fast bitwise verification approach with precomputed masks and greedy removal
to minimize comparator count while maintaining correctness over all 2^18 Boolean inputs.
"""

import time
import numpy as np

N = 18
NUM_BOOLS = 1 << N  # 262144
DEADLINE = 55.0  # seconds (safety margin before 60s limit)

# Precompute sorted mask: True if the state is sorted (all 0s before all 1s)
_SORTED_MASK = np.zeros(NUM_BOOLS, dtype=np.bool_)
for _k in range(N + 1):
    _val = ((1 << _k) - 1) << (N - _k)
    _SORTED_MASK[_val] = True

# Precompute masks for each wire to speed up verification
# mask[i] is a numpy array of shape (NUM_BOOLS,) with 1s at bit i
_WIRE_MASKS = [np.zeros(NUM_BOOLS, dtype=np.int32) for _ in range(N)]
for i in range(N):
    _WIRE_MASKS[i] = (1 << i)

# Precompute clear/set masks for comparators? 
# It's better to compute on the fly or cache.
# For each pair (i, j), we need:
# swap_mask = (bit_i == 1) & (bit_j == 0)
# new_val = (val & ~mask_i) | mask_j
# We can precompute ~mask_i and mask_j for each i, j.
_CLEAR_MASKS = [~(1 << i) for i in range(N)]
_SET_MASKS = [(1 << i) for i in range(N)]

def _verify_network(comparators):
    """Verify that the network sorts all 2^18 Boolean inputs.
    Returns True if valid, False otherwise.
    """
    if not comparators:
        return False
    current = np.arange(NUM_BOOLS, dtype=np.int32)
    for i, j in comparators:
        # Get bits
        bit_i = (current >> i) & 1
        bit_j = (current >> j) & 1
        # Where bit_i=1 and bit_j=0, we need to swap
        # This means we clear bit i and set bit j
        mask_swap = (bit_i == 1) & (bit_j == 0)
        if np.any(mask_swap):
            # Apply swap
            current = np.where(mask_swap, (current & _CLEAR_MASKS[i]) | _SET_MASKS[j], current)
    return np.all(_SORTED_MASK[current])

def _odd_even_transposition_network():
    """Generate the odd-even transposition sorting network for N wires."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _greedy_removal(comparators, deadline):
    """Try to remove redundant comparators one by one."""
    current = list(comparators)
    start_time = time.time()
    
    # Repeat until no more removals possible or time limit reached
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate backwards to avoid index shifting issues when removing
        for idx in range(len(current) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Try removing comparator at idx
            trial = current[:idx] + current[idx + 1:]
            
            # Verify
            if _verify_network(trial):
                current = trial
                improved = True
                # If we removed one, we restart the loop to check for new redundancies
                # Or we can continue, but restarting is safer for finding all
                # Since we restart the outer while loop, we break inner
                break
                
    return current

def construct():
    """Construct a sorting network for 18 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with odd-even transposition network (known good, 153 comparators)
    comparators = _odd_even_transposition_network()
    
    # Verify baseline
    if not _verify_network(comparators):
        return comparators
    
    # Optimize using greedy removal
    if time.time() < deadline - 5:
        comparators = _greedy_removal(comparators, deadline - 5)
    
    # Final verification
    if not _verify_network(comparators):
        # Fallback to known good network
        comparators = _odd_even_transposition_network()
    
    return comparators
