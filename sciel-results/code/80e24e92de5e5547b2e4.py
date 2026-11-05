"""
Strategy: Bit-Parallel Verification + Aggressive Greedy Removal.
The Zero-One Principle states that a sorting network is correct if and only if
it sorts all binary (0/1) inputs. There are 2^16 = 65,536 such inputs.
We can represent the state of all 65,536 inputs for all 16 wires simultaneously
using a numpy uint64 array of shape (16, 1024) where each uint64 holds 64 bits
corresponding to 64 different input vectors.

This allows us to verify the entire network in O(Comparators * Wires) operations
on 64-bit chunks, which is extremely fast. We use this to aggressively remove
comparators from the 120-comparator Odd-Even Transposition baseline.
"""

import numpy as np
import time
import random

N = 16
NUM_INPUTS = 1 << N  # 65536
CHUNKS = NUM_INPUTS // 64  # 1024 chunks of 64 inputs each

# Pre-compute the initial states for all 65536 binary inputs.
# State shape: (N, CHUNKS) where each element is uint64.
# Bit 'k' in chunk 'c' corresponds to input vector (c * 64 + k).
# We need to set bit 'w' of input 'v' to 1 if the w-th bit of v is 1.
# This is equivalent to: state[w, c] = bitmask of inputs in chunk c where bit w is set.

def _init_states():
    states = np.zeros((N, CHUNKS), dtype=np.uint64)
    # For each wire w, we need a mask for each chunk c.
    # Input index v = c * 64 + k.
    # Bit w of v is (v >> w) & 1.
    # So for a fixed w, the mask for chunk c has bit k set if ((c*64 + k) >> w) & 1 == 1.
    
    # Optimization: 
    # If w >= 6, then (v >> w) depends on c. 
    # If w < 6, then (v >> w) & 1 depends only on k (for the lower bits) and c (for higher bits if w+6 > 6? No).
    # Let's just compute it directly for simplicity, it's a one-time cost of 16*1024 ops.
    for w in range(N):
        for c in range(CHUNKS):
            mask = 0
            for k in range(64):
                v = c * 64 + k
                if (v >> w) & 1:
                    mask |= (1 << k)
            states[w, c] = mask
    return states

# Pre-compute once at module load? No, construct() is the entry point. 
# But we can cache it.
_INIT_STATES = None

def _get_states():
    global _INIT_STATES
    if _INIT_STATES is None:
        _INIT_STATES = _init_states()
    return _INIT_STATES

def _verify_network_bit_parallel(comparators):
    """
    Verifies if the network sorts all 65536 binary inputs.
    Returns True if sorted, False otherwise.
    """
    states = _get_states()
    # Copy initial states
    curr = states.copy()
    
    for i, j in comparators:
        # min(curr[i], curr[j]) and max(curr[i], curr[j])
        # min(a, b) = a & b
        # max(a, b) = a | b
        # This works bitwise!
        min_val = curr[i] & curr[j]
        max_val = curr[i] | curr[j]
        curr[i] = min_val
        curr[j] = max_val
    
    # Check if sorted: for all adjacent wires i, i+1, we must have curr[i] <= curr[i+1]
    # In bit-parallel, this means: for all bits k, if curr[i] has 1 at k, curr[i+1] must have 1 at k.
    # Equivalent to: (curr[i] & ~curr[i+1]) == 0 for all i.
    # Or: curr[i] | curr[i+1] == curr[i+1]
    for i in range(N - 1):
        # If there is any bit set in curr[i] but not in curr[i+1], it's not sorted
        if np.any(curr[i] & ~curr[i+1]):
            return False
    return True

def _verify_network_full(comparators):
    """
    Fallback verification using standard simulation if bit-parallel fails or for debugging.
    """
    # This is slow, so we only use it if we really need to be sure, 
    # but bit-parallel is mathematically equivalent for 0-1 inputs.
    # We'll trust bit-parallel.
    return _verify_network_bit_parallel(comparators)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Start with the 120-comparator Odd-Even Transposition network
    # This is a known valid sorting network.
    network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Verify baseline (should be True)
    if not _verify_network_bit_parallel(network):
        # Fallback: return the baseline if something went wrong with pre-computation
        return network

    best_network = network
    best_len = len(network)
    
    # Aggressive Greedy Removal
    # We will iterate through the network, trying to remove each comparator.
    # If the network remains valid, we keep the removal.
    # We repeat this until no more removals are possible or time runs out.
    
    # To be more effective, we can randomize the order of removal attempts
    # or restart from different points.
    
    # Strategy: 
    # 1. Try to remove comparators from end to start (later comparators are often redundant 
    #    if earlier ones did the work, but actually in sorting networks, structure matters).
    #    Actually, removing from the end is safer because it doesn't affect the state of 
    #    subsequent comparators (there are none). But removing an early comparator affects 
    #    all subsequent ones.
    #    Let's try removing from the end first.
    
    progress = True
    while progress and time.time() < deadline:
        progress = False
        # Iterate backwards to avoid index shifting issues when removing
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                break
                
            # Try removing comparator at idx
            candidate = network[:idx] + network[idx+1:]
            
            # Verify
            if _verify_network_bit_parallel(candidate):
                # Accept removal
                network = candidate
                best_network = network
                best_len = len(network)
                progress = True
                # Since we removed one, the list is shorter. 
                # We can continue the loop, but the indices have shifted.
                # Since we are iterating backwards, the next idx is idx-1, which is still valid 
                # for the new list (it corresponds to the same position relative to the end).
                # However, if we removed an element, the list length decreased.
                # The loop variable idx is controlled by the range. 
                # If we are at idx=5 and remove it, next is idx=4. 
                # In the new list, index 4 is the element that was at index 3 in the old list.
                # This is correct.
        
        # If we made progress, we might want to shuffle the network to find new removals?
        # No, greedy removal is deterministic. 
        # To escape local minima (if any exist for removal-only), we could try swapping comparators,
        # but removal is the primary goal.
        
        # If no progress was made in this pass, we can try a different strategy:
        # Randomly shuffle the order of comparators? No, that breaks the network.
        # Try removing a random comparator? 
        # The greedy pass from back-to-front is quite strong.
        
        # If stuck, try a "restart" with a slightly perturbed network?
        # Or just stop if we hit the known optimum of 60?
        if best_len <= 60:
            break

    return best_network

# To ensure the module loads quickly and doesn't do heavy work before construct() is called,
# we do not call _init_states() here. It is called lazily inside _get_states().

if __name__ == "__main__":
    # Quick self-test
    net = construct()
    print(f"Construct returned {len(net)} comparators.")
    print(f"Valid: {_verify_network_bit_parallel(net)}")
