"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    # 1. Generate initial network: Odd-Even Transposition (120 comparators)
    # This is a known correct sorting network.
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))

    # 2. Generate all 2^N zero-one inputs for verification
    # Using bit operations for speed
    num_inputs = 1 << N
    # Create an array of indices 0 to 2^N - 1
    indices = np.arange(num_inputs, dtype=np.uint32)
    # Expand bits to create the input matrix
    # inputs[i, j] is the j-th bit of index i
    # We want (i, j) to be the value of wire j for input i.
    # Standard binary representation: bit 0 is LSB.
    # So inputs[i, j] = (i >> j) & 1
    inputs = np.zeros((num_inputs, N), dtype=np.uint8)
    for j in range(N):
        inputs[:, j] = (indices >> j) & 1

    # 3. Helper function to check if a network is sorting using 0-1 principle
    # This uses NumPy vectorization to process all 65536 inputs simultaneously.
    def is_sorting(net):
        # Copy inputs to state
        state = inputs.copy()
        
        # Apply each comparator
        for i, j in net:
            # min goes to i, max goes to j
            min_val = np.minimum(state[:, i], state[:, j])
            max_val = np.maximum(state[:, i], state[:, j])
            state[:, i] = min_val
            state[:, j] = max_val
        
        # Check if all rows are sorted (non-decreasing)
        # A row is sorted if state[:, k] <= state[:, k+1] for all k
        # We check if there is any violation: state[:, k] > state[:, k+1]
        # If any violation exists, it's not sorting.
        # We can check all adjacent pairs at once.
        # diff = state[:, 1:] - state[:, :-1]
        # If any diff < 0, not sorted.
        # Since values are 0 or 1, diff is in {-1, 0, 1}.
        # So we just check if any diff == -1.
        # Or simply: np.any(state[:, :-1] > state[:, 1:])
        return not np.any(state[:, :-1] > state[:, 1:])

    # Verify initial network (should be true)
    if not is_sorting(network):
        # Fallback: return the known correct network
        return network

    # 4. Greedy Removal Search
    # Try to remove comparators one by one.
    # We iterate through the network in reverse order (often easier to remove later ones first? 
    # Actually, order doesn't matter much for greedy, but reverse might be slightly faster 
    # if we modify the list in place or rebuild it).
    # We'll use a list and rebuild or pop.
    # Since we need to check the whole network for each removal, 
    # the cost is O(len(net)) checks per removal attempt.
    # Total cost is roughly O(M^2 * C) where M is number of comparators and C is cost of is_sorting.
    # C is ~65536 * 16 operations in NumPy, which is fast.
    # M starts at 120. 120^2 = 14400 checks.
    # Each check is ~1-2 ms? Maybe less.
    # 14400 * 2ms = 28 seconds. This might be tight but feasible.
    # We can optimize by only checking if the comparator is "potentially" redundant.
    # But let's try the direct approach first with a time guard.

    best_network = network[:]
    best_len = len(network)
    
    # We'll try multiple passes. In each pass, we try to remove any comparator.
    # We stop when no more can be removed or time runs out.
    
    max_passes = 5 # Safety limit
    for _ in range(max_passes):
        if time.time() > deadline:
            break
            
        changed = False
        # Iterate backwards to avoid index shifting issues if we remove
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                break
                
            # Try removing network[idx]
            candidate = network[:idx] + network[idx+1:]
            
            # Check if candidate is still sorting
            if is_sorting(candidate):
                # It is sorting! Remove it.
                network = candidate
                changed = True
                # Update best
                if len(network) < best_len:
                    best_len = len(network)
                    best_network = network[:]
                
        if not changed:
            break

    # 5. Optional: Try to swap or reorder? 
    # Reordering doesn't change the count.
    # Swapping might allow further removals.
    # But greedy removal is usually quite effective.
    
    # Final check on best_network
    if is_sorting(best_network):
        return best_network
    else:
        # Should not happen, but fallback to original
        return network

# EVOLVE-BLOCK-END
