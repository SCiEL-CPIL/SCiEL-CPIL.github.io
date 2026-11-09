import time
import random
import numpy as np

N = 16

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    # 1. Generate Batcher's Odd-Even Mergesort network
    def batcher_odd_even_mergesort(n):
        if n == 1:
            return []
        
        m = n // 2
        # Recursively sort halves
        left = batcher_odd_even_mergesort(m)
        right = [(i + m, j + m) for i, j in batcher_odd_even_mergesort(m)]
        
        # Merge network
        merge = batcher_odd_even_merge(n)
        
        return left + right + merge

    def batcher_odd_even_merge(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        m = n // 2
        # Recursive calls on odd and even indices
        # Map sub-network indices to parent indices
        # Odd indices: 1, 3, ..., n-1
        # Even indices: 0, 2, ..., n-2
        
        # Get the merge network for size m
        sub_merge = batcher_odd_even_merge(m)
        
        # Apply to odd indices: (2*i + 1, 2*j + 1)
        odd_part = [(2*i + 1, 2*j + 1) for i, j in sub_merge]
        # Apply to even indices: (2*i, 2*j)
        even_part = [(2*i, 2*j) for i, j in sub_merge]
        
        # Final comparators: (i, i + m) for i in 0..m-1
        final_part = [(i, i + m) for i in range(m)]
        
        # Order matters? Usually odd then even or vice versa.
        # Standard is often odd then even, or interleaved?
        # Let's check standard definition.
        # Often: M(n) = M(n/2) on odds, M(n/2) on evens, then comparators.
        return odd_part + even_part + final_part

    network = batcher_odd_even_mergesort(N)
    
    # Verify initial network
    # Prepare inputs for verification
    num_inputs = 1 << N
    indices = np.arange(num_inputs, dtype=np.uint32)
    inputs = np.zeros((num_inputs, N), dtype=np.uint8)
    for j in range(N):
        inputs[:, j] = (indices >> j) & 1

    def is_sorting(net):
        state = inputs.copy()
        for i, j in net:
            min_val = np.minimum(state[:, i], state[:, j])
            max_val = np.maximum(state[:, i], state[:, j])
            state[:, i] = min_val
            state[:, j] = max_val
        return not np.any(state[:, :-1] > state[:, 1:])

    if not is_sorting(network):
        # Fallback to known correct if generation failed (shouldn't happen)
        # Generate a simple odd-even transposition as fallback
        network = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                network.append((i, i + 1))

    best_network = network[:]
    best_len = len(network)

    # 2. Greedy Removal Search
    # We will iterate multiple passes until no more removals are possible or time runs out
    while time.time() < deadline - 5.0: # Keep 5s for final checks
        changed = False
        # Iterate backwards
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline - 5.0:
                break
            
            # Try removing network[idx]
            candidate = network[:idx] + network[idx+1:]
            
            if is_sorting(candidate):
                network = candidate
                changed = True
                if len(network) < best_len:
                    best_len = len(network)
                    best_network = network[:]
                    
        if not changed:
            # If no removals in this pass, try some random swaps to escape local minima?
            # Or just break if we've done several passes?
            # Let's try a few random swaps to see if we can enable further removals.
            if len(network) > 50: # Only if we're not already very small
                swaps_attempted = 0
                while time.time() < deadline - 5.0 and swaps_attempted < 100:
                    # Pick two random indices
                    i1 = random.randint(0, len(network) - 1)
                    i2 = random.randint(0, len(network) - 1)
                    if i1 == i2:
                        continue
                    # Swap
                    candidate = network[:]
                    candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
                    
                    # Check if still sorting
                    if is_sorting(candidate):
                        network = candidate
                        # After a successful swap, try to remove again immediately
                        # We break out of the swap loop to go back to the removal loop
                        break
                    swaps_attempted += 1
                else:
                    # If we exhausted swaps without success, break the outer loop
                    break
            else:
                break

    # Final check
    if is_sorting(best_network):
        return best_network
    else:
        return network

# EVOLVE-BLOCK-END
