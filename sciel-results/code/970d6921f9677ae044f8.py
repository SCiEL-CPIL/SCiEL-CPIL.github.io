"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
# Deadline guard: evaluator kills after 90s, keep a safety margin
DEADLINE = 80.0  # seconds

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Precompute all Boolean inputs as a matrix of shape (2^N, N)
    # Each row is a distinct subset of wires set to 1
    # For N=18, 2^18 = 262144 rows, 18 cols -> manageable with numpy
    
    # We'll use a bit-representation approach
    # Instead of storing the full matrix (which is 262144*18 bits ~ 585KB, fine),
    # we can represent each input as an integer bitmask and apply comparators via bit ops.
    
    # Actually, let's use the standard approach: represent each of the 2^N inputs
    # as an integer from 0 to 2^N - 1, where bit i indicates wire i is 1.
    # A comparator (i, j) on a bitmask x:
    #   bit_i = (x >> i) & 1
    #   bit_j = (x >> j) & 1
    #   if bit_i == 0 and bit_j == 1: swap -> set bit_i=1, bit_j=0
    #   else: no change
    # We can vectorize this over all 2^N inputs using numpy.
    
    # Let's precompute for each comparator (i,j) the operation on all 2^N inputs.
    # For input x (integer), after comparator (i,j):
    #   If ((x>>i)&1) == 0 and ((x>>j)&1) == 1:
    #       x_new = x | (1<<i) | ~(1<<j)  (but careful with bits)
    #   More precisely:
    #       x_new = x
    #       if bit_i=0, bit_j=1: x_new = x | (1<<i) & ~(1<<j)
    #   This is: x_new = x if (bit_i >= bit_j) else x ^ (1<<i) ^ (1<<j)
    #   Wait: if bit_i=0, bit_j=1, we want bit_i=1, bit_j=0.
    #   x ^ (1<<i) ^ (1<<j) flips both bits. Since bit_i=0->1, bit_j=1->0. Correct.
    #   If bit_i=1, bit_j=0 or both same, no change.
    #   So: condition = ((x >> i) & 1) == 0 and ((x >> j) & 1) == 1
    #   Which is: ((x >> j) & 1) == 1 and ((x >> i) & 1) == 0
    #   Let mask_i = (x >> i) & 1, mask_j = (x >> j) & 1
    #   swap_mask = (mask_j & ~mask_i)  -> 1 if swap needed
    #   x_new = x ^ (swap_mask << i) ^ (swap_mask << j)
    
    # Let's precompute arrays for all inputs
    all_inputs = np.arange(1 << N, dtype=np.int64)
    
    # Function to apply a list of comparators to all_inputs and check if sorted
    # Sorted means: for each input, the bits are in non-decreasing order from wire 0 to N-1
    # i.e., if wire i has 1 and wire j has 0 with i < j, it's not sorted.
    # Actually, sorted means all 0s come before all 1s. So the bitmask should be of the form:
    # lower k bits are 0, upper N-k bits are 1. I.e., x = (2^N - 1) ^ ((2^k - 1)) for some k, 
    # or equivalently, x & (x + 1) == 0? No.
    # A bitmask is "sorted" (0s then 1s) iff x == ((1 << N) - 1) - ((1 << k) - 1) for some k in [0, N]
    # i.e., x is of the form: 000...0111...1
    # This is equivalent to: x & (x - 1) == 0 is NOT right.
    # Let's think: sorted means the set of 1-bits is a suffix.
    # For a given x, it's sorted iff for all i < j, if bit_i=1 then bit_j=1.
    # Equivalently, the bitmask x has the form: (2^N - 1) - (2^k - 1) = 2^N - 2^k for k in [0, N]
    # Wait: if k=0, x = 2^N - 1 (all 1s). If k=N, x = 0 (all 0s).
    # If k=1, x = 2^N - 2 (all 1s except bit 0).
    # So sorted values are: {2^N - 2^k : k = 0, 1, ..., N}
    # That's N+1 values.
    
    sorted_values = set([ (1 << N) - (1 << k) for k in range(N + 1) ])
    
    def is_sorted(inputs):
        """Check if all inputs in the array are sorted (bitmask form)."""
        # For each input x, check if x is in sorted_values
        # We can do this vectorized: x is sorted iff x == 2^N - 2^k for some k
        # Equivalently, x & (x + 1) == 0? Let's check:
        # x = 2^N - 2^k. x + 1 = 2^N - 2^k + 1.
        # For k=0: x=2^N-1, x+1=2^N. x & (x+1) = (2^N-1) & 2^N = 0. Good.
        # For k=1: x=2^N-2, x+1=2^N-1. x & (x+1) = (2^N-2) & (2^N-1) = 2^N-2 != 0. Bad.
        # So that doesn't work.
        
        # Alternative: x is sorted iff (x | (x + 1)) == (1 << N) - 1 or x == 0?
        # Let's just check against the set of sorted values.
        # Since there are only N+1 = 19 sorted values, we can check efficiently.
        
        # For each sorted value s, count how many inputs equal s.
        # Sum should be 2^N.
        count = 0
        for s in sorted_values:
            count += np.sum(inputs == s)
        return count == len(inputs)
    
    def apply_network(inputs, comparators):
        """Apply a list of comparators to an array of bitmasks."""
        for (i, j) in comparators:
            mask_i = (inputs >> i) & 1
            mask_j = (inputs >> j) & 1
            swap_mask = mask_j & (~mask_i)
            inputs = inputs ^ (swap_mask << i) ^ (swap_mask << j)
        return inputs
    
    # Start with a decent initial network
    # Odd-even transposition network
    initial_network = [(i, i + 1) for phase in range(N)
                       for i in range(phase % 2, N - 1, 2)]
    
    best_network = initial_network
    best_count = len(initial_network)
    
    # Verify initial network
    if is_sorted(apply_network(all_inputs.copy(), initial_network)):
        pass  # It's correct
    else:
        # This shouldn't happen, but if it does, we have a problem
        pass
    
    # Local search: try to remove comparators or swap them
    # Strategy: iterate through comparators, try removing each one, check if still correct
    # Also try swapping adjacent comparators
    
    rng = random.Random(42)
    
    # Precompute for speed: we'll work with a list and check periodically
    # To speed up, we can check on a subset of inputs first, then verify on all
    
    # For efficiency, let's use a smaller set of test inputs for quick checks
    # Then verify on all 2^N inputs when a candidate looks good
    # Actually, 2^18 = 262144, and each comparator application is O(2^N) with numpy
    # A network of ~100 comparators would be 100 * 262144 operations, which is fine.
    
    # Let's do a greedy removal: try to remove each comparator
    network = list(initial_network)
    
    # First, try removing comparators one by one
    changed = True
    while changed:
        changed = False
        for idx in range(len(network) - 1, -1, -1):
            if time.time() - start_time > DEADLINE - 5:
                break
            # Try removing network[idx]
            candidate = network[:idx] + network[idx+1:]
            if len(candidate) < len(network):
                # Quick check on a subset
                subset_size = min(4096, 1 << N)
                test_inputs = all_inputs[:subset_size].copy()
                result = apply_network(test_inputs, candidate)
                if is_sorted(result):
                    # Full check
                    full_result = apply_network(all_inputs.copy(), candidate)
                    if is_sorted(full_result):
                        network = candidate
                        changed = True
                        best_network = network
                        best_count = len(network)
        if time.time() - start_time > DEADLINE - 5:
            break
    
    # Now try swapping adjacent comparators to enable further removals
    for iteration in range(20):
        if time.time() - start_time > DEADLINE - 5:
            break
        improved = False
        for idx in range(len(network) - 1):
            if time.time() - start_time > DEADLINE - 5:
                break
            # Try swapping network[idx] and network[idx+1]
            candidate = network[:]
            candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
            # Check if this swap allows removing some other comparator
            # For now, just check if the swapped network is still valid
            # (it should be if the original was valid and the swap doesn't break ordering)
            # Actually, swapping comparators can break validity, so we need to check
            subset_size = min(4096, 1 << N)
            test_inputs = all_inputs[:subset_size].copy()
            result = apply_network(test_inputs, candidate)
            if is_sorted(result):
                full_result = apply_network(all_inputs.copy(), candidate)
                if is_sorted(full_result):
                    network = candidate
                    improved = True
        if not improved:
            break
    
    # Try random perturbations
    for _ in range(50):
        if time.time() - start_time > DEADLINE - 5:
            break
        candidate = network[:]
        # Randomly swap two comparators
        if len(candidate) >= 2:
            i = rng.randint(0, len(candidate) - 2)
            candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
            # Try to remove a random comparator
            if len(candidate) > 1:
                j = rng.randint(0, len(candidate) - 1)
                candidate = candidate[:j] + candidate[j+1:]
            subset_size = min(4096, 1 << N)
            test_inputs = all_inputs[:subset_size].copy()
            result = apply_network(test_inputs, candidate)
            if is_sorted(result):
                full_result = apply_network(all_inputs.copy(), candidate)
                if is_sorted(full_result):
                    if len(candidate) < len(network):
                        network = candidate
                        best_network = network
                        best_count = len(network)
    
    return best_network
# EVOLVE-BLOCK-END
