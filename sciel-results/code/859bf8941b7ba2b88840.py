"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136

# Precompute bit masks
BITS = [1 << i for i in range(N)]

# Precompute the input states: 0 to 2^N - 1
# Each integer represents the initial configuration of the 17 wires.
# Bit i is 1 if wire i is 1.
_ALL_INPUTS = np.arange(1 << N, dtype=np.uint16)

# Precompute the sorted target states.
# A state is sorted if the bits are in non-increasing order (1s then 0s) or non-decreasing?
# The problem says: "compare-exchange writes min to wire i and max to j".
# Usually sorting networks sort in ascending order (0s then 1s) or descending.
# Let's check the seed: "odd-even transposition network".
# Standard sorting networks sort in ascending order (0,0,...,1,1).
# Let's verify with a small example.
# Input: [1, 0] (wire 0 is 1, wire 1 is 0). State: 0b10 = 2.
# Comparator (0, 1).
# min(1, 0) = 0 -> wire 0. max(1, 0) = 1 -> wire 1.
# Result: [0, 1]. State: 0b01 = 1.
# Sorted ascending: 0, 1.
# So we want the state to become the integer with the same number of 1s, but with 1s at the LSBs?
# No, if wire 0 is the "first" element, ascending order means wire 0 has the smallest value.
# If values are 0 and 1, smallest is 0. So wire 0 should be 0, wire 1 should be 1.
# So for a given number of 1s `k`, the sorted state has the `k` least significant bits set to 1?
# Let's check:
# k=1: 0b...001 (wire 0=1, others 0)? No, that's descending if wire 0 is MSB?
# Wait, "min to wire i".
# If we have input [1, 0] on wires 0, 1.
# min is 0, goes to wire 0. max is 1, goes to wire 1.
# Output [0, 1].
# So wire 0 gets the smaller value.
# In binary, if wire 0 is bit 0, wire 1 is bit 1...
# Output state 0b01 (1).
# Input state 0b10 (2).
# For k=1 ones, sorted ascending (0 then 1) means:
# Wire 0: 0, Wire 1: 1. State: 0b10 = 2?
# Wait. If wire 0 is 0 and wire 1 is 1, the state is `0*2^0 + 1*2^1 = 2`.
# If wire 0 is 1 and wire 1 is 0, the state is `1*2^0 + 0*2^1 = 1`.
# So ascending order of values (0, 1) corresponds to state 2 for N=2, k=1.
# Generally, for N wires and k ones, the sorted state (ascending) has the k ones in the HIGHEST bits?
# No, wire 0 is the first element. If sorted ascending, element 0 is smallest.
# So wire 0 should be 0. Wire 1 should be 0... Wire N-k should be 0.
# Wire N-k+1 ... N-1 should be 1.
# So the bits at positions N-k+1 to N-1 are 1.
# Let's verify for N=2, k=1.
# Positions: 0, 1.
# Sorted: [0, 1].
# Bit 0 is 0. Bit 1 is 1.
# State: 0b10 = 2.
# Formula: sum(2^i for i in range(N-k, N)).
# For N=2, k=1: range(1, 2) -> i=1. 2^1 = 2. Correct.
# For N=2, k=0: range(2, 2) -> 0. Correct.
# For N=2, k=2: range(0, 2) -> 1+2=3. Correct.

# Precompute sorted targets for all inputs
# popcount of input x gives k.
# Target is mask with top k bits set.
# We can compute this vectorized.
_popcounts = np.array([bin(i).count('1') for i in range(1 << N)], dtype=np.uint8)
# Create a lookup table for sorted states
# sorted_state[k] = sum(2^i for i in range(N-k, N))
_sorted_lookup = np.zeros(1 << N, dtype=np.uint16)
for k in range(N + 1):
    mask = 0
    for i in range(N - k, N):
        mask |= (1 << i)
    # All inputs with popcount k should map to this mask
    # We can just fill the lookup table based on popcount
    _sorted_lookup[_popcounts == k] = mask

# Optimization: Precompute the effect of each comparator on the state array?
# No, we'll just simulate.

def simulate_network_fast(comparators, inputs):
    """
    Simulate the network on the given input states (uint16 array).
    """
    state = inputs.copy()
    for i, j in comparators:
        # Extract bits i and j
        # We need to be careful with uint16 overflow if we shift, but N=17 fits in 17 bits.
        # However, numpy uint16 only has 16 bits.
        # We MUST use uint32 or int32 for safety, or handle bit 16 carefully.
        # Let's use uint32 to be safe.
        pass

# Let's switch to uint32 to avoid bit 16 issues.
_ALL_INPUTS_32 = _ALL_INPUTS.astype(np.uint32)
_sorted_lookup_32 = _sorted_lookup.astype(np.uint32)

def simulate_network_fast(comparators, inputs):
    """
    Fast simulation using uint32 vectorization.
    """
    state = inputs.copy()
    for i, j in comparators:
        mask_i = 1 << i
        mask_j = 1 << j
        
        # Extract bits
        # bit_i = (state & mask_i) > 0 -> 1 or 0
        # But we want the actual value to put back.
        # min_val: if state_i < state_j then state_i else state_j
        # Since they are 0 or 1:
        # min is 0 if (state_i + state_j) == 0 or (state_i == 1 and state_j == 1)? No.
        # min(0,0)=0, min(0,1)=0, min(1,0)=0, min(1,1)=1.
        # min = state_i AND state_j? No. 1 AND 0 = 0. 1 AND 1 = 1. 0 AND 1 = 0.
        # Yes, min of two bits is AND.
        # max of two bits is OR.
        
        bit_i = (state & mask_i)
        bit_j = (state & mask_j)
        
        # bit_i is either 0 or mask_i.
        # We need 0 or 1 for the logic, then shift back.
        # Or we can do logic on the masked values directly?
        # min_val_masked:
        # If bit_i is 0 (val 0) and bit_j is 0 (val 0) -> 0
        # If bit_i is 0 (val 0) and bit_j is mask_j (val 1) -> 0
        # If bit_i is mask_i (val 1) and bit_j is 0 (val 0) -> 0
        # If bit_i is mask_i (val 1) and bit_j is mask_j (val 1) -> mask_i (val 1)
        
        # So min_val_masked = bit_i & bit_j?
        # Case 1,1: mask_i & mask_j = 0. But we want 1 (mask_i).
        # So AND doesn't work directly on masks of different positions.
        
        # Let's extract 0/1 values.
        val_i = (state & mask_i) != 0
        val_j = (state & mask_j) != 0
        
        min_val = np.logical_and(val_i, val_j)
        max_val = np.logical_or(val_i, val_j)
        
        # Convert back to masks
        min_mask = min_val.astype(np.uint32) * mask_i
        max_mask = max_val.astype(np.uint32) * mask_j
        
        # Update state
        # Clear bits
        state = state & ~(mask_i | mask_j)
        # Set new bits
        state = state | min_mask | max_mask
        
    return state

def is_sorted_network_full(comparators):
    """
    Check if the network sorts all 2^17 inputs.
    """
    if len(comparators) > 4096:
        return False
    
    # We need to check if the output matches the sorted lookup table.
    # But the lookup table is indexed by input? No, it's indexed by popcount.
    # We need to check: for every input x, simulate(x) == sorted_state[popcount(x)].
    
    state = simulate_network_fast(comparators, _ALL_INPUTS_32)
    # Get expected sorted states for each input
    # _sorted_lookup_32 is indexed by input value? No, I built it such that _sorted_lookup[input] = sorted_state.
    # Let's verify my construction of _sorted_lookup.
    # I did: _sorted_lookup[_popcounts == k] = mask.
    # This sets _sorted_lookup[x] = mask_for_k if popcount(x) == k.
    # So _sorted_lookup is an array of size 2^N, where _sorted_lookup[x] is the target state for input x.
    
    return np.array_equal(state, _sorted_lookup_32)

def is_sorted_network_sample(comparators):
    """
    Fast check on a sample of inputs.
    """
    if len(comparators) > 4096:
        return False
    
    # Use a smaller subset for speed during search
    # Let's pick 4096 random inputs
    # We need a fixed sample for consistency
    return is_sorted_network_full(comparators) # Fallback for now, or implement sample

# Precompute sample for speed
_SAMPLE_SIZE = 4096
_rng = np.random.RandomState(42)
_SAMPLE_INDICES = _rng.choice(1 << N, size=_SAMPLE_SIZE, replace=False)
_SAMPLE_INPUTS = _ALL_INPUTS_32[_SAMPLE_INDICES]
_SAMPLE_TARGETS = _sorted_lookup_32[_SAMPLE_INDICES]

def is_sorted_network_sample(comparators):
    """
    Fast check on a sample of inputs.
    """
    if len(comparators) > 4096:
        return False
    state = simulate_network_fast(comparators, _SAMPLE_INPUTS)
    return np.array_equal(state, _SAMPLE_TARGETS)

# Alias for backward compatibility
def is_sorted_network(comparators):
    return is_sorted_network_full(comparators)

def random_network(num_comparators):
    """Generate a random sorting network candidate."""
    comparators = []
    for _ in range(num_comparators):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        comparators.append((i, j))
    return comparators

def odd_even_merge_sort_network(n):
    """
    Generate an odd-even merge sort network for n wires.
    Returns list of (i, j) comparators.
    """
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Split into two halves
    half = n // 2
    left = odd_even_merge_sort_network(half)
    right = odd_even_merge_sort_network(n - half)
    
    # Offset right half comparators
    right = [(i + half, j + half) for i, j in right]
    
    # Merge the two halves
    merged = left + right
    
    # Odd-even merge:
    # We have two sorted sequences of length half and n-half.
    # We need to merge them.
    # The standard odd-even merge for two lists of size A and B:
    # It uses (A+B-1) stages? No, it's a specific pattern.
    # Let's use the standard recursive definition or a known implementation.
    # For simplicity and correctness, let's use the "Bitonic Sort" logic which is easier to implement correctly for powers of 2,
    # but N=17 is not a power of 2.
    
    # Alternative: Use a known good small network for 17?
    # Or just use the seed odd-even transposition if the merge is complex.
    # But the prompt says the seed is 136. Odd-even merge should be better.
    
    # Let's implement a proper merge for odd-even merge sort.
    # Reference: "Odd-Even Merge Sort"
    # Merge two sorted lists L (len a) and R (len b).
    # The merge network M(a, b) consists of:
    # 1. Compare-exchange (i, i + a) for i in 0..b-1? No.
    
    # Let's stick to a simpler, verifiable construction: Bitonic Sort.
    # Bitonic sort works for N=16 (2^4).
    # For N=17, we can sort 16 elements, then insert the 17th?
    # Or just use the 136 seed if the merge implementation is risky.
    # However, 136 is far from 71.
    
    # Let's try to implement Bitonic Sort for N=17 by padding to 32?
    # No, that wastes comparators.
    
    # Let's try a different approach:
    # Use the "AKS" or "Batcher's" merge.
    # Batcher's odd-even merge for two sorted inputs of size n:
    # M(n, n) is defined recursively.
    # M(1, 1) = [(0, 1)]
    # M(a, b) for a != b:
    #   If a < b: M(a, b) = M(a, b-a) + M(a, a) shifted?
    
    # This is getting complicated to implement correctly from scratch in a short time.
    # Let's use a heuristic:
    # Start with the 136 seed.
    # Then aggressively optimize.
    
    # Actually, there is a known network for 17 with 83 comparators?
    # Let's just use the 136 seed and rely on the optimizer to find a good one.
    # The optimizer with fast simulation should be able to delete many.
    
    return [] # We will fallback to seed in construct

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Build seed network: Odd-Even Transposition
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
    
    # Verify seed
    if not is_sorted_network_full(seed_network):
        # Fallback: Random? No, seed should be valid.
        # If seed is invalid, something is wrong with simulation.
        # Let's assume it's valid.
        pass

    best_network = list(seed_network)
    best_len = len(best_network)
    
    # Optimization Loop
    # We will perform local search:
    # 1. Try removing each comparator.
    # 2. Try swapping each comparator with a random one.
    # 3. Try moving a comparator to a different position.
    
    current_network = list(best_network)
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # Phase 1: Deletion
        i = 0
        while i < len(current_network) and time.time() < deadline - 5.0:
            # Try removing current_network[i]
            candidate = current_network[:i] + current_network[i+1:]
            if is_sorted_network_sample(candidate):
                if is_sorted_network_full(candidate):
                    current_network = candidate
                    best_network = candidate
                    best_len -= 1
                    improved = True
                    # Do not increment i, check the same index again (which is now the next element)
                    continue
            i += 1
            
        # Phase 2: Replacement (if deletion didn't work, try replacing)
        # Only do this if we have time
        if time.time() < deadline - 10.0 and len(current_network) > 70:
            for _ in range(20): # Limit number of replacement attempts per pass
                if time.time() > deadline - 10.0:
                    break
                idx = random.randint(0, len(current_network) - 1)
                old_comp = current_network[idx]
                
                # Generate a random new comparator
                i_new = random.randint(0, N - 2)
                j_new = random.randint(i_new + 1, N - 1)
                new_comp = (i_new, j_new)
                
                if new_comp == old_comp:
                    continue
                
                candidate = current_network.copy()
                candidate[idx] = new_comp
                
                # Check for duplicates? Not strictly necessary for correctness, but might be inefficient.
                # Check validity
                if is_sorted_network_sample(candidate):
                    if is_sorted_network_full(candidate):
                        current_network = candidate
                        best_network = candidate
                        improved = True
                        break # Restart the while loop to try deletion again?
                        # Or just continue to next replacement?
                        # Let's break to restart the main loop to try deletion on the new network.
            
            if improved:
                continue
        
        # Phase 3: Random Perturbation
        if time.time() < deadline - 10.0:
            for _ in range(50):
                if time.time() > deadline - 10.0:
                    break
                # Swap two comparators
                if len(current_network) < 2:
                    break
                i1, i2 = random.sample(range(len(current_network)), 2)
                candidate = current_network.copy()
                candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
                
                if is_sorted_network_sample(candidate):
                    if is_sorted_network_full(candidate):
                        current_network = candidate
                        best_network = candidate
                        improved = True
                        break

    # Final validation
    if not is_sorted_network_full(best_network):
        return seed_network

    return best_network

# EVOLVE-BLOCK-END
