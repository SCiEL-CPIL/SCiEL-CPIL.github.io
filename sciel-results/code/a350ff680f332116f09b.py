"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
_DEADLINE = 45  # seconds, safety margin before 90s kill

# EVOLVE-BLOCK-START
# Pre-compute expected sorted masks for each possible popcount
_EXPECTED_MASKS = {}
def _init_expected_masks():
    """Pre-compute expected sorted masks for all possible popcounts."""
    for k in range(N + 1):
        if k == 0:
            _EXPECTED_MASKS[k] = 0
        else:
            _EXPECTED_MASKS[k] = (1 << k) - 1
_init_expected_masks()

def _popcount(mask):
    """Count set bits in mask."""
    return bin(mask).count('1')

def _verify_fast(network):
    """Fast verification using zero-one principle with pre-computed expected masks."""
    for mask in range(1 << N):
        for i, j in network:
            vi = (mask >> i) & 1
            vj = (mask >> j) & 1
            if vi > vj:
                mask |= (1 << i)
                mask &= ~(1 << j)
            elif vi < vj:
                mask &= ~(1 << i)
                mask |= (1 << j)
        k = _popcount(mask)
        if mask != _EXPECTED_MASKS[k]:
            return False
    return True

def _verify_numpy(network):
    """Vectorized verification using NumPy for speed."""
    n_comps = len(network)
    if n_comps == 0:
        return True
    
    # All 2^N masks
    masks = np.arange(1 << N, dtype=np.int32)
    
    # Precompute bit values for faster access
    bit_vals = [1 << i for i in range(N)]
    
    # Process each comparator
    for i, j in network:
        bi = bit_vals[i]
        bj = bit_vals[j]
        
        # Extract bits
        vi = (masks & bi) != 0
        vj = (masks & bj) != 0
        
        # Case 1: vi=1, vj=0 -> swap (1,0) to (0,1)
        # This means bit i was 1, bit j was 0. After swap, bit i=0, bit j=1.
        # We can do this by XORing both bits.
        swap_10 = vi & ~vj
        if np.any(swap_10):
            masks[swap_10] ^= bi
            masks[swap_10] ^= bj
            
        # Case 2: vi=0, vj=1 -> swap (0,1) to (1,0)
        # This means bit i was 0, bit j was 1. After swap, bit i=1, bit j=0.
        swap_01 = ~vi & vj
        if np.any(swap_01):
            masks[swap_01] ^= bi
            masks[swap_01] ^= bj
    
    # Check if all masks are sorted
    # Instead of popcount loop, we can check if the mask is contiguous from LSB.
    # A mask is sorted if it is of the form (1 << k) - 1.
    # We can precompute a set of valid sorted masks or check against expected array.
    # Using precomputed expected masks is O(1) per mask if we had a lookup, 
    # but we have an array of 131k masks.
    
    # Efficient check: 
    # For each mask, if it is sorted, it must be in _EXPECTED_MASKS.
    # We can vectorize the check by comparing against a precomputed array of expected masks for all possible popcounts?
    # No, popcount varies.
    
    # Let's use a fast popcount method with NumPy if available, or just check the property:
    # A mask is sorted if (mask & (mask + 1)) == 0 is NOT sufficient (that's power of 2).
    # Sorted masks are 0, 1, 3, 7, ... 2^k - 1.
    # We can check if mask is in the set of valid masks.
    # Since N=17, there are only 18 valid masks.
    valid_masks_set = set(_EXPECTED_MASKS.values())
    
    # Check if all masks are in valid_masks_set
    # This is a set membership check for 131k elements. 
    # We can do this by checking if the mask is one of the 18 values.
    # np.isin is slow for large arrays.
    
    # Alternative: 
    # The sorted mask for a given popcount k is (1 << k) - 1.
    # We can compute popcount using a lookup table for 16-bit chunks?
    # Or simply: 
    # If mask is sorted, then mask is (1 << k) - 1.
    # This implies mask + 1 is a power of 2.
    # And mask & (mask + 1) == 0.
    # AND mask < 2^N.
    # AND mask is not 0? 0 is (1<<0)-1.
    
    # Let's test this condition:
    # m is sorted iff (m & (m + 1)) == 0
    # Is this true?
    # 0: 0 & 1 = 0. Yes.
    # 1: 1 & 2 = 0. Yes.
    # 3: 3 & 4 = 0. Yes.
    # 7: 7 & 8 = 0. Yes.
    # 2: 2 & 3 = 2 != 0. No.
    # 5: 5 & 6 = 4 != 0. No.
    # This condition is equivalent to "m is of the form 2^k - 1".
    # Which is exactly the set of sorted masks for N wires.
    
    return np.all((masks & (masks + 1)) == 0)

def _seed_network():
    """Odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _find_redundant_indices(network):
    """Find indices of comparators that appear more than once (redundant candidates)."""
    from collections import Counter
    counter = Counter(network)
    redundant = []
    for idx, comp in enumerate(network):
        if counter[comp] > 1:
            redundant.append(idx)
    return redundant

def construct():
    start_time = time.time()
    network = _seed_network()
    best = list(network)
    best_len = len(best)
    
    # Verify seed
    if not _verify_numpy(network):
        return best
    
    # Phase 1: Greedy removal passes (using fast numpy verifier)
    passes = 0
    max_passes = 50  # Increased passes due to faster verification
    
    while passes < max_passes:
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 10:
            break
        
        # Strategy 1: Reverse-order removal (end of network is often less critical)
        progress = False
        for attempt in range(len(network) - 1, -1, -1):
            if time.time() - start_time > _DEADLINE - 10:
                break
            candidate = list(network)
            candidate.pop(attempt)
            if _verify_numpy(candidate):
                network = candidate
                best = list(candidate)
                best_len = len(best)
                progress = True
                break # Restart loop from end again
        
        # Strategy 2: Forward-order removal
        if not progress:
            i = 0
            while i < len(network):
                if time.time() - start_time > _DEADLINE - 10:
                    break
                candidate = list(network)
                candidate.pop(i)
                if _verify_numpy(candidate):
                    network = candidate
                    best = list(candidate)
                    best_len = len(best)
                    progress = True
                else:
                    i += 1
        
        # Check if we made any progress this pass
        if not progress:
            # No progress, try one more pass with shuffled order
            if time.time() - start_time > _DEADLINE - 10:
                break
            indices = list(range(len(network)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() - start_time > _DEADLINE - 10:
                    break
                candidate = list(network)
                candidate.pop(idx)
                if _verify_numpy(candidate):
                    network = candidate
                    best = list(candidate)
                    best_len = len(best)
                    progress = True
                    break
        
        if not progress:
            break # If no strategy worked, break out of greedy loop
            
        passes += 1
    
    # Phase 2: Simulated annealing to escape local minima
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 5:
        # Use numpy verification for speed in SA
        current = list(network)
        current_len = len(current)
        sa_best = list(best)
        sa_best_len = best_len
        
        # Temperature schedule
        T = 2.0
        T_min = 0.01
        cooling = 0.99
        
        # Pre-compute all possible comparators
        all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
        
        steps = 0
        max_steps = 100000 # Increased steps
        
        while T > T_min and steps < max_steps:
            if time.time() - start_time > _DEADLINE - 5:
                break
            
            steps += 1
            step_type = random.random()
            
            if step_type < 0.5:
                # Try removing a random comparator
                if len(current) > 0:
                    idx = random.randint(0, len(current) - 1)
                    candidate = list(current)
                    candidate.pop(idx)
                    if _verify_numpy(candidate):
                        current = candidate
                        if len(current) < sa_best_len:
                            sa_best = list(current)
                            sa_best_len = len(current)
                    else:
                        # Accept with probability exp(-delta/T) where delta = 1 (worse)
                        if random.random() < np.exp(-1.0 / T):
                            current = candidate
            elif step_type < 0.8:
                # Try swapping two comparators
                if len(current) >= 2:
                    i, j = random.sample(range(len(current)), 2)
                    candidate = list(current)
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    if _verify_numpy(candidate):
                        current = candidate
                        if len(current) < sa_best_len:
                            sa_best = list(current)
                            sa_best_len = len(current)
            else:
                # Try replacing a comparator with a random one
                if len(current) > 0:
                    idx = random.randint(0, len(current) - 1)
                    new_comp = random.choice(all_comps)
                    candidate = list(current)
                    candidate[idx] = new_comp
                    if _verify_numpy(candidate):
                        current = candidate
                        if len(current) < sa_best_len:
                            sa_best = list(current)
                            sa_best_len = len(current)
                    else:
                        if random.random() < np.exp(-1.0 / T):
                            current = candidate
            
            T *= cooling
        
        # Update best if SA found something better
        if sa_best_len < best_len:
            best = sa_best
            best_len = sa_best_len
    
    return best
# EVOLVE-BLOCK-END
