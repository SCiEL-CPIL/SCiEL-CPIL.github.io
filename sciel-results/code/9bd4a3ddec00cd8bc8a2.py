"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(primes)
    
    max_n = 1000000
    window_size = 24
    max_val = max_n + window_size + 10  # Safety margin

    # Precompute masks for each number in the range
    # mask[m] is an integer 0..511 representing which primes divide m
    masks = np.zeros(max_val, dtype=np.int32)
    for idx, p in enumerate(primes):
        bit = 1 << idx
        multiples = np.arange(p, max_val, p)
        masks[multiples] |= bit

    # Precompute indices for each prime bit in the 512 possible mask values
    # bit_indices[j] contains all mask values (0-511) that have bit j set
    bit_indices = []
    for j in range(num_primes):
        bit_val = 1 << j
        indices = [i for i in range(512) if (i & bit_val) != 0]
        bit_indices.append(np.array(indices, dtype=np.int32))

    # Frequency array for mask values
    freq = np.zeros(512, dtype=np.int32)
    
    # Initialize window for n=1000 (numbers 1001 to 1024)
    start_idx = 1000 + 1
    end_idx = 1000 + 24 + 1
    initial_masks = masks[start_idx:end_idx]
    freq = np.bincount(initial_masks, minlength=512).astype(np.int32)
    
    # Helper to calculate distinct prime count
    def get_distinct_count():
        count = 0
        for j in range(num_primes):
            # Check if sum of frequencies for masks with bit j set is > 0
            if np.sum(freq[bit_indices[j]]) > 0:
                count += 1
        return count

    best_n = 1000
    best_count = get_distinct_count()
    
    # Slide the window
    for n in range(1001, max_n + 1):
        # Remove masks[n] (leaving window)
        # Add masks[n+24] (entering window)
        freq[masks[n]] -= 1
        freq[masks[n + 24]] += 1
        
        current_count = get_distinct_count()
        
        if current_count < best_count:
            best_count = current_count
            best_n = n
            if best_count == 1:
                break

    return best_n

# EVOLVE-BLOCK-END
