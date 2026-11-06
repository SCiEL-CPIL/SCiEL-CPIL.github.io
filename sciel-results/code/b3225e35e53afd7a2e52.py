"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(primes)
    
    # Precompute mask for each residue class modulo the product of small primes
    # Actually, let's just precompute masks for a range and slide
    
    # We'll search over n in [1000, 1000000]
    # For efficiency, precompute the mask for each number in a large range
    # mask[m] = bitmask of which primes < 24 divide m
    
    max_n = 1000000
    window_size = 24
    # We need masks for numbers up to max_n + 24
    max_val = max_n + 24 + 1  # +1 for safety
    
    # Precompute masks using numpy for speed
    masks = np.zeros(max_val, dtype=np.int32)
    for idx, p in enumerate(primes):
        bit = 1 << idx
        # For multiples of p, set the bit
        multiples = np.arange(p, max_val, p)
        masks[multiples] |= bit
    
    # Now slide a window of size 24 and compute the OR of masks
    # For each position i (representing n=i, window is i+1 to i+24)
    # We need OR of masks[i+1 : i+1+24]
    
    best_n = 1000
    best_count = 9  # max possible is 9
    
    # Use a frequency-based approach to maintain OR in O(1) per step
    # freq[bit] = count of numbers in window that have that bit set
    # A bit is in the OR if freq[bit] > 0
    
    # Initialize for window starting at n=1000 (i.e., numbers 1001 to 1024)
    freq = np.zeros(num_primes, dtype=np.int32)
    for i in range(1001, 1001 + window_size):
        m = masks[i]
        for b in range(num_primes):
            if m & (1 << b):
                freq[b] += 1
    
    current_count = np.sum(freq > 0)
    if current_count < best_count:
        best_count = current_count
        best_n = 1000
    
    # Slide the window
    for n in range(1001, max_n + 1):
        # Remove masks[n] (was at position n, now leaving window)
        # Window was [n, n+23], now becomes [n+1, n+24]
        # So remove masks[n], add masks[n+24]
        m_remove = masks[n]
        for b in range(num_primes):
            if m_remove & (1 << b):
                freq[b] -= 1
        
        m_add = masks[n + 24]
        for b in range(num_primes):
            if m_add & (1 << b):
                freq[b] += 1
        
        current_count = np.sum(freq > 0)
        if current_count < best_count:
            best_count = current_count
            best_n = n
            if best_count == 1:
                break  # Can't do better than 1
    
    return best_n

# EVOLVE-BLOCK-END
