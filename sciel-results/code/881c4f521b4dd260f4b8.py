"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # We need to check n in [1000, 1000000], so n+i in [1001, 1000024]
    # Precompute smooth parts for range [1001, 1000024]
    low = 1001
    high = 1000024
    size = high - low + 1
    
    # Create array of numbers
    nums = np.arange(low, high + 1, dtype=np.int64)
    
    # For each prime, compute the smooth part contribution
    smooth_parts = np.ones(size, dtype=np.int64)
    
    for p in primes:
        # Compute p^k for k >= 1
        power = p
        while power <= high:
            # Find all positions where nums is divisible by power
            mask = (nums % power == 0)
            # For those, we need to check if they're NOT divisible by power*p
            # Actually, let's think differently: for each number, the exponent of p
            # is the largest k such that p^k divides the number
            
            # For efficiency, let's process powers in descending order
            break
        # Better approach: for each prime, compute the p-part for each number
        # by repeatedly dividing
    
    # Let me redo this more carefully
    smooth_parts = np.ones(size, dtype=np.int64)
    
    for p in primes:
        # Compute the p-adic valuation for each number
        # For each number, divide by p as many times as possible
        # and track the product of p's removed
        current = nums.copy()
        p_part = np.ones(size, dtype=np.int64)
        
        while True:
            mask = (current % p == 0)
            if not np.any(mask):
                break
            p_part[mask] *= p
            current[mask] //= p
        
        smooth_parts *= p_part
    
    # Now smooth_parts[i] corresponds to number low + i
    # For each n in [1000, 1000000], the window is n+1 to n+24
    # which corresponds to indices (n+1-low) to (n+24-low) in smooth_parts
    # n+1-low = n+1-1001 = n-1000
    # n+24-low = n+24-1001 = n-977
    
    # So for n, we look at smooth_parts[n-1000 : n-977+1] = smooth_parts[n-1000 : n-976]
    
    best_n = 1000
    best_count = float('inf')
    
    # Use a sliding window approach
    # For efficiency, let's use a different approach: for each window, count distinct values
    # This is O(24 * N) which is fine for N ~ 1000000
    
    # Actually, let's be smarter. We can use a frequency dictionary approach with sliding window.
    
    from collections import defaultdict
    
    freq = defaultdict(int)
    # Initialize window for n=1000: indices 0 to 23
    for i in range(24):
        freq[smooth_parts[i]] += 1
    
    best_count = len(freq)
    best_n = 1000
    
    # Slide window
    for n in range(1001, 1000001):
        # Remove element at index (n-1000 - 24) = n-1024
        out_idx = n - 1024
        if out_idx >= 0:
            val_out = smooth_parts[out_idx]
            freq[val_out] -= 1
            if freq[val_out] == 0:
                del freq[val_out]
        
        # Add element at index (n-1000 + 23) = n-977
        in_idx = n - 977
        if in_idx < size:
            val_in = smooth_parts[in_idx]
            freq[val_in] += 1
        
        curr_count = len(freq)
        if curr_count < best_count:
            best_count = curr_count
            best_n = n
            if best_count == 1:
                break  # Can't do better than 1
    
    return best_n

# EVOLVE-BLOCK-END
