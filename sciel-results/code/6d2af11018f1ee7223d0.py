import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s buffer
    
    N_MAX = 1000000 + 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute smooth parts
    smooth_part = [1] * (N_MAX + 1)
    
    for p in primes:
        # For each multiple of p, multiply smooth_part by p as many times as p divides it
        # We can do this by iterating multiples and dividing
        for k in range(p, N_MAX + 1, p):
            x = k
            while x % p == 0:
                x //= p
                smooth_part[k] *= p
                
    # Sliding window to find min distinct count
    window_size = 24
    freq = {}
    distinct_count = 0
    
    # Initialize first window [1000+1, 1000+24] -> indices 1001 to 1024
    # We want n in [1000, 1000000]. Window is n+1 to n+24.
    # So we check windows starting at index 1001.
    
    min_distinct = float('inf')
    best_n = 1000
    
    # Add first 24 elements
    for i in range(1001, 1001 + window_size):
        val = smooth_part[i]
        if val in freq:
            freq[val] += 1
        else:
            freq[val] = 1
            distinct_count += 1
            
    if distinct_count < min_distinct:
        min_distinct = distinct_count
        best_n = 1000
        
    # Slide window
    # Next window starts at 1002, ends at 1025. n becomes 1001.
    # Remove 1001, add 1025.
    for n in range(1001, 1000001):
        if time.time() > deadline:
            break
            
        # Remove element at index n (which was n+1 for previous n-1? No.)
        # Previous window for n-1 was [n, n+23].
        # Current window for n is [n+1, n+24].
        # So we remove index n, add index n+24.
        
        out_val = smooth_part[n]
        freq[out_val] -= 1
        if freq[out_val] == 0:
            distinct_count -= 1
            # del freq[out_val] # optional
            
        in_val = smooth_part[n + 24]
        if in_val in freq:
            freq[in_val] += 1
        else:
            freq[in_val] = 1
            distinct_count += 1
            
        if distinct_count < min_distinct:
            min_distinct = distinct_count
            best_n = n
            # If we reach 2, it's likely the minimum possible (even/odd constraint)
            # But let's check if 1 is possible? No, 1 is odd, 2 is even.
            # Can we have all smooth parts = 1? No, even numbers have smooth part >= 2.
            # So min is at least 2.
            if min_distinct == 2:
                # We can break early if we are confident, but let's just record best
                # Actually, 2 is the theoretical minimum given parity.
                # We can break to save time? The loop is fast.
                pass

    return best_n

# EVOLVE-BLOCK-START
# The function construct is defined above
# EVOLVE-BLOCK-END
