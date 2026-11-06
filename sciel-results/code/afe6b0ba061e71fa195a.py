# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    MAX_N = 1000000
    MAX_VAL = MAX_N + 24
    
    # Precompute smooth_part[x] = product of p^a for p in primes where p^a || x
    # This is the "complete factor composed of primes strictly below 24"
    smooth_part = [1] * (MAX_VAL + 1)
    
    for p in primes:
        for multiple in range(p, MAX_VAL + 1, p):
            # Count how many times p divides multiple
            temp = multiple
            power = 1
            while temp % p == 0:
                power *= p
                temp //= p
            smooth_part[multiple] *= power
    
    # Now for each candidate n in [1000, 1000000-24], we want to minimize
    # the number of distinct values in {smooth_part[n+1], ..., smooth_part[n+24]}
    
    # Use a sliding window approach with a frequency counter for efficiency
    # Window size = 24, we slide from n=1000 to n=999976
    
    best_n = 1000
    best_count = float('inf')
    
    # Initialize frequency dict for window [1000+1, 1000+24] = [1001, 1024]
    freq = {}
    for i in range(1, 25):
        v = smooth_part[1000 + i]
        freq[v] = freq.get(v, 0) + 1
    
    current_distinct = len(freq)
    if current_distinct < best_count:
        best_count = current_distinct
        best_n = 1000
    
    # Slide the window
    # Window for n: indices n+1 to n+24
    # When we move from n to n+1: remove smooth_part[n+1], add smooth_part[n+25]
    
    for n in range(1001, MAX_N - 23):  # n goes from 1001 to 999976
        if time.time() - start_time > DEADLINE:
            break
        
        # Remove the element that leaves the window: smooth_part[n] (was n+1 in previous window)
        leaving = smooth_part[n]
        freq[leaving] -= 1
        if freq[leaving] == 0:
            del freq[leaving]
            current_distinct -= 1
        
        # Add the new element: smooth_part[n + 24]
        entering = smooth_part[n + 24]
        if entering in freq:
            freq[entering] += 1
        else:
            freq[entering] = 1
            current_distinct += 1
        
        if current_distinct < best_count:
            best_count = current_distinct
            best_n = n
            if best_count == 1:
                break  # Can't do better than 1
    
    return best_n
# EVOLVE-BLOCK-END
