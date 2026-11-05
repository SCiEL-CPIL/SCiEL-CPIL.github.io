import math

def construct():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    MAX_N = 1000000
    MAX_VAL = MAX_N + 24
    
    # Precompute smooth parts
    smooth = [1] * (MAX_VAL + 1)
    for i in range(2, MAX_VAL + 1):
        x = i
        s = 1
        for p in primes:
            if p * p > x:
                break
            while x % p == 0:
                s *= p
                x //= p
        # If x > 1, it's a prime > 23 (or 1), so it doesn't contribute to the smooth part
        smooth[i] = s
        
    best_n = 1000
    best_count = float('inf')
    
    # Search for the best n
    # We can optimize by skipping if the current best is very low, but let's just scan.
    for n in range(1000, MAX_N - 24 + 1):
        # Extract the window
        window = smooth[n+1 : n+25]
        count = len(set(window))
        if count < best_count:
            best_count = count
            best_n = n
            
        # Early termination if we hit the theoretical minimum?
        # The minimum number of distinct smooth parts in 24 consecutive integers.
        # If we find 1, we can't do better.
        # If we find 2, it's very good.
        # Let's just run the full scan, it's fast enough.
        
    return best_n
