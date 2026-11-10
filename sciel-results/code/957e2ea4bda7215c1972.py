import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    N = 1000000
    LOW = 10000
    
    # Build smallest prime factor array using NumPy
    spf = np.arange(N + 1, dtype=np.int64)
    spf[0] = 0
    spf[1] = 1
    
    limit = int(math.isqrt(N))
    for i in range(2, limit + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute largest prime factor for each number
    lpf = np.ones(N + 1, dtype=np.int64)
    for i in range(2, N + 1):
        x = i
        max_pf = 1
        while x > 1:
            p = int(spf[x])
            if p > max_pf:
                max_pf = p
            while x % p == 0:
                x //= p
        lpf[i] = max_pf
    
    if time.time() > deadline:
        return LOW
    
    # Vectorized scan for best consecutive pair
    window = np.maximum(lpf[LOW:N], lpf[LOW+1:N+1])
    best_idx = int(np.argmin(window))
    best_n = LOW + best_idx
    best_val = int(window[best_idx])
    
    # Check if we have time to try a different approach
    if time.time() < deadline - 10:
        # Try iterating over possible largest prime factor P in increasing order
        # For each P, find if there are two consecutive numbers in [LOW, N] 
        # that are both P-smooth (all prime factors <= P)
        
        # Get all primes up to N
        primes = [i for i in range(2, N + 1) if spf[i] == i]
        
        # Binary search for the minimum P such that there exists n in [LOW, N-1]
        # where both n and n+1 have all prime factors <= P
        # This is equivalent to: max(lpf[n], lpf[n+1]) <= P
        
        # We already found best_val = min over n of max(lpf[n], lpf[n+1])
        # So the answer is best_n with value best_val
        
        # Let's verify by checking if we can do better with a smarter search
        # Actually, the current approach already finds the global minimum
        # because we're computing lpf for all numbers and taking the min of max
        
        # The only improvement would be to find it faster, but we've already found it
        # So we just return the best we found
    
    return best_n
