import math

def construct():
    # Upper bound for n is 1000000, so we need lpf up to 1000001
    max_n = 1000000
    limit = max_n + 1
    
    # Sieve for largest prime factor
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (limit + 1)
    
    # Initialize
    for i in range(2, limit + 1):
        lpf[i] = i
        
    # Sieve
    # For each prime i, update its multiples
    # We only need to check up to sqrt(limit) for primality test optimization, 
    # but the loop structure below handles it by checking if lpf[i] == i
    for i in range(2, int(math.isqrt(limit)) + 1):
        if lpf[i] == i: # i is prime
            for j in range(i, limit + 1, i):
                lpf[j] = i
                
    # Find the best n in [10000, 1000000]
    best_n = 10000
    best_val = float('inf')
    
    for n in range(10000, max_n + 1):
        # Largest prime factor of n(n+1) is max(lpf[n], lpf[n+1])
        val = max(lpf[n], lpf[n+1])
        if val < best_val:
            best_val = val
            best_n = n
            # Optional: early termination if we find 2? 
            # But n and n+1 are consecutive, so one is even. 
            # The other must be odd. If the odd one is 1, n=0, not in range.
            # So min possible is at least 3? No, if n+1 is a power of 2, lpf[n+1]=2.
            # Then lpf[n] must be small.
            # E.g., n=1023 (3*11*31), n+1=1024 (2^10). max(31, 2)=31.
            # There might be smaller.
            # We must scan all to be sure, or break if best_val == 2? 
            # Impossible for n>=10000.
            # Can we break if best_val is very small?
            # The task asks to minimize. We just need the global min.
            
    return best_n
