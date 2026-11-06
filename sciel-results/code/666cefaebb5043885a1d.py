import math

def construct():
    N = 123457
    
    # Precompute largest prime factor for all numbers up to N
    lpf = [0] * (N + 1)
    lpf[1] = 1
    for i in range(2, N + 1):
        if lpf[i] == 0:
            # i is a prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    # Get list of unique prime values that appear as largest prime factors
    # Or just list of primes up to N
    primes = [i for i in range(2, N + 1) if lpf[i] == i]
    
    # Binary search on the prime bound B
    # We want the smallest B such that there exists a, b with a+b=N and lpf[a]<=B and lpf[b]<=B
    low = 0
    high = len(primes) - 1
    best_pair = None
    
    # Helper to check if a pair exists for a given bound B
    def find_pair(B):
        for a in range(1, N):
            if lpf[a] <= B and lpf[N - a] <= B:
                return (a, N - a)
        return None
    
    while low <= high:
        mid = (low + high) // 2
        B = primes[mid]
        pair = find_pair(B)
        if pair:
            best_pair = pair
            high = mid - 1
        else:
            low = mid + 1
            
    # If no pair found in binary search (should not happen), fallback to largest prime
    if best_pair is None:
        best_pair = find_pair(primes[-1])
        
    return [best_pair[0], best_pair[1]]
