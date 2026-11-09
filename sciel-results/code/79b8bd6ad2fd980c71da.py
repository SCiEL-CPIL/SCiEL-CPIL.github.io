import time
import math
from heapq import heappush, heappop

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 1000001
    
    # First, get all primes up to sqrt(N) for trial division
    sqrt_N = int(math.isqrt(N))
    is_prime = [True] * (sqrt_N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(sqrt_N)) + 1):
        if is_prime[i]:
            for j in range(i*i, sqrt_N+1, i):
                is_prime[j] = False
    small_primes = [i for i in range(2, sqrt_N+1) if is_prime[i]]
    
    def largest_prime_factor(x):
        """Compute largest prime factor of x using trial division by small primes"""
        if x <= 1:
            return 1
        lpf = 1
        for p in small_primes:
            if p * p > x:
                break
            if x % p == 0:
                lpf = p
                while x % p == 0:
                    x //= p
        if x > 1:
            lpf = x  # x is prime (or 1)
        return lpf
    
    # Strategy: Use a priority queue to generate numbers in order of their largest prime factor
    # We want to find consecutive pairs (n, n+1) where both have small lpf
    
    # Alternative: Scan with early termination - if we find a pair with lpf <= B, 
    # we can stop increasing B once we've found the minimum
    
    # Let's do a smart scan: for each n, compute lpf(n) and lpf(n+1), track the max
    # But use memoization and early stopping
    
    # Better approach: Precompute lpf for all numbers using a segmented sieve-like method
    # but only for the range we care about, using the small primes
    
    # Actually, let's just do the full scan but with a faster lpf computation
    # Cache lpf values to avoid recomputation
    
    lpf_cache = {}
    
    def get_lpf(x):
        if x in lpf_cache:
            return lpf_cache[x]
        result = largest_prime_factor(x)
        lpf_cache[x] = result
        return result
    
    best_n = 10000
    best_val = float('inf')
    
    # Scan all n in [10000, 1000000]
    for n in range(10000, 1000001):
        if time.time() > deadline:
            break
        
        lpf_n = get_lpf(n)
        lpf_np1 = get_lpf(n + 1)
        current_max = max(lpf_n, lpf_np1)
        
        if current_max < best_val:
            best_val = current_max
            best_n = n
    
    return best_n
