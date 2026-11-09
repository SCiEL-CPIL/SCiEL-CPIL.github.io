import math
import time
import bisect
from collections import defaultdict

def construct():
    start_time = time.time()
    # 110s deadline to be safe
    deadline = start_time + 110.0
    
    # Strategy: Search a larger range than the current program (10^9) 
    # but small enough to allow O(M^2) counting.
    # M(L) ~ L^(1/3). 
    # For L=10^13, M ~ 2e4, M^2 ~ 4e8 (too slow in Python).
    # For L=10^12, M ~ 1e4, M^2 ~ 1e8 (fast).
    # Let's try L=10^12 first, and if time permits, we could extend, 
    # but fixed L is simpler and safer.
    
    L = 10**12
    
    # 1. Generate powerful numbers up to L
    # Powerful numbers are of the form a^2 * b^3 where b is square-free.
    # b <= L^(1/3)
    # a <= sqrt(L / b^3)
    
    # Generate primes up to L^(1/3) for square-free check
    limit_primes = int(L ** (1.0/3.0)) + 2
    is_prime = [True] * (limit_primes + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit_primes ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_primes + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    
    def is_square_free(n):
        if n < 2:
            return True
        for p in primes:
            if p * p > n:
                break
            if n % (p * p) == 0:
                return False
        return True
        
    powerful_set = set()
    max_b = int(L ** (1.0/3.0))
    for b in range(1, max_b + 1):
        if not is_square_free(b):
            continue
        b3 = b ** 3
        if b3 > L:
            break
        max_a = int(math.isqrt(L // b3))
        for a in range(1, max_a + 1):
            val = a * a * b3
            powerful_set.add(val)
            
    powerful_list = sorted(powerful_set)
    powerful_set = set(powerful_list)
    M = len(powerful_list)
    
    # 2. Count representations
    # We use a dictionary to count sums.
    # To speed up, we only iterate a <= b.
    counts = {}
    best_N = 0
    best_count = 0
    
    p_list = powerful_list
    L_half = L // 2 # We only care about N <= L? 
    # Actually, if a,b <= L, then N <= 2L. 
    # But we want to find the N with max representations.
    # We can just count all sums.
    
    for i, a in enumerate(p_list):
        if time.time() > deadline:
            break
        # b must be >= a
        # We can just iterate from i to end
        for j in range(i, M):
            b = p_list[j]
            N = a + b
            # Update count
            if N in counts:
                c = counts[N] + 1
                counts[N] = c
            else:
                counts[N] = 1
                c = 1
            
            if c > best_count:
                best_count = c
                best_N = N
                
    if best_N == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # 3. Retrieve pairs for best_N
    pairs = []
    p_set = powerful_set
    for a in p_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in p_set:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
