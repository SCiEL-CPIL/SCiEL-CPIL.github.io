import math
import time
import bisect
from collections import defaultdict

def construct():
    start_time = time.time()
    # 110s deadline to be safe
    deadline = start_time + 110.0
    
    # 1. Generate powerful numbers up to L
    # We aim for L = 10^9. 
    # If generation or counting takes too long, we might need to adjust, 
    # but 10^9 is a good balance.
    L = 10**9
    
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
    
    # 2. Count representations
    counts = {}
    best_N = 0
    best_count = 0
    
    # We iterate a in powerful_list where a <= L/2
    # For each a, we find b in powerful_list where a <= b <= L - a
    # and increment counts[a+b]
    
    # Optimization: Local variables
    p_list = powerful_list
    p_set = powerful_set
    L_half = L // 2
    
    for a in p_list:
        if a > L_half:
            break
            
        # Check time
        if time.time() > deadline:
            break
            
        # Find range of b
        # b >= a and b <= L - a
        left_idx = bisect.bisect_left(p_list, a)
        right_idx = bisect.bisect_right(p_list, L - a)
        
        for i in range(left_idx, right_idx):
            b = p_list[i]
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
        # Fallback for very small cases or errors
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # 3. Retrieve pairs for best_N
    pairs = []
    for a in p_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in p_set:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
