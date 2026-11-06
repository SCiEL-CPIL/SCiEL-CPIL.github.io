import time
import sys
import numpy as np
from math import isqrt

def _generate_powerful_numbers(limit):
    """
    Generate all powerful numbers up to limit.
    A powerful number is of the form a^2 * b^3 where b is square-free.
    """
    if limit < 1:
        return []
    
    powerful = set()
    powerful.add(1)
    
    # b must be square-free and b^3 <= limit
    # max_b such that b^3 <= limit
    max_b = int(round(limit ** (1.0 / 3.0))) + 2
    if max_b < 1:
        max_b = 1

    # We need to check if b is square-free.
    # Primes up to isqrt(max_b) are sufficient to check square-freeness of b.
    sieve_limit = isqrt(max_b) + 1
    if sieve_limit < 2:
        primes = []
    else:
        is_prime = bytearray(b'\x01') * (sieve_limit + 1)
        is_prime[0] = 0
        if sieve_limit >= 1:
            is_prime[1] = 0
        for i in range(2, isqrt(sieve_limit) + 1):
            if is_prime[i]:
                for j in range(i * i, sieve_limit + 1, i):
                    is_prime[j] = 0
        primes = [i for i in range(2, sieve_limit + 1) if is_prime[i]]
        
    for b in range(1, max_b + 1):
        # Check if b is square-free
        t = b
        is_sf = True
        for p in primes:
            pp = p * p
            if pp > t:
                break
            if t % pp == 0:
                is_sf = False
                break
        if not is_sf:
            continue
            
        b3 = b * b * b
        if b3 > limit:
            continue
            
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
            else:
                break
                
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Increased limits to allow for larger K (number of powerful numbers)
    # K ~ C * L^(1/3). 
    # L=5e11 -> K~20000. Pairs ~ 2e8. Memory ~ 4-8GB. Time ~ 10-20s.
    # L=1e11 -> K~12000. Pairs ~ 7e7. Memory ~ 2-4GB. Time ~ 5-10s.
    limits_to_try = [500_000_000_000, 100_000_000_000, 50_000_000_000, 10_000_000_000]
    
    best_result = {'N': 5, 'pairs': [[1, 4]]}
    
    for limit in limits_to_try:
        try:
            if time.time() > deadline - 40:
                break
                
            gen_start = time.time()
            powerful = _generate_powerful_numbers(limit)
            
            if time.time() > deadline - 30:
                continue
                
            K = len(powerful)
            if K == 0:
                continue
                
            pow_arr = np.array(powerful, dtype=np.int64)
            
            # Estimate memory usage for counts dict.
            # Number of pairs is K*(K+1)/2.
            # Each dict entry takes ~50-100 bytes.
            # If K=20000, pairs=2e8. 2e8 * 100B = 20GB. Too much.
            # If K=10000, pairs=5e7. 5e7 * 100B = 5GB. Risky.
            # If K=5000, pairs=1.25e7. 1.25e7 * 100B = 1.25GB. Safe.
            
            # We need to be conservative with K to avoid MemoryError.
            # Let's cap K at 8000 to be safe (~32M pairs, ~1.5GB).
            if K > 8000:
                continue
            
            counts = {}
            # Optimized loop
            for i in range(K):
                a = pow_arr[i]
                # Use local reference for speed
                for j in range(i, K):
                    s = a + pow_arr[j]
                    if s in counts:
                        counts[s] += 1
                    else:
                        counts[s] = 1
                        
            if not counts:
                continue
                
            best_N = max(counts, key=counts.get)
            best_count = counts[best_N]
            
            if best_N > 10**18:
                continue
            
            pow_set = set(powerful)
            pairs = []
            half_N = best_N // 2
            end_idx = np.searchsorted(pow_arr, half_N, side='right')
            
            for a in pow_arr[:end_idx]:
                b = best_N - a
                if b in pow_set:
                    pairs.append([int(a), int(b)])
            
            if pairs:
                if len(pairs) > len(best_result['pairs']):
                    best_result = {'N': best_N, 'pairs': pairs}
                    
            return best_result

        except MemoryError:
            continue
        except Exception as e:
            continue
            
    return best_result

if __name__ == "__main__":
    res = construct()
    print(res['N'], len(res['pairs']))
