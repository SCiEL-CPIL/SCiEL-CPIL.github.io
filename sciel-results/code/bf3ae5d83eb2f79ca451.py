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
    max_b = int(round(limit ** (1.0 / 3.0))) + 2
    if max_b < 1:
        max_b = 1

    # Primes up to isqrt(max_b)
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
    deadline = start_time + 110 # Safety margin
    
    # Try a few limits to balance coverage and speed
    # L=10^14 gives K~46000, pairs~1e9. Chunked processing should handle it.
    limits_to_try = [10**14, 10**13, 10**12]
    
    best_result = {'N': 5, 'pairs': [[1, 4]]}
    best_count = 1
    
    for limit in limits_to_try:
        try:
            if time.time() > deadline - 30:
                break
                
            gen_start = time.time()
            powerful = _generate_powerful_numbers(limit)
            K = len(powerful)
            
            if time.time() > deadline - 20:
                continue
                
            if K == 0:
                continue
                
            pow_arr = np.array(powerful, dtype=np.int64)
            
            # Chunked counting
            counts = {}
            chunk_size = 1000
            
            for i in range(0, K, chunk_size):
                if time.time() > deadline - 10:
                    break
                    
                chunk = pow_arr[i:i+chunk_size]
                # Compute sums
                sums = chunk[:, None] + pow_arr[None, :]
                # Mask for a <= b
                mask = chunk[:, None] <= pow_arr[None, :]
                
                # Extract valid sums
                valid_sums = sums[mask]
                
                if len(valid_sums) == 0:
                    continue
                
                # Count unique sums
                unique_sums, unique_counts = np.unique(valid_sums, return_counts=True)
                
                for s, c in zip(unique_sums, unique_counts):
                    s_int = int(s)
                    if s_int > 10**18:
                        continue
                    if s_int in counts:
                        counts[s_int] += int(c)
                    else:
                        counts[s_int] = int(c)
                        
            if not counts:
                continue
                
            # Find best N
            best_N = max(counts, key=counts.get)
            current_best_count = counts[best_N]
            
            if current_best_count > best_count:
                best_count = current_best_count
                pow_set = set(powerful)
                pairs = []
                half_N = best_N // 2
                end_idx = np.searchsorted(pow_arr, half_N, side='right')
                
                for a in pow_arr[:end_idx]:
                    b = best_N - a
                    if b in pow_set:
                        pairs.append([int(a), int(b)])
                        
                best_result = {'N': best_N, 'pairs': pairs}
                
        except MemoryError:
            continue
        except Exception as e:
            continue
            
    return best_result

if __name__ == "__main__":
    res = construct()
    print(res['N'], len(res['pairs']))
