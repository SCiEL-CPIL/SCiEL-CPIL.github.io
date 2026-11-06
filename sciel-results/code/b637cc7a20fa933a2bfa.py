import time
import sys
import numpy as np
from math import isqrt
from collections import defaultdict

def _generate_powerful_numbers(limit):
    """
    Generate all powerful numbers up to limit.
    A powerful number is of the form u^2 * v^3 where v is square-free.
    """
    if limit < 1:
        return []
    
    powerful = set()
    powerful.add(1)
    
    # v must be square-free and v^3 <= limit
    max_v = int(round(limit ** (1.0 / 3.0))) + 2
    if max_v < 1:
        max_v = 1

    # Primes up to isqrt(max_v) for square-free check
    sieve_limit = isqrt(max_v) + 1
    if sieve_limit < 2:
        sieve_limit = 2
    
    # Simple sieve for primes up to sieve_limit
    is_prime = bytearray(b'\x01') * (sieve_limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, isqrt(sieve_limit) + 1):
        if is_prime[i]:
            for j in range(i * i, sieve_limit + 1, i):
                is_prime[j] = 0
    primes = [i for i in range(2, sieve_limit + 1) if is_prime[i]]
    
    def _is_square_free(n):
        if n == 1:
            return True
        t = n
        for p in primes:
            if p * p > t:
                break
            if t % (p * p) == 0:
                return False
        return True

    for v in range(1, max_v + 1):
        if not _is_square_free(v):
            continue
            
        v3 = v * v * v
        if v3 > limit:
            continue
            
        max_u = isqrt(limit // v3)
        for u in range(1, max_u + 1):
            val = u * u * v3
            if val <= limit:
                powerful.add(val)
            else:
                break
                
    return sorted(powerful)

def _is_powerful(n):
    """
    Check if n is powerful using exact factorization.
    Efficiently checks if the square-free part is 1.
    """
    if n == 1:
        return True
    if n <= 0:
        return False
    
    t = n
    # Check factor 2
    if t % 2 == 0:
        exp = 0
        while t % 2 == 0:
            t //= 2
            exp += 1
        if exp == 1:
            return False
            
    p = 3
    while p * p <= t:
        if t % p == 0:
            exp = 0
            while t % p == 0:
                t //= p
                exp += 1
            if exp == 1:
                return False
        p += 2
        
    if t > 1:
        return False
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Strategy:
    # 1. Generate powerful numbers up to a limit L.
    # 2. Use a dictionary to count sums of pairs (a, b) where a <= b.
    # 3. Find the N with the maximum count.
    # 4. Verify all pairs with exact factorization and extract them.
    
    # The number of powerful numbers up to L is roughly C * L^(1/3).
    # For L = 10^12, K ~ 10^4. Pairs ~ 5 * 10^7.
    # For L = 10^13, K ~ 2 * 10^4. Pairs ~ 2 * 10^8.
    
    # We want to maximize the peak count, so we want a large K.
    # But we are limited by time.
    
    # Let's try a few limits, starting with a large one.
    # If it takes too long, we fall back to smaller ones.
    
    limits_to_try = [
        20_000_000_000_000,  # 2e13
        10_000_000_000_000,  # 1e13
        5_000_000_000_000,   # 5e12
        2_000_000_000_000,   # 2e12
        1_000_000_000_000,   # 1e12
        500_000_000_000,     # 5e11
        200_000_000_000,     # 2e11
    ]
    
    best_result = {'N': 5, 'pairs': [[1, 4]]}
    
    for limit in limits_to_try:
        if time.time() > deadline - 20:
            break
            
        try:
            gen_start = time.time()
            powerful = _generate_powerful_numbers(limit)
            gen_time = time.time() - gen_start
            
            if time.time() > deadline - 15:
                continue
                
            K = len(powerful)
            if K == 0:
                continue
                
            # Estimate time for O(K^2) loop
            # Roughly 100ns per iteration in Python
            est_time = (K * (K + 1) / 2) * 1e-7
            if est_time > 10:  # Too slow
                continue
                
            pow_arr = np.array(powerful, dtype=np.int64)
            
            # Count sums
            counts = defaultdict(int)
            
            # Use a simple nested loop but optimized
            # We can use numpy to generate all sums for a fixed i?
            # sums = pow_arr[i] + pow_arr[i:]
            # Then use np.unique to get counts?
            # That might be faster.
            
            # Let's try the numpy approach for speed.
            # For each i, compute sums = a_i + pow_arr[i:]
            # unique_sums, unique_counts = np.unique(sums, return_counts=True)
            # Then add to global counts.
            
            # This is O(K^2 log K) total, but the log factor is small and numpy is fast.
            
            for i in range(K):
                if time.time() > deadline - 10:
                    break
                a = pow_arr[i]
                sums = a + pow_arr[i:]
                unique_sums, unique_counts = np.unique(sums, return_counts=True)
                for s, c in zip(unique_sums, unique_counts):
                    counts[s] += c
                    
            if not counts:
                continue
                
            best_N = max(counts, key=counts.get)
            best_count = counts[best_N]
            
            if best_N > 10**18:
                continue
            
            # Extract pairs and verify with exact factorization
            pow_set = set(powerful)
            
            # We need primes up to sqrt(best_N) for factorization.
            # sqrt(10^18) = 10^9. That's too many primes to sieve.
            # Instead, we can factorize using trial division up to cbrt(n) or so,
            # and handle the remaining factor.
            
            # Actually, for checking if n is powerful, we can do:
            # 1. Check if n is a perfect square times a cube of a square-free number?
            #    No, the definition is: for every prime p|n, p^2|n.
            #    This is equivalent to: the square-free part of n is 1.
            #    i.e., n = m^2 * k where k is powerful? No.
            #    n is powerful iff for all p|n, v_p(n) >= 2.
            
            # Efficient check:
            # Let n = s * t^2 where s is square-free.
            # n is powerful iff s = 1.
            # So we just need to extract the square-free part.
            
            pairs = []
            half_N = best_N // 2
            end_idx = np.searchsorted(pow_arr, half_N, side='right')
            
            for a in pow_arr[:end_idx]:
                b = best_N - a
                if b in pow_set:
                    # Verify with exact factorization
                    # Since a and b are from our generated list, they are powerful by construction.
                    # But the task says "checked by exact factorization".
                    # We should verify.
                    if _is_powerful(int(a)) and _is_powerful(int(b)):
                        pairs.append([int(a), int(b)])
                        
            if pairs:
                if len(pairs) > len(best_result['pairs']):
                    best_result = {'N': best_N, 'pairs': pairs}
                    
            # Return immediately if we found a good result from the largest limit.
            # The score is the number of pairs.
            # Larger limit -> more powerful numbers -> potentially higher peak count.
            return best_result
            
        except Exception as e:
            continue
            
    return best_result

if __name__ == "__main__":
    res = construct()
    print(res['N'], len(res['pairs']))
