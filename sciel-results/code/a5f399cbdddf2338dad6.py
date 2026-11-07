import time
import math
import sys
from fractions import Fraction
from collections import defaultdict
import bisect

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    MAX_N = 10_000_000
    
    # Linear sieve for sigma
    # sigma[n] = sigma of n
    # For a prime p: sigma[p] = 1 + p
    # For n = p^k * m where gcd(p, m) = 1: sigma[n] = sigma[m] * (1 + p + ... + p^k)
    
    is_prime = [True] * (MAX_N + 1)
    primes = []
    sigma = [0] * (MAX_N + 1)
    sigma[1] = 1
    # p_power[n] = highest power of smallest prime dividing n
    # Actually, let's use a different approach:
    # spf = smallest prime factor
    # p_power[n] = p^k where p^k is the highest power of spf[n] dividing n
    # Then sigma[n] = sigma[n // p_power[n]] * (p_power[n] * spf[n] - 1) // (spf[n] - 1)
    
    spf = [0] * (MAX_N + 1)
    p_pow = [0] * (MAX_N + 1)
    
    for i in range(2, MAX_N + 1):
        if spf[i] == 0:
            spf[i] = i
            p_pow[i] = i
            primes.append(i)
            sigma[i] = 1 + i
        for p in primes:
            if p > spf[i] or i * p > MAX_N:
                break
            spf[i * p] = p
            if p == spf[i]:
                p_pow[i * p] = p_pow[i] * p
                break
            else:
                p_pow[i * p] = p
    
    for n in range(2, MAX_N + 1):
        if spf[n] != 0 and p_pow[n] != 0:
            p = spf[n]
            pp = p_pow[n]
            m = n // pp
            # sigma[n] = sigma[m] * (pp * p - 1) // (p - 1)
            # pp * p = p^(k+1)
            sigma[n] = sigma[m] * (pp * p - 1) // (p - 1)
    
    # Check time
    if time.time() > deadline:
        return [11, 6]
    
    # Group by sigma
    groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        groups[sigma[n]].append(n)
    
    if time.time() > deadline:
        return [11, 6]
    
    best_diff = None
    best_pair = None
    
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
        if time.time() > deadline:
            break
        
        nums_sorted = sorted(nums)
        k = len(nums_sorted)
        
        for b in nums_sorted:
            target = 3 * b // 2
            # We want a close to 3b/2, so a close to target
            # Search in nums_sorted for values near target
            pos = bisect.bisect_left(nums_sorted, target)
            
            for idx in [pos, pos - 1]:
                if 0 <= idx < k:
                    a = nums_sorted[idx]
                    if a == b:
                        continue
                    diff = abs(2 * a - 3 * b)
                    if diff == 0:
                        continue
                    val = Fraction(diff, 2 * b)
                    if best_diff is None or val < best_diff:
                        best_diff = val
                        best_pair = (a, b)
    
    if best_pair is None:
        return [11, 6]
    
    a, b = best_pair
    return [int(a), int(b)]

# EVOLVE-BLOCK-END
