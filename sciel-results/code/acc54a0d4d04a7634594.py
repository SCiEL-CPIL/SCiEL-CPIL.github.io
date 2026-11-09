import math
import time
import numpy as np
from collections import defaultdict
from itertools import product

def _sigma(n):
    if n == 1:
        return 1
    result = 1
    temp = n
    # factor out 2
    cnt = 0
    while temp % 2 == 0:
        temp //= 2
        cnt += 1
    if cnt:
        result *= (2**(cnt+1) - 1)
    # odd primes
    p = 3
    while p * p <= temp:
        if temp % p == 0:
            cnt = 0
            while temp % p == 0:
                temp //= p
                cnt += 1
            result *= (p**(cnt+1) - 1) // (p - 1)
        p += 2
    if temp > 1:
        result *= (temp + 1)
    return result

def _factorize(n):
    """Factorize n into prime factors. Returns dict {prime: exponent}."""
    if n == 1:
        return {}
    factors = {}
    temp = n
    # factor out 2
    cnt = 0
    while temp % 2 == 0:
        temp //= 2
        cnt += 1
    if cnt:
        factors[2] = cnt
    # odd primes
    p = 3
    while p * p <= temp:
        if temp % p == 0:
            cnt = 0
            while temp % p == 0:
                temp //= p
                cnt += 1
            factors[p] = cnt
        p += 2
    if temp > 1:
        factors[temp] = 1
    return factors

def _divisors_from_factorization(factors):
    """Generate all divisors from prime factorization."""
    if not factors:
        return [1]
    primes = list(factors.keys())
    exponents = [factors[p] for p in primes]
    
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divs = []
        for d in divisors:
            for exp in range(e + 1):
                new_divs.append(d * (p ** exp))
        divisors = new_divs
    return divisors

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Known solution
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    # Strategy 1: Use vectorized sieve to compute sigma(k) for all k up to N efficiently
    N = 5000000
    sigma_arr = np.zeros(N + 1, dtype=np.int64)
    
    for d in range(1, N + 1):
        sigma_arr[d::d] += d
        if time.time() > deadline - 60.0:
            break
    
    # Group by M = k * sigma(k)
    groups = defaultdict(list)
    
    for k in range(1, N + 1):
        if time.time() > deadline - 30.0:
            break
        M = k * int(sigma_arr[k])
        if M <= 10**18:
            groups[M].append(k)
            if len(groups[M]) > best_count:
                best_count = len(groups[M])
                best_M = M
                best_k = groups[M][:]
    
    # Strategy 2: Search for highly composite M by constructing from small primes
    # and finding k values that satisfy k*sigma(k) = M
    
    # Small primes to consider
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    
    # Generate candidate M values as products of small primes with various exponents
    # Focus on M that are "highly composite" - many divisors
    candidate_Ms = set()
    
    # Add some known highly composite numbers
    highly_composite = [
        2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840, 1260, 
        1680, 2520, 5040, 7560, 10080, 15120, 20160, 25200, 27720, 45360,
        50400, 55440, 83160, 110880, 166320, 221760, 277200, 332640, 498960,
        554400, 665280, 720720, 1081080, 1441440, 2162160, 2882880, 3603600,
        4324320, 6486480, 7207200, 8648640, 10810800, 14414400, 17297280,
        21621600, 32432400, 36756720, 43243200, 61261200, 73513440, 110270160,
        122522400, 147026880, 183783600, 245053760, 294053760, 367567200,
        551350800, 698377680, 735134400, 1102701600, 1396755360, 1470268800,
        2205403200, 2795510720, 3675672000, 4410806400, 6983776800, 13967553600,
        20951330400, 27935107200, 41902660800, 55870214400, 69837768000,
        110270160000, 139675536000, 220540320000, 329310528000, 441080640000,
        698377680000, 1102701600000, 2205403200000, 4410806400000, 6983776800000
    ]
    for hc in highly_composite:
        if hc <= 10**18:
            candidate_Ms.add(hc)
    
    # Generate more candidates by multiplying small primes
    for p1 in [2, 3, 5, 7, 11, 13]:
        for p2 in [2, 3, 5, 7, 11, 13]:
            for p3 in [2, 3, 5, 7]:
                for e1 in range(1, 8):
                    for e2 in range(1, 6):
                        for e3 in range(1, 4):
                            M = (p1 ** e1) * (p2 ** e2) * (p3 ** e3)
                            if M <= 10**18:
                                candidate_Ms.add(M)
    
    # For each candidate M, find k values such that k*sigma(k) = M
    # This means k divides M and sigma(k) = M/k
    for M in candidate_Ms:
        if time.time() > deadline - 10.0:
            break
        if M > 10**18:
            continue
        
        # Find all divisors of M
        divisors = _divisors_from_factorization(_factorize(M))
        
        ks = []
        for k in divisors:
            if k > math.isqrt(M):
                continue
            if M % k == 0:
                sigma_k = M // k
                if _sigma(k) == sigma_k:
                    ks.append(k)
        
        if len(ks) > best_count:
            best_count = len(ks)
            best_M = M
            best_k = ks
    
    return {'M': int(best_M), 'k': [int(x) for x in best_k]}
