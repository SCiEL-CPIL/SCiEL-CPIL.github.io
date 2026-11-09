"""Approximate a ratio inside a divisor-sum fiber."""
import time
import math
from fractions import Fraction
from collections import defaultdict
import bisect

# EVOLVE-BLOCK-START
MAX_N = 10_000_000
TARGET_NUM, TARGET_DEN = 3, 2
DEADLINE_FRACTION = 0.88  # safety margin

def _build_spf(n):
    """Build smallest prime factor array up to n using sieve."""
    spf = list(range(n + 1))
    spf[0] = 0
    if n >= 1:
        spf[1] = 1
    for i in range(2, int(n**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, n + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _sigma_from_spf(spf, n):
    """Compute sigma(n) using smallest prime factorization."""
    if n == 1:
        return 1
    result = 1
    while n > 1:
        p = spf[n]
        power_sum = 1
        power = 1
        while n % p == 0:
            n //= p
            power *= p
            power_sum += power
        result *= power_sum
    return result

def construct():
    t_start = time.time()
    deadline = t_start + 90.0 * DEADLINE_FRACTION
    
    # Build SPF table
    spf = _build_spf(MAX_N)
    
    # Compute all sigmas
    sigmas = [0] * (MAX_N + 1)
    sigmas[1] = 1
    for n in range(2, MAX_N + 1):
        sigmas[n] = _sigma_from_spf(spf, n)
    
    # Group indices by sigma value
    sigma_groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        sigma_groups[sigmas[n]].append(n)
    
    # For each sigma group, find the best pair
    best_score_num = 1  # numerator of |a/b - 3/2|
    best_score_den = 2  # denominator
    best_pair = (1, 1)
    
    # Sort groups by size descending to process larger groups first
    # Larger groups have more candidates and are more likely to contain close pairs
    groups_sorted = sorted(sigma_groups.items(), key=lambda x: -len(x[1]))
    
    for s, group in groups_sorted:
        if len(group) < 2:
            continue
        
        group.sort()
        
        # For each b in group, find a in group closest to 3b/2
        for b in group:
            target = (3 * b) / 2
            idx = bisect.bisect_left(group, target)
            
            # Check idx and idx-1
            candidates = []
            if idx < len(group):
                candidates.append(group[idx])
            if idx > 0:
                candidates.append(group[idx - 1])
            
            for a in candidates:
                # Compute |a/b - 3/2| = |2a - 3b| / (2b)
                diff = abs(2 * a - 3 * b)
                if diff == 0:
                    # Exact match is invalid per contract
                    continue
                den = 2 * b
                # Compare diff/den with best_score_num/best_score_den
                # diff * best_score_den < best_score_num * den
                lhs = diff * best_score_den
                rhs = best_score_num * den
                if lhs < rhs:
                    best_score_num = diff
                    best_score_den = den
                    best_pair = (a, b)
        
        if time.time() > deadline - 5.0:
            break
    
    a, b = best_pair
    return [int(a), int(b)]
# EVOLVE-BLOCK-END
