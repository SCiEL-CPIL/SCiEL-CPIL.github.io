"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import Counter

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using recursive prime-based generation."""
    from math import isqrt
    max_p = isqrt(limit)
    # Sieve primes up to max_p
    if max_p < 2:
        return [1]
    is_prime = [True] * (max_p + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(max_p) + 1):
        if is_prime[i]:
            for j in range(i * i, max_p + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, max_p + 1) if is_prime[i]]
    
    result = set([1])
    
    def gen(idx, current):
        if current > limit:
            return
        if idx >= len(primes):
            if current > 1:
                result.add(current)
            return
        p = primes[idx]
        # Option 1: don't use prime p
        gen(idx + 1, current)
        # Option 2: use p^2, p^3, p^4, ...
        val = current * p * p
        while val <= limit:
            result.add(val)
            gen(idx + 1, val)
            val *= p
    
    gen(0, 1)
    return sorted(result)

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 135s deadline, 15s safety margin
    
    # Generate powerful numbers up to a larger limit for denser coverage
    L = 10_000_000
    powerful = _generate_powerful(L)
    n_pow = len(powerful)
    pow_set = set(powerful)
    
    # O(n^2) pair enumeration with early termination
    sum_counts = Counter()
    
    for i in range(n_pow):
        a = powerful[i]
        for j in range(i, n_pow):
            b = powerful[j]
            s = a + b
            if s > 10**18:
                break
            sum_counts[s] += 1
        
        # Check time every 100 outer iterations
        if i % 100 == 99 and time.time() > deadline - 30:
            break
    
    # Find the N with the most pairs
    if not sum_counts:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    best_N = max(sum_counts, key=lambda x: sum_counts[x])
    
    # Collect all pairs for best_N
    pairs = []
    for a in powerful:
        b = best_N - a
        if b >= a and b in pow_set:
            pairs.append([a, b])
    
    # Remove duplicates and sort
    pairs = sorted(set(tuple(p) for p in pairs))
    pairs = [[p[0], p[1]] for p in pairs]
    
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
