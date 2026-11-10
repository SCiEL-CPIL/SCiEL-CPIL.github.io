"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial, gcd
from itertools import combinations
from functools import lru_cache

def _factorize(n):
    """Return prime factorization as dict {prime: exponent}."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s limit
    
    n = 12
    target = factorial(n)
    target_factors = _factorize(target)
    
    # Generate all divisors of 12! that are > 12
    primes = sorted(target_factors.keys())
    max_exps = [target_factors[p] for p in primes]
    
    # Generate all divisors > 12
    def gen_divisors(idx, current):
        if idx == len(primes):
            if current > n:
                yield current
            return
        p = primes[idx]
        e = 0
        while e <= max_exps[idx]:
            yield from gen_divisors(idx + 1, current)
            current *= p
            e += 1
            if current > target:
                break
    
    all_divisors = sorted(set(gen_divisors(0, 1)))
    
    # Pre-compute exponent vectors for all divisors
    div_vecs = []
    for d in all_divisors:
        fac = _factorize(d)
        vec = tuple(fac.get(p, 0) for p in primes)
        div_vecs.append((d, vec))
    
    target_vec = tuple(target_factors.get(p, 0) for p in primes)
    num_primes = len(primes)
    
    # Use dynamic programming: dp[remaining_vec] = (max_factor, factors_tuple)
    # But we need to ensure distinctness, so we process divisors in order
    
    # Better approach: BFS/DFS with memoization on (divisor_index, remaining_vec)
    # State: (idx, remaining_vec) -> (min_max_factor, factors_used)
    
    from collections import defaultdict
    
    # dp[idx][remaining_vec] = (best_max, factors)
    # We'll use a dictionary for sparse storage
    
    dp = {}
    
    def solve(idx, remaining_vec):
        """Returns (min_max_factor, factors_tuple) or None."""
        if all(x == 0 for x in remaining_vec):
            return (0, ())
        
        if idx >= len(div_vecs):
            return None
        
        key = (idx, remaining_vec)
        if key in dp:
            return dp[key]
        
        best = None
        
        # Option 1: Skip this divisor
        res_skip = solve(idx + 1, remaining_vec)
        if res_skip is not None:
            best = res_skip
        
        # Option 2: Use this divisor
        d, dvec = div_vecs[idx]
        if all(dvec[j] <= remaining_vec[j] for j in range(num_primes)):
            new_remaining = tuple(remaining_vec[j] - dvec[j] for j in range(num_primes))
            res_use = solve(idx + 1, new_remaining)
            if res_use is not None:
                max_f, factors = res_use
                candidate_max = max(d, max_f)
                candidate_factors = (d,) + factors
                if best is None or candidate_max < best[0]:
                    best = (candidate_max, candidate_factors)
        
        dp[key] = best
        return best
    
    try:
        result = solve(0, target_vec)
        if result is not None:
            max_f, factors = result
            return list(factors)
    except Exception:
        pass
    
    # Fallback: return the single factor 12!
    return [factorial(12)]

# EVOLVE-BLOCK-END
