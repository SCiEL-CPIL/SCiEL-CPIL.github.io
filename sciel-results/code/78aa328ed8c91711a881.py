"""Factorial decompositions into distinct factors above the index.

Return 1..12 DISTINCT integers all greater than 12, with product exactly 12!.
Minimize the LARGEST factor; the number of factors is variable.

Strategy:
1. Start from a known valid explicit factorization.
2. Use a pruned DFS that iterates divisors of the remaining product in increasing
   order, aggressively cutting branches where the next factor would exceed the
   current best maximum or where the remaining product cannot be formed by
   distinct larger integers.
3. Local improvement: try splitting the current max factor into two smaller
   distinct factors to reduce the maximum.
"""
import math
import time

T12 = math.factorial(12)  # 479001600


def _generate_divisors(n):
    """Generate all divisors of n that are > 12 and < n, sorted."""
    # Factorize n: 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    exps = [10, 5, 2, 1, 1]
    
    divisors = set()
    
    def rec(i, current):
        if i == len(primes):
            if 12 < current < n:
                divisors.add(current)
            return
        p = primes[i]
        e = exps[i]
        val = 1
        for _ in range(e + 1):
            rec(i + 1, current * val)
            val *= p
    
    rec(0, 1)
    divisors.add(n)  # include n itself
    return sorted(divisors)


# Precompute all divisors of 12!
_DIVISORS = _generate_divisors(T12)
_DIVISOR_SET = set(_DIVISORS)


def valid_solution(factors):
    """Check if factors is a valid solution: 1-12 distinct integers > 12, product = T12."""
    if not (1 <= len(factors) <= 12):
        return False
    if len(set(factors)) != len(factors):
        return False
    if any(f < 13 for f in factors):
        return False
    p = 1
    for f in factors:
        p *= f
    return p == T12


def _dfs(remaining, min_factor, count, factors, best_max, deadline):
    """
    DFS: build factors in increasing order.
    remaining: product still to be factored
    min_factor: next factor must be > this
    count: number of factors chosen so far
    factors: list of factors chosen so far
    best_max: current best maximum (mutable via return)
    deadline: time deadline
    
    Returns (best_max, best_factors) or None if no improvement.
    """
    best_factors = None
    best_m = best_max
    
    if time.time() > deadline:
        return None
    
    if remaining == 1:
        if count >= 1:
            cur_max = max(factors)
            if cur_max < best_m:
                best_m = cur_max
                best_factors = factors[:]
        return (best_m, best_factors)
    
    if count >= 12:
        return None
    
    # Try taking 'remaining' as the last factor if valid
    if remaining > min_factor and remaining > 12 and remaining in _DIVISOR_SET:
        cur_max = max(max(factors) if factors else 0, remaining)
        if cur_max < best_m:
            best_m = cur_max
            best_factors = factors + [remaining]
    
    # Try each divisor of remaining that is > min_factor and < remaining
    for d in _DIVISORS:
        if d <= min_factor:
            continue
        if d >= remaining:
            break
        if remaining % d != 0:
            continue
        # Pruning: if d >= best_m, no point (all subsequent are >= d)
        if d >= best_m:
            break
        
        new_rem = remaining // d
        
        # Quick prune: if new_rem <= 1, skip
        if new_rem <= 1:
            continue
        
        # The next factor must be > d, so the minimum product of remaining factors
        # is at least (d+1). If new_rem <= d, no divisor > d can divide it.
        if new_rem <= d:
            continue
        
        # Recurse
        result = _dfs(new_rem, d, count + 1, factors + [d], best_m, deadline)
        if result is not None:
            best_m, bf = result
            if bf is not None:
                best_factors = bf
            # If d >= best_m, no further divisors will help
            if d >= best_m:
                break
    
    return (best_m, best_factors)


def _local_improve(factors, deadline):
    """Try to reduce the max factor by splitting it into two smaller distinct factors."""
    if not factors:
        return factors
    
    best = list(factors)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        cur_max = max(best)
        others = [f for f in best if f != cur_max]
        
        # Try splitting cur_max into a*b where a > 12, b > 12, a != b,
        # and neither a nor b is in others
        for a in range(13, math.isqrt(cur_max) + 1):
            if time.time() > deadline:
                break
            if cur_max % a == 0:
                b = cur_max // a
                if b > a and a >= 13 and b >= 13:
                    if a not in others and b not in others:
                        new_set = others + [a, b]
                        if valid_solution(new_set):
                            if max(new_set) < cur_max:
                                best = new_set
                                improved = True
                                break
    
    return best


def construct():
    """Construct a valid factorization of 12! into distinct integers > 12,
    minimizing the largest factor."""
    deadline = time.time() + 80  # safety margin under 90s
    
    # Start with a known valid explicit set
    # 144*64*77*25*27 = 479001600 = 12!
    candidate = [144, 64, 77, 25, 27]
    if valid_solution(candidate):
        best = candidate
        best_max = max(candidate)
    else:
        best = [T12]
        best_max = T12
    
    # Local improvement: try to reduce max by splitting
    if time.time() < deadline:
        best = _local_improve(best, deadline)
        best_max = max(best) if best else T12
    
    # Full DFS search to find better solutions
    if time.time() < deadline - 2:
        result = _dfs(T12, 12, 0, [], best_max, deadline)
        if result is not None:
            new_max, new_factors = result
            if new_factors is not None and new_max < best_max:
                best = new_factors
                best_max = new_max
    
    # Final verification
    if best and valid_solution(best):
        return best
    return [T12]
