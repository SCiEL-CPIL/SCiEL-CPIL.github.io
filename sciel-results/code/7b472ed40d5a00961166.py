"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial

_START_TIME = time.time()
_TIMEOUT = 85  # safety margin before 90s kill

def _time_left():
    return _TIMEOUT - (time.time() - _START_TIME)

# Prime factorization of 12! = 479001600
# 12! = 2^10 * 3^5 * 5^2 * 7 * 11
_PRIMES = [2, 3, 5, 7, 11]
_EXPONENTS = [10, 5, 2, 1, 1]

def _divisors_from_exp(exp):
    """Generate all divisors from a prime-exponent vector."""
    divs = [1]
    for p, e in zip(_PRIMES, exp):
        new_divs = []
        pw = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * pw)
            pw *= p
        divs = new_divs
    return sorted(divs)

def _divisors_gt(remaining, min_val):
    """Generate divisors of remaining that are >= min_val, using prime factorization."""
    # Factor remaining into primes
    exp = [0] * len(_PRIMES)
    rem = remaining
    for i, p in enumerate(_PRIMES):
        while rem % p == 0:
            exp[i] += 1
            rem //= p
    if rem != 1:
        # Should not happen since remaining always divides 12!
        return [remaining] if remaining >= min_val else []
    divs = _divisors_from_exp(exp)
    return [d for d in divs if d >= min_val]

def _min_possible_max(remaining, min_val):
    """
    Lower bound on the max factor needed to factor 'remaining' into
    distinct factors each >= min_val.
    """
    if remaining == 1:
        return 0
    if remaining >= min_val:
        # Could be a single factor
        return remaining
    # Need at least 2 factors: min_val and remaining/min_val
    # The max would be at least remaining/min_val
    # But we need distinct factors, so this is a rough lower bound
    return remaining // min_val + 1

def _dfs(remaining, min_val, max_bound, chosen, path, best):
    """
    DFS to find distinct factors in [min_val, max_bound] with product = remaining.
    Returns True if found, updates best.
    """
    if _time_left() < 1:
        return False
    
    if remaining == 1:
        # Found a complete decomposition
        max_factor = max(path) if path else 0
        if max_factor < best['max']:
            best['max'] = max_factor
            best['factors'] = list(path)
        return True
    
    if len(path) >= 12:
        return False
    
    # Get divisors of remaining that are >= min_val and <= max_bound
    divs = _divisors_gt(remaining, min_val)
    divs = [d for d in divs if d <= max_bound]
    
    if not divs:
        return False
    
    for d in divs:
        if d in chosen:
            continue
        
        # Branch and bound: if d alone exceeds current best max, skip
        if d >= best['max']:
            continue
        
        new_max = max(d, max(path) if path else 0)
        if new_max >= best['max']:
            continue
        
        rem2 = remaining // d
        
        if rem2 == 1:
            # This is the last factor
            if new_max < best['max']:
                best['max'] = new_max
                best['factors'] = list(path) + [d]
                return True
        else:
            # Need more factors. The next factor must be >= d+1 (for distinctness/ordering)
            # and the remaining rem2 must be factorable
            # Pruning: rem2 must be >= (d+1) if we need at least one more factor
            if rem2 < d + 1:
                continue
            
            chosen.add(d)
            path.append(d)
            
            if _dfs(rem2, d + 1, max_bound, chosen, path, best):
                path.pop()
                chosen.remove(d)
                return True
            
            path.pop()
            chosen.remove(d)
    
    return False

def _search_for_max(M):
    """
    Search for a decomposition of 12! into distinct factors in [13, M].
    Returns the factors if found, None otherwise.
    """
    target = factorial(12)
    best = {'max': M + 1, 'factors': None}
    chosen = set()
    path = []
    
    if _dfs(target, 13, M, chosen, path, best):
        return best['factors']
    return None

def construct():
    target = factorial(12)  # 479001600
    
    # Binary search on M (the maximum factor)
    # Lower bound: 13
    # Upper bound: target (trivial solution)
    # But we can narrow it: with k factors all >= 13, product >= 13^k
    # 13^1 = 13, 13^2 = 169, ..., 13^12 is huge
    # The minimal possible M is at least 13
    
    # Start with a reasonable upper bound
    # If we use 2 factors: M >= sqrt(12!) ≈ 21886
    # If we use 3 factors: M >= (12!)^(1/3) ≈ 782
    # If we use 4 factors: M >= (12!)^(1/4) ≈ 263
    # If we use 5 factors: M >= (12!)^(1/5) ≈ 136
    # If we use 6 factors: M >= (12!)^(1/6) ≈ 96
    # If we use 7 factors: M >= (12!)^(1/7) ≈ 73
    # If we use 8 factors: M >= (12!)^(1/8) ≈ 58
    # If we use 9 factors: M >= (12!)^(1/9) ≈ 48
    # If we use 10 factors: M >= (12!)^(1/10) ≈ 41
    # If we use 11 factors: M >= (12!)^(1/11) ≈ 35
    # If we use 12 factors: M >= (12!)^(1/12) ≈ 31
    
    # Let's try to find the minimal M by searching from low to high
    # But use binary search for efficiency
    
    lo = 13
    hi = target
    best_result = [target]  # Trivial solution
    
    # First, try to find a good solution with a moderate M
    # Try M values that are likely to work
    # Based on the analysis, M around 100-300 should work with 4-6 factors
    
    # Do a linear search from a reasonable lower bound
    # This is more effective than binary search because we want the MINIMAL M
    # and the search space is small enough
    
    # Try M from 13 upward, but skip to likely ranges
    # Actually, let's do a smarter search: try M values that are divisors of 12!
    
    # Get all divisors of 12! that are >= 13
    all_divs = _divisors_gt(target, 13)
    
    # Sort them
    all_divs.sort()
    
    # Binary search on the sorted divisors
    # Find the smallest M in all_divs such that a decomposition exists
    lo_idx = 0
    hi_idx = len(all_divs) - 1
    best_M = target
    best_factors = [target]
    
    while lo_idx <= hi_idx and _time_left() > 5:
        mid_idx = (lo_idx + hi_idx) // 2
        M = all_divs[mid_idx]
        
        result = _search_for_max(M)
        if result is not None:
            best_M = M
            best_factors = result
            hi_idx = mid_idx - 1
        else:
            lo_idx = mid_idx + 1
    
    return best_factors
# EVOLVE-BLOCK-END
