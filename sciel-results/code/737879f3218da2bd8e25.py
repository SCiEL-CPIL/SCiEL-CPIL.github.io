"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from itertools import combinations

P = {'n': 40, 'maximum_size': 18}

def _factorize(n):
    """Return dict of prime->exponent for n."""
    d = {}
    d2 = dict(d)
    f = 2
    while f * f <= n:
        while n % f == 0:
            d[f] = d.get(f, 0) + 1
            n //= f
        f += 1
    if n > 1:
        d[n] = d.get(n, 0) + 1
    return d

def _check_distinct_subsets(nums):
    """Check if all 2^k subset products are distinct."""
    k = len(nums)
    products = set()
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= nums[i]
        if prod in products:
            return False
        products.add(prod)
    return True

def _greedy_search(nums, deadline):
    """Greedy: try to add numbers one by one, starting from a good base."""
    # Sort by "usefulness" - prefer numbers that introduce new prime dimensions
    def score(n):
        f = _factorize(n)
        # Score by number of distinct primes it introduces
        return -len(f)
    
    sorted_nums = sorted(nums, key=score)
    result = []
    for n in sorted_nums:
        if time.time() > deadline:
            break
        candidate = result + [n]
        if len(candidate) <= 20:  # Limit subset check size
            if _check_distinct_subsets(candidate):
                result.append(n)
    return result

def _backtrack_search(nums, current, remaining, best, deadline):
    """Backtracking search for maximum valid subset."""
    if time.time() > deadline:
        return best
    if not remaining:
        if len(current) > len(best):
            best = current[:]
        return best
    
    # Pruning: even if we add all remaining, can't beat best
    if len(current) + len(remaining) <= len(best):
        return best
    
    for i, n in enumerate(remaining):
        if time.time() > deadline:
            break
        candidate = current + [n]
        if len(candidate) <= 20:
            if _check_distinct_subsets(candidate):
                new_remaining = remaining[i+1:]
                best = _backtrack_search(candidate, new_remaining, best, deadline)
        # Also try skipping n
        # (implicitly handled by loop continuing)
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    all_nums = list(range(2, 41))
    
    # Start with known good 16-element solution
    base = [11, 13, 17, 19, 23, 29, 31, 37, 2, 4, 16, 3, 9, 5, 25, 7]
    best = base[:]
    
    # Try to improve by adding one more element
    for n in all_nums:
        if time.time() > deadline:
            break
        if n in best:
            continue
        candidate = best + [n]
        if len(candidate) <= 20:
            if _check_distinct_subsets(candidate):
                best = candidate[:]
                break
    
    # Try different base configurations
    # Strategy: try various combinations of small-prime powers
    small_primes_options = [
        [2, 4, 16],  # 2^1, 2^2, 2^4
        [2, 4, 8],   # 2^1, 2^2, 2^3
        [2, 8, 16],  # 2^1, 2^3, 2^4
        [4, 8, 16],  # 2^2, 2^3, 2^4
        [2, 4, 32],  # 2^1, 2^2, 2^5
    ]
    three_options = [
        [3, 9],
        [3, 27],
        [9, 27],
    ]
    five_options = [
        [5, 25],
        [5],
    ]
    seven_options = [
        [7],
    ]
    
    for sp in small_primes_options:
        for tp in three_options:
            for fp in five_options:
                for gp in seven_options:
                    if time.time() > deadline:
                        break
                    small = sp + tp + fp + gp
                    candidate = [11, 13, 17, 19, 23, 29, 31, 37] + small
                    if len(candidate) > len(best):
                        if _check_distinct_subsets(candidate):
                            best = candidate[:]
    
    # Try to add extra composites to the best found
    for n in all_nums:
        if time.time() > deadline:
            break
        if n in best:
            continue
        candidate = best + [n]
        if len(candidate) <= 20:
            if _check_distinct_subsets(candidate):
                best = candidate[:]
    
    return list(best)
# EVOLVE-BLOCK-END
