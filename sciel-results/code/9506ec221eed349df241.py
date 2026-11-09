import time
from itertools import combinations

def _prime_factors(n):
    """Return dict of prime -> exponent for n."""
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

def _subset_products_distinct(nums):
    """Check if all 2^k subset products are distinct."""
    k = len(nums)
    if k > 20:
        return False  # too many to enumerate efficiently
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

def _valid_set_size(nums):
    """Check validity using exponent-vector approach for efficiency."""
    k = len(nums)
    if k > 20:
        return False
    # Enumerate all subset products
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

def construct():
    start = time.time()
    deadline = start + 50.0  # safety margin before 90s limit

    # Precompute prime factorizations for all numbers 2..40
    all_nums = list(range(2, 41))
    factorizations = {n: _prime_factors(n) for n in all_nums}
    
    # All primes in 2..40
    primes = [p for p in all_nums if len(factorizations[p]) == 1 and list(factorizations[p].values())[0] == 1]
    
    # Start with the known 16-element solution
    best_set = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    best_size = len(best_set)
    
    # Verify the 16-set is valid
    if not _valid_set_size(best_set):
        # Should not happen, but fallback
        best_set = primes
        best_size = len(best_set)
    
    # Try to find larger sets using branch-and-bound
    # Strategy: try adding one more element to the 16-set
    # Numbers not in best_set that could potentially be added
    in_best = set(best_set)
    remaining = [n for n in all_nums if n not in in_best]
    
    # For each remaining number, check if adding it to best_set keeps validity
    for n in remaining:
        candidate = best_set + [n]
        if len(candidate) <= 20:  # keep enumeration feasible
            if _valid_set_size(candidate):
                best_set = candidate
                best_size = len(candidate)
                break  # found one, restart search with larger base
    
    # Now try to find even larger sets by searching more broadly
    # Use a greedy + local improvement approach
    # Try different orderings and additions
    
    # Attempt: start from primes and greedily add prime powers
    # Then try to add composites
    
    # More systematic: try all pairs of additions to the 16-set
    for n1 in remaining:
        if time.time() > deadline - 10:
            break
        for n2 in remaining:
            if n1 >= n2:
                continue
            candidate = best_set + [n1, n2]
            if len(candidate) <= 20:
                if _valid_set_size(candidate):
                    best_set = candidate
                    best_size = len(candidate)
                    break
        if time.time() > deadline - 10:
            break
        if best_size > 16:
            break
    
    # Try triple additions if we have time
    if time.time() < deadline - 20:
        for n1 in remaining:
            if time.time() > deadline - 15:
                break
            for n2 in remaining:
                if n1 >= n2:
                    continue
                for n3 in remaining:
                    if n2 >= n3:
                        continue
                    candidate = best_set + [n1, n2, n3]
                    if len(candidate) <= 20:
                        if _valid_set_size(candidate):
                            best_set = candidate
                            best_size = len(candidate)
                            break
                if time.time() > deadline - 15:
                    break
            if best_size > 17:
                break
    
    # Final verification
    if _valid_set_size(best_set):
        return list(best_set)
    
    # Fallback to known valid 16-set
    return [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
