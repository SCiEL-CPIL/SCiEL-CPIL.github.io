import time

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _sieve(n):
    if n < 2:
        return []
    is_p = [True] * (n + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_p[i]:
            for j in range(i*i, n+1, i):
                is_p[j] = False
    return [i for i in range(n+1) if is_p[i]]

def _valid_subset(subset):
    """Check if all 2^k subset products of the given distinct ints are distinct."""
    k = len(subset)
    if k > 20:
        return False
    products = set()
    # Use iterative doubling
    prods = [1]
    for x in subset:
        new_prods = [p * x for p in prods]
        prods = prods + new_prods
    if len(prods) != len(set(prods)):
        return False
    return True

def construct():
    """
    Returns up to 18 distinct integers in [2, 40] such that all 2^k subset products
    (including the empty product 1) are distinct. Maximize k.
    
    Strategy:
    1. Use primes >= 7 as 'free' elements (each contributes a unique prime factor).
    2. For primes 2, 3, 5, select powers within [2,40] such that exponent subset sums are distinct.
    3. Optionally include composite numbers that are products of distinct primes (squarefree)
       to add more elements, verifying the full subset-product distinctness.
    
    The current best known is 16. We attempt to reach 17 or 18 by adding squarefree composites.
    """
    deadline = time.time() + 85.0  # 5s safety margin before 90s kill
    
    primes = _sieve(40)
    primes_2_3_5 = [2, 3, 5]
    primes_ge_7 = [p for p in primes if p >= 7]
    
    # Base: powers of 2, 3, 5 with distinct subset-sum exponents
    # p=2: {2, 4, 16} -> exponents {1,2,4} -> subset sums 0..7
    # p=3: {3, 9} -> exponents {1,2} -> subset sums 0..3
    # p=5: {5, 25} -> exponents {1,2} -> subset sums 0..3
    base_powers = [2, 4, 16, 3, 9, 5, 25]
    
    # All primes >= 7 in [2,40]
    free_primes = primes_ge_7  # 7, 11, 13, 17, 19, 23, 29, 31, 37 -> 9 primes
    
    base_set = base_powers + free_primes
    # base_set has 6 + 9 = 15 elements
    # Wait, 2,4,16,3,9,5,25 is 7 elements. 7+9=16. Correct.
    
    # Try to add more elements from [2,40] \ base_set
    # Candidates: composite numbers not in base_set
    all_candidates = list(range(2, 41))
    base_set_frozenset = frozenset(base_set)
    extra_candidates = [x for x in all_candidates if x not in base_set_frozenset]
    
    # Sort candidates: prefer squarefree numbers with large prime factors first?
    # Actually, we want to add as many as possible. Let's try greedy addition.
    
    def _try_add(candidate, current_set):
        """Check if adding candidate to current_set preserves distinct subset products."""
        # Efficient check: the new subset products are old_products U (old_products * candidate)
        # We need all to be distinct.
        # Since current_set is valid, old_products are distinct.
        # We need:
        # 1. candidate not in old_products (i.e., candidate is not a product of a subset)
        # 2. No two old_products p1, p2 exist such that p1 * candidate == p2
        #    i.e., p2 / p1 == candidate
        #    i.e., candidate is not a ratio of two subset products
        
        # Generate all subset products of current_set
        prods = [1]
        for x in current_set:
            prods += [p * x for p in prods]
        
        prod_set = set(prods)
        
        # Check 1: candidate not in prod_set
        if candidate in prod_set:
            return False
        
        # Check 2: for each p in prods, p * candidate should not be in prod_set
        # (unless p * candidate == p, which implies candidate==1, impossible)
        for p in prods:
            if p * candidate in prod_set:
                return False
        
        return True
    
    # Greedy: try to add candidates one by one
    current_set = list(base_set)
    
    # Sort extra candidates: try smaller ones first? Or larger?
    # Smaller numbers are more likely to be products of existing subset elements.
    # Let's just try all in some order.
    # Actually, let's try to find ANY set of size 17 or 18.
    
    # If we have 16 elements, we need to add 1 or 2 more.
    # Try adding each candidate one at a time to see if 17 is possible.
    for cand in extra_candidates:
        if _try_add(cand, current_set):
            current_set.append(cand)
            break
    
    # Now try to add one more if we got to 17
    if len(current_set) == 17:
        for cand in extra_candidates:
            if cand not in current_set:
                if _try_add(cand, current_set):
                    current_set.append(cand)
                    break
    
    # Verify final result
    if not _valid_subset(current_set):
        # Fallback to known valid 16
        return [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Ensure types are Python int
    return [int(x) for x in current_set]

# Quick self-test
if __name__ == "__main__":
    result = construct()
    print(f"Found {len(result)} elements: {result}")
    # Verify
    prods = [1]
    for x in result:
        prods += [p * x for p in prods]
    assert len(prods) == len(set(prods)), "Subset products not distinct!"
    print("Valid!")
