"""Rewritten program: maximize distinct integers in 2..40 with all subset products distinct."""
# EVOLVE-BLOCK-START
import time
import itertools

_TIME_LIMIT = 55.0  # safety margin before 60s kill

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

def _all_subset_products_distinct(selected):
    """Check that all 2^k subset products of 'selected' are distinct."""
    k = len(selected)
    if k == 0:
        return True
    # Generate all subset products using bitmask
    products = set()
    n = 1 << k
    for mask in range(n):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= selected[i]
        if prod in products:
            return False
        products.add(prod)
    return True

def construct():
    start_time = time.time()
    
    # Candidate pool: integers 2..40
    candidates = list(range(2, 41))
    
    # We want to find the maximum subset of candidates such that all subset products are distinct.
    # This is equivalent to: the prime exponent vectors of the selected numbers are linearly independent
    # over the rationals (i.e., no non-trivial integer linear combination of exponent vectors equals zero).
    # Actually, it's stronger: all 2^k subset products must be distinct, which means the vectors are
    # affinely independent in the sense that no two different subsets have the same product.
    # This is equivalent to the vectors being linearly independent over Q (since if a subset product
    # equals another, their difference is a non-trivial integer relation).
    
    # Key insight: two subsets have the same product iff the symmetric difference gives a multiplicative
    # relation. So we need the exponent vectors to be linearly independent over Z (or Q).
    
    # Let's get the prime factorization exponent vectors for each candidate.
    primes = [p for p in range(2, 41) if _is_prime(p)]
    # primes in 2..40: 2,3,5,7,11,13,17,19,23,29,31,37 -> 12 primes
    
    def factorize(n):
        """Return exponent vector as tuple indexed by primes list."""
        vec = []
        for p in primes:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            vec.append(e)
        return tuple(vec)
    
    # Precompute exponent vectors
    exp_vecs = {n: factorize(n) for n in candidates}
    
    # Greedy approach: try to build a large set.
    # Strategy: start with all numbers, then remove ones that create dependencies.
    # Better: use a greedy heuristic that tries to include as many as possible.
    
    # Approach: Try different orderings and use a maximum independent set style search.
    # Since we have at most 39 candidates, and we want up to 18, let's use a smart greedy + local search.
    
    # First, let's try a simple greedy: sort candidates by "independence potential" and add if it doesn't break.
    
    def can_add(current_set, new_elem):
        """Check if adding new_elem to current_set keeps all subset products distinct."""
        test_set = list(current_set) + [new_elem]
        return _all_subset_products_distinct(test_set)
    
    # Greedy with multiple orderings
    best_set = []
    
    # Try several orderings:
    # 1. Primes first (most independent)
    # 2. By number of prime factors (fewer = more independent)
    # 3. Random orderings
    
    orderings = []
    
    # Ordering 1: primes first, then by number of distinct prime factors
    def num_distinct_pf(n):
        return sum(1 for p in primes if n % p == 0)
    
    order1 = sorted(candidates, key=lambda x: (0 if _is_prime(x) else 1, num_distinct_pf(x), x))
    orderings.append(order1)
    
    # Ordering 2: just by number of distinct prime factors
    order2 = sorted(candidates, key=lambda x: (num_distinct_pf(x), x))
    orderings.append(order2)
    
    # Ordering 3: by total prime factors (with multiplicity)
    def total_pf(n):
        count = 0
        for p in primes:
            while n % p == 0:
                n //= p
                count += 1
        return count
    order3 = sorted(candidates, key=lambda x: (total_pf(x), x))
    orderings.append(order3)
    
    # Run greedy on each ordering
    for order in orderings:
        current = []
        for elem in order:
            if time.time() - start_time > _TIME_LIMIT - 5:
                break
            if can_add(current, elem):
                current.append(elem)
        if len(current) > len(best_set):
            best_set = current[:]
    
    # Now try to improve best_set by adding more elements
    # Local search: try adding any candidate not in best_set
    improved = True
    while improved and (time.time() - start_time < _TIME_LIMIT - 10):
        improved = False
        for elem in candidates:
            if elem not in best_set:
                if can_add(best_set, elem):
                    best_set.append(elem)
                    improved = True
                    break
    
    # Try to swap: remove one element, try to add two others
    for i in range(len(best_set)):
        if time.time() - start_time > _TIME_LIMIT - 5:
            break
        removed = best_set[i]
        temp_set = best_set[:i] + best_set[i+1:]
        # Try to add any two elements
        for e1 in candidates:
            if e1 in temp_set or e1 == removed:
                continue
            if can_add(temp_set, e1):
                new_set = temp_set + [e1]
                # Try to add one more
                for e2 in candidates:
                    if e2 in new_set or e2 == removed or e2 == e1:
                        continue
                    if can_add(new_set, e2):
                        if len(new_set) > len(best_set):
                            best_set = new_set
                            improved = True
                            break
                if improved:
                    break
        if improved:
            break
    
    # Final check and return
    if not _all_subset_products_distinct(best_set):
        # Fallback: just return primes
        best_set = [p for p in range(2, 41) if _is_prime(p)]
    
    return list(best_set)
# EVOLVE-BLOCK-END
