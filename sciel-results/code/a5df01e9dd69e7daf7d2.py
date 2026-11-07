"""Rewritten program: maximize number of distinct integers in 2..40 with distinct subset products."""
# EVOLVE-BLOCK-START
import time

P = {'n': 40, 'maximum_size': 18}

# Precompute primes in 2..40
def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = 5
    while d * d <= n:
        if n % d == 0 or n % (d + 2) == 0:
            return False
        d += 6
    return True

PRIMES = [a for a in range(2, 41) if _is_prime(a)]
# Primes: 2,3,5,7,11,13,17,19,23,29,31,37 -> 12 primes

# Precompute exponent vectors for all numbers 2..40
# Prime index mapping
PRIME_INDEX = {p: i for i, p in enumerate(PRIMES)}
NUM_PRIMES = len(PRIMES)

def _exponent_vector(n):
    """Return the exponent vector of n as a tuple of length NUM_PRIMES."""
    vec = [0] * NUM_PRIMES
    for i, p in enumerate(PRIMES):
        while n % p == 0:
            vec[i] += 1
            n //= p
    return tuple(vec)

EXP_VECTORS = {n: _exponent_vector(n) for n in range(2, 41)}

def _check_distinct_subset_products(S):
    """
    Check if all 2^k subset products of S are distinct.
    S is a list of integers.
    """
    if not S:
        return True
    products = {1}
    for x in S:
        new_products = set()
        for p in products:
            np = p * x
            if np in products or np in new_products:
                return False
            new_products.add(np)
        products.update(new_products)
    return True

def _try_add(products, x):
    """
    Try to add x to the set of products.
    Returns None if collision, else list of new products.
    """
    new_products = []
    for p in products:
        np = p * x
        if np in products or np in new_products:
            return None
        new_products.append(np)
    return new_products

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin before 60s limit
    
    # Strategy: use exponent vectors to check multiplicative independence.
    # Two subsets have the same product iff their exponent vectors sum to the same value.
    # So we need all 2^k subset sums of exponent vectors to be distinct.
    
    # Precompute all numbers 2..40 with their exponent vectors
    all_numbers = list(range(2, 41))
    
    # Group numbers by their "prime support" - numbers that are powers of the same prime
    # can be analyzed together
    # For each prime, collect all its powers in 2..40
    prime_powers = {}
    for n in all_numbers:
        vec = EXP_VECTORS[n]
        # Find which primes have non-zero exponents
        supports = [i for i, e in enumerate(vec) if e > 0]
        if len(supports) == 1:
            p_idx = supports[0]
            p = PRIMES[p_idx]
            if p not in prime_powers:
                prime_powers[p] = []
            prime_powers[p].append(n)
        # else: composite with multiple prime factors
    
    # For each prime, we can select at most 2 powers (since 3 powers a^i, a^j, a^k
    # with i<j<k will have a^i * a^k = a^(i+k) and if i+k=j, collision; but more generally
    # we need to check subset sums)
    
    # Let's do a greedy search using the product set approach but with smart ordering.
    # Start with the best known construction: 16 elements.
    
    # Known good set: pick 2 powers from primes 2,3,5 and the prime itself from others
    # For prime 2: pick 2 and 16 (vectors (1,0,...) and (4,0,...) -> subset sums: 0,1,4,5 distinct)
    # For prime 3: pick 3 and 9 (vectors (0,1,0,...) and (0,2,0,...) -> subset sums: 0,1,2,3 distinct)
    # For prime 5: pick 5 and 25 (vectors (0,0,1,0,...) and (0,0,2,0,...) -> subset sums: 0,1,2,3 distinct)
    # For primes 7,11,13,17,19,23,29,31,37: pick the prime itself
    
    # Let's verify this gives 2+2+2+9 = 15 elements
    # Can we get 16? Maybe by swapping one prime for a composite?
    
    # Let's do a proper search: greedy with backtracking on the prime power groups.
    
    best_result = []
    best_size = 0
    
    # Candidate set: all numbers 2..40
    # We'll use a greedy approach: try to build the set by adding elements one at a time,
    # prioritizing elements that are "independent" (don't create collisions).
    
    # Use the product set method for correctness
    def greedy_build(candidates, existing_set, existing_products):
        """Greedily add elements from candidates to existing_set."""
        current_set = list(existing_set)
        current_products = set(existing_products)
        
        for x in candidates:
            if x in current_set:
                continue
            new_prods = _try_add(current_products, x)
            if new_prods is not None:
                current_set.append(x)
                current_products.update(new_prods)
        
        return current_set, current_products
    
    # Strategy 1: Start with all primes, then try to add prime powers
    base = list(PRIMES)
    base_products = {1}
    for x in base:
        new_prods = _try_add(base_products, x)
        if new_prods is not None:
            base_products.update(new_prods)
    
    # Now try to add prime powers in order of "value" (fewer prime factors first)
    # Prime powers: 4,8,9,16,25,27,32
    prime_power_list = [4, 8, 9, 16, 25, 27, 32]
    
    # For each prime power, check if it can be added
    # But we need to be careful: adding 4 means we should consider removing 2 or keeping both
    # Since 2 and 4 can coexist (2*4=8, and 8 is not in the set unless we add it), let's try
    
    # Actually, let's try multiple strategies and keep the best
    
    # Strategy A: Greedy add prime powers to prime base
    set_a, prods_a = greedy_build(prime_power_list, base, base_products)
    
    # Strategy B: Try different combinations of prime powers
    # For prime 2: choose from {2,4,8,16,32} - pick 2
    # For prime 3: choose from {3,9,27} - pick 2
    # For prime 5: choose from {5,25} - pick 2
    # For other primes: pick the prime
    
    # Let's enumerate: for each prime with multiple powers, try all pairs
    from itertools import product as iproduct
    
    # Powers per prime
    p2_powers = [2, 4, 8, 16, 32]
    p3_powers = [3, 9, 27]
    p5_powers = [5, 25]
    other_primes = [7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    best_set = []
    best_prods = None
    
    for p2_a, p2_b in [(a, b) for i, a in enumerate(p2_powers) for b in p2_powers[i+1:]]:
        for p3_a, p3_b in [(a, b) for i, a in enumerate(p3_powers) for b in p3_powers[i+1:]]:
            for p5_a, p5_b in [(a, b) for i, a in enumerate(p5_powers) for b in p5_powers[i+1:]]:
                candidate = [p2_a, p2_b, p3_a, p3_b, p5_a, p5_b] + other_primes
                # Check if this set has distinct subset products
                if _check_distinct_subset_products(candidate):
                    if len(candidate) > len(best_set):
                        best_set = candidate
                        # Compute products for further extension
                        best_prods = {1}
                        for x in best_set:
                            new_prods = _try_add(best_prods, x)
                            if new_prods is not None:
                                best_prods.update(new_prods)
    
    # Now try to extend best_set with remaining numbers
    if best_prods is not None:
        remaining = [x for x in range(2, 41) if x not in best_set]
        extended_set, extended_prods = greedy_build(remaining, best_set, best_prods)
        if len(extended_set) > len(best_set):
            best_set = extended_set
    
    # Also try strategy A
    if len(set_a) > len(best_set):
        best_set = set_a
    
    # Try to extend further with any remaining numbers
    if best_set:
        best_prods_final = {1}
        for x in best_set:
            new_prods = _try_add(best_prods_final, x)
            if new_prods is not None:
                best_prods_final.update(new_prods)
        
        remaining = [x for x in range(2, 41) if x not in best_set]
        final_set, _ = greedy_build(remaining, best_set, best_prods_final)
        if len(final_set) > len(best_set):
            best_set = final_set
    
    # Final verification
    if not _check_distinct_subset_products(best_set):
        # Fallback to primes only
        best_set = list(PRIMES)
    
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
