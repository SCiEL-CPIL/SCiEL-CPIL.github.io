"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 40, 'maximum_size': 18}

def _prime_factorization(n, primes):
    """Return dict of prime -> exponent for n."""
    factors = {}
    for p in primes:
        if p * p > n:
            break
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _subset_products_distinct(selected, primes):
    """Check if all 2^k subset products of selected are distinct.
    Uses the fact that subset products are distinct iff the prime-exponent
    vectors are linearly independent over the non-negative integers,
    which for our small numbers we can verify by checking all subset sums.
    """
    k = len(selected)
    if k == 0:
        return True
    # Get prime exponent vectors for each selected number
    vectors = []
    for num in selected:
        fac = _prime_factorization(num, primes)
        vec = {}
        for i, p in enumerate(primes):
            if p in fac:
                vec[i] = fac[p]
        vectors.append(vec)
    
    # Enumerate all subset sums and check for duplicates
    # For k up to ~18, 2^18 = 262144 which is manageable
    seen = set()
    seen.add((0,))  # empty product = 1
    
    # Use a more efficient approach: represent each subset product by its
    # prime exponent tuple
    n_primes = len(primes)
    
    def get_product_exponents(mask):
        """Get the prime exponent vector for the product of elements whose bits are set in mask."""
        exps = [0] * n_primes
        for i in range(k):
            if mask & (1 << i):
                vec = vectors[i]
                for idx, val in vec.items():
                    exps[idx] += val
        return tuple(exps)
    
    # Check all 2^k subsets
    for mask in range(1, 1 << k):
        prod_exps = get_product_exponents(mask)
        if prod_exps in seen:
            return False
        seen.add(prod_exps)
    return True

def construct():
    import time
    deadline = time.time() + 55  # 55s safety margin within 60s limit
    
    n = P['n']
    max_size = P['maximum_size']
    
    # Generate primes up to n
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i in range(2, limit + 1) if sieve[i]]
    
    primes = get_primes(n)
    
    # Candidate numbers: 2..n
    candidates = list(range(2, n + 1))
    
    # Greedy: try to add numbers in an order that maximizes count.
    # Strategy: sort by "difficulty" - prefer numbers that are less likely
    # to cause collisions. Primes are safest, then prime powers, then composites.
    # Actually, let's try a different approach: use a heuristic ordering.
    
    # Key insight: two numbers a, b cause a collision if a*b = c*d for some
    # distinct pairs, or more generally if the multiset of prime exponents
    # allows a non-trivial linear relation with coefficients in {-1, 0, 1}.
    # For subset products to be distinct, we need: for any two distinct subsets S, T,
    # the product over S != product over T. This is equivalent to: the numbers
    # are multiplicatively independent in the sense that no non-trivial subset
    # product equals another.
    
    # A simpler sufficient condition: all numbers are prime -> always works.
    # But we can do better.
    
    # Greedy approach: start with empty set, try adding candidates in a smart order.
    # Order: try numbers that have fewer prime factors first (primes, then semiprimes, etc.)
    
    def num_prime_factors(num, primes_list):
        fac = _prime_factorization(num, primes_list)
        return sum(fac.values())
    
    # Sort candidates by number of prime factors (ascending), then by value
    candidates_sorted = sorted(candidates, key=lambda x: (num_prime_factors(x, primes), x))
    
    selected = []
    
    for cand in candidates_sorted:
        if time.time() > deadline:
            break
        if len(selected) >= max_size:
            break
        # Try adding cand to selected
        trial = selected + [cand]
        if _subset_products_distinct(trial, primes):
            selected.append(cand)
    
    # If we didn't reach max_size, try a different strategy:
    # Remove some composites and try to add more primes or other numbers
    if len(selected) < max_size and time.time() < deadline - 5:
        # Try swapping: for each composite in selected, try removing it and adding others
        for i in range(len(selected) - 1, -1, -1):
            if time.time() > deadline:
                break
            if len(selected) >= max_size:
                break
            # Try removing selected[i] and see if we can add more
            trial_base = selected[:i] + selected[i+1:]
            for cand in candidates_sorted:
                if time.time() > deadline:
                    break
                if len(trial_base) >= max_size:
                    break
                if cand in trial_base:
                    continue
                trial = trial_base + [cand]
                if _subset_products_distinct(trial, primes):
                    trial_base.append(cand)
            # If trial_base is longer, use it
            if len(trial_base) > len(selected):
                selected = trial_base
    
    return list(selected)
# EVOLVE-BLOCK-END
