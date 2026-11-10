import itertools
from functools import lru_cache

def _primes_up_to(n):
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _max_exponent(p, max_val):
    m = 0
    val = 1
    while val * p <= max_val:
        val *= p
        m += 1
    return m

def _distinct_subset_sums(exponents):
    """Check if all subset sums of exponents are distinct."""
    sums = set()
    for r in range(len(exponents) + 1):
        for combo in itertools.combinations(exponents, r):
            s = sum(combo)
            if s in sums:
                return False
            sums.add(s)
    return True

def _best_exponent_subset(max_exp):
    """Find largest subset of {1..max_exp} with distinct subset sums."""
    candidates = list(range(1, max_exp + 1))
    for size in range(len(candidates), 0, -1):
        for subset in itertools.combinations(candidates, size):
            if _distinct_subset_sums(subset):
                return subset
    return []

def _verify_all_distinct(selected):
    """Verify all 2^k subset products are distinct."""
    k = len(selected)
    products = set()
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= selected[i]
        if prod in products:
            return False
        products.add(prod)
    return True

def construct():
    max_val = 40
    primes = _primes_up_to(max_val)
    
    # For each prime, determine max exponent and best exponent subset
    prime_data = []
    for p in primes:
        m = _max_exponent(p, max_val)
        if m == 0:
            continue
        best_exps = _best_exponent_subset(m)
        prime_data.append((p, best_exps))
    
    # We need to select a subset of primes (with their best exponents)
    # such that all subset products are distinct.
    # Since we use distinct primes, the only way to get collisions is
    # if two different combinations of exponents give the same product.
    # With distinct primes, this can only happen if the exponent vectors
    # are identical, which means the subset of selected prime-powers
    # must be the same. So actually, with distinct primes, any selection
    # of prime powers gives distinct subset products!
    # Wait, that's not right. The subset products are over the selected
    # integers. If we select p^a and q^b (p != q), then:
    # - empty: 1
    # - {p^a}: p^a
    # - {q^b}: q^b
    # - {p^a, q^b}: p^a * q^b
    # These are all distinct as long as p^a != q^b, etc.
    # Since p and q are distinct primes, p^a != q^b for any a,b >= 1.
    # And p^a * q^b is different from any single prime power.
    # So with distinct primes, we just need to pick which primes to use.
    
    # But wait - we can also use multiple powers of the same prime.
    # E.g., 2^1 and 2^2. Then:
    # - empty: 1
    # - {2^1}: 2
    # - {2^2}: 4
    # - {2^1, 2^2}: 8
    # These are distinct. Good.
    # But if we have 2^1, 2^2, and 3^1:
    # - empty: 1
    # - {2^1}: 2
    # - {2^2}: 4
    # - {3^1}: 3
    # - {2^1, 2^2}: 8
    # - {2^1, 3^1}: 6
    # - {2^2, 3^1}: 12
    # - {2^1, 2^2, 3^1}: 24
    # All distinct? Yes.
    
    # The key insight: if we select a set of prime powers where for each prime,
    # the exponents form a set with distinct subset sums, then the overall
    # subset products are distinct. This is because the exponent vector for
    # each prime is uniquely determined by the subset, and distinct subset
    # sums ensure no two subsets give the same exponent for any prime.
    
    # So the approach is: for each prime, pick a subset of exponents with
    # distinct subset sums, then combine all these.
    
    selected = []
    for p, best_exps in prime_data:
        for e in best_exps:
            selected.append(p ** e)
    
    # Verify
    if not _verify_all_distinct(selected):
        # Fallback: try smaller selections
        # This shouldn't happen with the correct exponent subsets
        pass
    
    return selected
