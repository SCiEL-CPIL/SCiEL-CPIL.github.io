<<<<<<< SEARCH
import random
import time

def _prime_factors(n):
    """Return tuple of prime exponents for n."""
    factors = []
    d = 2
    while d * d <= n:
        count = 0
        while n % d == 0:
            count += 1
            n //= d
        if count:
            factors.append((d, count))
        d += 1
    if n > 1:
        factors.append((n, 1))
    return tuple(factors)

def _subset_products_set(nums):
    """Return the set of all 2^k subset products."""
    products = {1}
    for num in nums:
        new_products = set()
        for p in products:
            new_products.add(p * num)
        products.update(new_products)
    return products

def _check_collision_incremental(current_products, new_num):
    """Check if adding new_num creates a collision. Returns (valid, new_products_set)."""
    # For each existing product p, p * new_num must not already be in current_products
    # and must not collide with any other p' * new_num
    new_products = set()
    for p in current_products:
        np = p * new_num
        if np in current_products:
            return False, None
        if np in new_products:
            return False, None
        new_products.add(np)
    return True, current_products | new_products

def _verify_subset_products(nums):
    """Verify all 2^k subset products are distinct."""
    k = len(nums)
    if k > 20:
        return False
    products = _subset_products_set(nums)
    return len(products) == (1 << k)
=======
import random
import time

# Primes up to 40
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_NPRIMES = len(_PRIMES)

def _prime_factor_vector(n):
    """Return tuple of exponents for each prime in _PRIMES."""
    vec = [0] * _NPRIMES
    for i, p in enumerate(_PRIMES):
        while n % p == 0:
            vec[i] += 1
            n //= p
    return tuple(vec)

# Precompute factor vectors for all numbers 2..40
_FACTOR_VECS = {}
for _n in range(2, 41):
    _FACTOR_VECS[_n] = _prime_factor_vector(_n)

def _subset_exponent_sets(nums):
    """Return set of all 2^k subset exponent-sum vectors."""
    vectors = {(0,) * _NPRIMES}
    for num in nums:
        v = _FACTOR_VECS[num]
        new_vectors = set()
        for sv in vectors:
            nv = tuple(sv[i] + v[i] for i in range(_NPRIMES))
            new_vectors.add(nv)
        vectors.update(new_vectors)
    return vectors

def _check_collision_exponent(current_vectors, new_num):
    """Check if adding new_num creates a collision using exponent vectors.
    Returns (valid, new_vectors_set)."""
    v = _FACTOR_VECS[new_num]
    new_vectors = set()
    for sv in current_vectors:
        nv = tuple(sv[i] + v[i] for i in range(_NPRIMES))
        if nv in current_vectors:
            return False, None
        if nv in new_vectors:
            return False, None
        new_vectors.add(nv)
    return True, current_vectors | new_vectors

def _verify_subset_products(nums):
    """Verify all 2^k subset products are distinct."""
    k = len(nums)
    if k > 20:
        return False
    vectors = _subset_exponent_sets(nums)
    return len(vectors) == (1 << k)
>>>>>>> REPLACE

