"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt, prod as math_prod
from functools import lru_cache

# Precompute primes and prime factorizations for 2..40
def _factorize(n):
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

# All numbers 2..40 and their prime exponent vectors
_NUMBERS = list(range(2, 41))
_FACTORS = {n: _factorize(n) for n in _NUMBERS}

# Get all primes up to 40
_PRIMES = sorted(set(p for n in _NUMBERS for p in _FACTORS[n]))
_PRIME_TO_IDX = {p: i for i, p in enumerate(_PRIMES)}
_NUM_PRIMES = len(_PRIMES)

# Convert to vectors for faster comparison
def _to_vec(factors):
    v = [0] * _NUM_PRIMES
    for p, e in factors.items():
        v[_PRIME_TO_IDX[p]] = e
    return tuple(v)

_VECTORS = {n: _to_vec(_FACTORS[n]) for n in _NUMBERS}

# Two numbers conflict if one is a multiple of the other in terms of prime exponents
# Actually, for subset products to be distinct, we need that no non-empty subset of the chosen
# numbers has product 1 when considering multiplicative relations.
# The key insight: if we represent each number by its prime exponent vector, then subset products
# are distinct iff the sum of any non-empty subset of vectors is not equal to the sum of any other
# non-empty subset. This is equivalent to: the vectors are linearly independent over the integers
# in the sense that no integer linear combination with coefficients in {-1, 0, 1} equals zero,
# except the trivial one. But actually it's stronger: all 2^k subset sums must be distinct.
# 
# A simpler check: two numbers a, b conflict if a divides b or b divides a (in terms of prime exponents),
# because then {a, b} and {b} would give products a*b and b... no wait, that's not right either.
#
# Actually the condition is: for any two distinct subsets S1, S2 of the chosen set, prod(S1) != prod(S2).
# This is equivalent to: for any non-empty subset T, prod(T) != 1, AND for any two distinct non-empty subsets
# with T1 != T2, prod(T1) != prod(T2).
#
# The most efficient check: maintain a set of all subset products. When adding a new number x,
# the new products are {p * x for p in existing_products}. We need none of these to be in existing_products.

def _can_add(x, current_products):
    """Check if x can be added without creating duplicate products."""
    # For each existing product p, check if p * x is already in current_products
    # Equivalently: for each p in current_products, check if p * x in current_products
    # Or: for each q in current_products, check if q % x == 0 and q // x in current_products
    for p in current_products:
        if p * x in current_products:
            return False
    return True

def _can_add_fast(x, current_products):
    """Faster check using modular arithmetic."""
    for p in current_products:
        q = p * x
        if q in current_products:
            return False
    return True

def construct():
    deadline = time.time() + 55  # 55 second budget with safety margin
    
    best_set = []
    best_products = set()
    
    # Strategy: greedy with backtracking + random restarts
    # First, try a deterministic greedy approach
    # Then use random restarts with different orderings
    
    def greedy_build(order, start_set=None, start_products=None):
        """Build a set greedily following the given order."""
        if start_set is not None:
            selected = list(start_set)
            products = set(start_products)
        else:
            selected = []
            products = {1}
        
        for x in order:
            if x in selected:
                continue
            # Check if we can add x
            can_add = True
            for p in products:
                if p * x in products:
                    can_add = False
                    break
            if can_add:
                selected.append(x)
                new_products = set(products)
                for p in products:
                    new_products.add(p * x)
                products = new_products
        
        return selected, products
    
    # Deterministic greedy: try numbers in different orders
    # Order 1: smallest first
    order1 = sorted(_NUMBERS)
    # Order 2: largest first  
    order2 = sorted(_NUMBERS, reverse=True)
    # Order 3: primes first, then composites
    primes_first = sorted(_NUMBERS, key=lambda n: (len(_FACTORS[n]) > 1, n))
    # Order 4: by number of prime factors (fewer first)
    by_pf_count = sorted(_NUMBERS, key=lambda n: (len(_FACTORS[n]), n))
    
    for order in [order1, order2, primes_first, by_pf_count]:
        if time.time() > deadline:
            break
        s, p = greedy_build(order)
        if len(s) > len(best_set):
            best_set = s
            best_products = p
    
    # Random restarts with local search
    rng = random.Random(42)
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Create a random ordering with some bias
        order = list(_NUMBERS)
        rng.shuffle(order)
        
        # Sometimes bias towards numbers with fewer prime factors
        if restart % 3 == 0:
            order.sort(key=lambda n: (len(_FACTORS[n]), rng.random()))
        elif restart % 3 == 1:
            order.sort(key=lambda n: n)
            # Perturb
            for _ in range(3):
                i = rng.randint(0, len(order)-2)
                order[i], order[i+1] = order[i+1], order[i]
        else:
            # Random
            pass
        
        s, p = greedy_build(order)
        if len(s) > len(best_set):
            best_set = s
            best_products = p
        
        # Local search: try to add more numbers by removing some
        if len(s) < 18 and time.time() < deadline:
            # Try to improve by swapping
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(len(s)):
                    # Try removing s[i] and adding a different number
                    removed = s[i]
                    # Recompute products without removed
                    rem_products = {1}
                    for x in s:
                        if x == removed:
                            continue
                        new_rem = set(rem_products)
                        for p2 in rem_products:
                            new_rem.add(p2 * x)
                        rem_products = new_rem
                    
                    for x in _NUMBERS:
                        if x in s:
                            continue
                        # Check if x can be added
                        can_add = True
                        for p2 in rem_products:
                            if p2 * x in rem_products:
                                can_add = False
                                break
                        if can_add:
                            new_s = s[:i] + s[i+1:] + [x]
                            new_p = set(rem_products)
                            for p2 in rem_products:
                                new_p.add(p2 * x)
                            if len(new_s) > len(s):
                                s = new_s
                                p = new_p
                                improved = True
                                break
                    if improved:
                        break
    
    # Final verification
    if best_set:
        # Verify all subset products are distinct
        prods = {1}
        for x in best_set:
            new_prods = set(prods)
            for p in prods:
                new_prods.add(p * x)
            prods = new_prods
        if len(prods) != 2 ** len(best_set):
            # Fallback to a known valid set
            best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
