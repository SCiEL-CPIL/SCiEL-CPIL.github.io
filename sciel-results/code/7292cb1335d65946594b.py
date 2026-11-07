"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

# Increase recursion limit just in case, though depth is small
sys.setrecursionlimit(10000)

_START_TIME = time.time()
_DEADLINE = _START_TIME + 55.0  # Safety margin for 60s limit

def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def _can_add(prods, x):
    """
    Check if adding x to a set with existing subset products 'prods'
    keeps all subset products distinct.
    
    The new products will be {p * x for p in prods}.
    For the new set to be valid, none of these new products can collide
    with existing products in 'prods'.
    (Collisions among new products p1*x == p2*x imply p1==p2, which is impossible
    since prods are distinct and x != 0).
    """
    for p in prods:
        if p * x in prods:
            return False
    return True

def construct():
    """
    Returns a list of distinct integers in 2..40 such that all 2^k subset products are distinct.
    Maximizes the number of selected integers.
    
    Strategy:
    1. Start with all primes in [2, 40]. This set is always valid because 
       prime factorizations are unique.
    2. Attempt to add composite numbers one by one.
    3. Order composites by "risk": those with fewer prime factors are generally 
       safer to add earlier? Or perhaps those that are powers of primes?
       Actually, let's just sort composites by value, or by number of prime factors.
       Let's try sorting by number of prime factors (with multiplicity) ascending, 
       then by value. This tends to add "simple" composites first.
    4. For each composite, check if it can be added to the current set of subset products.
    5. If yes, add it and update the subset products.
    6. If no, skip it.
    
    This greedy approach is very fast and likely finds a near-optimal solution.
    """
    
    candidates = list(range(2, 41))
    
    primes = [x for x in candidates if _is_prime(x)]
    composites = [x for x in candidates if not _is_prime(x)]
    
    # Sort composites by number of prime factors (with multiplicity)
    def prime_factor_count(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    composites.sort(key=lambda x: (prime_factor_count(x), x))
    
    # Start with primes
    current_set = primes
    prods = {1}
    for p in primes:
        new_prods = set(prods)
        for prod in prods:
            new_prods.add(prod * p)
        prods = new_prods
        
    # Try to add composites
    for c in composites:
        if time.time() > _DEADLINE:
            break
        if _can_add(prods, c):
            current_set.append(c)
            # Update prods
            new_prods = set(prods)
            for prod in prods:
                new_prods.add(prod * c)
            prods = new_prods
            
    # Ensure types are correct
    return [int(x) for x in current_set]
# EVOLVE-BLOCK-END
