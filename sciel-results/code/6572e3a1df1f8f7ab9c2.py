"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

sys.setrecursionlimit(10000)

_START_TIME = time.time()
_DEADLINE = _START_TIME + 55.0  # Safety margin for 60s limit

def _can_add(products, x):
    """
    Check if adding x to a set with existing subset products 'products'
    keeps all subset products distinct.
    
    New products will be {p * x for p in products}.
    For validity: none of p*x can collide with existing products.
    (p1*x == p2*x implies p1==p2, so new products are distinct among themselves.)
    """
    for p in products:
        if p * x in products:
            return False
    return True

def _add_number(products, x):
    """Return new set of products after adding x."""
    new_prods = set(products)
    for p in products:
        new_prods.add(p * x)
    return new_prods

def _greedy_construct(candidates):
    """Greedy: add candidates in given order if possible."""
    products = {1}
    selected = []
    for x in candidates:
        if _can_add(products, x):
            products = _add_number(products, x)
            selected.append(x)
    return selected

def _backtrack_search(candidates, products, selected, depth, best):
    """Backtracking search to find larger sets."""
    if time.time() > _DEADLINE:
        return
    
    if len(selected) > len(best[0]):
        best[0] = list(selected)
    
    if depth >= len(candidates):
        return
    
    # Try including candidates[depth]
    x = candidates[depth]
    if _can_add(products, x):
        new_prods = _add_number(products, x)
        selected.append(x)
        _backtrack_search(candidates, new_prods, selected, depth + 1, best)
        selected.pop()
    
    # Try excluding candidates[depth]
    _backtrack_search(candidates, products, selected, depth + 1, best)

def construct():
    """
    Find the largest set of distinct integers in 2..40 such that all
    2^k subset products (including empty product = 1) are distinct.
    """
    # Candidates: all integers from 2 to 40
    candidates = list(range(2, 41))
    
    # First, try a greedy approach with different orderings
    # Strategy: try primes first (they're most likely to be independent),
    # then composites
    
    def is_prime(n):
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
    
    primes = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    
    # Greedy with primes first
    best = _greedy_construct(primes + composites)
    
    # Also try greedy with composites interleaved differently
    # Sort by number of prime factors (fewer first)
    def num_prime_factors(x):
        count = 0
        d = 2
        n = x
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    sorted_by_pf = sorted(candidates, key=lambda x: (num_prime_factors(x), x))
    best2 = _greedy_construct(sorted_by_pf)
    if len(best2) > len(best):
        best = best2
    
    # Try a few more greedy orderings
    # Reverse order
    best3 = _greedy_construct(candidates[::-1])
    if len(best3) > len(best):
        best = best3
    
    # Now try backtracking to see if we can do better
    # Use a limited backtracking to avoid timeout
    # Start with the best greedy solution and try to improve
    
    # Try backtracking on the full candidate list with time limit
    # To make it feasible, limit the branching
    if time.time() < _DEADLINE - 5:
        # Use backtracking with a reduced candidate set for speed
        # Focus on promising candidates
        # Try backtracking with primes + some composites
        bt_candidates = primes + composites[:10]  # Limit composites for speed
        
        best_bt = [best]  # mutable container
        _backtrack_search(bt_candidates, {1}, [], 0, best_bt)
        
        if len(best_bt[0]) > len(best):
            best = best_bt[0]
    
    # Verify the result
    # Check all subset products are distinct
    def verify(nums):
        products = {1}
        for x in nums:
            new_prods = set()
            for p in products:
                new_prods.add(p * x)
            if len(products | new_prods) != len(products) + len(new_prods):
                return False
            products = products | new_prods
        return True
    
    if not verify(best):
        # Fallback to primes only
        best = primes
    
    # Ensure return type is list of Python ints
    return [int(x) for x in best]

# EVOLVE-BLOCK-END
