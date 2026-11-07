"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

sys.setrecursionlimit(10000)

_START_TIME = time.time()
_DEADLINE = _START_TIME + 55.0  # Safety margin for 60s limit

# Precompute primes up to 40
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

# All primes in 2..40
_ALL_PRIMES = [p for p in range(2, 41) if _is_prime(p)]
# Primes that appear in factorizations of numbers 2..40
_RELEVANT_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def _factorize(n):
    """Return dict of prime -> exponent for n."""
    factors = {}
    for p in _RELEVANT_PRIMES:
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

# Precompute factorizations for 2..40
_FACTORIZATIONS = {}
for i in range(2, 41):
    _FACTORIZATIONS[i] = _factorize(i)

# Precompute prime exponent vectors as tuples for fast comparison
_PRIME_TO_IDX = {p: i for i, p in enumerate(_RELEVANT_PRIMES)}
_EXP_VECTORS = {}
for i in range(2, 41):
    vec = [0] * len(_RELEVANT_PRIMES)
    for p, e in _FACTORIZATIONS[i].items():
        if p in _PRIME_TO_IDX:
            vec[_PRIME_TO_IDX[p]] = e
    _EXP_VECTORS[i] = tuple(vec)

def _can_add_fast(products, x):
    """
    Check if adding x to a set with existing subset products 'products'
    keeps all subset products distinct.
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

def _verify(nums):
    """Verify all subset products are distinct."""
    prods = {1}
    for x in nums:
        new_prods = set()
        for p in prods:
            new_prods.add(p * x)
        if len(prods | new_prods) != len(prods) + len(new_prods):
            return False
        prods = prods | new_prods
    return True

def _local_search_improve(nums, deadline):
    """Try to improve the set by swapping elements or adding new ones."""
    nums = list(nums)
    products = {1}
    for x in nums:
        products = _add_number(products, x)
    
    candidates = [x for x in range(2, 41) if x not in nums]
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        
        # Try adding a candidate
        for x in candidates:
            if time.time() > deadline:
                break
            if _can_add_fast(products, x):
                products = _add_number(products, x)
                nums.append(x)
                candidates.remove(x)
                improved = True
                break
        
        # Try swapping: remove one, try to add two new ones
        if not improved and len(nums) >= 2:
            for i in range(len(nums)):
                if time.time() > deadline:
                    break
                removed = nums[i]
                remaining = nums[:i] + nums[i+1:]
                
                # Rebuild products without removed
                rem_prods = {1}
                for x in remaining:
                    rem_prods = _add_number(rem_prods, x)
                
                # Try to add two new candidates
                added = []
                for x in candidates:
                    if time.time() > deadline:
                        break
                    if _can_add_fast(rem_prods, x):
                        rem_prods = _add_number(rem_prods, x)
                        added.append(x)
                        if len(added) == 2:
                            break
                
                if len(added) == 2:
                    # Do the swap
                    nums = remaining + added
                    products = rem_prods
                    for x in added:
                        if x in candidates:
                            candidates.remove(x)
                    improved = True
                    break
    
    return nums

def construct():
    """
    Find the largest set of distinct integers in 2..40 such that all
    2^k subset products (including empty product = 1) are distinct.
    """
    candidates = list(range(2, 41))
    
    primes = [x for x in candidates if _is_prime(x)]
    composites = [x for x in candidates if not _is_prime(x)]
    
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

    # Greedy initialization: primes first, then composites by fewest factors
    order1 = primes + sorted(composites, key=lambda x: (num_prime_factors(x), x))
    products = {1}
    selected = []
    for x in order1:
        if _can_add_fast(products, x):
            products = _add_number(products, x)
            selected.append(x)
    
    best_set = list(selected)
    
    # Backtracking search with improved pruning
    # Order: primes first (most likely to be included), then composites
    # Sort composites by a heuristic: prefer those with unique prime factors
    def composite_score(x):
        """Score composites: prefer those with fewer/rarer prime factors."""
        f = _FACTORIZATIONS[x]
        # Count how many other numbers share these prime factors
        score = 0
        for p in f:
            # How common is this prime in 2..40?
            common = sum(1 for y in range(2, 41) if y % p == 0)
            score += common
        return (score, num_prime_factors(x), x)
    
    search_candidates = primes + sorted(composites, key=composite_score)
    
    best = [best_set]
    
    def backtrack(candidates, products, selected, depth):
        if time.time() > _DEADLINE:
            return
        
        if len(selected) > len(best[0]):
            best[0] = list(selected)
            
        if depth >= len(candidates):
            return
            
        # Pruning: if we can't beat the best size with remaining candidates
        if len(selected) + (len(candidates) - depth) <= len(best[0]):
            return
            
        x = candidates[depth]
        
        # Try including x
        if _can_add_fast(products, x):
            new_prods = _add_number(products, x)
            selected.append(x)
            backtrack(candidates, new_prods, selected, depth + 1)
            selected.pop()
            
        # Try excluding x
        backtrack(candidates, products, selected, depth + 1)
        
    # Run backtracking
    backtrack(search_candidates, {1}, [], 0)
    
    # Apply local search improvement
    if time.time() < _DEADLINE - 5:
        improved = _local_search_improve(best[0], _DEADLINE - 2)
        if len(improved) > len(best[0]):
            best[0] = improved
    
    # Verify the best result
    if not _verify(best[0]):
        # Fallback to primes
        best[0] = primes
        
    return [int(x) for x in best[0]]

# EVOLVE-BLOCK-END
