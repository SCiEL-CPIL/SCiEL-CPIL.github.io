import time
import random
from math import gcd

# EVOLVE-BLOCK-START

# Precompute prime factorizations for 2..40
# We'll use a different approach: track which "collision patterns" have been used

def _prime_factorize(n):
    """Return prime factorization of n as a dict {prime: exponent}."""
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

# Precompute factorizations
_FACTORS = {n: _prime_factorize(n) for n in range(2, 41)}

def _can_add_fast(current_products, candidate):
    """
    Check if adding candidate keeps all subset products distinct.
    Uses the fact that a collision occurs iff candidate * p1 == p2 for some p1, p2 in products.
    Equivalently, p2/p1 == candidate for some p1 != p2 in products.
    """
    # Check: for any p in current_products, is p * candidate already in current_products?
    # This is equivalent to: does {x * candidate for x in current_products} intersect current_products?
    new_products = {x * candidate for x in current_products}
    return len(new_products & current_products) == 0

def _greedy_build(candidates, deadline, rng):
    """
    Greedy build: add candidates in given order if they don't create collisions.
    Returns (result_list, product_set).
    """
    result = []
    current_products = {1}
    for c in candidates:
        if time.time() > deadline:
            break
        if _can_add_fast(current_products, c):
            result.append(c)
            current_products |= {x * c for x in current_products}
    return result, current_products

def _local_search(result, current_products, all_candidates, deadline, rng, max_swaps=300):
    """
    Try to improve the set by swapping elements out and in, or adding new elements.
    Returns (best_result, best_products, best_size).
    """
    best_size = len(result)
    best_result = list(result)
    best_products = set(current_products)
    
    # First, try to add any remaining candidates
    for c in all_candidates:
        if time.time() > deadline:
            break
        if c not in result and _can_add_fast(current_products, c):
            result.append(c)
            current_products |= {x * c for x in current_products}
            best_size = len(result)
            best_result = list(result)
            best_products = set(current_products)
    
    # Then try swaps
    for _ in range(max_swaps):
        if time.time() > deadline:
            break
        
        if not result:
            break
        
        out_idx = rng.randrange(len(result))
        out_elem = result[out_idx]
        
        # Find candidates not in result
        available = [c for c in all_candidates if c not in result]
        if not available:
            break
        
        in_elem = rng.choice(available)
        
        # Check: remove out_elem, add in_elem
        new_set = result[:out_idx] + result[out_idx+1:] + [in_elem]
        
        # Check validity
        products = {1}
        valid = True
        for n in new_set:
            new_products = {x * n for x in products}
            if new_products & products:
                valid = False
                break
            products |= new_products
        
        if valid and len(new_set) > best_size:
            best_size = len(new_set)
            best_result = new_set
            best_products = products
            result = list(new_set)
            current_products = products
    
    return best_result, best_products, best_size

def _randomized_greedy(all_candidates, deadline, rng, n_restarts=80):
    """
    Multiple randomized greedy restarts to find larger valid sets.
    Uses weighted randomization favoring numbers with fewer prime factors.
    """
    best_result = []
    best_products = {1}
    best_size = 0
    
    # Precompute weights: prefer numbers with fewer distinct prime factors
    weights = []
    for c in all_candidates:
        # Count distinct prime factors
        distinct_primes = len(_FACTORS[c])
        # Inverse weight: fewer primes = higher priority
        w = 1.0 / (distinct_primes + 0.5)
        weights.append(w)
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Weighted random shuffle
        shuffled = list(all_candidates)
        # Use rng to create a weighted order
        # Simple approach: sort by random key with bias
        keyed = [(c, rng.random() / (weights[i] + 0.1)) for i, c in enumerate(shuffled)]
        keyed.sort(key=lambda x: x[1])
        shuffled = [k[0] for k in keyed]
        
        result, products = _greedy_build(shuffled, deadline, rng)
        
        if len(result) > best_size:
            best_size = len(result)
            best_result = list(result)
            best_products = products
    
    # Local search on the best result
    if best_size > 0 and time.time() < deadline:
        best_result, best_products, best_size = _local_search(
            best_result, best_products, all_candidates, deadline, rng
        )
    
    return best_result, best_products

def construct():
    """
    Returns a list of distinct integers in the range 2..40 such that
    all 2^k subset products are distinct. Maximizes k.
    """
    t0 = time.time()
    deadline = t0 + 55.0  # 55 second budget (5s margin before 60s kill)
    
    rng = random.Random(42)
    
    all_candidates = list(range(2, 41))
    
    # Phase 1: Fast deterministic greedy with sorted order (quick baseline)
    def count_prime_factors(n):
        return len(_FACTORS[n])
    
    sorted_candidates = sorted(all_candidates, key=count_prime_factors)
    base_result, base_products = _greedy_build(sorted_candidates, deadline, rng)
    
    # Phase 2: Randomized greedy with multiple restarts
    rand_result, rand_products = _randomized_greedy(all_candidates, deadline, rng, n_restarts=120)
    
    # Pick the better one
    if len(rand_result) > len(base_result):
        result = rand_result
        current_products = rand_products
    else:
        result = base_result
        current_products = base_products
    
    # Phase 3: More local search iterations on the best result
    if time.time() < deadline:
        result, current_products, _ = _local_search(
            result, current_products, all_candidates, deadline, rng, max_swaps=400
        )
    
    # Ensure all elements are Python int
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
