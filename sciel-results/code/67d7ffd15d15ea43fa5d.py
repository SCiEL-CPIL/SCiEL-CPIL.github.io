import time
import random
from math import gcd

# EVOLVE-BLOCK-START

def _is_valid_set(nums):
    """Check if all 2^k subset products are distinct."""
    products = {1}
    for n in nums:
        new_products = {x * n for x in products}
        if new_products & products:
            return False
        products |= new_products
    return True

def _can_add(current_products, candidate):
    """Check if adding candidate keeps all subset products distinct."""
    new_products = {x * candidate for x in current_products}
    return len(new_products & current_products) == 0

def _compute_products(nums):
    """Compute the set of all subset products."""
    products = {1}
    for n in nums:
        products |= {x * n for x in products}
    return products

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
        if _can_add(current_products, c):
            result.append(c)
            current_products |= {x * c for x in current_products}
    return result, current_products

def _local_search(result, current_products, all_candidates, deadline, rng, max_swaps=200):
    """
    Try to improve the set by swapping elements out and in.
    Returns (best_result, best_products, best_size).
    """
    best_size = len(result)
    best_result = list(result)
    best_products = set(current_products)
    
    for _ in range(max_swaps):
        if time.time() > deadline:
            break
        
        # Try removing a random element and adding a random candidate
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
        # New products = (products without out_elem contributions) ∪ (those * in_elem)
        # This is tricky; let's just rebuild for simplicity
        new_set = result[:out_idx] + result[out_idx+1:] + [in_elem]
        
        # Check validity efficiently
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

def _randomized_greedy(all_candidates, deadline, rng, n_restarts=50):
    """
    Multiple randomized greedy restarts to find larger valid sets.
    """
    best_result = []
    best_products = {1}
    best_size = 0
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        shuffled = list(all_candidates)
        rng.shuffle(shuffled)
        
        result, products = _greedy_build(shuffled, deadline, rng)
        
        if len(result) > best_size:
            best_size = len(result)
            best_result = list(result)
            best_products = products
    
    # Local search on the best result
    if best_size > 0:
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
    deadline = t0 + 50.0  # 50 second budget (10s margin before 60s kill)
    
    rng = random.Random(42)
    
    all_candidates = list(range(2, 41))
    
    # Phase 1: Fast deterministic greedy with sorted order (quick baseline)
    def count_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    sorted_candidates = sorted(all_candidates, key=count_prime_factors)
    base_result, base_products = _greedy_build(sorted_candidates, deadline, rng)
    
    # Phase 2: Randomized greedy with multiple restarts
    rand_result, rand_products = _randomized_greedy(all_candidates, deadline, rng, n_restarts=100)
    
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
            result, current_products, all_candidates, deadline, rng, max_swaps=500
        )
    
    # Ensure all elements are Python int
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
