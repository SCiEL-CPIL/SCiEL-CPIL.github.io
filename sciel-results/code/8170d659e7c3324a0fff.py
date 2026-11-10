"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from math import isqrt

DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _subset_products_collide(existing_products, x):
    """Check if adding x to a set with existing subset products causes a collision.
    existing_products: set of all subset products of current set (includes 1 for empty product)
    Returns True if adding x would cause a collision.
    """
    new_products = set()
    for p in existing_products:
        new_products.add(p)
        new_products.add(p * x)
    return len(new_products) != 2 * len(existing_products)

def _greedy_select(candidates, order, deadline):
    """Greedy: add candidates in given order if no collision."""
    selected = []
    products = {1}  # subset products of empty set
    for x in order:
        if time.time() > deadline:
            break
        if x in selected:
            continue
        # Check collision
        new_products = set()
        for p in products:
            new_products.add(p)
            new_products.add(p * x)
        if len(new_products) == 2 * len(products):
            selected.append(x)
            products = new_products
    return selected, products

def _try_improve(selected, products, candidates_pool, deadline):
    """Try to swap out a selected element for an unselected one to gain size, or just try adding."""
    # First try: can we add any unselected candidate?
    for x in candidates_pool:
        if time.time() > deadline:
            break
        if x in selected:
            continue
        new_products = set()
        for p in products:
            new_products.add(p)
            new_products.add(p * x)
        if len(new_products) == 2 * len(products):
            return selected + [x], new_products
    return selected, products

def _local_search(selected, products, all_candidates, deadline):
    """Local search: try removing one and adding one to potentially increase size."""
    best = (selected[:], products.copy())
    best_size = len(selected)
    
    for i, remove_x in enumerate(selected):
        if time.time() > deadline:
            break
        # Remove remove_x
        new_selected = selected[:i] + selected[i+1:]
        # Recompute products without remove_x
        new_products = {1}
        for x in new_selected:
            new_products = new_products | {p * x for p in new_products}
        
        # Try adding each unselected candidate
        for add_x in all_candidates:
            if time.time() > deadline:
                break
            if add_x in new_selected:
                continue
            test_products = set()
            for p in new_products:
                test_products.add(p)
                test_products.add(p * add_x)
            if len(test_products) == 2 * len(new_products):
                if len(new_selected) + 1 > best_size:
                    best = (new_selected + [add_x], test_products)
                    best_size = len(new_selected) + 1
                elif len(new_selected) + 1 == best_size:
                    # Same size, keep as alternative
                    pass
    return best

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    all_candidates = list(range(2, 41))
    
    # Strategy 1: Greedy with different orderings
    # Order by: ascending, descending, by number of prime factors, etc.
    
    # Prime factor count heuristic: numbers with more distinct prime factors are "costlier"
    def prime_factor_count(n):
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
    
    # Try multiple orderings
    orderings = [
        sorted(all_candidates),  # ascending
        sorted(all_candidates, reverse=True),  # descending
        sorted(all_candidates, key=lambda x: prime_factor_count(x)),  # fewest prime factors first
        sorted(all_candidates, key=lambda x: (-prime_factor_count(x), x)),  # most prime factors first
        sorted(all_candidates, key=lambda x: (x % 2 == 0, x)),  # odd first
        sorted(all_candidates, key=lambda x: (x % 2 == 1, x)),  # even first
        # Mix: primes first, then composites
        sorted(all_candidates, key=lambda x: (not all(x%d for d in range(2, isqrt(x)+1)) and x > 1, x)),
    ]
    
    best_result = []
    best_products = {1}
    
    for order in orderings:
        if time.time() > deadline:
            break
        result, products = _greedy_select(all_candidates, order, deadline)
        if len(result) > len(best_result):
            best_result = result
            best_products = products
    
    # Local search to improve
    if time.time() < deadline:
        improved, improved_products = _local_search(best_result, best_products, all_candidates, deadline)
        if len(improved) > len(best_result):
            best_result = improved
            best_products = improved_products
    
    # Try more aggressive local search if time permits
    if time.time() < deadline - 5:
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            improved, improved_products = _local_search(best_result, best_products, all_candidates, deadline - 2)
            if len(improved) > len(best_result):
                best_result = improved
                best_products = improved_products
            else:
                break
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best_result]
# EVOLVE-BLOCK-END
