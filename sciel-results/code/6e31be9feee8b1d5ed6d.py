"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

sys.setrecursionlimit(100000)

_START_TIME = time.time()
_DEADLINE = _START_TIME + 55.0  # Safety margin for 60s limit

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

def _greedy_construct():
    """
    Greedy construction: iterate through candidate numbers and add them
    if they maintain the distinct subset product property.
    """
    products = {1}
    selected = []
    for a in range(2, 41):
        if time.time() > _DEADLINE:
            break
        if _can_add(products, a):
            products = _add_number(products, a)
            selected.append(a)
    return selected, products

def _backtrack_search(candidates, current_products, current_selected, target_size):
    """
    Backtrack search to find a set of size target_size.
    """
    if len(current_selected) == target_size:
        return current_selected
    
    if time.time() > _DEADLINE:
        return None
    
    for i in range(len(candidates)):
        a = candidates[i]
        if _can_add(current_products, a):
            new_products = _add_number(current_products, a)
            result = _backtrack_search(candidates[i+1:], new_products, current_selected + [a], target_size)
            if result is not None:
                return result
    return None

def construct():
    # First, try greedy to get a baseline
    selected, products = _greedy_construct()
    best_size = len(selected)
    best_result = selected
    
    if best_size >= 18:
        return list(best_result)
    
    # Try to improve with backtracking, starting from the greedy solution
    # and trying to add more elements or swap for better ones
    
    # Strategy: Try to build up to 18 elements
    # We'll try different starting points and search strategies
    
    # First, let's see what the greedy gave us
    # Then try to find a larger set using backtracking with pruning
    
    # Candidate pool: all numbers 2-40
    all_candidates = list(range(2, 41))
    
    # Try to find a set of size 18 using backtracking
    # Start with empty set and try to build up
    # This might be too slow, so we use the greedy solution as a guide
    
    # Alternative: start with the greedy solution and try to swap/extend
    # For now, let's try a more systematic approach
    
    # Try backtracking from scratch with a limit
    # But first, let's try a smarter greedy: try primes first, then composites
    products = {1}
    selected = []
    
    # Try primes first (they're more likely to maintain distinctness)
    for a in range(2, 41):
        if time.time() > _DEADLINE:
            break
        if _is_prime(a):
            if _can_add(products, a):
                products = _add_number(products, a)
                selected.append(a)
    
    # Then try composites
    for a in range(2, 41):
        if time.time() > _DEADLINE:
            break
        if not _is_prime(a) and a not in selected:
            if _can_add(products, a):
                products = _add_number(products, a)
                selected.append(a)
    
    if len(selected) > best_size:
        best_size = len(selected)
        best_result = selected
    
    if best_size >= 18:
        return list(best_result)
    
    # Try a more exhaustive search with backtracking
    # Start from the best greedy solution and try to extend it
    # or find a different combination
    
    # Let's try backtracking to find a set of size best_size + 1
    # We'll use the candidates not in the current solution
    remaining = [a for a in all_candidates if a not in selected]
    
    for target in range(best_size + 1, 19):
        if time.time() > _DEADLINE:
            break
        
        # Try to find a set of size 'target'
        # Use backtracking with the current products as a starting point
        # But we need to be careful: we can't just add to the current set
        # We need to search for a completely new set of size 'target'
        
        # For efficiency, let's try a limited backtracking search
        # Start with a few promising elements and build up
        
        # This is getting complex, so let's try a simpler approach:
        # Randomized restarts with greedy
        import random
        random.seed(42)
        
        for attempt in range(100):
            if time.time() > _DEADLINE:
                break
            
            # Shuffle candidates
            shuffled = all_candidates[:]
            random.shuffle(shuffled)
            
            products = {1}
            sel = []
            for a in shuffled:
                if _can_add(products, a):
                    products = _add_number(products, a)
                    sel.append(a)
            
            if len(sel) > best_size:
                best_size = len(sel)
                best_result = sel
                if best_size >= 18:
                    return list(best_result)
    
    return list(best_result)
# EVOLVE-BLOCK-END
