import time
from functools import lru_cache

_P0 = time.time()
_DEADLINE = _P0 + 85.0

def _subset_products_set(nums):
    prods = {1}
    for x in nums:
        add = set()
        for p in prods:
            add.add(p * x)
        prods = prods | add
    return prods

def _can_add(prods, x):
    """Check if adding x to a set with existing products prods keeps all distinct."""
    for p in prods:
        if p * x in prods:
            return False
    return True

def _add_to_prods(prods, x):
    """Return new product set after adding x."""
    new_prods = set(prods)
    for p in prods:
        new_prods.add(p * x)
    return new_prods

def construct():
    deadline = _DEADLINE
    candidates = list(range(2, 41))
    
    # First, identify primes
    def is_prime(n):
        if n < 2:
            return False
        for d in range(2, int(n**0.5) + 1):
            if n % d == 0:
                return False
        return True
    
    primes = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Start with all primes, then try to add composites
    # This gives us at least 12
    cur_set = primes[:]
    cur_prods = _subset_products_set(cur_set)
    if len(cur_set) > best_size:
        best_set = cur_set[:]
        best_size = len(cur_set)
    
    # Try to add composites to the prime set
    for x in composites:
        if time.time() > deadline - 5:
            break
        if _can_add(cur_prods, x):
            cur_set.append(x)
            cur_prods = _add_to_prods(cur_prods, x)
            if len(cur_set) > best_size:
                best_set = cur_set[:]
                best_size = len(cur_set)
    
    # Strategy 2: Backtracking search with different orderings
    # Try to find larger sets by exploring the space
    
    def backtrack(current_set, current_prods, remaining, depth=0):
        nonlocal best_set, best_size, deadline
        
        if time.time() > deadline - 2:
            return
        
        if len(current_set) > best_size:
            best_set = current_set[:]
            best_size = len(current_set)
        
        if not remaining:
            return
        
        # Pruning: if we can't beat best_size even with all remaining, stop
        if len(current_set) + len(remaining) <= best_size:
            return
        
        for i, x in enumerate(remaining):
            if time.time() > deadline - 2:
                return
            
            if _can_add(current_prods, x):
                new_prods = _add_to_prods(current_prods, x)
                backtrack(current_set + [x], new_prods, remaining[i+1:], depth+1)
            
            # Also try skipping x (implicit in the recursion)
            # But we need to be careful about the order
    
    # Use a smarter ordering: try composites first that are "independent"
    # Sort composites by some heuristic
    
    # Try backtracking with different candidate orderings
    orderings = [
        composites + primes,
        primes + composites,
        sorted(composites, key=lambda x: x) + primes,
        sorted(composites, key=lambda x: -x) + primes,
    ]
    
    for order in orderings:
        if time.time() > deadline - 3:
            break
        backtrack([], {1}, order)
    
    # Ensure best_set is a list of Python ints
    return list(best_set)
