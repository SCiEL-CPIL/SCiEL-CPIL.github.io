import time
import itertools
import random

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
    for p in prods:
        if (p * x) in prods:
            return False
    return True

def _try_add(selected, prods, x):
    if not _can_add(prods, x):
        return False, selected, prods
    new_prods = set(prods)
    for p in prods:
        new_prods.add(p * x)
    new_selected = selected + [x]
    return True, new_selected, new_prods

def _is_valid_set(nums):
    prods = {1}
    for x in nums:
        if not _can_add(prods, x):
            return False
        new_prods = set(prods)
        for p in prods:
            new_prods.add(p * x)
        prods = new_prods
    return True

def construct():
    # Primes in 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Targeted search for 19: Pure powers of primes
    # 2: 2, 4, 8, 16, 32 (exponents 1, 2, 4, 8, 16 -> unique subset sums)
    # 3: 3, 9, 27 (exponents 1, 2, 4 -> unique subset sums)
    # 5: 5, 25 (exponents 1, 2 -> unique subset sums)
    # 7: 7
    # 11, 13, 17, 19, 23, 29, 31, 37: primes
    candidate_19 = [2, 4, 8, 16, 32, 3, 9, 27, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    if _is_valid_set(candidate_19):
        return candidate_19

    # Fallback: Greedy + Local Search
    composites = [c for c in range(2, 41) if c not in primes]
    
    best_set = []
    best_len = 0
    
    # Greedy baseline
    selected = []
    prods = {1}
    for x in sorted(range(2, 41)):
        if time.time() > _DEADLINE:
            break
        ok, selected, prods = _try_add(selected, prods, x)
        if ok:
            pass
    if len(selected) > best_len:
        best_set = selected
        best_len = len(selected)
        
    # Random search for improvement
    rng = random.Random(42)
    max_attempts = 5000
    attempts = 0
    
    while attempts < max_attempts and time.time() < _DEADLINE:
        attempts += 1
        # Try to build a set of size 18 by picking 6 composites
        # Prioritize "pure powers" which are more likely to be independent
        priority_composites = [4, 8, 16, 32, 9, 27, 25, 49, 121, 169, 289, 361]
        available = [c for c in priority_composites if c in composites]
        if len(available) < 6:
             available = composites
             
        c_sub = rng.sample(available, 6)
        current_prods = _subset_products_set(primes)
        valid = True
        selected_c = []
        
        for c in c_sub:
            if not _can_add(current_prods, c):
                valid = False
                break
            new_prods = set(current_prods)
            for p in current_prods:
                new_prods.add(p * c)
            current_prods = new_prods
            selected_c.append(c)
            
        if valid:
            final_set = primes + selected_c
            if _is_valid_set(final_set):
                return final_set

    return best_set
