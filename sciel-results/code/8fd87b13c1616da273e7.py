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
    
    # Initial 19-set
    candidate_19 = [2, 4, 8, 16, 32, 3, 9, 27, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    if not _is_valid_set(candidate_19):
        best_set = []
        prods = {1}
        for x in range(2, 41):
            if _can_add(prods, x):
                best_set.append(x)
                for p in prods:
                    prods.add(p * x)
        return best_set

    best_set = candidate_19[:]
    best_len = len(best_set)
    all_nums = list(range(2, 41))
    
    # 1. Try to add any remaining number to the 19-set
    prods_best = _subset_products_set(best_set)
    for x in all_nums:
        if x in best_set:
            continue
        if _can_add(prods_best, x):
            best_set.append(x)
            best_len += 1
            prods_best = _subset_products_set(best_set)
            # If we added one, try to add another immediately
            for y in all_nums:
                if y in best_set:
                    continue
                if _can_add(prods_best, y):
                    best_set.append(y)
                    best_len += 1
                    break
            break
            
    if best_len >= 20:
        return best_set

    # 2. Systematic Swap Search
    # Goal: Find S' of size 19 such that S' + {z} is valid for some z.
    # This implies S' is a subset of a valid set of size 20.
    
    # Precompute prods for S \ {x} for all x in S
    prods_minus_map = {}
    for x in best_set:
        set_minus = [y for y in best_set if y != x]
        prods_minus_map[x] = _subset_products_set(set_minus)
        
    # For each x in S, and each y not in S:
    # Check if y can be added to S \ {x}
    # If yes, S' = (S \ {x}) U {y} is valid.
    # Then check if any z not in S' can be added to S'.
    
    found_better = False
    for x in best_set:
        if time.time() > _DEADLINE:
            break
        prods_minus = prods_minus_map[x]
        for y in all_nums:
            if y in best_set:
                continue
            if _can_add(prods_minus, y):
                # S' is valid
                set_prime = [z for z in best_set if z != x] + [y]
                prods_prime = set(prods_minus)
                for p in prods_minus:
                    prods_prime.add(p * y)
                
                # Check if we can add any z to S'
                for z in all_nums:
                    if z in set_prime:
                        continue
                    if _can_add(prods_prime, z):
                        # Found a valid set of size 20
                        best_set = set_prime + [z]
                        best_len = 20
                        found_better = True
                        break
                if found_better:
                    break
        if found_better:
            break
            
    if found_better:
        return best_set

    # 3. If no size 20 found via single swap, try random local search
    # This is a fallback
    rng = random.Random(42)
    max_attempts = 10000
    attempts = 0
    current_set = best_set[:]
    current_prods = _subset_products_set(current_set)
    current_len = len(current_set)
    
    while attempts < max_attempts and time.time() < _DEADLINE:
        attempts += 1
        # Random add
        candidates = [x for x in all_nums if x not in current_set]
        if candidates:
            x = rng.choice(candidates)
            if _can_add(current_prods, x):
                new_prods = set(current_prods)
                for p in current_prods:
                    new_prods.add(p * x)
                current_set.append(x)
                current_prods = new_prods
                current_len += 1
                if current_len > best_len:
                    best_set = current_set[:]
                    best_len = current_len
        else:
            # Random swap
            if current_set:
                x_remove = rng.choice(current_set)
                candidates = [x for x in all_nums if x not in current_set]
                if candidates:
                    x_add = rng.choice(candidates)
                    # Check validity of swap
                    set_minus = [z for z in current_set if z != x_remove]
                    prods_minus = _subset_products_set(set_minus)
                    if _can_add(prods_minus, x_add):
                        # Valid swap, but size is same.
                        # We only accept if it allows an addition?
                        # For now, we accept it to escape local minima.
                        new_set = set_minus + [x_add]
                        new_prods = set(prods_minus)
                        for p in prods_minus:
                            new_prods.add(p * x_add)
                        current_set = new_set
                        current_prods = new_prods

    return best_set
