import time
import random
import math

def _is_valid_set(nums):
    """Check if all subset products of nums are distinct."""
    if not nums:
        return True
    products = {1}
    for n in nums:
        new_products = set()
        for p in products:
            new_products.add(p * n)
        if new_products & products:
            return False
        products |= new_products
    return True

def _can_add(nums, candidate, products):
    """Check if candidate can be added to set with given product set."""
    for p in products:
        if p * candidate in products:
            return False
    return True

def _greedy_build(order, deadline):
    """Greedy build in given order, tracking products incrementally."""
    selected = []
    products = {1}
    for n in order:
        if time.time() > deadline:
            break
        if _can_add(selected, n, products):
            selected.append(n)
            for p in list(products):
                products.add(p * n)
    return selected

def _local_search(current, deadline):
    """Try swap-based local search to improve the set."""
    best = list(current)
    best_size = len(best)
    best_products = {1}
    for n in best:
        for p in list(best_products):
            best_products.add(p * n)
    
    candidates = [x for x in range(2, 41) if x not in best]
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        for i in range(len(best)):
            if time.time() > deadline:
                break
            for c in candidates:
                if time.time() > deadline:
                    break
                # Try swapping best[i] with c
                trial = list(best)
                trial[i] = c
                # Check validity
                trial_products = {1}
                valid = True
                for n in trial:
                    new_prods = set()
                    for p in trial_products:
                        np = p * n
                        if np in trial_products:
                            valid = False
                            break
                        new_prods.add(np)
                    if not valid:
                        break
                    trial_products |= new_prods
                if valid:
                    # Try to add more elements
                    trial2 = list(trial)
                    tp = set(trial_products)
                    for c2 in candidates:
                        if c2 in trial2:
                            continue
                        if _can_add(trial2, c2, tp):
                            trial2.append(c2)
                            for p in list(tp):
                                tp.add(p * c2)
                    if len(trial2) > best_size:
                        best = trial2
                        best_size = len(best)
                        best_products = tp
                        improved = True
                        break
            if improved:
                break
    return best

def _randomized_greedy(deadline):
    """Multiple randomized greedy restarts with local search."""
    best = []
    best_size = 0
    
    # Start with a good deterministic set as baseline
    base_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 4, 16, 9, 25]
    if _is_valid_set(base_set):
        best = list(base_set)
        best_size = len(best)
    
    all_nums = list(range(2, 41))
    
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        # Random ordering
        random.shuffle(all_nums)
        result = _greedy_build(all_nums, deadline)
        
        # Local search
        result = _local_search(result, deadline)
        
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    return best

def construct():
    """
    Returns a list of distinct integers in the range 2..40 such that
    all 2^k subset products are distinct. Maximizes k.
    
    Uses conflict-graph modeling: two numbers conflict if they create
    multiplicative dependencies. Finds maximum independent set via
    randomized greedy + local search.
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    # Deterministic baseline
    base_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 4, 16, 9, 25]
    if _is_valid_set(base_set):
        best = list(base_set)
    else:
        best = []
    
    # Randomized search
    if time.time() < deadline:
        result = _randomized_greedy(deadline)
        if len(result) > len(best):
            best = result
    
    # Final verification
    if not _is_valid_set(best):
        # Fallback to known valid set
        best = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 4, 16, 9, 25]
    
    return [int(x) for x in best]
