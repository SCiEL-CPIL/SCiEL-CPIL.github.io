import time
import random
from math import gcd

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

def _can_add(products, candidate):
    """Check if candidate can be added given current product set."""
    for p in products:
        if p * candidate in products:
            return False
    return True

def _add_to_products(products, n):
    """Return new product set after adding n."""
    new_products = set()
    for p in products:
        new_products.add(p * n)
    return products | new_products

def _greedy_build(order, deadline):
    """Greedy build in given order, tracking products incrementally."""
    selected = []
    products = {1}
    for n in order:
        if time.time() > deadline:
            break
        if _can_add(products, n):
            selected.append(n)
            products = _add_to_products(products, n)
    return selected, products

def _try_extend(selected, products, candidates, deadline):
    """Try to greedily add more elements to selected."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for c in candidates:
            if c in selected:
                continue
            if _can_add(products, c):
                selected.append(c)
                products = _add_to_products(products, c)
                changed = True
                break
    return selected, products

def _local_search_swap(current, deadline):
    """Diversity-driven swap: replace elements that cause the most conflicts.
    After each swap, try to extend greedily."""
    best = list(current)
    best_products = {1}
    for n in best:
        best_products = _add_to_products(best_products, n)
    best_size = len(best)
    
    candidates = [x for x in range(2, 41) if x not in best]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Score each element by how many candidates it conflicts with
        conflict_scores = []
        for i, elem in enumerate(best):
            # Remove elem temporarily
            remaining = best[:i] + best[i+1:]
            rem_products = {1}
            for n in remaining:
                rem_products = _add_to_products(rem_products, n)
            # Count how many candidates can be added
            addable = 0
            for c in candidates:
                if c in remaining:
                    continue
                if _can_add(rem_products, c):
                    addable += 1
            conflict_scores.append((addable, i, elem))
        
        # Try swapping the element with fewest addable candidates (most restrictive)
        conflict_scores.sort(key=lambda x: x[0])
        for addable, i, elem in conflict_scores:
            if time.time() > deadline:
                break
            remaining = best[:i] + best[i+1:]
            rem_products = {1}
            for n in remaining:
                rem_products = _add_to_products(rem_products, n)
            # Try adding each candidate
            for c in candidates:
                if c in remaining:
                    continue
                if _can_add(rem_products, c):
                    trial = remaining + [c]
                    trial_products = _add_to_products(rem_products, c)
                    # Try to extend
                    trial2, tp2 = _try_extend(list(trial), trial_products, 
                                             [x for x in range(2,41) if x not in trial], deadline)
                    if len(trial2) > best_size:
                        best = trial2
                        best_products = tp2
                        best_size = len(best)
                        candidates = [x for x in range(2, 41) if x not in best]
                        improved = True
                        break
            if improved:
                break
    return best

def _randomized_search(deadline):
    """Multi-phase search combining randomized greedy, heuristics, and local search."""
    best = []
    best_size = 0
    all_nums = list(range(2, 41))
    
    # Prime factorization for heuristics
    def factorize(n):
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
    
    prime_factors = {n: factorize(n) for n in all_nums}
    
    # Phase 1: Deterministic baseline with primes first
    base_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    products = {1}
    for n in base_set:
        if _can_add(products, n):
            products = _add_to_products(products, n)
    base_set, products = _try_extend(base_set, products, all_nums, deadline)
    if len(base_set) > best_size:
        best = list(base_set)
        best_size = len(best)
    
    # Phase 2: Randomized greedy restarts
    random.seed(42)
    while time.time() < deadline - 10:
        shuffled = all_nums[:]
        random.shuffle(shuffled)
        result, _ = _greedy_build(shuffled, deadline)
        result, _ = _try_extend(result, {1}, all_nums, deadline)
        if len(result) > best_size:
            best = result
            best_size = len(best)
    
    # Phase 3: Heuristic-ordered greedy
    sort_keys = [
        lambda x: x,
        lambda x: -x,
        lambda x: len(prime_factors[x]),
        lambda x: -len(prime_factors[x]),
        lambda x: max(prime_factors[x].values(), default=0),
        lambda x: sum(prime_factors[x].values()),
        lambda x: -sum(prime_factors[x].values()),
    ]
    for sort_key in sort_keys:
        if time.time() > deadline - 10:
            break
        ordered = sorted(all_nums, key=sort_key)
        result, _ = _greedy_build(ordered, deadline)
        result, _ = _try_extend(result, {1}, all_nums, deadline)
        if len(result) > best_size:
            best = result
            best_size = len(best)
    
    # Phase 4: Local search on best set
    if time.time() < deadline - 5:
        result = _local_search_swap(best, deadline)
        if len(result) > best_size:
            best = result
            best_size = len(best)
    
    # Phase 5: More randomized restarts with local search polish
    while time.time() < deadline - 2:
        shuffled = all_nums[:]
        random.shuffle(shuffled)
        result, _ = _greedy_build(shuffled, deadline)
        result, _ = _try_extend(result, {1}, all_nums, deadline)
        if len(result) > best_size:
            result = _local_search_swap(result, deadline)
            best = result
            best_size = len(best)
    
    return best

def construct():
    """
    Returns a list of distinct integers in the range 2..40 such that
    all 2^k subset products are distinct. Maximizes k.
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    best = _randomized_search(deadline)
    
    # Final verification
    if not _is_valid_set(best):
        # Fallback to known valid set
        fallback = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 4, 16, 9, 25]
        if _is_valid_set(fallback):
            best = fallback
        else:
            best = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    return [int(x) for x in best]
