import time
import random
from math import isqrt

def _can_add(products, cand):
    """
    Check if adding cand to a set with given product set keeps all products distinct.
    products is a set of all subset products of the current set.
    cand is the new candidate integer.
    """
    for p in products:
        if p * cand in products:
            return False
    return True

def _add_element(products, cand):
    """Add element to products set, return new products set."""
    new_prods = {p * cand for p in products}
    return products | new_prods

def _prime_factor_count(n):
    """Count total prime factors with multiplicity."""
    count = 0
    d = 2
    while d * d <= n:
        while n % d == 0:
            count += 1
            n //= d
        d += 1
    if n > 1:
        count += 1
    return count

def _distinct_prime_count(n):
    """Count distinct prime factors."""
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

def _is_prime_power(n):
    """Check if n is a power of a prime."""
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            m = n
            while m % d == 0:
                m //= d
            return m == 1
        d += 1
    return True  # n is prime

def _local_search(current_set, products, candidates, deadline):
    """
    Try to improve current_set via swap operations (1-for-1 and 2-for-1/2).
    """
    k = len(current_set)
    if k >= 18:
        return current_set, products
    
    current_set = list(current_set)
    products = set(products)
    
    # Try 1-for-1 swaps
    for _ in range(3):  # Multiple passes
        if time.time() > deadline:
            break
        improved = False
        shuffled_indices = list(range(len(current_set)))
        random.shuffle(shuffled_indices)
        
        for i in shuffled_indices:
            if time.time() > deadline:
                break
            removed = current_set[i]
            remaining = [x for j, x in enumerate(current_set) if j != i]
            
            # Compute products of remaining
            rem_products = {1}
            for elem in remaining:
                rem_products = _add_element(rem_products, elem)
            
            # Try adding each candidate
            shuffled_cands = [c for c in candidates if c not in current_set]
            random.shuffle(shuffled_cands)
            
            for cand in shuffled_cands:
                if _can_add(rem_products, cand):
                    new_set = remaining + [cand]
                    new_products = _add_element(rem_products, cand)
                    return new_set, new_products
        if not improved:
            break
    
    # Try 2-for-1 swaps: remove two elements, add one
    if time.time() > deadline:
        return current_set, products
    
    for _ in range(2):  # Fewer passes for expensive 2-for-1
        if time.time() > deadline:
            break
        n = len(current_set)
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                removed1 = current_set[i]
                removed2 = current_set[j]
                remaining = [x for k, x in enumerate(current_set) if k != i and k != j]
                
                rem_products = {1}
                for elem in remaining:
                    rem_products = _add_element(rem_products, elem)
                
                # Try adding each candidate
                shuffled_cands = [c for c in candidates if c not in current_set]
                random.shuffle(shuffled_cands)
                
                for cand in shuffled_cands:
                    if _can_add(rem_products, cand):
                        new_set = remaining + [cand]
                        new_products = _add_element(rem_products, cand)
                        return new_set, new_products
    
    return current_set, products

def _greedy_construct(candidates, order_key, deadline):
    """Greedy construction with a given ordering."""
    ordered = sorted(candidates, key=order_key)
    current_set = []
    products = {1}
    
    for cand in ordered:
        if time.time() > deadline:
            break
        if _can_add(products, cand):
            current_set.append(cand)
            products = _add_element(products, cand)
    
    return current_set, products

def construct():
    """
    Find the largest subset of {2, ..., 40} such that all subset products are distinct.
    """
    deadline = time.time() + 55  # Safety margin
    best_set = []
    best_len = 0
    
    all_candidates = list(range(2, 41))
    
    # Precompute prime power candidates (likely to be independent)
    prime_powers = [x for x in all_candidates if _is_prime_power(x)]
    
    # Strategy 1: Random restarts with greedy + local search
    for restart in range(5000):
        if time.time() > deadline:
            break
        if best_len >= 18:
            break
        
        # Choose candidate pool: mix of prime powers and other numbers
        if restart % 3 == 0:
            # Focus on prime powers
            pool = prime_powers[:]
        elif restart % 3 == 1:
            # All candidates
            pool = all_candidates[:]
        else:
            # Prime powers + numbers with few distinct prime factors
            pool = [x for x in all_candidates if _distinct_prime_count(x) <= 2]
        
        random.shuffle(pool)
        
        current_set = []
        products = {1}
        
        for cand in pool:
            if time.time() > deadline:
                break
            if _can_add(products, cand):
                current_set.append(cand)
                products = _add_element(products, cand)
        
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
            if best_len >= 18:
                break
        
        # Local search to improve
        if len(current_set) >= best_len and len(current_set) < 18:
            improved_set, improved_products = _local_search(
                current_set, products, all_candidates, deadline
            )
            if len(improved_set) > best_len:
                best_len = len(improved_set)
                best_set = improved_set[:]
                if best_len >= 18:
                    break
    
    # Strategy 2: Deterministic orderings
    if best_len < 18 and time.time() < deadline:
        orderings = [
            lambda x: (_distinct_prime_count(x), _prime_factor_count(x), x),
            lambda x: (_distinct_prime_count(x), -x),
            lambda x: (_prime_factor_count(x), x),
            lambda x: x,
            lambda x: -x,
            lambda x: (not _is_prime_power(x), _distinct_prime_count(x), x),
        ]
        
        for order_key in orderings:
            if time.time() > deadline or best_len >= 18:
                break
            
            current_set, products = _greedy_construct(all_candidates, order_key, deadline)
            
            if len(current_set) > best_len:
                best_len = len(current_set)
                best_set = current_set[:]
                if best_len >= 18:
                    break
            
            # Local search
            if len(current_set) >= best_len and len(current_set) < 18:
                improved_set, improved_products = _local_search(
                    current_set, products, all_candidates, deadline
                )
                if len(improved_set) > best_len:
                    best_len = len(improved_set)
                    best_set = improved_set[:]
                    if best_len >= 18:
                        break
    
    # Strategy 3: If still not 18, try focused search on prime powers only
    if best_len < 18 and time.time() < deadline:
        for _ in range(2000):
            if time.time() > deadline:
                break
            if best_len >= 18:
                break
            
            shuffled_pp = prime_powers[:]
            random.shuffle(shuffled_pp)
            
            current_set = []
            products = {1}
            
            for cand in shuffled_pp:
                if time.time() > deadline:
                    break
                if _can_add(products, cand):
                    current_set.append(cand)
                    products = _add_element(products, cand)
            
            if len(current_set) > best_len:
                best_len = len(current_set)
                best_set = current_set[:]
                if best_len >= 18:
                    break
    
    # Final validation
    if best_set:
        prods = {1}
        valid = True
        for i in best_set:
            new_prods = {p * i for p in prods}
            if new_prods & prods:
                valid = False
                break
            prods |= new_prods
        
        if not valid:
            # Fallback to known valid set
            best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    if not best_set:
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Ensure all elements are Python int
    return [int(x) for x in best_set]
