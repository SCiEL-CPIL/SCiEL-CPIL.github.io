"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random

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
    # Check if adding x creates any collision: x*p1 == p2 for p1,p2 in products
    # i.e., for each p in products, check if p*x is already in products
    prods = products
    for p in prods:
        if p * x in prods:
            return False
    return True

def _add_number(products, x):
    new_prods = set(products)
    for p in products:
        new_prods.add(p * x)
    return new_prods

def _compute_products(selected):
    products = {1}
    for x in selected:
        products = _add_number(products, x)
    return products

def _num_distinct_primes(n):
    count = 0
    d = 2
    m = n
    while d * d <= m:
        if m % d == 0:
            count += 1
            while m % d == 0:
                m //= d
            d += 1
        else:
            d += 1
    if m > 1:
        count += 1
    return count

def _greedy_construct(order):
    products = {1}
    selected = []
    for a in order:
        if time.time() > _DEADLINE:
            break
        if _can_add(products, a):
            products = _add_number(products, a)
            selected.append(a)
    return selected, products

def _hill_climb(selected, all_candidates, max_iters=100):
    best_selected = list(selected)
    best_products = _compute_products(best_selected)
    best_size = len(best_selected)
    
    for _ in range(max_iters):
        if time.time() > _DEADLINE:
            break
        improved = False
        remaining = [a for a in all_candidates if a not in best_selected]
        
        # Try single insertions
        for a in remaining:
            if time.time() > _DEADLINE:
                break
            if _can_add(best_products, a):
                new_sel = best_selected + [a]
                new_prod = _add_number(best_products, a)
                best_selected = new_sel
                best_products = new_prod
                best_size = len(best_selected)
                remaining.remove(a)
                improved = True
        
        # Try swap: remove one, add one or more
        if not improved:
            for i in range(len(best_selected)):
                if time.time() > _DEADLINE:
                    break
                rem = [x for j, x in enumerate(best_selected) if j != i]
                rem_prod = _compute_products(rem)
                for a in remaining:
                    if _can_add(rem_prod, a):
                        new_sel = rem + [a]
                        new_prod = _add_number(rem_prod, a)
                        # Try to add more after the swap
                        for b in remaining:
                            if b == a or b in new_sel:
                                continue
                            if _can_add(new_prod, b):
                                new_sel.append(b)
                                new_prod = _add_number(new_prod, b)
                                break
                        if len(new_sel) > best_size:
                            best_size = len(new_sel)
                            best_selected = new_sel
                            best_products = new_prod
                            improved = True
                            break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_selected, best_products

def construct():
    all_candidates = list(range(2, 41))
    
    best_size = 0
    best_result = []
    best_products = {1}
    
    # Multiple heuristic orderings
    orderings = [
        all_candidates,
        list(reversed(all_candidates)),
        [a for a in all_candidates if _is_prime(a)] + [a for a in all_candidates if not _is_prime(a)],
        sorted(all_candidates, key=_num_distinct_primes),
        sorted(all_candidates, key=_num_distinct_primes, reverse=True),
        sorted(all_candidates, key=lambda x: (x, _num_distinct_primes(x))),
        sorted(all_candidates, key=lambda x: -x),
        [2,3,5,7,11,13,17,19,23,29,31,37,4,6,8,9,10,12,14,15,16,18,20,21,22,24,25,26,27,28,30,32,33,34,35,36,38,39,40],
    ]
    
    for order in orderings:
        if time.time() > _DEADLINE:
            break
        selected, products = _greedy_construct(order)
        if len(selected) > best_size:
            best_size = len(selected)
            best_result = selected
            best_products = products
        if best_size >= 18:
            return list(best_result)
    
    # Hill climb from best greedy result
    if best_size < 18:
        best_result, best_products = _hill_climb(best_result, all_candidates, max_iters=100)
        best_size = len(best_result)
        if best_size >= 18:
            return list(best_result)
    
    # Random restarts with hill climbing
    random.seed(42)
    for attempt in range(500):
        if time.time() > _DEADLINE:
            break
        shuffled = all_candidates[:]
        random.shuffle(shuffled)
        selected, products = _greedy_construct(shuffled)
        if len(selected) > best_size:
            best_size = len(selected)
            best_result = selected
            best_products = products
            if best_size >= 18:
                return list(best_result)
        sel, prod = _hill_climb(selected, all_candidates, max_iters=30)
        if len(sel) > best_size:
            best_size = len(sel)
            best_result = sel
            best_products = prod
            if best_size >= 18:
                return list(best_result)
    
    # Final polish: try to add any remaining candidates
    remaining = [a for a in all_candidates if a not in best_result]
    for a in remaining:
        if time.time() > _DEADLINE:
            break
        if _can_add(best_products, a):
            best_result.append(a)
            best_products = _add_number(best_products, a)
            best_size = len(best_result)
            if best_size >= 18:
                break
    
    return list(best_result)
# EVOLVE-BLOCK-END
