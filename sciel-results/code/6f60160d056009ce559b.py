"""
Strategy: Greedy construction + local improvement for maximum sunflower-free
4-uniform family on [1,16].

Key improvements over baseline:
1. Smarter ordering: sort sets by "conflict potential" (number of pairs they
   would conflict with if added early)
2. Local improvement phase: try swapping one set out for one or two new sets
3. More diverse random restarts with different seeds
4. Optimized inner loop with precomputed intersection properties
"""

import time
import random
from itertools import combinations

def _check_conflict(family, x):
    """Check if adding x to family creates a sunflower triple."""
    m = len(family)
    for i in range(m):
        a = family[i]
        ax = a & x
        for j in range(i + 1, m):
            b = family[j]
            if (a & b) == ax and (b & x) == ax:
                return True
    return False

def _greedy_construct(all_sets, order, deadline):
    """Greedy: add sets in given order if no conflict."""
    family = []
    for idx in order:
        if time.time() > deadline:
            break
        x = all_sets[idx]
        if not _check_conflict(family, x):
            family.append(x)
    return family

def _local_improve(family, all_sets, all_set_indices, deadline, max_rounds=10):
    """
    Local improvement: try to swap out one set for another (or two) to increase
    family size. We try:
    1. Remove one set, try adding one new set that wasn't in family
    2. Remove one set, try adding two new sets
    """
    family_set = set(family)
    improved = True
    rounds = 0
    
    while improved and rounds < max_rounds and time.time() < deadline:
        improved = False
        rounds += 1
        current_size = len(family)
        
        # Try removing each set and adding one replacement
        for i in range(len(family)):
            if time.time() > deadline:
                break
            removed = family[i]
            family_without = family[:i] + family[i+1:]
            
            # Try adding each set not in family
            best_addition = None
            for idx in all_set_indices:
                if time.time() > deadline:
                    break
                candidate = all_sets[idx]
                if candidate in family_set:
                    continue
                if not _check_conflict(family_without, candidate):
                    # Check it's not the same as removed
                    if candidate != removed:
                        best_addition = candidate
                        break
            
            if best_addition is not None:
                # Perform the swap
                family = family_without + [best_addition]
                family_set.remove(removed)
                family_set.add(best_addition)
                improved = True
        
        # Try removing each set and adding two replacements
        for i in range(len(family)):
            if time.time() > deadline:
                break
            removed = family[i]
            family_without = family[:i] + family[i+1:]
            
            found_pair = False
            candidates = [all_sets[idx] for idx in all_set_indices 
                         if all_sets[idx] not in family_set and all_sets[idx] != removed]
            
            for ci in range(len(candidates)):
                if time.time() > deadline:
                    break
                c1 = candidates[ci]
                if not _check_conflict(family_without, c1):
                    family_temp = family_without + [c1]
                    for cj in range(ci + 1, len(candidates)):
                        if time.time() > deadline:
                            break
                        c2 = candidates[cj]
                        if not _check_conflict(family_temp, c2):
                            family = family_without + [c1, c2]
                            family_set.remove(removed)
                            family_set.add(c1)
                            family_set.add(c2)
                            found_pair = True
                            improved = True
                            break
                    if found_pair:
                        break
            
            if found_pair:
                continue
        
    return family

def _shuffle_order(n, seed):
    """Create a shuffled order with given seed."""
    order = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(order)
    return order

def construct():
    start_time = time.time()
    deadline = start_time + 78.0  # safety margin (90s limit, leave buffer)
    
    # Generate all 4-subsets of [1,16] as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_sets.append(mask)
    
    n_total = len(all_sets)  # 1820
    all_set_indices = list(range(n_total))
    
    # Precompute: for each set, count how many other sets it "conflicts with"
    # in the sense that they share a large intersection (more likely to form
    # sunflowers with a third set). Use this for ordering.
    # A set with fewer high-intersection partners is "safer" to add early.
    
    # Compute conflict score: for each set, count pairs (i,j) where 
    # all_sets[i] & all_sets[j] has size >= 2 and both intersect our set
    # in a way that could form sunflowers. Simplified: count sets with 
    # intersection size >= 2.
    conflict_scores = [0] * n_total
    for i in range(n_total):
        for j in range(i + 1, n_total):
            inter_size = bin(all_sets[i] & all_sets[j]).count('1')
            if inter_size >= 2:
                conflict_scores[i] += 1
                conflict_scores[j] += 1
    
    # Order 1: by increasing conflict score (safer sets first)
    order_by_conflict = sorted(range(n_total), key=lambda x: conflict_scores[x])
    
    # Order 2: reverse conflict score
    order_by_conflict_rev = order_by_conflict[::-1]
    
    # Order 3: natural
    order_natural = list(range(n_total))
    
    # Multiple random orders
    random_orders = []
    for seed in [42, 123, 456, 789, 1000, 2024]:
        random_orders.append(_shuffle_order(n_total, seed))
    
    all_orders = [order_by_conflict, order_by_conflict_rev, order_natural] + random_orders
    
    best_family = []
    best_size = 0
    
    for trial, order in enumerate(all_orders):
        if time.time() > deadline:
            break
        
        # Phase 1: Greedy construction
        family = _greedy_construct(all_sets, order, deadline)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)
        
        # Phase 2: Local improvement (if time permits)
        if time.time() < deadline - 5.0 and len(family) >= best_size:
            family = _local_improve(family, all_sets, all_set_indices, deadline, max_rounds=5)
            
            if len(family) > best_size:
                best_size = len(family)
                best_family = list(family)
    
    # Final local improvement on the best family if time permits
    if time.time() < deadline - 3.0 and len(best_family) > 0:
        best_family = _local_improve(best_family, all_sets, all_set_indices, deadline, max_rounds=3)
    
    # Convert bitmasks back to sorted lists
    result = []
    for mask in best_family:
        elements = []
        for i in range(16):
            if mask & (1 << i):
                elements.append(i + 1)
        result.append(sorted(elements))
    
    return result
