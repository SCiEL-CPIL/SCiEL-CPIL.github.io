"""Rewritten program for sunflower-free 4-uniform family on [1,16]."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def _precompute_pair_intersections(all_sets):
    """Precompute pairwise intersections for all sets."""
    n = len(all_sets)
    pair_int = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            v = all_sets[i] & all_sets[j]
            pair_int[i][j] = v
            pair_int[j][i] = v
    return pair_int

def _check_sunflower_fast(new_idx, family_list, all_sets, pair_int):
    """
    Check if adding new_idx to family creates a sunflower.
    Uses grouping by intersection with new set.
    family_list must be a list (not set) for consistent iteration.
    """
    s = all_sets[new_idx]
    # Group family members by their intersection with s
    groups = defaultdict(list)
    for fi in family_list:
        inter = all_sets[fi] & s
        groups[inter].append(fi)
    
    # For each group with >= 2 members, check if any pair has intersection == group key
    for inter, members in groups.items():
        if len(members) < 2:
            continue
        # Check all pairs in this group
        for a_pos in range(len(members)):
            for b_pos in range(a_pos+1, len(members)):
                a = members[a_pos]
                b = members[b_pos]
                if pair_int[a][b] == inter:
                    return True
    return False

def _greedy_construct(all_sets, pair_int, order, deadline):
    """Run greedy construction with given order."""
    family = []
    family_set = set()
    
    for idx in order:
        if time.time() > deadline:
            break
        if idx in family_set:
            continue
        if not _check_sunflower_fast(idx, family, all_sets, pair_int):
            family.append(idx)
            family_set.add(idx)
    
    return family

def _local_search_swap(family, all_sets, pair_int, deadline, max_iterations=500):
    """
    Local search: try to improve family by swap moves (remove one, add one)
    or by adding if possible. Multiple swaps per iteration allowed.
    """
    family_set = set(family)
    n_total = len(all_sets)
    improved = True
    iter_count = 0
    
    while improved and time.time() < deadline and iter_count < max_iterations:
        improved = False
        iter_count += 1
        
        # First, try adding any set that doesn't conflict
        for idx in range(n_total):
            if time.time() > deadline:
                break
            if idx in family_set:
                continue
            if not _check_sunflower_fast(idx, family, all_sets, pair_int):
                family.append(idx)
                family_set.add(idx)
                improved = True
                break
        
        if improved:
            continue
        
        # Try multiple swaps in this iteration
        random.shuffle(family)
        for remove_idx in family:
            if time.time() > deadline:
                break
            
            temp_family = [x for x in family if x != remove_idx]
            temp_set = set(temp_family)
            
            # Find best candidate to add
            best_add = -1
            for add_idx in range(n_total):
                if add_idx in temp_set:
                    continue
                if not _check_sunflower_fast(add_idx, temp_family, all_sets, pair_int):
                    best_add = add_idx
                    break
            
            if best_add != -1:
                family = temp_family
                family.append(best_add)
                family_set = set(family)
                improved = True
                break
    
    return family

def _local_search_aggressive(family, all_sets, pair_int, deadline, max_iterations=300):
    """
    More aggressive local search: try removing 1-2 elements and adding 1-2.
    """
    family_set = set(family)
    n_total = len(all_sets)
    improved = True
    iter_count = 0
    
    while improved and time.time() < deadline and iter_count < max_iterations:
        improved = False
        iter_count += 1
        
        # Try single add
        for idx in range(n_total):
            if time.time() > deadline:
                break
            if idx in family_set:
                continue
            if not _check_sunflower_fast(idx, family, all_sets, pair_int):
                family.append(idx)
                family_set.add(idx)
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-remove-1-add
        random.shuffle(family)
        for remove_idx in family:
            if time.time() > deadline:
                break
            
            temp_family = [x for x in family if x != remove_idx]
            temp_set = set(temp_family)
            
            best_add = -1
            for add_idx in range(n_total):
                if add_idx in temp_set:
                    continue
                if not _check_sunflower_fast(add_idx, temp_family, all_sets, pair_int):
                    best_add = add_idx
                    break
            
            if best_add != -1:
                family = temp_family
                family.append(best_add)
                family_set = set(family)
                improved = True
                break
        
        if improved:
            continue
        
        # Try 2-remove-2-add (heavier but can escape local optima)
        if len(family) >= 4:
            random.shuffle(family)
            # Try removing first two random elements
            r1 = family[0]
            r2 = family[1]
            temp_family = [x for x in family if x != r1 and x != r2]
            temp_set = set(temp_family)
            
            # Try adding two elements
            added = []
            for add_idx in range(n_total):
                if time.time() > deadline:
                    break
                if add_idx in temp_set or add_idx in added:
                    continue
                trial_family = temp_family + added
                if not _check_sunflower_fast(add_idx, trial_family, all_sets, pair_int):
                    added.append(add_idx)
                    if len(added) == 2:
                        break
            
            if len(added) == 2:
                # Verify the full family is valid
                new_family = temp_family + added
                # Quick check: ensure no sunflowers in new_family
                valid = True
                for i in range(len(new_family)):
                    for j in range(i+1, len(new_family)):
                        for k in range(j+1, len(new_family)):
                            a, b, c = new_family[i], new_family[j], new_family[k]
                            if pair_int[a][b] == pair_int[a][c] == pair_int[b][c]:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    family = new_family
                    family_set = set(family)
                    improved = True
    
    return family

def construct():
    start_time = time.time()
    deadline = start_time + 86  # 4s safety margin before 90s kill
    
    # Represent sets as bitmasks for speed
    all_sets = []
    for c in combinations(range(1, 17), 4):
        mask = 0
        for x in c:
            mask |= (1 << (x - 1))
        all_sets.append(mask)
    
    n_total = len(all_sets)  # C(16,4) = 1820
    
    # Precompute pair intersections
    pair_int = _precompute_pair_intersections(all_sets)
    
    best_family = []
    best_size = 0
    
    # Multiple greedy passes with different orderings
    n_attempts = 30
    for attempt in range(n_attempts):
        if time.time() > deadline - 20:
            break
        
        if attempt == 0:
            order = list(range(n_total))
        elif attempt == 1:
            order = list(range(n_total - 1, -1, -1))
        elif attempt == 2:
            # Sort by popcount of complement (all same for 4-sets, so use index)
            order = sorted(range(n_total), key=lambda i: all_sets[i])
        elif attempt == 3:
            # Sort by sum of elements
            order = sorted(range(n_total), key=lambda i: sum(bin(all_sets[i]).count('1') * (i % 17)))
        else:
            order = list(range(n_total))
            random.shuffle(order)
        
        family = _greedy_construct(all_sets, pair_int, order, deadline - 20)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
        
        if best_size >= 250:
            break
    
    # Local search on the best family - basic
    if best_size > 0 and time.time() < deadline - 10:
        improved_family = _local_search_swap(best_family, all_sets, pair_int, deadline - 8, max_iterations=500)
        if len(improved_family) > best_size:
            best_size = len(improved_family)
            best_family = improved_family[:]
    
    # Aggressive local search
    if best_size > 0 and time.time() < deadline - 5:
        improved_family = _local_search_aggressive(best_family, all_sets, pair_int, deadline - 3, max_iterations=300)
        if len(improved_family) > best_size:
            best_size = len(improved_family)
            best_family = improved_family[:]
    
    # Convert back to sorted lists
    result = []
    for idx in best_family:
        mask = all_sets[idx]
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(elems)
    
    return result
# EVOLVE-BLOCK-END
