"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _lcm(a, b):
    return a // gcd(a, b) * b

def _build_lcm_table():
    """Precompute LCM table for all pairs in [1,128]."""
    lcm_table = {}
    for a in range(1, 129):
        for b in range(a, 129):
            lcm_table[(a, b)] = _lcm(a, b)
    return lcm_table

def _get_lcm(a, b, lcm_table):
    """Get LCM from precomputed table."""
    if a <= b:
        return lcm_table[(a, b)]
    else:
        return lcm_table[(b, a)]

def _validate_set(selected_list, lcm_table):
    """Full validation that no bad triple exists."""
    n = len(selected_list)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a, b, c = selected_list[i], selected_list[j], selected_list[k]
                if _get_lcm(a,b,lcm_table) == _get_lcm(a,c,lcm_table) == _get_lcm(b,c,lcm_table):
                    return False
    return True

def _is_valid_addition(selected_list, x, lcm_table):
    """Check if adding x to selected_list creates a bad triple using precomputed LCM table."""
    n = len(selected_list)
    for i in range(n):
        a = selected_list[i]
        lax = _get_lcm(a, x, lcm_table)
        for j in range(i+1, n):
            b = selected_list[j]
            if _get_lcm(a, b, lcm_table) == lax == _get_lcm(b, x, lcm_table):
                return False
    return True

def _conflict_count(selected_list, x, lcm_table):
    """Count how many pairs in selected_list would form a bad triple with x."""
    n = len(selected_list)
    conflicts = 0
    for i in range(n):
        a = selected_list[i]
        lax = _get_lcm(a, x, lcm_table)
        for j in range(i+1, n):
            b = selected_list[j]
            if _get_lcm(a, b, lcm_table) == lax == _get_lcm(b, x, lcm_table):
                conflicts += 1
    return conflicts

def _greedy_min_conflict(elements, start_set, lcm_table, deadline):
    """Greedy construction choosing element with minimum conflict count at each step."""
    selected = list(start_set) if start_set else []
    selected_set = set(selected)
    candidates = [x for x in elements if x not in selected_set]
    
    while candidates:
        if time.time() > deadline:
            break
        
        best_x = None
        min_conflicts = float('inf')
        
        for x in candidates:
            if time.time() > deadline:
                break
            conflicts = _conflict_count(selected, x, lcm_table)
            if conflicts < min_conflicts:
                min_conflicts = conflicts
                best_x = x
                if conflicts == 0:
                    break
        
        if best_x is not None and min_conflicts == 0:
            selected.append(best_x)
            selected_set.add(best_x)
            candidates.remove(best_x)
        elif best_x is not None and _is_valid_addition(selected, best_x, lcm_table):
            selected.append(best_x)
            selected_set.add(best_x)
            candidates.remove(best_x)
        else:
            break
    
    return selected

def _local_search_improve(selected, lcm_table, deadline):
    """Try to improve the set by removing one and adding one or two."""
    selected = list(selected)
    current_size = len(selected)
    selected_set = set(selected)
    all_elements = list(range(1, 129))
    improvement_found = True
    
    while improvement_found and time.time() < deadline:
        improvement_found = False
        
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            removed = selected[i]
            temp_set = selected[:i] + selected[i+1:]
            temp_set_set = selected_set - {removed}
            
            # Try adding one element
            best_add = None
            for a in all_elements:
                if time.time() > deadline:
                    break
                if a in temp_set_set:
                    continue
                if _is_valid_addition(temp_set, a, lcm_table):
                    best_add = a
                    break
            
            if best_add is not None:
                selected = temp_set + [best_add]
                selected_set = set(selected)
                current_size = len(selected)
                improvement_found = True
                break
        
        if not improvement_found and time.time() < deadline:
            # Try removing one and adding two
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                removed = selected[i]
                temp_set = selected[:i] + selected[i+1:]
                temp_set_set = selected_set - {removed}
                
                added = []
                for a in all_elements:
                    if time.time() > deadline:
                        break
                    if a in temp_set_set or a in added:
                        continue
                    if _is_valid_addition(temp_set + added, a, lcm_table):
                        added.append(a)
                        if len(added) == 2:
                            break
                
                if len(added) >= 2:
                    selected = temp_set + added
                    selected_set = set(selected)
                    current_size = len(selected)
                    improvement_found = True
                    break
    
    return selected

def _omega(x):
    """Count total prime factors with multiplicity."""
    c = 0
    n = x
    d = 2
    while d * d <= n:
        while n % d == 0:
            c += 1
            n //= d
        d += 1
    if n > 1:
        c += 1
    return c

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    lcm_table = _build_lcm_table()
    all_elements = list(range(1, 129))
    best_set = []
    best_size = 0
    
    # Strategy 1: Start with all odds (64 elements), verify, then add evens
    odds = list(range(1, 129, 2))
    
    # Check if all odds are valid (should be, but verify)
    odds_valid = True
    n_odds = len(odds)
    for i in range(n_odds):
        if time.time() > deadline:
            odds_valid = False
            break
        for j in range(i+1, n_odds):
            for k in range(j+1, n_odds):
                a, b, c = odds[i], odds[j], odds[k]
                if _get_lcm(a,b,lcm_table) == _get_lcm(a,c,lcm_table) == _get_lcm(b,c,lcm_table):
                    odds_valid = False
                    break
            if not odds_valid:
                break
        if not odds_valid:
            break
    
    if odds_valid:
        candidate = list(odds)
    else:
        candidate = _greedy_min_conflict(odds, [], lcm_table, deadline)
    
    # Add evens greedily with minimum conflict
    evens = list(range(2, 129, 2))
    for x in evens:
        if time.time() > deadline:
            break
        if x not in candidate:
            if _is_valid_addition(candidate, x, lcm_table):
                candidate.append(x)
    
    if len(candidate) > best_size:
        best_set = list(candidate)
        best_size = len(candidate)
    
    # Local search improvement
    if time.time() < deadline:
        improved = _local_search_improve(best_set, lcm_table, deadline)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
    
    # Strategy 2: Try different orderings
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Ordering by omega (prime factor count) - fewer factors first
        sorted_omega = sorted(all_elements, key=lambda x: _omega(x))
        if time.time() < deadline:
            candidate2 = _greedy_min_conflict(sorted_omega, [], lcm_table, deadline)
            if len(candidate2) > best_size:
                best_set = list(candidate2)
                best_size = len(candidate2)
                if time.time() < deadline:
                    improved = _local_search_improve(best_set, lcm_table, deadline)
                    if len(improved) > best_size:
                        best_set = improved
                        best_size = len(best_set)
        
        # Random restarts with structured seeds
        num_restarts = min(10, int(remaining_time / 6))
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            random.seed(12345 + r * 7919)
            
            # Mix of strategies: sometimes start with odds, sometimes random
            if r % 3 == 0:
                # Start with odds, shuffle evens
                shuffled_odds = odds[:]
                random.shuffle(shuffled_odds)
                shuffled_evens = evens[:]
                random.shuffle(shuffled_evens)
                order = shuffled_odds + shuffled_evens
            elif r % 3 == 1:
                # Pure random
                order = all_elements[:]
                random.shuffle(order)
            else:
                # By value range: small, medium, large
                small = list(range(1, 43))
                medium = list(range(43, 86))
                large = list(range(86, 129))
                random.shuffle(small)
                random.shuffle(medium)
                random.shuffle(large)
                order = small + medium + large
            
            candidate = _greedy_min_conflict(order, [], lcm_table, deadline)
            if len(candidate) > best_size:
                best_set = list(candidate)
                best_size = len(candidate)
                if time.time() < deadline:
                    improved = _local_search_improve(best_set, lcm_table, deadline)
                    if len(improved) > best_size:
                        best_set = improved
                        best_size = len(best_set)
    
    # Final validation
    if not _validate_set(best_set, lcm_table):
        # Fallback: greedy from scratch
        best_set = _greedy_min_conflict(all_elements, [], lcm_table, deadline)
    
    if not best_set:
        best_set = [1]
    
    return best_set
# EVOLVE-BLOCK-END
