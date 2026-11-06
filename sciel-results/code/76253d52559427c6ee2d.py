"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _lcm(a, b):
    return a // gcd(a, b) * b

def _validate_set(selected_list):
    """Full validation that no bad triple exists."""
    n = len(selected_list)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a, b, c = selected_list[i], selected_list[j], selected_list[k]
                if _lcm(a,b) == _lcm(a,c) == _lcm(b,c):
                    return False
    return True

def _is_valid_addition(selected_list, x, lcm_table):
    """Check if adding x to selected_list creates a bad triple using precomputed LCM table."""
    for i in range(len(selected_list)):
        for j in range(i+1, len(selected_list)):
            a, b = selected_list[i], selected_list[j]
            aab = (a, b) if a <= b else (b, a)
            aax = (a, x) if a <= x else (x, a)
            bbx = (b, x) if b <= x else (x, b)
            if lcm_table[aab] == lcm_table[aax] == lcm_table[bbx]:
                return False
    return True

def _build_lcm_table():
    """Precompute LCM table for all pairs in [1,128]."""
    lcm_table = {}
    for a in range(1, 129):
        for b in range(a, 129):
            lcm_table[(a, b)] = _lcm(a, b)
    return lcm_table

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
            conflicts = 0
            for i in range(len(selected)):
                for j in range(i+1, len(selected)):
                    a, b = selected[i], selected[j]
                    aab = (a, b) if a <= b else (b, a)
                    aax = (a, x) if a <= x else (x, a)
                    bbx = (b, x) if b <= x else (x, b)
                    if lcm_table[aab] == lcm_table[aax] == lcm_table[bbx]:
                        conflicts += 1
                        if conflicts >= min_conflicts:
                            break
                if conflicts >= min_conflicts:
                    break
            
            if conflicts < min_conflicts:
                min_conflicts = conflicts
                best_x = x
        
        if best_x is not None:
            if min_conflicts == 0:
                selected.append(best_x)
                selected_set.add(best_x)
                candidates.remove(best_x)
            elif _is_valid_addition(selected, best_x, lcm_table):
                selected.append(best_x)
                selected_set.add(best_x)
                candidates.remove(best_x)
            else:
                candidates.remove(best_x)
        else:
            break
    
    return selected

def _local_search_improve(selected, lcm_table, deadline):
    """Try to improve the set by removing one and adding one."""
    selected = list(selected)
    current_size = len(selected)
    selected_set = set(selected)
    improvement_found = True
    
    while improvement_found and time.time() < deadline:
        improvement_found = False
        
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            removed = selected[i]
            temp_set = selected[:i] + selected[i+1:]
            temp_set_set = selected_set - {removed}
            
            for a in range(1, 129):
                if time.time() > deadline:
                    break
                if a in temp_set_set:
                    continue
                if not _is_valid_addition(temp_set, a, lcm_table):
                    continue
                
                new_set = temp_set + [a]
                if len(new_set) > current_size:
                    selected = new_set
                    current_size = len(selected)
                    selected_set = set(selected)
                    improvement_found = True
                    break
            if improvement_found:
                break
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    lcm_table = _build_lcm_table()
    all_elements = list(range(1, 129))
    best_set = []
    best_size = 0
    
    # Strategy 1: Start with odds, then add evens greedily
    odds = list(range(1, 129, 2))
    
    # Check if all odds are valid
    odds_valid = True
    n_odds = len(odds)
    for i in range(n_odds):
        if time.time() > deadline:
            odds_valid = False
            break
        for j in range(i+1, n_odds):
            for k in range(j+1, n_odds):
                a, b, c = odds[i], odds[j], odds[k]
                aab = (a, b) if a <= b else (b, a)
                aac = (a, c) if a <= c else (c, a)
                bbc = (b, c) if b <= c else (c, b)
                if lcm_table[aab] == lcm_table[aac] == lcm_table[bbc]:
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
    
    evens = list(range(2, 129, 2))
    for x in evens:
        if time.time() > deadline:
            break
        if x not in candidate:
            if _is_valid_addition(candidate, x, lcm_table):
                candidate.append(x)
    
    if len(candidate) > best_size:
        best_set = candidate
        best_size = len(candidate)
    
    if time.time() < deadline:
        improved = _local_search_improve(best_set, lcm_table, deadline)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
    
    # Strategy 2: Random restarts
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        num_restarts = min(15, int(remaining_time / 4))
        for _ in range(num_restarts):
            if time.time() > deadline:
                break
            random.seed(random.randint(0, 2**31))
            shuffled = all_elements[:]
            random.shuffle(shuffled)
            candidate = _greedy_min_conflict(shuffled, [], lcm_table, deadline)
            if len(candidate) > best_size:
                best_set = candidate
                best_size = len(candidate)
                if time.time() < deadline:
                    improved = _local_search_improve(best_set, lcm_table, deadline)
                    if len(improved) > best_size:
                        best_set = improved
                        best_size = len(best_set)
    
    # Final validation
    if not _validate_set(best_set):
        best_set = _greedy_min_conflict(all_elements, [], lcm_table, deadline)
    
    if not best_set:
        best_set = [1]
    
    return best_set
# EVOLVE-BLOCK-END
