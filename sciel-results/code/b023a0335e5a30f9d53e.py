"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import gcd, lcm as math_lcm

def _lcm(a, b):
    return a // gcd(a, b) * b

def _has_bad_triple(selected_set, selected_list):
    """Check if selected_list contains any bad triple."""
    n = len(selected_list)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a, b, c = selected_list[i], selected_list[j], selected_list[k]
                if _lcm(a,b) == _lcm(a,c) == _lcm(b,c):
                    return True
    return False

def _is_valid_addition(selected_list, x, lcm_cache):
    """Check if adding x to selected_list creates a bad triple."""
    for i in range(len(selected_list)):
        for j in range(i+1, len(selected_list)):
            a, b = selected_list[i], selected_list[j]
            lab = _lcm(a, b)
            lax = _lcm(a, x)
            lbx = _lcm(b, x)
            if lab == lax == lbx:
                return False
    return True

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

def _greedy_with_order(elements, start_set=None, deadline=None):
    """Greedy construction with given element order."""
    selected = list(start_set) if start_set else []
    selected_set = set(selected)
    for x in elements:
        if x in selected_set:
            continue
        if deadline and time.time() > deadline:
            break
        if _is_valid_addition(selected, x, None):
            selected.append(x)
            selected_set.add(x)
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    # Precompute all LCMs for pairs in [1,128]
    lcm_table = {}
    for a in range(1, 129):
        for b in range(a, 129):
            lcm_table[(a,b)] = _lcm(a, b)
    
    def get_lcm(a, b):
        if a > b:
            a, b = b, a
        return lcm_table[(a,b)]
    
    # Strategy 1: Start with all odd numbers (64 elements)
    odds = list(range(1, 129, 2))
    
    # Check if odds alone is valid
    odds_valid = True
    for i in range(len(odds)):
        for j in range(i+1, len(odds)):
            for k in range(j+1, len(odds)):
                a, b, c = odds[i], odds[j], odds[k]
                if get_lcm(a,b) == get_lcm(a,c) == get_lcm(b,c):
                    odds_valid = False
                    break
            if not odds_valid:
                break
        if not odds_valid:
            break
    
    best_set = list(odds) if odds_valid else []
    
    if not odds_valid:
        # Build odds greedily
        best_set = _greedy_with_order(odds, deadline=deadline)
    
    # Try to add even numbers one by one
    evens = list(range(2, 129, 2))
    for x in evens:
        if time.time() > deadline:
            break
        if x not in best_set:
            if _is_valid_addition(best_set, x, None):
                best_set.append(x)
    
    # Local search: try swapping elements to increase size
    current_size = len(best_set)
    improvement_found = True
    while improvement_found and time.time() < deadline:
        improvement_found = False
        best_set_set = set(best_set)
        
        # Try removing one element and adding two
        for i in range(len(best_set)):
            if time.time() > deadline:
                break
            removed = best_set[i]
            temp_set = best_set[:i] + best_set[i+1:]
            temp_set_set = best_set_set - {removed}
            
            # Try adding two elements
            for a in range(1, 129):
                if time.time() > deadline:
                    break
                if a in temp_set_set:
                    continue
                if not _is_valid_addition(temp_set, a, None):
                    continue
                temp_set_with_a = temp_set + [a]
                temp_set_set_with_a = temp_set_set | {a}
                
                for b in range(a+1, 129):
                    if time.time() > deadline:
                        break
                    if b in temp_set_set_with_a:
                        continue
                    if not _is_valid_addition(temp_set_with_a, b, None):
                        continue
                    
                    new_set = temp_set_with_a + [b]
                    if len(new_set) > current_size:
                        best_set = new_set
                        current_size = len(new_set)
                        best_set_set = set(best_set)
                        improvement_found = True
                        break
                if improvement_found:
                    break
            if improvement_found:
                break
    
    # Final validation
    if not _validate_set(best_set):
        # Fallback to greedy
        best_set = _greedy_with_order(range(1, 129), deadline=deadline)
    
    if not best_set:
        best_set = [1]
    
    return best_set
# EVOLVE-BLOCK-END
