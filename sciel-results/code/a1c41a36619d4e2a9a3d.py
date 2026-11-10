"""Combined program: integer LCM subset-sum validation + B&B + local search polish."""
import time
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START
def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    result = lst[0]
    for v in lst[1:]:
        result = _lcm(result, v)
        if result > 10**18:
            break
    return result

def _can_form_sum(target_int, values):
    """Check if target_int can be formed as a subset sum of values using meet-in-the-middle."""
    n = len(values)
    if n == 0:
        return target_int == 0
    if n == 1:
        return target_int == values[0]
    if n <= 2:
        return target_int in values or target_int == values[0] + values[1]
    
    mid = n // 2
    left = values[:mid]
    right = values[mid:]
    
    # Generate all subset sums for left
    left_sums = {0}
    for v in left:
        new_sums = set()
        for s in left_sums:
            ns = s + v
            if ns <= target_int:
                new_sums.add(ns)
        left_sums |= new_sums
    
    # Check if any right subset sum completes target
    right_sums = {0}
    for v in right:
        new_sums = set()
        for s in right_sums:
            ns = s + v
            if ns <= target_int:
                new_sums.add(ns)
        right_sums |= new_sums
    
    for rs in right_sums:
        if (target_int - rs) in left_sums:
            return True
    return False

def _is_valid_addition(x, selected):
    """Check if adding x to selected creates a violation."""
    if not selected:
        return True
    
    # Check 1: 1/x is not subset sum of reciprocals of selected
    all_vals = [x] + list(selected)
    lcm_val = _lcm_list(all_vals)
    target_int = lcm_val // x
    candidate_vals = [lcm_val // y for y in selected]
    if _can_form_sum(target_int, candidate_vals):
        return False
    
    # Check 2: For each y in selected, adding x doesn't create 1/y = sum of subset of (selected\{y}) ∪ {x}
    # New subsets must include x, so: 1/y = 1/x + sum(1/z for z in subset of (selected\{y,x}))
    # => 1/y - 1/x = sum(1/z for z in subset of (selected\{y,x}))
    for y in selected:
        if y == x:
            continue
        remaining = [z for z in selected if z != y and z != x]
        if not remaining:
            continue
        
        all_for_check = [y, x] + remaining
        lcm_check = _lcm_list(all_for_check)
        target = lcm_check // y - lcm_check // x
        if target <= 0:
            continue
        
        candidate_vals = [lcm_check // z for z in remaining]
        if _can_form_sum(target, candidate_vals):
            return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    # Elements 1 to 24, process from largest to smallest
    elements = list(range(24, 0, -1))
    n = len(elements)
    
    best_set = []
    best_size = 0
    
    # Phase 1: Greedy from large to small
    greedy_set = []
    for x in elements:
        if time.time() > deadline:
            break
        if _is_valid_addition(x, greedy_set):
            greedy_set.append(x)
    
    best_set = list(greedy_set)
    best_size = len(greedy_set)
    
    # Phase 2: Branch and bound
    def branch_and_bound(idx, current_set, current_size):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        remaining_count = n - idx
        if current_size + remaining_count <= best_size:
            return
        
        if idx == n:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
        
        x = elements[idx]
        
        # Try including x first (prefer larger elements)
        if _is_valid_addition(x, current_set):
            current_set.append(x)
            branch_and_bound(idx + 1, current_set, current_size + 1)
            current_set.pop()
        
        # Try excluding x
        branch_and_bound(idx + 1, current_set, current_size)
    
    branch_and_bound(0, [], 0)
    
    # Phase 3: Local search polish - try to improve by swapping elements
    if time.time() < deadline - 5:
        current_set = set(best_set)
        excluded = set(range(1, 25)) - current_set
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for add_elem in list(excluded):
                if time.time() > deadline - 5:
                    break
                for remove_elem in list(current_set):
                    new_set = (current_set - {remove_elem}) | {add_elem}
                    if len(new_set) == len(current_set):
                        # Check if new_set is valid
                        # Quick check: just verify add_elem is valid with new_set minus add_elem
                        others = [x for x in new_set if x != add_elem]
                        if _is_valid_addition(add_elem, others):
                            # Need to check all elements, not just add_elem
                            valid = True
                            for elem in new_set:
                                others_for_elem = [x for x in new_set if x != elem]
                                all_vals = [elem] + others_for_elem
                                lcm_val = _lcm_list(all_vals)
                                target_int = lcm_val // elem
                                candidate_vals = [lcm_val // y for y in others_for_elem]
                                if _can_form_sum(target_int, candidate_vals):
                                    valid = False
                                    break
                            if valid:
                                current_set = new_set
                                excluded = set(range(1, 25)) - current_set
                                improved = True
                                break
                if improved:
                    break
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
    
    # Ensure nonempty
    if not best_set:
        best_set = [24]
    
    return best_set
# EVOLVE-BLOCK-END
