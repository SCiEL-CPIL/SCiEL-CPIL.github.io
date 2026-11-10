"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from functools import reduce
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
        if result > 10**18:  # cap to avoid huge numbers
            break
    return result

def _can_form_sum(target_int, values):
    """Check if target_int can be formed as a subset sum of values.
    Uses meet-in-the-middle for larger sets."""
    n = len(values)
    if n == 0:
        return target_int == 0
    if n == 1:
        return target_int == values[0]
    if n <= 20:
        # Meet in the middle
        mid = n // 2
        left = values[:mid]
        right = values[mid:]
        
        # Generate all subset sums for left
        left_sums = set()
        for mask in range(1 << len(left)):
            s = 0
            for i in range(len(left)):
                if mask & (1 << i):
                    s += left[i]
            left_sums.add(s)
        
        # Check if any right subset sum completes target
        for mask in range(1 << len(right)):
            s = 0
            for i in range(len(right)):
                if mask & (1 << i):
                    s += right[i]
            if target_int - s in left_sums:
                return True
        return False
    else:
        # For larger sets, use a simpler approach: greedy check won't work,
        # but we can use a bounded DP or just check if target is in the set
        # Actually for n > 20, use a different strategy
        # Check if target_int is achievable via subset sum - use a set-based approach
        # but cap the number of reachable sums
        reachable = {0}
        for v in values:
            new_reachable = set()
            for s in reachable:
                ns = s + v
                if ns <= target_int:
                    new_reachable.add(ns)
            reachable |= new_reachable
            if target_int in reachable:
                return True
        return False

def _is_valid_addition(x, selected):
    """Check if adding x to selected creates a violation.
    Violation: 1/x = sum(1/y for y in subset of selected)
    OR for any y in selected: 1/y = sum(1/z for z in subset of (selected \ {y}) ∪ {x})
    
    Actually, we need to check:
    1. 1/x is not a subset sum of reciprocals of selected
    2. For each y in selected, 1/y is not a subset sum of reciprocals of (selected \ {y}) ∪ {x}
    
    But condition 2 is equivalent to: adding x doesn't create a new subsum for any existing y.
    The only new subsets that can be formed are those that include x.
    So for each y in selected, check if 1/y = 1/x + sum(1/z for z in subset of (selected \ {y, x}))
    i.e., 1/y - 1/x = sum(1/z for z in subset of (selected \ {y, x}))
    """
    if not selected:
        return True
    
    # Check 1: 1/x is not subset sum of reciprocals of selected
    all_vals = [x] + list(selected)
    lcm_val = _lcm_list(all_vals)
    target_int = lcm_val // x
    candidate_vals = [lcm_val // y for y in selected]
    if _can_form_sum(target_int, candidate_vals):
        return False
    
    # Check 2: For each y in selected, 1/y - 1/x is not subset sum of reciprocals of (selected \ {y, x})
    # 1/y - 1/x = (x - y) / (x*y)
    # In integer terms with LCM of all: (lcm_val / y) - (lcm_val / x) = lcm_val * (x-y)/(x*y)
    # This should be an integer if lcm_val is divisible by x*y
    for y in selected:
        if y == x:
            continue
        # remaining = selected without y and x
        remaining = [z for z in selected if z != y and z != x]
        if not remaining:
            # Check if 1/y - 1/x == 0, i.e., x == y, but they're distinct
            continue
        
        # We need: 1/y - 1/x = sum(1/z for z in subset of remaining)
        # 1/y - 1/x = (x - y) / (x * y)
        # Multiply by LCM of [y, x] + remaining:
        all_for_check = [y, x] + remaining
        lcm_check = _lcm_list(all_for_check)
        # 1/y * lcm_check = lcm_check // y
        # 1/x * lcm_check = lcm_check // x
        # So target = lcm_check//y - lcm_check//x
        target = lcm_check // y - lcm_check // x
        if target <= 0:
            # 1/y - 1/x <= 0 means y >= x, so 1/y <= 1/x
            # The sum of positive reciprocals can't equal a non-positive number
            # Unless target == 0, which means x == y, impossible
            continue
        
        candidate_vals = [lcm_check // z for z in remaining]
        if _can_form_sum(target, candidate_vals):
            return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    # Elements to consider: 1 to 24
    # Process from largest to smallest for better pruning
    elements = list(range(24, 0, -1))
    
    best_set = []
    best_size = 0
    
    # Upper bound: at most 24 elements
    # Greedy first to get a good initial bound
    greedy_set = []
    for x in elements:
        if _is_valid_addition(x, greedy_set):
            greedy_set.append(x)
    
    best_set = list(greedy_set)
    best_size = len(greedy_set)
    
    # Now try to improve with branch and bound
    # State: current index in elements, current selected set
    # We process elements in order, deciding include/exclude
    
    def branch_and_bound(idx, current_set, current_size):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        # Upper bound: current_size + (number of remaining elements)
        remaining_count = len(elements) - idx
        if current_size + remaining_count <= best_size:
            return
        
        if idx == len(elements):
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
        
        x = elements[idx]
        
        # Try including x
        if _is_valid_addition(x, current_set):
            current_set.append(x)
            branch_and_bound(idx + 1, current_set, current_size + 1)
            current_set.pop()
        
        # Try excluding x
        branch_and_bound(idx + 1, current_set, current_size)
    
    # Start branch and bound
    branch_and_bound(0, [], 0)
    
    return best_set
# EVOLVE-BLOCK-END
