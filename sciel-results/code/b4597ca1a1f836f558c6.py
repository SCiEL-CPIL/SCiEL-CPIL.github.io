"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    return reduce(_lcm, lst, 1)

def _is_valid_subset(selected):
    """Check if selected is reciprocal-subsum-free using integer arithmetic."""
    if len(selected) < 2:
        return True
    lcm_val = _lcm_list(selected)
    # Convert each element x to lcm_val // x (integer)
    vals = [lcm_val // x for x in selected]
    n = len(selected)
    for i in range(n):
        target = vals[i]
        others = [vals[j] for j in range(n) if j != i]
        # Check if target can be formed as sum of a non-empty subset of others
        # Use meet-in-the-middle for efficiency
        m = len(others)
        if m == 0:
            continue
        half = m // 2
        left = others[:half]
        right = others[half:]
        # Generate all subset sums of left
        left_sums = set()
        for mask in range(1, 1 << len(left)):  # non-empty
            s = 0
            for b in range(len(left)):
                if mask & (1 << b):
                    s += left[b]
            left_sums.add(s)
        # Also include 0 (empty subset) for combining
        left_sums.add(0)
        # Generate all subset sums of right
        right_sums = set()
        for mask in range(1, 1 << len(right)):  # non-empty
            s = 0
            for b in range(len(right)):
                if mask & (1 << b):
                    s += right[b]
            right_sums.add(s)
        # Check if any combination gives target
        for rs in right_sums:
            if target - rs in left_sums:
                return False
        # Also check if target is in left_sums or right_sums directly (already covered)
    return True

def construct():
    deadline = time.time() + 80  # 80s deadline, 10s safety margin
    all_elements = list(range(1, 25))
    
    # Start with all elements, greedily remove problematic ones
    current = all_elements[:]
    
    # First, check if full set is valid
    if _is_valid_subset(current):
        return current
    
    # Greedy removal: remove the element that causes the most conflicts
    # For each element, count how many other elements' reciprocal relations it violates
    while current:
        if time.time() > deadline - 10:
            break
        if _is_valid_subset(current):
            break
        
        # Find which elements are "responsible" for violations
        # For each element x in current, check if removing x makes it valid
        # But that's expensive. Instead, find violations and remove one of the involved elements.
        
        lcm_val = _lcm_list(current)
        vals = [lcm_val // x for x in current]
        n = len(current)
        
        violation_found = False
        for i in range(n):
            if time.time() > deadline - 10:
                break
            target = vals[i]
            others = [vals[j] for j in range(n) if j != i]
            m = len(others)
            if m == 0:
                continue
            half = m // 2
            left = others[:half]
            right = others[half:]
            left_sums = {0}
            for mask in range(1, 1 << len(left)):
                s = 0
                for b in range(len(left)):
                    if mask & (1 << b):
                        s += left[b]
                left_sums.add(s)
            right_sums = set()
            for mask in range(1, 1 << len(right)):
                s = 0
                for b in range(len(right)):
                    if mask & (1 << b):
                        s += right[b]
                right_sums.add(s)
            
            for rs in right_sums:
                if target - rs in left_sums:
                    # Violation found: current[i]'s reciprocal = sum of subset of others
                    # Remove current[i] (the target element)
                    current.pop(i)
                    violation_found = True
                    break
            if violation_found:
                break
        
        if not violation_found:
            # Shouldn't happen if _is_valid_subset said invalid, but safety
            break
    
    # Try to add back removed elements
    removed = [x for x in all_elements if x not in current]
    for x in removed:
        if time.time() > deadline - 5:
            break
        trial = current + [x]
        if _is_valid_subset(trial):
            current = trial
    
    # Final greedy improvement: try swapping
    if time.time() < deadline - 5:
        improved = True
        while improved:
            improved = False
            for i, x in enumerate(current):
                if time.time() > deadline - 5:
                    break
                for y in removed:
                    if time.time() > deadline - 5:
                        break
                    trial = current[:i] + current[i+1:] + [y]
                    if len(trial) == len(current) and _is_valid_subset(trial):
                        # Check if this is actually an improvement (it's same size, so skip)
                        # Only accept if we can then add more
                        pass
            # Try adding any remaining
            for x in removed[:]:
                if time.time() > deadline - 5:
                    break
                trial = current + [x]
                if _is_valid_subset(trial):
                    current = trial
                    removed.remove(x)
                    improved = True
    
    if not current:
        current = [24]
    
    return current
# EVOLVE-BLOCK-END
