"""Program for large reciprocal-subsum-free sets using backtracking."""
import time
from fractions import Fraction
from math import gcd
from itertools import combinations
import random
from functools import lru_cache

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    result = 1
    for x in lst:
        result = _lcm(result, x)
    return result

def _can_add(k, selected):
    """
    Check if we can add k to the selected set.
    Condition: For every x in selected, 1/x must NOT be equal to the sum of reciprocals
    of any non-empty subset of (selected \ {x}) U {k}.
    
    When we add k to S to form S' = S U {k}:
    1. We must check if 1/k is a sum of reciprocals of a subset of S. 
    2. For each existing x in S, we must check if 1/x is a sum of reciprocals of a 
       subset of S' \ {x}.
       S' \ {x} = (S \ {x}) U {k}.
       The subsets of S' \ {x} are either subsets of S \ {x} (already safe) OR subsets 
       that include k.
       So we only need to check if 1/x is a sum of reciprocals of a subset T of S \ {x} 
       such that 1/x = sum(1/y for y in T) + 1/k.
       This is equivalent to: sum(1/y for y in T) = 1/x - 1/k.
    """
    # Check 1: Is 1/k a subset sum of reciprocals of S?
    if _is_subset_sum(1, k, selected):
        return False
        
    # Check 2: For each x in S, is (1/x - 1/k) a subset sum of reciprocals of S \ {x}?
    for x in selected:
        # target = 1/x - 1/k = (k - x) / (x * k)
        if x >= k:
            # 1/x - 1/k <= 0, skip
            continue
        
        # target = (k - x) / (x * k)
        # We need: sum of 1/y for y in T = (k-x)/(x*k)
        # Multiply both sides by x*k: sum of x*k/y for y in T = k - x
        # But x*k/y may not be integer. Let's use the general integer conversion.
        remaining = [y for y in selected if y != x]
        if _is_subset_sum_frac(1, x, 1, k, remaining):
            return False
            
    return True

def _is_subset_sum(num, den, elements):
    """
    Check if num/den can be formed by a sum of reciprocals of a non-empty subset of elements.
    All values are positive integers.
    """
    if not elements:
        return False
    
    # Quick checks
    # If any single element's reciprocal equals num/den
    for x in elements:
        if num * x == den:
            return True
    
    # Convert to integer subset sum:
    # We want: sum(1/y for y in T) = num/den
    # Multiply by den: sum(den/y for y in T) = num
    # But den/y may not be integer. Use LCM approach.
    
    # Compute LCM of all elements and den
    lcm_val = _lcm_list(elements + [den])
    
    # Now: sum(lcm_val / y for y in T) = num * lcm_val / den
    # Let target_int = num * lcm_val // den
    # Let values = [lcm_val // y for y in elements]
    
    target_int = num * lcm_val // den
    values = [lcm_val // y for y in elements]
    
    # Check if target_int is achievable as subset sum of values
    return _integer_subset_sum(target_int, values)

def _is_subset_sum_frac(num1, den1, num2, den2, elements):
    """
    Check if (num1/den1 - num2/den2) can be formed by a sum of reciprocals of a non-empty subset of elements.
    """
    # Compute the difference
    # diff = num1/den1 - num2/den2 = (num1*den2 - num2*den1) / (den1*den2)
    num = num1 * den2 - num2 * den1
    den = den1 * den2
    
    if num <= 0:
        return False
    
    # Simplify
    g = gcd(num, den)
    num //= g
    den //= g
    
    return _is_subset_sum(num, den, elements)

def _integer_subset_sum(target, values):
    """
    Check if target can be formed by a sum of a non-empty subset of values.
    Uses meet-in-the-middle for larger sets.
    """
    if not values:
        return False
    
    # Quick check: if any single value equals target
    for v in values:
        if v == target:
            return True
    
    # If sum of all values < target, return False
    if sum(values) < target:
        return False
    
    # If min value > target, return False
    if min(values) > target:
        return False
    
    n = len(values)
    
    if n <= 20:
        # Use meet-in-the-middle
        half = n // 2
        left = values[:half]
        right = values[half:]
        
        # Generate all subset sums for left
        left_sums = set()
        for mask in range(1 << len(left)):
            s = 0
            for i in range(len(left)):
                if mask & (1 << i):
                    s += left[i]
                    if s > target:
                        break
            left_sums.add(s)
        
        # Check if any right subset sum completes a left sum to target
        for mask in range(1 << len(right)):
            s = 0
            for i in range(len(right)):
                if mask & (1 << i):
                    s += right[i]
                    if s > target:
                        break
            if target - s in left_sums:
                # Ensure non-empty subset (mask or corresponding left mask is non-zero)
                # Since we started from mask 1 for left (excluding empty), and here mask can be 0
                # But left_sums was generated from masks 0 to 2^half-1, including 0 (empty sum = 0)
                # We need at least one element total.
                if mask != 0 or (target - s) != 0:
                    return True
                # If mask == 0 and target - s == 0, that means empty subset, which is invalid
                # But target > 0 so target - s = 0 means s = target, and mask=0 means s=0, so target=0, contradiction
                # Actually if target > 0, and mask=0, s=0, then we need target in left_sums.
                # left_sums includes 0 (from empty left subset). If target is in left_sums from a non-empty left subset, that's fine.
                # The issue is only if both are empty.
                # Since target > 0, if target is in left_sums, it came from a non-empty left subset (since empty gives 0).
                # So this is fine.
                return True
        
        return False
    else:
        # For larger sets, use a simpler DFS with pruning
        # Sort in descending order for better pruning
        sorted_vals = sorted(values, reverse=True)
        prefix_sums = [0] * (n + 1)
        for i in range(n):
            prefix_sums[i + 1] = prefix_sums[i] + sorted_vals[i]
        
        def dfs(idx, current_sum, used):
            if current_sum == target:
                return used  # Return True if used at least one element
            if idx >= n:
                return False
            if current_sum + prefix_sums[n] - prefix_sums[idx] < target:
                return False
            if current_sum > target:
                return False
            
            # Include
            if dfs(idx + 1, current_sum + sorted_vals[idx], used or True):
                return True
            # Exclude
            if dfs(idx + 1, current_sum, used):
                return True
            return False
        
        return dfs(0, 0, False)

def _validate_set(S):
    """
    Verify that S is a valid reciprocal-subsum-free set.
    """
    S = set(S)
    for x in S:
        others = S - {x}
        # Check if 1/x is a subset sum of reciprocals of others
        if _is_subset_sum(1, x, list(others)):
            return False
    return True

def _try_additions(S):
    """
    Try to add any element from 1..24 not in S.
    Returns (new_set, improved_flag).
    """
    S = set(S)
    for c in range(1, 25):
        if c not in S:
            if _can_add(c, list(S)):
                S.add(c)
                return S, True
    return S, False

def construct():
    """
    Construct the largest subset of [1, 24] that is reciprocal-subsum-free.
    Strategy: Multiple random restarts with greedy construction, followed by local search (additions).
    """
    best_set = set()
    best_size = 0
    
    # Use a fixed seed for reproducibility in evaluation, but vary order
    random.seed(42)
    
    # Phase 1: Try several different orderings (greedy heuristics)
    # 1. Descending (24..1)
    # 2. Ascending (1..24)
    # 3. Random shuffles
    
    orderings = []
    orderings.append(list(range(24, 0, -1)))
    orderings.append(list(range(1, 25)))
    
    # Add more random shuffles since the subset sum check is now faster
    for _ in range(50):
        order = list(range(1, 25))
        random.shuffle(order)
        orderings.append(order)
    
    for order in orderings:
        if _time_remaining() < 5.0:
            break
            
        S_temp = []
        for c in order:
            if _can_add(c, S_temp):
                S_temp.append(c)
                
        if len(S_temp) > best_size:
            best_set = set(S_temp)
            best_size = len(S_temp)
            
    # Phase 2: Local search on the best set found
    # Try to add more elements if possible
    improved = True
    while improved and _time_remaining() > 5.0:
        best_set, improved = _try_additions(best_set)
        best_size = len(best_set)
        
    # Phase 3: Try swap-based local search
    # If we can't add, try removing one element and adding two
    if _time_remaining() > 10.0:
        for _ in range(3):  # A few rounds of swap search
            if _time_remaining() < 10.0:
                break
            improved = False
            for x in list(best_set):
                if _time_remaining() < 10.0:
                    break
                candidate = best_set - {x}
                # Try to add elements to candidate
                for c in range(1, 25):
                    if c not in candidate:
                        if _can_add(c, list(candidate)):
                            new_set = candidate | {c}
                            if len(new_set) > len(best_set):
                                best_set = new_set
                                best_size = len(best_set)
                                improved = True
                                break
            if not improved:
                break
    
    return list(best_set)

# Quick test
if __name__ == "__main__":
    result = construct()
    print(f"Result size: {len(result)}")
    print(f"Result: {sorted(result)}")
    print(f"Valid: {_validate_set(result)}")
