"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from functools import reduce
from itertools import combinations
from fractions import Fraction

# EVOLVE-BLOCK-START
def _lcm(a, b):
    return a * b // gcd(a, b)

def _check_reciprocal_subsum(target_x, others):
    """Check if 1/target_x equals sum of 1/y for some subset of others."""
    if not others:
        return False
    all_vals = [target_x] + list(others)
    lcm_val = reduce(_lcm, all_vals)
    target_int = lcm_val // target_x
    candidate_vals = [lcm_val // y for y in others]
    n = len(candidate_vals)
    if n == 0:
        return False
    half = n // 2
    left = candidate_vals[:half]
    right = candidate_vals[half:]
    left_sums = {0}
    for v in left:
        left_sums = left_sums | {s + v for s in left_sums}
    right_sums = {0}
    for v in right:
        right_sums = right_sums | {s + v for s in right_sums}
    for s in left_sums:
        if target_int - s in right_sums:
            return True
    return False

def _is_valid_set(selected):
    """Check if selected set is reciprocal-subsum-free."""
    selected = list(selected)
    for i, x in enumerate(selected):
        others = selected[:i] + selected[i+1:]
        if _check_reciprocal_subsum(x, others):
            return False
    return True

def _precompute_forbidden_patterns():
    """
    Precompute all forbidden patterns.
    A forbidden pattern is a tuple (x, frozenset(S)) where x is in [1..24],
    S is a subset of [1..24]\\{x}, and sum(1/s for s in S) = 1/x.
    If all elements in {x} ∪ S are selected, the set is invalid.
    """
    forbidden = []
    for x in range(1, 25):
        # Find all subsets S of [1..24]\{x} where sum(1/s) = 1/x
        others = [y for y in range(1, 25) if y != x]
        # Use LCM-based integer arithmetic
        # 1/x = sum(1/s) => LCM/x = sum(LCM/s)
        # Try subset sizes 1, 2, 3, 4 (larger subsets unlikely to sum to 1/x for x>=1)
        for size in range(1, 5):
            for S in combinations(others, size):
                # Check if sum(1/s for s in S) == 1/x
                # Compute as Fraction for correctness
                total = Fraction(0, 1)
                for s in S:
                    total += Fraction(1, s)
                if total == Fraction(1, x):
                    forbidden.append((x, frozenset(S)))
    return forbidden

def _branch_and_bound_optimized(ordered_elements, selected_set, forbidden_patterns, best_set, deadline):
    """
    Optimized B&B with precomputed forbidden patterns.
    """
    if time.time() > deadline:
        return best_set
    
    # Upper bound: current selected + remaining
    remaining_count = len([e for e in ordered_elements if e not in selected_set])
    if len(selected_set) + remaining_count <= len(best_set):
        return best_set
    
    if not ordered_elements:
        if len(selected_set) > len(best_set):
            best_set = list(selected_set)
        return best_set
    
    # Find the next candidate to branch on
    x = None
    idx = -1
    for i, e in enumerate(ordered_elements):
        if e not in selected_set:
            x = e
            idx = i
            break
    
    if x is None:
        if len(selected_set) > len(best_set):
            best_set = list(selected_set)
        return best_set
    
    remaining = ordered_elements[idx + 1:]
    
    # Branch 1: Include x
    new_selected = selected_set | {x}
    
    # Check if adding x creates any violation with forbidden patterns
    valid = True
    for (fx, fs) in forbidden_patterns:
        # Check if fx is in new_selected and all of fs are in new_selected
        if fx in new_selected and fs.issubset(new_selected):
            valid = False
            break
    
    if valid:
        result = _branch_and_bound_optimized(remaining, new_selected, forbidden_patterns, best_set, deadline)
        if len(result) > len(best_set):
            best_set = result
    
    # Branch 2: Exclude x
    result = _branch_and_bound_optimized(remaining, selected_set, forbidden_patterns, best_set, deadline)
    if len(result) > len(best_set):
        best_set = result
    
    return best_set

def _try_ilp(forbidden_patterns, deadline):
    """
    Try to solve using ILP via scipy.optimize.milp.
    """
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        # Binary variables x_1, ..., x_24
        # Maximize sum(x_i)
        # Subject to: for each forbidden pattern (fx, fs), NOT(all in {fx} ∪ fs are selected)
        # i.e., sum of x_i for i in {fx} ∪ fs <= |{fx} ∪ fs| - 1
        
        n_vars = 24
        c = -np.ones(n_vars)  # minimize -sum = maximize sum
        
        A_rows = []
        b_lower = []
        b_upper = []
        
        for (fx, fs) in forbidden_patterns:
            elements = [fx] + list(fs)
            # Sum of x_i for i in elements <= len(elements) - 1
            row = np.zeros(n_vars)
            for elem in elements:
                row[elem - 1] = 1.0  # 0-indexed
            A_rows.append(row)
            b_lower.append(-np.inf)
            b_upper.append(len(elements) - 1)
        
        if not A_rows:
            return list(range(1, 25))
        
        A = np.array(A_rows)
        b_lo = np.array(b_lower)
        b_hi = np.array(b_upper)
        
        constraints = LinearConstraint(A, b_lo, b_hi)
        integrality = np.ones(n_vars)  # all binary
        bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
        
        result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if result.success and result.x is not None:
            selected = [i + 1 for i in range(n_vars) if result.x[i] > 0.5]
            if selected and _is_valid_set(selected):
                return selected
    except Exception:
        pass
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    # Precompute forbidden patterns
    forbidden_patterns = _precompute_forbidden_patterns()
    
    # Greedy initial solution: start with large elements (fewer conflicts)
    greedy_set = []
    for x in sorted(range(1, 25), reverse=True):
        test_set = set(greedy_set) | {x}
        valid = True
        for (fx, fs) in forbidden_patterns:
            if fx in test_set and fs.issubset(test_set):
                valid = False
                break
        if valid:
            greedy_set.append(x)
    
    best_set = greedy_set
    
    # B&B with elements in descending order
    elements_for_bb = list(range(24, 0, -1))
    result = _branch_and_bound_optimized(elements_for_bb, set(), forbidden_patterns, best_set, deadline)
    if len(result) > len(best_set):
        best_set = result
    
    # If we have time, try ILP
    if time.time() < deadline - 5:
        ilp_result = _try_ilp(forbidden_patterns, deadline)
        if ilp_result and len(ilp_result) > len(best_set):
            best_set = ilp_result
    
    # Final verification
    if not _is_valid_set(best_set):
        # Fallback to a known valid set
        best_set = list(range(13, 25))
    
    return best_set
# EVOLVE-BLOCK-END
