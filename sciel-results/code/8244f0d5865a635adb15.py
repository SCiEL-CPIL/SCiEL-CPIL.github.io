"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from functools import reduce
from itertools import combinations

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

def _build_conflict_graph():
    """
    Build a conflict graph on [1..24].
    Edge (i, j) exists if selecting both i and j creates a reciprocal-subsum violation
    in at least one direction (1/i = sum of subset of {j} or 1/j = sum of subset of {i}).
    
    For two elements {i, j}:
    - 1/i = 1/j? Only if i == j, impossible for distinct.
    - So no direct 2-element conflict.
    
    For three elements {i, j, k}:
    - 1/i = 1/j + 1/k?
    - 1/j = 1/i + 1/k?
    - 1/k = 1/i + 1/j?
    
    So conflicts arise from triples. We need a different approach.
    
    Instead, we'll use a branch-and-bound that tracks the current selected set
    and checks validity incrementally.
    """
    return None

def _max_independent_set_branch_and_bound(elements, selected, best_set, deadline):
    """
    Branch and bound to find maximum reciprocal-subsum-free subset.
    
    elements: list of remaining candidate elements (sorted)
    selected: list of currently selected elements
    best_set: best valid set found so far
    deadline: time limit
    """
    if time.time() > deadline:
        return best_set
    
    # Upper bound: current selected + all remaining
    if len(selected) + len(elements) <= len(best_set):
        return best_set
    
    if not elements:
        if len(selected) > len(best_set):
            best_set = list(selected)
        return best_set
    
    # Choose branching element: pick the one with most conflicts
    # to prune early. We'll use a heuristic: pick the smallest element
    # since small elements have more reciprocal relationships.
    x = elements[0]
    remaining = elements[1:]
    
    # Branch 1: Include x
    new_selected = selected + [x]
    if _is_valid_set(new_selected):
        # Filter remaining to only those compatible with x
        # (but compatibility depends on the full set, so we just recurse)
        result = _max_independent_set_branch_and_bound(remaining, new_selected, best_set, deadline)
        if len(result) > len(best_set):
            best_set = result
    
    # Branch 2: Exclude x
    result = _max_independent_set_branch_and_bound(remaining, selected, best_set, deadline)
    if len(result) > len(best_set):
        best_set = result
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    # Elements 1-24
    all_elements = list(range(1, 25))
    
    # Greedy initial solution: start with large elements (fewer conflicts)
    greedy_set = []
    for x in sorted(all_elements, reverse=True):
        test_set = greedy_set + [x]
        if _is_valid_set(test_set):
            greedy_set = test_set
    
    best_set = greedy_set
    
    # Branch and bound search
    # Order elements: try including smaller elements first (more constraints, prune early)
    # But we want to find large sets, so let's try a different ordering
    
    # Use B&B with elements sorted in a smart order
    # Start with the greedy solution as lower bound
    # Try to improve by searching for larger sets
    
    # Strategy: enumerate subsets of increasing size, check validity
    # Start from the greedy solution size and go up
    
    current_best_size = len(best_set)
    
    # Try to find a set of size current_best_size + 1, then + 2, etc.
    # This is expensive, so use B&B
    
    # Better: use B&B with a good branching order
    # Sort elements by "conflict potential" - elements that conflict with many others first
    # For now, use reverse order (largest first) since large elements are more independent
    
    # Run B&B with elements in descending order
    elements_for_bb = list(range(24, 0, -1))
    
    result = _max_independent_set_branch_and_bound(elements_for_bb, [], best_set, deadline)
    if len(result) > len(best_set):
        best_set = result
    
    # If we have time, try another pass with different ordering
    if time.time() < deadline - 10:
        elements_for_bb2 = list(range(1, 25))
        result2 = _max_independent_set_branch_and_bound(elements_for_bb2, [], best_set, deadline)
        if len(result2) > len(best_set):
            best_set = result2
    
    return best_set
# EVOLVE-BLOCK-END
