"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from functools import reduce
from math import gcd

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)

    # Precompute LCM of all denominators for exact integer arithmetic
    lcm_all = 1
    for d in denominators:
        lcm_all = lcm_all * d // gcd(lcm_all, d)

    # Integer values: value[i] = lcm_all // denominators[i]
    values = [lcm_all // d for d in denominators]
    total_value = sum(values)

    # We want to select a subset S such that sum_{i in S} values[i] < lcm_all
    # and minimize lcm_all - sum_{i in S} values[i]
    # i.e., minimize the deficit integer = lcm_all - sum(values[i] for i in S)
    # subject to sum(values[i] for i in S) <= lcm_all - 1

    # This is equivalent to: find subset with sum as close to lcm_all as possible but strictly below.
    # Let target = lcm_all - 1. We want max subset sum <= target.
    # Then deficit = lcm_all - subset_sum.

    # Since total_value > lcm_all, we can also think of it as:
    # Remove a subset R from all such that sum(values) - sum(values[i] for i in R) < lcm_all
    # i.e., sum(values[i] for i in R) > total_value - lcm_all
    # and we want to MINIMIZE such sum (to minimize deficit).
    # Deficit = lcm_all - (total_value - sum_R) = sum_R - (total_value - lcm_all)
    # So minimize sum_R subject to sum_R > total_value - lcm_all, sum_R integer.
    # i.e., find min sum_R >= total_value - lcm_all + 1.

    excess = total_value - lcm_all  # this is positive
    # Find subset R with minimum sum >= excess + 1
    # Then selected = all \ R, deficit = (sum_R) - excess

    # Use meet-in-the-middle for subset sum (n=47 is large, but we can use DP on smaller sets)
    # Actually n=47 is too large for exact subset sum in general.
    # But values are specific: lcm_all // d for d in 2..48.
    # Let's think differently.

    # Alternative: Greedy + local search
    # Start with greedy selection (as in seed), then try local improvements.

    # Greedy: add elements in order if sum stays < lcm_all
    selected = set()
    current_sum = 0
    for i in range(n):
        if current_sum + values[i] < lcm_all:
            selected.add(i)
            current_sum += values[i]

    best_deficit = lcm_all - current_sum
    best_selected = set(selected)

    # Local search: try to improve by swapping or adding/removing
    # Try to add any unselected element if it keeps sum < lcm_all
    improved = True
    while improved and time.time() < deadline:
        improved = False
        unselected = [i for i in range(n) if i not in selected]
        for i in unselected:
            if current_sum + values[i] < lcm_all:
                new_sum = current_sum + values[i]
                new_deficit = lcm_all - new_sum
                if new_deficit < best_deficit:
                    selected.add(i)
                    current_sum = new_sum
                    best_deficit = new_deficit
                    best_selected = set(selected)
                    improved = True
                    break

    # Try removing elements and replacing with others to reduce deficit
    # For each selected element, try removing it and adding one or more unselected
    # This is complex; let's do a simpler local search:
    # Try pairs: remove one selected, add one unselected

    if time.time() < deadline:
        for _ in range(200000):  # limit iterations
            if time.time() > deadline:
                break
            # Try random swap
            if not selected or not [i for i in range(n) if i not in selected]:
                break
            sel_list = list(selected)
            unsel_list = [i for i in range(n) if i not in selected]
            if not sel_list or not unsel_list:
                break
            import random
            ri = random.randint(0, len(sel_list) - 1)
            rj = random.randint(0, len(unsel_list) - 1)
            a = sel_list[ri]
            b = unsel_list[rj]
            new_sum = current_sum - values[a] + values[b]
            if new_sum < lcm_all:
                new_deficit = lcm_all - new_sum
                if new_deficit < best_deficit:
                    selected.remove(a)
                    selected.add(b)
                    current_sum = new_sum
                    best_deficit = new_deficit
                    best_selected = set(selected)

    # Convert to list of indices
    result = sorted(best_selected)
    if not result:
        # Fallback: just pick index 0 (denominator 2)
        result = [0]

    return result
# EVOLVE-BLOCK-END
