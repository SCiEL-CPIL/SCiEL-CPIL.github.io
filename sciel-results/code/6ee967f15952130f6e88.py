"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm, gcd

def construct():
    deadline = time.time() + 55  # 55s safety margin for 60s limit
    
    # Precompute compatibility: compat[a][b] = True if lcm(a,b) > 256
    # We only need to check pairs, so use a set of incompatible pairs
    incompatible = set()
    for a in range(2, 257):
        for b in range(a+1, 257):
            if lcm(a, b) <= 256:
                incompatible.add((a, b))
                incompatible.add((b, a))
    
    def is_compatible(x, selected_set):
        """Check if x can be added to selected_set (which is a set of ints)"""
        for y in selected_set:
            if (x, y) in incompatible:
                return False
        return True
    
    def reciprocal_sum(subset):
        return sum(1.0 / x for x in subset)
    
    # Strategy 1: Greedy smallest first (seed)
    selected1 = []
    selected1_set = set()
    for x in range(2, 257):
        if is_compatible(x, selected1_set):
            selected1.append(x)
            selected1_set.add(x)
    
    # Strategy 2: Greedy largest first
    selected2 = []
    selected2_set = set()
    for x in range(256, 1, -1):
        if is_compatible(x, selected2_set):
            selected2.append(x)
            selected2_set.add(x)
    
    # Strategy 3: Greedy by reciprocal (smallest first is already this)
    # Strategy 4: Try to add numbers that are "incompatible" with few others
    # Count how many numbers each x is incompatible with
    incompat_count = {}
    for x in range(2, 257):
        count = 0
        for y in range(2, 257):
            if x != y and (x, y) in incompatible:
                count += 1
        incompat_count[x] = count
    
    # Sort by (incompat_count, x) - prefer numbers with fewer conflicts, then smaller
    order3 = sorted(range(2, 257), key=lambda x: (incompat_count[x], x))
    selected3 = []
    selected3_set = set()
    for x in order3:
        if is_compatible(x, selected3_set):
            selected3.append(x)
            selected3_set.add(x)
    
    # Strategy 5: Local search on the best candidate
    best = max([selected1, selected2, selected3], key=reciprocal_sum)
    best_set = set(best)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try removing one element and adding another
        for i, x in enumerate(list(best_set)):
            trial_set = best_set - {x}
            # Try to add a number not in trial_set
            for y in range(2, 257):
                if y in trial_set:
                    continue
                if is_compatible(y, trial_set):
                    new_set = trial_set | {y}
                    if reciprocal_sum(new_set) > reciprocal_sum(best_set) + 1e-12:
                        best_set = new_set
                        improved = True
                        break
            if improved:
                break
        # Try swapping two elements
        if not improved and time.time() < deadline - 5:
            for x in list(best_set):
                trial_set = best_set - {x}
                for y in list(best_set):
                    if y == x:
                        continue
                    trial_set2 = trial_set - {y}
                    # Try to add two numbers
                    for z in range(2, 257):
                        if z in trial_set2:
                            continue
                        if is_compatible(z, trial_set2):
                            trial_set3 = trial_set2 | {z}
                            for w in range(2, 257):
                                if w in trial_set3:
                                    continue
                                if is_compatible(w, trial_set3):
                                    new_set = trial_set3 | {w}
                                    if reciprocal_sum(new_set) > reciprocal_sum(best_set) + 1e-12:
                                        best_set = new_set
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    return list(best_set)
# EVOLVE-BLOCK-END
