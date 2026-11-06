"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _check_constraint(selected):
    """Check if selected set satisfies the constraint."""
    n = len(selected)
    if n < 4:
        return True
    for i in range(n):
        a = selected[i]
        others = [selected[j] for j in range(n) if j != i]
        m = len(others)
        if m < 3:
            continue
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _can_add_fast(selected, x):
    """Fast check if x can be added to selected without violating constraint."""
    m = len(selected)
    
    # Check 1: x divides sum of any 3 from selected
    if m >= 3:
        for t in combinations(selected, 3):
            if sum(t) % x == 0:
                return False
    
    # Check 2: for each a in selected, check triples from (selected - {a}) + {x} that include x
    for i in range(m):
        a = selected[i]
        # Triples with x and 2 from selected (not a)
        others_no_a = [selected[j] for j in range(m) if j != i]
        if len(others_no_a) >= 2:
            for t2 in combinations(others_no_a, 2):
                s3 = x + t2[0] + t2[1]
                if s3 % a == 0:
                    return False
    return True

def _greedy_smart(order):
    """Greedy selection following the given order."""
    selected = []
    for x in order:
        if _can_add_fast(selected, x):
            selected.append(x)
    return selected

def _local_improve(selected, deadline):
    """Try to improve by removing one element and adding another."""
    selected = list(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(selected)
        for i in range(n):
            without_i = [selected[j] for j in range(n) if j != i]
            current_set = set(selected)
            for x in range(192, 0, -1):
                if x in current_set:
                    continue
                if _can_add_fast(without_i, x):
                    selected = without_i + [x]
                    improved = True
                    break
            if improved or time.time() >= deadline:
                break
    return selected

def _multi_swap_improve(selected, deadline):
    """Try removing 2 elements and adding 2 different ones."""
    selected = list(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(selected)
        if n < 2:
            break
        current_set = set(selected)
        for i in range(n):
            for j in range(i+1, n):
                without_ij = [selected[k] for k in range(n) if k != i and k != j]
                found = False
                for x in range(192, 0, -1):
                    if x in current_set:
                        continue
                    if _can_add_fast(without_ij, x):
                        without_ij_new = without_ij + [x]
                        for y in range(x-1, 0, -1):
                            if y in set(without_ij_new):
                                continue
                            if _can_add_fast(without_ij_new, y):
                                selected = without_ij_new + [y]
                                improved = True
                                found = True
                                break
                    if found or time.time() >= deadline:
                        break
                if found or time.time() >= deadline:
                    break
            if improved or time.time() >= deadline:
                break
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 80
    rng = random.Random(42)
    
    orderings = [
        list(range(192, 0, -1)),
        list(range(1, 193)),
        [x for x in range(192, 0, -1) if x > 96] + [x for x in range(96, 0, -1) if x > 48],
        [x for x in range(192, 0, -1) if x > 128] + [x for x in range(128, 0, -1) if x > 64],
    ]
    
    for _ in range(3):
        order = list(range(1, 193))
        rng.shuffle(order)
        orderings.append(order)
    
    best = []
    for order in orderings:
        if time.time() >= deadline:
            break
        result = _greedy_smart(order)
        if len(result) > len(best):
            best = result
    
    if best and time.time() < deadline:
        improved = _local_improve(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    if best and time.time() < deadline:
        improved2 = _multi_swap_improve(best, deadline)
        if len(improved2) > len(best):
            best = improved2
    
    if best and not _check_constraint(best):
        best = _greedy_smart(list(range(192, 0, -1)))
    
    return best
# EVOLVE-BLOCK-END
