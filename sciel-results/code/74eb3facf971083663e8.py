"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _is_valid(selected):
    """Check if no element in selected divides the sum of any 3 distinct others."""
    n = len(selected)
    for i, a in enumerate(selected):
        others = [v for j, v in enumerate(selected) if j != i]
        for combo in combinations(others, 3):
            if sum(combo) % a == 0:
                return False
    return True

def _count_conflicts(selected):
    """For each element, count how many triples of others it divides. Return dict elem->count."""
    n = len(selected)
    conflicts = {v: 0 for v in selected}
    for i, a in enumerate(selected):
        others = [v for j, v in enumerate(selected) if j != i]
        for combo in combinations(others, 3):
            s = sum(combo)
            if s % a == 0:
                conflicts[a] += 1
    return conflicts

def _remove_most_conflicting(selected):
    """Remove the element with the most conflicts. Return new list."""
    if not selected:
        return selected
    conflicts = _count_conflicts(selected)
    worst = max(selected, key=lambda x: conflicts[x])
    return [v for v in selected if v != worst]

def _try_add(selected, x):
    """Check if x can be added to selected without violating constraints."""
    if x in selected:
        return False
    n = len(selected)
    # Check if x divides sum of any 3 from selected
    if n >= 3:
        for combo in combinations(selected, 3):
            if sum(combo) % x == 0:
                return False
    # Check if any a in selected divides x + b + c for b,c in selected, b!=c, b,c != a
    for a in selected:
        others = [v for v in selected if v != a]
        if len(others) >= 2:
            for i in range(len(others)):
                for j in range(i + 1, len(others)):
                    if (x + others[i] + others[j]) % a == 0:
                        return False
    return True

def _greedy_from_dense():
    """Start with a dense set, remove conflicting elements, then try to add back."""
    # Start with numbers 100-192 (93 numbers) - these are relatively safe as divisors
    selected = list(range(100, 193))
    
    # Iteratively remove the most conflicting element
    max_iterations = 200
    for _ in range(max_iterations):
        if not selected:
            break
        conflicts = _count_conflicts(selected)
        if all(c == 0 for c in conflicts.values()):
            break
        worst = max(selected, key=lambda x: conflicts[x])
        selected = [v for v in selected if v != worst]
    
    # Now try to add back removed numbers (and others) greedily
    removed = [x for x in range(1, 193) if x not in selected]
    # Sort by descending value (larger numbers are safer to add)
    removed.sort(reverse=True)
    for x in removed:
        if _try_add(selected, x):
            selected.append(x)
    
    return selected

def _local_improve(selected, deadline):
    """Try swaps and additions to improve the set."""
    best = list(selected)
    best_size = len(best)
    
    # Try additions
    for x in range(192, 0, -1):
        if time.time() > deadline:
            break
        if x not in best and _try_add(best, x):
            best.append(x)
            best_size += 1
    
    # Try swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best)):
            if time.time() > deadline:
                break
            for x in range(192, 0, -1):
                if x in best:
                    continue
                candidate = [v for j, v in enumerate(best) if j != i] + [x]
                if _is_valid(candidate):
                    best = candidate
                    improved = True
                    break
            if improved:
                break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin
    
    # Primary approach: dense start + conflict removal
    selected = _greedy_from_dense()
    
    # Local improvement
    if time.time() < deadline:
        selected = _local_improve(selected, deadline)
    
    return selected
# EVOLVE-BLOCK-END
