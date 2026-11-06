"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _check_no_4ap(S):
    """Verify that set S contains no nonconstant 4-term arithmetic progression."""
    S_set = set(S)
    for a in S:
        for d in range(1, (512 - a) // 3 + 1):
            if (a + d in S_set and a + 2*d in S_set and a + 3*d in S_set):
                return False
    return True

def _greedy_construct():
    """Greedy construction: add elements in order, skipping those that would create a 4-AP."""
    selected = set()
    for x in range(1, 513):
        # Check if adding x would create a 4-AP
        # x could be: a, a+d, a+2d, a+3d
        # Case 1: x is a (first term): need x+d, x+2d, x+3d in selected
        ok = True
        for d in range(1, (512 - x) // 3 + 1):
            if x+d in selected and x+2*d in selected and x+3*d in selected:
                ok = False
                break
        if not ok:
            continue
        # Case 2: x is a+d (second term): need a, a+2d, a+3d in selected => x-d, x+d, x+2d
        for d in range(1, (x - 1) // 1):
            if x - d >= 1 and x - d in selected and x + d in selected and x + 2*d in selected:
                ok = False
                break
        if not ok:
            continue
        # Case 3: x is a+2d (third term): need a, a+d, a+3d in selected => x-2d, x-d, x+d
        for d in range(1, (x - 1) // 2 + 1):
            if x - 2*d >= 1 and x - 2*d in selected and x - d in selected and x + d in selected:
                ok = False
                break
        if not ok:
            continue
        # Case 4: x is a+3d (fourth term): need a, a+d, a+2d in selected => x-3d, x-2d, x-d
        for d in range(1, (x - 1) // 3 + 1):
            if x - 3*d >= 1 and x - 3*d in selected and x - 2*d in selected and x - d in selected:
                ok = False
                break
        if not ok:
            continue
        selected.add(x)
    return selected

def _local_search_improve(S, time_limit=55.0):
    """Local search: try to add elements or swap to improve size."""
    S_set = set(S)
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        # Try adding each element not in S
        for x in range(1, 513):
            if x in S_set:
                continue
            if _check_no_4ap(S_set | {x}):
                S_set.add(x)
                improved = True
        # Try single swaps: remove one, add another
        if not improved:
            for x in list(S_set):
                for y in range(1, 513):
                    if y in S_set:
                        continue
                    new_S = (S_set - {x}) | {y}
                    if _check_no_4ap(new_S):
                        # Only accept if size doesn't decrease (it won't, it's a swap)
                        S_set = new_S
                        improved = True
                        break
                if time.time() - start_time > time_limit:
                    break
    return S_set

def _pair_swap_local_search(S_set, time_limit=55.0):
    """Pair-swap local search: remove two elements, add two different ones."""
    start_time = time.time()
    improved = True
    elements = list(S_set)
    while improved and time.time() - start_time < time_limit:
        improved = False
        n = len(elements)
        for i in range(n):
            for j in range(i + 1, n):
                # Try removing elements[i] and elements[j], adding two new ones
                candidates = [x for x in range(1, 513) if x not in S_set]
                random.shuffle(candidates)
                for c1 in candidates[:20]:
                    for c2 in candidates[:20]:
                        if c1 == c2:
                            continue
                        new_S = (S_set - {elements[i], elements[j]}) | {c1, c2}
                        if _check_no_4ap(new_S):
                            S_set = new_S
                            elements = list(S_set)
                            improved = True
                            break
                    if improved:
                        break
                if time.time() - start_time > time_limit:
                    break
            if time.time() - start_time > time_limit:
                break
    return S_set

def _randomized_greedy(seed, time_limit=50.0):
    """Randomized greedy: process elements in random order."""
    random.seed(seed)
    order = list(range(1, 513))
    random.shuffle(order)
    selected = set()
    for x in order:
        if _check_no_4ap(selected | {x}):
            selected.add(x)
        if time.time() - start_time > time_limit:
            break
    return selected

def construct():
    start_time = time.time()
    deadline = 58.0  # Safety margin before 60s limit
    
    # Start with deterministic greedy
    best_set = _greedy_construct()
    best_size = len(best_set)
    
    # Try local search improvement
    if time.time() - start_time < deadline - 10:
        improved = _local_search_improve(set(best_set), time_limit=deadline - time.time() - 2)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
    
    # Try pair swaps
    if time.time() - start_time < deadline - 5:
        improved = _pair_swap_local_search(set(best_set), time_limit=deadline - time.time() - 1)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(improved)
    
    # Try a few randomized greedy starts with local search
    for seed in range(1, 10):
        if time.time() - start_time > deadline - 8:
            break
        random.seed(seed)
        order = list(range(1, 513))
        random.shuffle(order)
        selected = set()
        for x in order:
            if _check_no_4ap(selected | {x}):
                selected.add(x)
        if len(selected) > best_size:
            # Improve this candidate
            if time.time() - start_time < deadline - 3:
                improved = _local_search_improve(set(selected), time_limit=deadline - time.time() - 1)
                if len(improved) > best_size:
                    best_set = improved
                    best_size = len(improved)
    
    # Final verification
    if not _check_no_4ap(best_set):
        # Fallback to simple greedy
        best_set = _greedy_construct()
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
