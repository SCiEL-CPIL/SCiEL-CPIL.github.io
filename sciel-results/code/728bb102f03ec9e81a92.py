"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _init_sums(a):
    s = set()
    for i, x in enumerate(a):
        for y in a[i:]:
            s.add(x + y)
    return s

def _can_add(a, sums, x):
    new = {x + y for y in a} | {2 * x}
    if len(new) != len(a) + 1:
        return False
    return not (new & sums)

def _greedy_with_order(a_init, order, deadline):
    a = list(a_init)
    sums = _init_sums(a)
    a_set = set(a)
    for x in order:
        if time.time() > deadline:
            break
        if x in a_set:
            continue
        if _can_add(a, sums, x):
            a.append(x)
            a_set.add(x)
            new = {x + y for y in a[:-1]} | {2 * x}
            sums |= new
    return a

def _local_search(a, deadline, max_iters=5000):
    """Try to improve by swapping one element out and another in."""
    a = list(a)
    a_set = set(a)
    sums = _init_sums(a)
    best = len(a)
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        # Try removing each element and adding a better one
        improved = False
        for i in range(len(a)):
            if time.time() > deadline:
                break
            removed = a[i]
            # Remove from set and sums
            a2 = a[:i] + a[i+1:]
            a2_set = set(a2)
            # Recompute sums for a2
            sums2 = set()
            for j, x in enumerate(a2):
                for y in a2[j:]:
                    sums2.add(x + y)
            # Try to add candidates not in a2
            for x in range(1, 285):
                if x in a2_set:
                    continue
                if _can_add(a2, sums2, x):
                    # Check if this is an improvement (same size, but maybe better positioned)
                    # We accept any valid addition to maintain size
                    a = a2 + [x]
                    a_set = set(a)
                    sums = sums2 | {x + y for y in a2} | {2 * x}
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return a

def _backtrack_greedy(a_init, deadline, max_depth=3):
    """Greedy with limited backtracking when stuck."""
    a = list(a_init)
    sums = _init_sums(a)
    a_set = set(a)
    
    # Find all valid candidates
    candidates = [x for x in range(1, 285) if x not in a_set and _can_add(a, sums, x)]
    
    if not candidates:
        return a
    
    # Try greedy first
    for x in sorted(candidates):
        if time.time() > deadline:
            break
        if x not in a_set and _can_add(a, sums, x):
            a.append(x)
            a_set.add(x)
            new = {x + y for y in a[:-1]} | {2 * x}
            sums |= new
            candidates = [c for c in candidates if c != x and _can_add(a, sums, c)]
    
    # If we're stuck, try backtracking: remove last element and try next candidate
    if candidates and len(a) > len(a_init):
        last = a.pop()
        a_set.discard(last)
        # Recompute sums
        sums = _init_sums(a)
        # Try remaining candidates
        for x in candidates:
            if time.time() > deadline:
                break
            if x not in a_set and _can_add(a, sums, x):
                a.append(x)
                a_set.add(x)
                new = {x + y for y in a[:-1]} | {2 * x}
                sums |= new
                break
    
    return a

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    a_init = [1, 2, 5, 11]
    best = list(a_init)
    best_sums = _init_sums(best)
    
    # Strategy 1: Simple greedy (original)
    a = list(a_init)
    sums = _init_sums(a)
    a_set = set(a)
    for x in range(12, 285):
        if time.time() > deadline:
            break
        if x not in a_set and _can_add(a, sums, x):
            a.append(x)
            a_set.add(x)
            new = {x + y for y in a[:-1]} | {2 * x}
            sums |= new
    if len(a) > len(best):
        best = a
        best_sums = sums
    
    # Strategy 2: Greedy with random orderings
    random.seed(42)
    for trial in range(20):
        if time.time() > deadline:
            break
        order = list(range(1, 285))
        random.shuffle(order)
        a = _greedy_with_order(a_init, order, deadline)
        if len(a) > len(best):
            best = a
            best_sums = _init_sums(best)
    
    # Strategy 3: Backtracking greedy
    if time.time() < deadline:
        a = _backtrack_greedy(a_init, deadline)
        if len(a) > len(best):
            best = a
            best_sums = _init_sums(best)
    
    # Strategy 4: Local search from best
    if time.time() < deadline - 5:
        a = _local_search(best, deadline, max_iters=3000)
        if len(a) > len(best):
            best = a
            best_sums = _init_sums(best)
    
    # Strategy 5: More random restarts if time remains
    if time.time() < deadline - 10:
        for trial in range(50):
            if time.time() > deadline:
                break
            order = list(range(1, 285))
            random.shuffle(order)
            a = _greedy_with_order(a_init, order, deadline)
            if len(a) > len(best):
                best = a
                best_sums = _init_sums(best)
    
    # Final local search on best if time remains
    if time.time() < deadline - 2:
        a = _local_search(best, deadline, max_iters=2000)
        if len(a) > len(best):
            best = a
    
    return best
# EVOLVE-BLOCK-END
