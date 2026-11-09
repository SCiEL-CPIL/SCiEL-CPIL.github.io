"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import random
import itertools

DEADLINE = 330.0
MARGIN = 5.0


def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _is_iso(a, b, c):
    d1 = _d2(a, b)
    d2 = _d2(a, c)
    d3 = _d2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3


def _can_add(p, chosen):
    """Check if point p can be added to chosen without creating an isosceles triangle."""
    n = len(chosen)
    for i in range(n):
        for j in range(i + 1, n):
            if _is_iso(p, chosen[i], chosen[j]):
                return False
    return True


def _is_maximal(chosen, all_points):
    """Check if the set is maximal (no more points can be added)."""
    chosen_set = set(chosen)
    for p in all_points:
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            return False
    return True


def _random_maximal(ordered_points, start_time=None):
    """Build a maximal no-isosceles set by considering points in a random order."""
    if start_time is None:
        start_time = time.time()
    chosen = []
    for p in ordered_points:
        if time.time() - start_time > DEADLINE - MARGIN:
            break
        if _can_add(p, chosen):
            chosen.append(p)
    return chosen


def _local_search(chosen, all_points, start_time=None):
    """Try to improve the set by swapping points. Returns improved set or None."""
    if start_time is None:
        start_time = time.time()
    chosen_set = set(chosen)
    n = len(chosen)
    improved = False
    
    # Try removing one point and adding a different one
    for idx in range(n):
        if time.time() - start_time > DEADLINE - MARGIN:
            return None
        removed = chosen[idx]
        remaining = [c for i, c in enumerate(chosen) if i != idx]
        
        # Try to add a point that was previously blocked
        for p in all_points:
            if p in chosen_set:
                continue
            if _can_add(p, remaining):
                # Check if this gives a smaller or equal set (we want minimal)
                # Actually we want to see if we can remove a point entirely
                pass
    
    # Better: try removing each point and see if the set is still maximal
    # If removing a point still leaves a maximal set, we have a smaller maximal set
    for idx in range(n):
        if time.time() - start_time > DEADLINE - MARGIN:
            break
        removed = chosen[idx]
        remaining = [c for i, c in enumerate(chosen) if i != idx]
        if _is_maximal(remaining, all_points):
            return remaining
    
    return None


def _greedy_with_lookahead(ordered_points, all_points, start_time=None):
    """Greedy with a small lookahead: prefer points that block more other points."""
    if start_time is None:
        start_time = time.time()
    chosen = []
    chosen_set = set()
    
    for p in ordered_points:
        if time.time() - start_time > DEADLINE - MARGIN:
            break
        if _can_add(p, chosen):
            chosen.append(p)
            chosen_set.add(p)
    
    return chosen


def construct():
    start_time = time.time()
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    best = None
    best_size = float('inf')
    
    # Strategy 1: Multiple random-order greedy constructions
    random.seed(42)
    num_trials = 200
    for trial in range(num_trials):
        if time.time() - start_time > DEADLINE - MARGIN:
            break
        ordered = list(all_points)
        random.shuffle(ordered)
        result = _random_maximal(ordered, start_time)
        if result:
            if len(result) < best_size:
                best_size = len(result)
                best = result
    
    # Strategy 2: Local search on the best result
    if best is not None:
        improved = _local_search(best, all_points, start_time)
        if improved is not None:
            if len(improved) < best_size:
                best_size = len(improved)
                best = improved
    
    # Strategy 3: Try more random trials with different seeds
    for trial in range(num_trials):
        if time.time() - start_time > DEADLINE - MARGIN:
            break
        seed = 1000 + trial
        random.seed(seed)
        ordered = list(all_points)
        random.shuffle(ordered)
        result = _random_maximal(ordered, start_time)
        if result:
            if len(result) < best_size:
                best_size = len(result)
                best = result
            # Quick local search on improved results
            if len(result) <= best_size:
                improved = _local_search(result, all_points, start_time)
                if improved is not None and len(improved) < best_size:
                    best_size = len(improved)
                    best = improved
    
    if best is None:
        # Fallback: simple greedy
        best = _random_maximal(all_points, start_time)
    
    return best
# EVOLVE-BLOCK-END
