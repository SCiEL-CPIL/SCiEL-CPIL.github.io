"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import time


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _can_add_fast(p, chosen):
    """
    Check if p can be added to chosen without forming an isosceles triangle.
    Optimized to check O(k) for 'apex at p' and O(k^2) for other cases with early exit.
    """
    if not chosen:
        return True
    
    dists = []
    seen = set()
    # Check if d(p,a) == d(p,b) for any a,b in chosen
    for a in chosen:
        d = _dist2(p, a)
        if d in seen:
            return False
        seen.add(d)
        dists.append(d)
        
    # Check if d(p,a) == d(a,b) or d(p,b) == d(a,b) for any pair a,b in chosen
    k = len(chosen)
    for i in range(k):
        a = chosen[i]
        dp_a = dists[i]
        for j in range(i + 1, k):
            b = chosen[j]
            d_ab = _dist2(a, b)
            if dp_a == d_ab or dists[j] == d_ab:
                return False
    return True


def _greedy_maximal(order, chosen, deadline):
    """Greedy maximal construction given an ordering of points."""
    for p in order:
        if time.time() > deadline:
            return None
        if _can_add_fast(p, chosen):
            chosen.append(p)
    return chosen


def _is_maximal(chosen, all_points):
    """Check if no point can be added to chosen without creating an isosceles triangle."""
    chosen_set = set(chosen)
    for p in all_points:
        if p in chosen_set:
            continue
        if _can_add_fast(p, chosen):
            return False
    return True


def _prune(chosen, all_points, deadline):
    """Try to remove points that are not essential for maximality."""
    if time.time() > deadline:
        return chosen
    chosen = list(chosen)
    i = 0
    while i < len(chosen):
        if time.time() > deadline:
            break
        p = chosen[i]
        candidate = chosen[:i] + chosen[i+1:]
        if _is_maximal(candidate, all_points):
            chosen = candidate
        else:
            i += 1
    return chosen


def _local_search(chosen, all_points, deadline):
    """Try to reduce set size by removing pairs and checking maximality."""
    if time.time() > deadline:
        return chosen
    chosen = list(chosen)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        k = len(chosen)
        for i in range(k):
            if time.time() > deadline:
                return chosen
            for j in range(i+1, k):
                if time.time() > deadline:
                    return chosen
                # Try removing both chosen[i] and chosen[j]
                candidate = [chosen[m] for m in range(k) if m != i and m != j]
                if _is_maximal(candidate, all_points):
                    chosen = candidate
                    improved = True
                    break
            if improved:
                break
    return _prune(chosen, all_points, deadline)


def _try_swap(chosen, all_points, deadline):
    """Try swapping a point in chosen with a point not in chosen to maintain maximality."""
    if time.time() > deadline:
        return chosen
    chosen = list(chosen)
    chosen_set = set(chosen)
    k = len(chosen)
    
    for i in range(k):
        if time.time() > deadline:
            break
        p = chosen[i]
        # Try removing p and adding some other point q
        base = [chosen[m] for m in range(k) if m != i]
        # Find points that can be added to base
        for q in all_points:
            if time.time() > deadline:
                return chosen
            if q in chosen_set:
                continue
            if _can_add_fast(q, base):
                candidate = base + [q]
                if _is_maximal(candidate, all_points):
                    return candidate
    return chosen


def construct():
    deadline = time.time() + 280  # safety margin before 330s kill
    all_points = list(itertools.product(range(N), repeat=2))
    
    best = None
    best_size = float('inf')
    
    # First: deterministic row-major greedy
    result = _greedy_maximal(all_points, [], deadline)
    if result is not None:
        result = _prune(result, all_points, deadline)
        result = _local_search(result, all_points, deadline)
        best = result
        best_size = len(result)
    
    # Randomized restarts with different orderings
    rng = random.Random(42)
    num_trials = 5000
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        # Shuffle the order
        order = all_points[:]
        rng.shuffle(order)
        chosen = []
        result = _greedy_maximal(order, chosen, deadline)
        if result is None:
            break
        # Prune and local search the result
        result = _prune(result, all_points, deadline)
        result = _local_search(result, all_points, deadline)
        if len(result) < best_size:
            best = result
            best_size = len(result)
            if best_size <= 8:
                break
    
    # Final polish: try local search on best result
    if best is not None and time.time() < deadline:
        best = _local_search(best, all_points, deadline)
    
    return best if best is not None else []
# EVOLVE-BLOCK-END
