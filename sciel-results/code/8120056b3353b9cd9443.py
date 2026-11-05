N = 16

# EVOLVE-BLOCK-START
import random
import itertools
import time
from collections import defaultdict


def _dist_sq(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def _creates_isosceles(p, chosen):
    """Check if adding p to chosen creates any isosceles triangle."""
    n = len(chosen)
    for i in range(n):
        a = chosen[i]
        d_pa = _dist_sq(p, a)
        for j in range(i + 1, n):
            b = chosen[j]
            d_pb = _dist_sq(p, b)
            d_ab = _dist_sq(a, b)
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                return True
    return False


def _is_maximal(chosen, all_points):
    """Check if chosen is maximal: no point can be added."""
    chosen_set = set(chosen)
    for p in all_points:
        if p in chosen_set:
            continue
        if not _creates_isosceles(p, chosen):
            return False
    return True


def _blocking_power(p, chosen, all_points):
    """Count how many other points are blocked by adding p to chosen."""
    count = 0
    for q in all_points:
        if q == p or q in chosen:
            continue
        # Check if q would be blocked (i.e., adding q to chosen+p creates isosceles)
        # Actually, q is blocked if q cannot be added to chosen (without p)
        # But we want: how many points become impossible to add AFTER adding p
        # A point q is blocked by p if: q is not in chosen, q != p,
        # and either q already can't be added to chosen, OR adding q to chosen+p creates isosceles
        # Simpler: count points that CANNOT be added to chosen ∪ {p}
        if _creates_isosceles(q, chosen + [p]):
            count += 1
    return count


def _greedy_with_order(order, all_points, max_points=None):
    """Greedy construction following a given order."""
    chosen = []
    for p in order:
        if not _creates_isosceles(p, chosen):
            chosen.append(p)
        if max_points and len(chosen) >= max_points:
            break
    return chosen


def _local_improve(chosen, all_points, time_limit):
    """Try to reduce the set by removing points and re-adding others."""
    chosen_list = list(chosen)
    start = time.time()
    
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        n = len(chosen_list)
        for i in range(n):
            # Try removing chosen_list[i]
            candidate = chosen_list[:i] + chosen_list[i+1:]
            # Check if still no-isosceles (it is, since we're removing)
            # Check if still maximal
            if _is_maximal(candidate, all_points):
                chosen_list = candidate
                improved = True
                break
    return chosen_list


def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    all_points = list(itertools.product(range(N), repeat=2))
    
    best = None
    best_len = float('inf')
    
    # Strategy 1: Multiple greedy restarts with different orderings
    rng = random.Random(42)
    
    # Precompute some orderings
    orderings = []
    
    # Row-major
    orderings.append(list(all_points))
    
    # Random orderings
    for _ in range(20):
        order = list(all_points)
        rng.shuffle(order)
        orderings.append(order)
    
    # Diagonal ordering
    diag_order = []
    for s in range(2 * N - 1):
        for x in range(N):
            y = s - x
            if 0 <= y < N:
                diag_order.append((x, y))
    orderings.append(diag_order)
    
    # Reverse row-major
    orderings.append(list(reversed(all_points)))
    
    # Column-major
    col_order = [(x, y) for y in range(N) for x in range(N)]
    orderings.append(col_order)
    
    # Spiral-like ordering from center
    center = (N // 2, N // 2)
    spiral_order = sorted(all_points, key=lambda p: _dist_sq(p, center))
    orderings.append(spiral_order)
    
    # Random spatial clusterings
    for _ in range(10):
        anchor = rng.choice(all_points)
        order = sorted(all_points, key=lambda p: _dist_sq(p, anchor) + rng.random() * 10)
        orderings.append(order)
    
    # Run greedy for each ordering
    for order in orderings:
        if time.time() > deadline:
            break
        chosen = _greedy_with_order(order, all_points)
        if len(chosen) < best_len:
            # Check maximality
            if _is_maximal(chosen, all_points):
                best = chosen
                best_len = len(chosen)
    
    # Strategy 2: Local improvement on best found
    if best is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 10:
            improved = _local_improve(best, all_points, remaining_time - 5)
            if len(improved) < best_len:
                best = improved
                best_len = len(improved)
    
    # Strategy 3: Try a few more random restarts with local improvement
    for seed in range(50):
        if time.time() > deadline - 5:
            break
        rng2 = random.Random(seed * 1337 + 7)
        order = list(all_points)
        rng2.shuffle(order)
        chosen = _greedy_with_order(order, all_points)
        if len(chosen) < best_len:
            if _is_maximal(chosen, all_points):
                best = chosen
                best_len = len(chosen)
                remaining = deadline - time.time()
                if remaining > 5:
                    improved = _local_improve(best, all_points, remaining - 2)
                    if len(improved) < best_len:
                        best = improved
                        best_len = len(improved)
    
    if best is None:
        # Fallback: simple greedy
        best = _greedy_with_order(all_points, all_points)
    
    return best
# EVOLVE-BLOCK-END
