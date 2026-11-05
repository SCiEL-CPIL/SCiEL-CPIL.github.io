"""Minimal maximal no-isosceles set on [0, N)^2."""
import time
import random
import itertools

N = 16
DEADLINE = time.time() + 280.0  # safety margin before 330s kill


def _dist_sq(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy


def _creates_isosceles(p, chosen):
    """Check if adding p to chosen creates any isosceles triangle with two existing points."""
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


def _is_maximal(chosen):
    """Check if the set is maximal: no point can be added."""
    chosen_set = set(chosen)
    for p in itertools.product(range(N), repeat=2):
        if p in chosen_set:
            continue
        if not _creates_isosceles(p, chosen):
            return False
    return True


def _greedy(order, chosen=None):
    """Greedy construction following a given point order."""
    if chosen is None:
        chosen = []
    for p in order:
        if not _creates_isosceles(p, chosen):
            chosen.append(p)
    return chosen


def _local_search(chosen, rng, max_iters=200):
    """Try to reduce size by removing a point and re-filling minimally."""
    best = list(chosen)
    for _ in range(max_iters):
        if time.time() > DEADLINE:
            break
        # Try removing each point
        for i in range(len(best)):
            candidate = [p for j, p in enumerate(best) if j != i]
            # Check if still maximal; if not, try greedy completion
            if not _is_maximal(candidate):
                # Try to complete greedily
                chosen_set = set(candidate)
                remaining = [p for p in itertools.product(range(N), repeat=2)
                             if p not in chosen_set]
                rng.shuffle(remaining)
                completed = list(candidate)
                for p in remaining:
                    if not _creates_isosceles(p, completed):
                        completed.append(p)
                # Check if completed is maximal and smaller
                if len(completed) < len(best) and _is_maximal(completed):
                    best = completed
                    break
        else:
            break
        if time.time() > DEADLINE:
            break
    return best


def _multi_start_greedy(rng, n_starts=50):
    """Multiple random-order greedy constructions."""
    all_points = list(itertools.product(range(N), repeat=2))
    best = None
    best_len = float('inf')
    for _ in range(n_starts):
        if time.time() > DEADLINE:
            break
        order = list(all_points)
        rng.shuffle(order)
        result = _greedy(order)
        if _is_maximal(result) and len(result) < best_len:
            best = result
            best_len = len(result)
    return best


def _constrained_first_greedy(rng, n_starts=30):
    """Greedy where we pick the point that blocks the most others."""
    all_points = list(itertools.product(range(N), repeat=2))
    best = None
    best_len = float('inf')
    for _ in range(n_starts):
        if time.time() > DEADLINE:
            break
        # Randomize initial seed
        seed_points = rng.sample(all_points, k=min(3, len(all_points)))
        chosen = []
        for p in seed_points:
            if not _creates_isosceles(p, chosen):
                chosen.append(p)
        # Now greedily add points, preferring those that are "most constrained"
        # (i.e., have fewest valid partners remaining)
        available = [p for p in all_points if p not in set(chosen)]
        while available:
            if time.time() > DEADLINE:
                break
            # Score each available point by how many it would block
            best_p = None
            best_score = -1
            # Sample a subset for speed
            sample = available if len(available) <= 200 else rng.sample(available, 200)
            for p in sample:
                if _creates_isosceles(p, chosen):
                    continue
                # Count how many other available points this would block
                blocks = 0
                for q in sample:
                    if q == p:
                        continue
                    if _creates_isosceles(q, chosen + [p]):
                        blocks += 1
                if blocks > best_score:
                    best_score = blocks
                    best_p = p
            if best_p is None:
                break
            chosen.append(best_p)
            available.remove(best_p)
        if _is_maximal(chosen) and len(chosen) < best_len:
            best = chosen
            best_len = len(chosen)
    return best


def construct():
    rng = random.Random(42)
    best = None
    best_len = float('inf')

    # Phase 1: Multi-start random greedy
    result = _multi_start_greedy(rng, n_starts=100)
    if result is not None:
        best = result
        best_len = len(result)

    # Phase 2: Constrained-first greedy
    result = _constrained_first_greedy(rng, n_starts=40)
    if result is not None and len(result) < best_len:
        best = result
        best_len = len(result)

    # Phase 3: Local search on best
    if best is not None:
        improved = _local_search(best, rng, max_iters=100)
        if len(improved) < best_len:
            best = improved
            best_len = len(improved)

    # Phase 4: More random starts if time permits
    if time.time() < DEADLINE - 30:
        result = _multi_start_greedy(rng, n_starts=200)
        if result is not None and len(result) < best_len:
            best = result
            best_len = len(result)
        if best is not None:
            improved = _local_search(best, rng, max_iters=100)
            if len(improved) < best_len:
                best = improved
                best_len = len(improved)

    # Phase 5: Final local search
    if best is not None and time.time() < DEADLINE - 10:
        improved = _local_search(best, rng, max_iters=200)
        if len(improved) < best_len:
            best = improved

    return best if best is not None else []
