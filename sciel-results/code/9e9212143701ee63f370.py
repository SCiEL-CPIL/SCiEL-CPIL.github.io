"""Combined: simulated annealing with swap moves + local removal refinement."""
import time
import random
import itertools

N = 16
DEADLINE = time.time() + 290  # Safety margin before 330s kill

ALL_POINTS = list(itertools.product(range(N), repeat=2))
ALL_POINTS_SET = set(ALL_POINTS)


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _is_isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3


def _can_add(point, chosen):
    """Check if point can be added to chosen without creating isosceles triangle."""
    for a, b in itertools.combinations(chosen, 2):
        if _is_isosceles(point, a, b):
            return False
    return True


def _is_maximal(chosen):
    """Check if no grid point can be added to chosen."""
    chosen_set = set(chosen)
    for p in ALL_POINTS:
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            return False
    return True


def _is_valid(chosen):
    """Check if chosen is isosceles-free."""
    for a, b, c in itertools.combinations(chosen, 3):
        if _is_isosceles(a, b, c):
            return False
    return True


def _greedy_random(seed_val):
    """Randomized greedy construction."""
    rng = random.Random(seed_val)
    points = ALL_POINTS[:]
    rng.shuffle(points)
    chosen = []
    for p in points:
        if _can_add(p, chosen):
            chosen.append(p)
    return chosen


def _local_remove(chosen):
    """Try removing points one at a time; if the set is still maximal, keep the removal.
    Note: for a true maximal set, removing any single point makes it non-maximal
    (that point can be re-added). But after a swap move, the set may not be truly
    minimal-maximal, so this can still help in some cases."""
    current = list(chosen)
    improved = True
    while improved:
        improved = False
        for i in range(len(current)):
            candidate = current[:i] + current[i + 1:]
            if _is_maximal(candidate):
                current = candidate
                improved = True
                break
    return current


def _swap_move(chosen, rng):
    """Try one swap: remove a random point, add a random other point.
    Returns new set if valid and smaller-or-equal, else None."""
    if len(chosen) < 2:
        return None
    # Remove a random point
    idx = rng.randrange(len(chosen))
    removed = chosen[idx]
    without = chosen[:idx] + chosen[idx + 1:]
    # Try to add a random point (not already in set, not the removed one)
    candidates = [p for p in ALL_POINTS if p not in set(without)]
    if not candidates:
        return None
    add = candidates[rng.randrange(len(candidates))]
    if add == removed:
        return None
    # Check if add can be added to without
    if not _can_add(add, without):
        return None
    new_set = without + [add]
    # Only accept if size decreased or stayed same (we prefer smaller)
    if len(new_set) < len(chosen):
        return new_set
    return None


def _simulated_annealing(initial_set, rng, time_budget):
    """Simulated annealing with swap moves to find smaller maximal sets."""
    current = list(initial_set)
    current_size = len(current)
    best = list(current)
    best_size = current_size
    T = 1.0
    cooling = 0.995
    start = time.time()

    while time.time() - start < time_budget and T > 0.01:
        # Try a swap move
        new_set = _swap_move(current, rng)
        if new_set is None:
            T *= cooling
            continue
        new_size = len(new_set)
        delta = new_size - current_size
        if delta < 0 or rng.random() < pow(2.718281828, -delta / T):
            current = new_set
            current_size = new_size
            if current_size < best_size:
                best = list(current)
                best_size = current_size
        T *= cooling

    return best, best_size


def construct():
    start_time = time.time()
    rng = random.Random(42)

    best_set = None
    best_size = float('inf')

    # Baseline: deterministic row-major greedy
    baseline = []
    for p in ALL_POINTS:
        if _can_add(p, baseline):
            baseline.append(p)
    if _is_valid(baseline) and _is_maximal(baseline):
        best_set = baseline
        best_size = len(baseline)

    # Try multiple random restarts with simulated annealing refinement
    attempt = 0
    while time.time() < DEADLINE:
        attempt += 1
        # Check time
        remaining = DEADLINE - time.time()
        if remaining < 5:
            break

        # Random greedy start
        candidate = _greedy_random(attempt * 7 + 13)
        if not _is_valid(candidate):
            continue
        size = len(candidate)
        if size >= best_size:
            continue

        # Check maximality
        if not _is_maximal(candidate):
            # Try to fix by adding points
            for p in ALL_POINTS:
                if p not in candidate and _can_add(p, candidate):
                    candidate.append(p)
            size = len(candidate)
            if size >= best_size:
                continue
            if not _is_maximal(candidate):
                continue

        # Local removal refinement
        candidate = _local_remove(candidate)
        size = len(candidate)
        if size < best_size:
            best_set = candidate[:]
            best_size = size

        # Simulated annealing refinement (use remaining time budget)
        sa_time = min(30, remaining - 5)
        if sa_time > 2:
            sa_best, sa_size = _simulated_annealing(candidate, rng, sa_time)
            if sa_size < best_size:
                # Verify and refine
                sa_best = _local_remove(sa_best)
                sa_size = len(sa_best)
                if sa_size < best_size and _is_valid(sa_best) and _is_maximal(sa_best):
                    best_set = sa_best
                    best_size = sa_size

        # Early termination if very small
        if best_size <= 10:
            break

    return best_set if best_set is not None else baseline
