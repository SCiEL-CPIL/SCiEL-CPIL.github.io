"""Minimum maximal no-isosceles set via simulated annealing."""
import itertools
import random
import time

N = 16
DEADLINE = 320.0  # safety margin before 330s kill

# Precompute all grid points
ALL_POINTS = [(x, y) for x in range(N) for y in range(N)]
NPOINTS = N * N


def _dist_sq(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _creates_iso(p, chosen):
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


def _is_valid(chosen):
    """Check that no three points in chosen form an isosceles triangle."""
    n = len(chosen)
    for i in range(n):
        a = chosen[i]
        for j in range(i + 1, n):
            b = chosen[j]
            d_ab = _dist_sq(a, b)
            for k in range(j + 1, n):
                c = chosen[k]
                d_ac = _dist_sq(a, c)
                d_bc = _dist_sq(b, c)
                if d_ab == d_ac or d_ab == d_bc or d_ac == d_bc:
                    return False
    return True


def _is_maximal(chosen):
    """Check that no grid point can be added without creating an isosceles triangle."""
    chosen_set = set(chosen)
    for p in ALL_POINTS:
        if p in chosen_set:
            continue
        if not _creates_iso(p, chosen):
            return False
    return True


def _greedy_construct(order=None):
    """Greedy construction in given order (or row-major)."""
    if order is None:
        order = list(ALL_POINTS)
    chosen = []
    for p in order:
        if not _creates_iso(p, chosen):
            chosen.append(p)
    return chosen


def _shrink_for_minimality(chosen):
    """Try to remove points that are redundant for maximality.
    A point q is redundant if removing it still leaves the set maximal."""
    chosen = list(chosen)
    improved = True
    while improved:
        improved = False
        for idx in range(len(chosen) - 1, -1, -1):
            candidate = chosen[idx]
            trial = chosen[:idx] + chosen[idx + 1:]
            if not trial:
                continue
            # Check if trial is still valid (no isosceles)
            if not _is_valid(trial):
                continue
            # Check if trial is still maximal
            if _is_maximal(trial):
                chosen = trial
                improved = True
                break  # restart scan since list changed
    return chosen


def construct():
    start_time = time.time()
    
    # Start with greedy construction
    current = _greedy_construct()
    current = _shrink_for_minimality(current)
    best = list(current)
    best_size = len(best)
    
    if time.time() - start_time > DEADLINE:
        return best
    
    # Simulated annealing: try random swaps and insertions/deletions
    random.seed(42)
    temp = 10.0
    cooling = 0.995
    min_temp = 0.1
    step_count = 0
    
    while temp > min_temp and time.time() - start_time < DEADLINE:
        step_count += 1
        
        # Generate a neighbor
        r = random.random()
        if r < 0.4:
            # Try to remove a random point and re-shrink
            if len(current) > 1:
                idx = random.randint(0, len(current) - 1)
                trial = current[:idx] + current[idx + 1:]
                if _is_valid(trial) and _is_maximal(trial):
                    trial = _shrink_for_minimality(trial)
                    if len(trial) <= len(current):
                        current = trial
                        if len(current) < best_size:
                            best = list(current)
                            best_size = len(best)
                            temp = min(temp * 1.1, 15.0)  # reward improvement
                else:
                    # Reject but maybe accept with probability
                    if random.random() < temp / max(1, len(current)):
                        pass  # don't accept invalid
        elif r < 0.7:
            # Try a random swap: remove one, add another
            if len(current) > 1:
                remove_idx = random.randint(0, len(current) - 1)
                # Find a point we can add
                chosen_set = set(current)
                candidates = [p for p in ALL_POINTS if p not in chosen_set]
                if candidates:
                    add_p = random.choice(candidates)
                    trial = current[:remove_idx] + current[remove_idx + 1:]
                    if not _creates_iso(add_p, trial):
                        trial.append(add_p)
                        if _is_maximal(trial):
                            trial = _shrink_for_minimality(trial)
                            if len(trial) <= len(current):
                                current = trial
                                if len(current) < best_size:
                                    best = list(current)
                                    best_size = len(best)
                                    temp = min(temp * 1.1, 15.0)
        else:
            # Restart with random order greedy
            order = list(ALL_POINTS)
            random.shuffle(order)
            trial = _greedy_construct(order)
            trial = _shrink_for_minimality(trial)
            if len(trial) < len(current):
                current = trial
                if len(current) < best_size:
                    best = list(current)
                    best_size = len(best)
                    temp = min(temp * 1.1, 15.0)
            elif len(trial) == len(current) and random.random() < 0.1:
                # Accept equal-size with small probability for diversity
                current = trial
        
        temp *= cooling
        
        # Periodic check
        if step_count % 500 == 0 and time.time() - start_time > DEADLINE - 30:
            break
    
    # Final shrink on best
    best = _shrink_for_minimality(best)
    
    return best
