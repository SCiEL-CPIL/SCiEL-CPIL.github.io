"""Rewritten program: local-search circle packing in the unit square."""
import time
import math
import numpy as np

N = 26
DEADLINE = 115.0  # safety margin before the 120 s kill


def _max_radius(x, y, other_idx, xs, ys, rs):
    """Largest radius at (x, y) given other circles and square walls."""
    r = min(x, 1.0 - x, y, 1.0 - y)
    for j in other_idx:
        d = math.hypot(x - xs[j], y - ys[j])
        r = min(r, d - rs[j])
    return max(r, 1e-9)


def _score(xs, ys, rs):
    return float(np.sum(rs))


def _validate(xs, ys, rs, tol=1e-7):
    if len(xs) != N:
        return False
    for i in range(N):
        if xs[i] < -tol or xs[i] > 1.0 + tol or ys[i] < -tol or ys[i] > 1.0 + tol:
            return False
        if rs[i] < 0:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            d = math.hypot(xs[i] - xs[j], ys[i] - ys[j])
            if d < rs[i] + rs[j] - tol:
                return False
    return True


def _clamp(x, y, r):
    x = min(max(x, r), 1.0 - r)
    y = min(max(y, r), 1.0 - r)
    return x, y


def _local_search(xs, ys, rs, deadline):
    """Iterative improvement: move each circle to maximize its own radius."""
    best_score = _score(xs, ys, rs)
    n = len(xs)
    while time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            old_r = rs[i]
            # Try moving circle i in small steps; pick best radius
            # Use a few random perturbations plus a local grid search
            best_r = old_r
            best_x, best_y = xs[i], ys[i]
            # Random perturbations
            for _ in range(20):
                dx = np.random.uniform(-0.05, 0.05)
                dy = np.random.uniform(-0.05, 0.05)
                nx, ny = xs[i] + dx, ys[i] + dy
                # Clamp to feasible region given current radius
                nx, ny = _clamp(nx, ny, old_r)
                other = [j for j in range(n) if j != i]
                new_r = _max_radius(nx, ny, other, xs, ys, rs)
                if new_r > best_r + 1e-12:
                    best_r = new_r
                    best_x, best_y = nx, ny
            if best_r > old_r + 1e-12:
                xs[i], ys[i], rs[i] = best_x, best_y, best_r
                improved = True
                new_score = _score(xs, ys, rs)
                if new_score > best_score:
                    best_score = new_score
        if not improved:
            # Try larger random jumps to escape local optima
            for _ in range(50):
                if time.time() >= deadline:
                    break
                i = np.random.randint(n)
                old_x, old_y, old_r = xs[i], ys[i], rs[i]
                nx = np.random.uniform(0.05, 0.95)
                ny = np.random.uniform(0.05, 0.95)
                nx, ny = _clamp(nx, ny, old_r)
                other = [j for j in range(n) if j != i]
                new_r = _max_radius(nx, ny, other, xs, ys, rs)
                if new_r > old_r + 1e-12:
                    xs[i], ys[i], rs[i] = nx, ny, new_r
                    improved = True
                    new_score = _score(xs, ys, rs)
                    if new_score > best_score:
                        best_score = new_score
            if not improved:
                break
    return xs, ys, rs, best_score


def _initial_grid():
    """Create an initial grid-based packing."""
    side = int(math.ceil(math.sqrt(N)))
    xs, ys, rs = [], [], []
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            xs.append(x)
            ys.append(y)
            rs.append(r)
            count += 1
    return np.array(xs), np.array(ys), np.array(rs)


def _initial_random():
    """Create a random initial packing with small circles."""
    xs, ys, rs = [], [], []
    for _ in range(N):
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        r = 0.02
        xs.append(x)
        ys.append(y)
        rs.append(r)
    return np.array(xs), np.array(ys), np.array(rs)


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    # Try multiple restarts with different initializations
    best_xs, best_ys, best_rs = None, None, None
    best_score = -1.0

    # Start with grid initialization
    xs, ys, rs = _initial_grid()
    # Run local search on grid
    xs, ys, rs, score = _local_search(xs, ys, rs, deadline)
    if score > best_score:
        best_score = score
        best_xs, best_ys, best_rs = xs.copy(), ys.copy(), rs.copy()

    # Try several random restarts
    while time.time() < deadline - 5.0:
        xs, ys, rs = _initial_random()
        xs, ys, rs, score = _local_search(xs, ys, rs, deadline)
        if score > best_score:
            best_score = score
            best_xs, best_ys, best_rs = xs.copy(), ys.copy(), rs.copy()

    # Final validation
    if best_xs is not None:
        if not _validate(best_xs, best_ys, best_rs):
            # Fallback to grid if validation fails
            xs, ys, rs = _initial_grid()
            best_xs, best_ys, best_rs = xs, ys, rs

    result = np.column_stack([best_xs, best_ys, best_rs])
    return result
