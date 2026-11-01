"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Best known packing radius
    best_known_r = 0.1258254895304

    # Start with a hexagonal lattice arrangement as initial guess
    points = _hexagonal_init(N)
    best_points = points.copy()
    best_score = _score(points)

    # Simulated annealing / gradient-based optimization
    points = _optimize(points, deadline)
    score = _score(points)
    if score > best_score:
        best_points = points.copy()
        best_score = score

    # Final polish with smaller steps
    points = _polish(points, deadline)
    score = _score(points)
    if score > best_score:
        best_points = points.copy()
        best_score = score

    # Ensure all points are strictly inside the unit circle
    dists = np.linalg.norm(best_points, axis=1)
    mask = dists < 1.0
    if np.any(~mask):
        # Clamp points that are outside
        for i in range(len(best_points)):
            if dists[i] >= 1.0:
                scale = (1.0 - 1e-10) / max(dists[i], 1e-10)
                best_points[i] *= scale

    # Final check: ensure valid output
    if not np.all(np.isfinite(best_points)) or best_points.shape != (N, 2):
        # Fallback to hexagonal
        return _hexagonal_init(N)

    return best_points


def _hexagonal_init(n):
    """Generate n points in a hexagonal lattice pattern inside unit circle."""
    # Try different spacings to find one that gives at least n points
    for attempt in range(100):
        spacing = 0.25 - attempt * 0.001
        if spacing <= 0:
            break
        dx = spacing
        dy = spacing * np.sqrt(3) / 2
        points = []
        # Generate hexagonal lattice points
        for j in range(-20, 21):
            for i in range(-20, 21):
                x = i * dx + (j % 2) * dx / 2
                y = j * dy
                if np.sqrt(x*x + y*y) <= 1.0 - spacing/2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= n:
            # Select the n points closest to center
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:n]
            return points[idx]
    # Fallback: uniform random
    return np.random.randn(n, 2) * 0.5


def _score(points):
    """Compute the score: largest common radius for non-overlapping circles inside unit circle."""
    n = len(points)
    if n < 2:
        return 0.0
    # Minimum distance between any two centers / 2
    min_dist = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_dist:
                min_dist = d
    r1 = min_dist / 2.0
    # Minimum distance from any center to boundary
    dists = np.linalg.norm(points, axis=1)
    r2 = 1.0 - np.max(dists)
    return min(r1, r2)


def _optimize(points, deadline):
    """Simulated annealing optimization."""
    n = len(points)
    current = points.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score

    # Initial temperature based on current score
    T = 0.01
    T_min = 1e-8
    cooling = 0.999

    step_size = 0.05
    n_iter = 0

    while time.time() < deadline - 10.0:
        # Random perturbation
        delta = np.random.randn(n, 2) * step_size
        new_points = current + delta

        # Keep points inside unit circle
        dists = np.linalg.norm(new_points, axis=1)
        for i in range(n):
            if dists[i] > 1.0:
                scale = 1.0 / dists[i]
                new_points[i] *= scale

        new_score = _score(new_points)

        # Acceptance criterion
        if new_score >= current_score or np.random.random() < np.exp((new_score - current_score) / max(T, T_min)):
            current = new_points
            current_score = new_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score

        T *= cooling
        if T < T_min:
            T = T_min
            step_size *= 0.95

        n_iter += 1

    return best


def _polish(points, deadline):
    """Final high-precision polish pass."""
    n = len(points)
    current = points.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score

    step_size = 0.01
    T = 0.001
    T_min = 1e-10
    cooling = 0.999

    while time.time() < deadline - 5.0:
        delta = np.random.randn(n, 2) * step_size
        new_points = current + delta

        dists = np.linalg.norm(new_points, axis=1)
        for i in range(n):
            if dists[i] > 1.0:
                scale = 1.0 / dists[i]
                new_points[i] *= scale

        new_score = _score(new_points)

        if new_score >= current_score or np.random.random() < np.exp((new_score - current_score) / max(T, T_min)):
            current = new_points
            current_score = new_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score

        T *= cooling
        if T < T_min:
            T = T_min
            step_size *= 0.98

    return best

# EVOLVE-BLOCK-END
