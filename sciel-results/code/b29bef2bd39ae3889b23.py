"""Rewritten program: local search from a perturbed grid initialisation."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.spatial import cKDTree

_DEADLINE = 140.0  # wall-clock guard (seconds), leaving margin under 150s kill.


def _radius(pts):
    """Compute the packing radius for centres pts (N,2) in [-0.5,0.5]^2."""
    n = pts.shape[0]
    if n == 0:
        return 0.0
    # Distance to walls: min over coordinates of (0.5 - |x|)
    wall = 0.5 - np.max(np.abs(pts), axis=1)
    r_wall = float(np.min(wall))
    if r_wall <= 0:
        return 0.0
    # Pairwise distances
    tree = cKDTree(pts)
    # Query all pairs (excluding self)
    dists, _ = tree.query(pts, k=2)
    min_pair = float(np.min(dists[:, 1]))
    return min(r_wall, min_pair / 2.0)


def _initial_grid(n):
    """Create a near-square grid initialisation inside [-0.5, 0.5]^2."""
    m = int(math.ceil(math.sqrt(n)))
    # We need exactly n points; use m x m grid and take first n
    xs = (np.arange(m) + 0.5) / m - 0.5
    ys = (np.arange(m) + 0.5) / m - 0.5
    pts = np.array([(x, y) for y in ys for x in xs])[:n]
    return pts


def _perturb(pts, sigma=0.01):
    """Add small random perturbation to points."""
    return pts + np.random.randn(*pts.shape) * sigma


def _local_search(pts, max_iter=20000, step_init=0.005, step_min=1e-6,
                  time_limit=None, deadline=None):
    """
    Simple hill-climbing local search.
    At each step, pick a random point, try moving it in a random direction
    by a small step, accept if radius improves.
    """
    best = _radius(pts)
    current = pts.copy()
    step = step_init
    n = len(current)
    t_start = time.time()

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        if time_limit is not None and (time.time() - t_start) > time_limit:
            break

        # Pick a random point
        idx = np.random.randint(n)
        # Random direction
        angle = np.random.uniform(0, 2 * math.pi)
        dx = step * math.cos(angle)
        dy = step * math.sin(angle)

        new_pt = current[idx].copy()
        new_pt[0] += dx
        new_pt[1] += dy

        # Keep inside square
        new_pt[0] = np.clip(new_pt[0], -0.5, 0.5)
        new_pt[1] = np.clip(new_pt[1], -0.5, 0.5)

        trial = current.copy()
        trial[idx] = new_pt
        r = _radius(trial)

        if r > best:
            best = r
            current = trial
            # Occasionally increase step slightly if improving
            step = min(step * 1.05, 0.01)
        else:
            # Decrease step
            step = max(step * 0.99, step_min)

    return current, best


def _simulated_annealing(pts, max_iter=50000, T_init=0.002, T_min=1e-6,
                         deadline=None):
    """
    Simulated annealing: accept worse moves with probability exp(-delta/T).
    """
    best = _radius(pts)
    current = pts.copy()
    T = T_init
    n = len(current)
    t_start = time.time()

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        if (time.time() - t_start) > 120:
            break

        idx = np.random.randint(n)
        angle = np.random.uniform(0, 2 * math.pi)
        step = 0.003
        dx = step * math.cos(angle)
        dy = step * math.sin(angle)

        new_pt = current[idx].copy()
        new_pt[0] = np.clip(new_pt[0] + dx, -0.5, 0.5)
        new_pt[1] = np.clip(new_pt[1] + dy, -0.5, 0.5)

        trial = current.copy()
        trial[idx] = new_pt
        r = _radius(trial)

        if r > best:
            best = r
            current = trial
        else:
            delta = best - r
            if delta < 0 or np.random.rand() < math.exp(-delta / max(T, 1e-12)):
                current = trial

        # Cool
        T = max(T * 0.999, T_min)

    return current, best


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE

    np.random.seed(42)

    # Start with grid
    pts = _initial_grid(N)
    best_pts = pts.copy()
    best_r = _radius(pts)

    # Phase 1: Local search from grid
    pts, r = _local_search(pts, max_iter=30000, step_init=0.004,
                           step_min=1e-6, deadline=deadline)
    if r > best_r:
        best_r = r
        best_pts = pts.copy()

    if time.time() > deadline - 30:
        return best_pts

    # Phase 2: Simulated annealing from best
    pts2, r2 = _simulated_annealing(best_pts, max_iter=40000,
                                    T_init=0.0015, T_min=1e-6,
                                    deadline=deadline)
    if r2 > best_r:
        best_r = r2
        best_pts = pts2.copy()

    if time.time() > deadline - 20:
        return best_pts

    # Phase 3: Final local search polish
    pts3, r3 = _local_search(best_pts, max_iter=20000, step_init=0.002,
                             step_min=1e-7, deadline=deadline)
    if r3 > best_r:
        best_r = r3
        best_pts = pts3.copy()

    # Ensure all points are strictly inside the square
    best_pts = np.clip(best_pts, -0.5, 0.5)

    return best_pts
# EVOLVE-BLOCK-END
