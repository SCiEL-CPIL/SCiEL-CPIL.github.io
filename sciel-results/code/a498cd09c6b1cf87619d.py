"""Rewritten program: optimize 50 circle centres in the unit disk via Nelder-Mead."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _score(points):
    """Return the packing radius for a set of centres (higher is better)."""
    if points.shape[0] != N:
        return 0.0
    # Check finite
    if not np.all(np.isfinite(points)):
        return 0.0
    # Boundary constraint: each centre must be inside unit circle
    dists_to_origin = np.sqrt(np.sum(points ** 2, axis=1))
    boundary_margin = 1.0 - np.max(dists_to_origin)
    if boundary_margin <= 0:
        return 0.0
    # Pairwise distances
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    if min_pair <= 0:
        return 0.0
    return min(min_pair / 2.0, boundary_margin)


def _objective(x):
    """Negative packing radius to be minimized."""
    pts = x.reshape(N, 2)
    # Penalize points outside the disk heavily
    r = np.sqrt(np.sum(pts ** 2, axis=1))
    outside = np.maximum(0.0, r - 1.0)
    if np.any(outside > 0):
        return 1.0 + np.sum(outside ** 2)
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    return -min_pair / 2.0


def _hex_init():
    """Generate a hexagonal lattice initial configuration for N points."""
    # Use a hexagonal lattice with spacing chosen to fit N points in unit disk
    # Try several spacings and pick one that gives at least N points inside
    best_pts = None
    for trial in range(50):
        spacing = 2.0 / (np.sqrt(N) + 1.0 + 0.02 * trial)
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x ** 2 + y ** 2 <= (1.0 - spacing / 2.0) ** 2:
                    rows.append((x, y))
        pts = np.array(rows)
        if len(pts) >= N:
            # Sort by radius and take first N
            radii = np.sqrt(np.sum(pts ** 2, axis=1))
            idx = np.argsort(radii, kind="stable")[:N]
            candidate = pts[idx]
            if best_pts is None or _score(candidate) > _score(best_pts):
                best_pts = candidate
    if best_pts is not None:
        return best_pts
    # Fallback: random
    return _random_init()


def _random_init(seed=None):
    """Generate random initial configuration."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((N, 2))
    for i in range(N):
        while True:
            x = rng.uniform(-1, 1)
            y = rng.uniform(-1, 1)
            if x ** 2 + y ** 2 <= 0.9 ** 2:
                pts[i] = [x, y]
                break
    return pts


def construct():
    """Construct centres of N equal circles inside the unit circle."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit

    # Start with hexagonal lattice initialization
    pts = _hex_init()
    best_score = _score(pts)
    best_pts = pts.copy()

    # Try multiple Nelder-Mead optimizations with different initializations
    attempts = 0
    max_attempts = 30

    while time.time() < deadline and attempts < max_attempts:
        attempts += 1

        # For first attempt use hex init, subsequent use perturbed versions
        if attempts == 1:
            x0 = best_pts.flatten()
        elif attempts <= 5:
            # Small perturbation of best solution
            x0 = best_pts.flatten() + np.random.default_rng(attempts).normal(0, 0.05, size=N * 2)
            # Clamp to disk
            for i in range(N):
                x, y = x0[2 * i], x0[2 * i + 1]
                r = np.sqrt(x ** 2 + y ** 2)
                if r > 0.95:
                    scale = 0.95 / r
                    x0[2 * i] = x * scale
                    x0[2 * i + 1] = y * scale
        else:
            # Random restart
            x0 = _random_init(seed=attempts).flatten()

        # Run Nelder-Mead
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'maxfev': 10000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
        except Exception:
            continue

        if time.time() > deadline:
            break

        x_opt = result.x.reshape(N, 2)
        # Ensure points are inside disk
        r = np.sqrt(np.sum(x_opt ** 2, axis=1))
        for i in range(N):
            if r[i] > 0.99:
                scale = 0.99 / r[i]
                x_opt[i] *= scale

        score = _score(x_opt)
        if score > best_score:
            best_score = score
            best_pts = x_opt.copy()

        # Check if we've reached a good solution
        if best_score >= 0.1258:
            break

    # Final validation
    final_score = _score(best_pts)
    if final_score <= 0:
        # Fallback to hex init
        best_pts = _hex_init()

    return best_pts

# EVOLVE-BLOCK-END
