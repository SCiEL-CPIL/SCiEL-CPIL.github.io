"""Program for minimizing max-to-min pairwise distance ratio for 16 points in 2D."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 16
D = 2


def _pairwise_distances(pts):
    """Compute all pairwise Euclidean distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def _score(pts):
    """Compute (max/min)^2 for pairwise distances."""
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    dmin = d.min()
    dmax = d.max()
    if dmin <= 1e-12:
        return np.inf
    return (dmax / dmin) ** 2


def _objective(pts_flat):
    """Objective function: (max/min)^2."""
    pts = pts_flat.reshape(N, 2)
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    dmin = d.min()
    dmax = d.max()
    if dmin <= 1e-12:
        return 1e10
    return (dmax / dmin) ** 2


def _make_hexagonal_cluster(n, spread=6):
    """Generate a hexagonal lattice cluster of n points centered at origin."""
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])

    pts = []
    for i in range(-spread, spread + 1):
        for j in range(-spread, spread + 1):
            p = i * v1 + j * v2
            pts.append(p)
    pts = np.array(pts)

    # Center the cluster
    pts -= pts.mean(axis=0)

    # Sort by distance to center and pick the closest n
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    return pts[order[:n]]


def _make_circular_config(n, radius=1.0):
    """Points equally spaced on a circle, plus center if odd, or two concentric rings."""
    if n == 16:
        # 1 on center + 7 on inner ring + 8 on outer ring
        pts = [np.array([0.0, 0.0])]
        r1 = 1.0
        r2 = 2.0
        for k in range(7):
            theta = 2 * np.pi * k / 7
            pts.append(r1 * np.array([np.cos(theta), np.sin(theta)]))
        for k in range(8):
            theta = 2 * np.pi * k / 8 + np.pi / 8
            pts.append(r2 * np.array([np.cos(theta), np.sin(theta)]))
        return np.array(pts)
    # Generic fallback
    angles = 2 * np.pi * np.arange(n) / n
    return radius * np.column_stack([np.cos(angles), np.sin(angles)])


def _make_grid_config(n):
    """4x4 grid, centered."""
    x, y = np.meshgrid(np.arange(4), np.arange(4))
    pts = np.column_stack([x.ravel(), y.ravel()]).astype(float)
    pts -= pts.mean(axis=0)
    return pts


def _make_random_config(n, seed=0):
    """Random points in unit square, centered."""
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2))
    pts -= pts.mean(axis=0)
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    best_pts = None
    best_score = np.inf

    def update_best(pts, score):
        nonlocal best_pts, best_score
        if score < best_score:
            best_score = score
            best_pts = pts.copy()

    # 1. Generate diverse initial candidates
    candidates = [
        _make_hexagonal_cluster(N),
        _make_circular_config(N),
        _make_grid_config(N),
        _make_random_config(N, seed=1),
        _make_random_config(N, seed=2),
        _make_random_config(N, seed=3),
        _make_random_config(N, seed=4),
        _make_random_config(N, seed=5),
    ]

    # Evaluate all candidates
    for c in candidates:
        s = _score(c)
        update_best(c, s)

    # 2. Differential Evolution - global search in 32D parameter space
    if time.time() < deadline - 30:
        try:
            bounds = [(-5, 5)] * (N * 2)
            # Use bounded DE with limited iterations to stay within time budget
            result = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-4,
                polish=False,
                init='sobol',
            )
            if result.fun < best_score:
                new_pts = result.x.reshape(N, 2)
                new_score = _score(new_pts)
                update_best(new_pts, new_score)
        except Exception:
            pass

    # 3. Local refinement with Powell on best candidates
    opt_inits = []
    # Include top candidates from initial set
    scored_candidates = [( _score(c), c) for c in candidates]
    scored_candidates.sort(key=lambda x: x[0])
    for sc, c in scored_candidates[:3]:
        opt_inits.append(c)
    if best_pts is not None:
        opt_inits.append(best_pts)

    for init_pts in opt_inits:
        if time.time() > deadline - 15:
            break
        try:
            result = minimize(
                _objective,
                init_pts.ravel(),
                method='Powell',
                options={'maxiter': 2000, 'maxfev': 50000, 'xtol': 1e-10, 'ftol': 1e-10},
            )
            if result.fun < best_score:
                new_pts = result.x.reshape(N, 2)
                new_score = _score(new_pts)
                update_best(new_pts, new_score)
        except Exception:
            pass

    # 4. Nelder-Mead polish on the best result
    if best_pts is not None and time.time() < deadline - 5:
        try:
            result = minimize(
                _objective,
                best_pts.ravel(),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True},
            )
            if result.fun < best_score:
                new_pts = result.x.reshape(N, 2)
                new_score = _score(new_pts)
                update_best(new_pts, new_score)
        except Exception:
            pass

    # 5. Second round of DE with tighter bounds around best if we have time
    if best_pts is not None and time.time() < deadline - 20:
        try:
            center = best_pts.ravel()
            span = np.max(np.abs(center)) + 1.0
            tight_bounds = [(c - span, c + span) for c in center]
            result = differential_evolution(
                _objective,
                tight_bounds,
                seed=123,
                maxiter=40,
                popsize=12,
                tol=1e-5,
                polish=False,
                init='sobol',
            )
            if result.fun < best_score:
                new_pts = result.x.reshape(N, 2)
                new_score = _score(new_pts)
                update_best(new_pts, new_score)
        except Exception:
            pass

    # Final refinement
    if best_pts is not None and time.time() < deadline - 3:
        try:
            result = minimize(
                _objective,
                best_pts.ravel(),
                method='Powell',
                options={'maxiter': 3000, 'maxfev': 80000, 'xtol': 1e-12, 'ftol': 1e-12},
            )
            if result.fun < best_score:
                new_pts = result.x.reshape(N, 2)
                new_score = _score(new_pts)
                update_best(new_pts, new_score)
        except Exception:
            pass

    if best_pts is None:
        return _make_hexagonal_cluster(N)

    return best_pts
