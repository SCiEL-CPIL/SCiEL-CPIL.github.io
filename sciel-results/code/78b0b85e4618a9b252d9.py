import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _score(points):
    n = len(points)
    pts = np.array(points, dtype=np.float64)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(n, dtype=bool)
    dists = dists[mask]
    diameter = np.max(dists)
    if diameter < 1e-12:
        return -np.inf
    log_prod = np.sum(np.log(dists))
    log_score = log_prod - n * (n - 1) * np.log(diameter)
    return np.exp(log_score)

def _neg_log_score_from_polar(polar_flat):
    """Convert polar (r, theta) pairs to Cartesian and compute negative log score."""
    n = 8
    polar = polar_flat.reshape(n, 2)
    # Handle negative radii by flipping angle
    radii = polar[:, 0]
    thetas = polar[:, 1]
    # Clamp radii to be non-negative for stability
    radii = np.abs(radii)
    pts = np.column_stack([radii * np.cos(thetas), radii * np.sin(thetas)])
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(n, dtype=bool)
    dists = dists[mask]
    diameter = np.max(dists)
    if diameter < 1e-12:
        return 1e10
    log_prod = np.sum(np.log(dists))
    log_score = log_prod - n * (n - 1) * np.log(diameter)
    return -log_score

def _neg_log_score_from_cartesian(cart_flat):
    n = 8
    pts = cart_flat.reshape(n, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(n, dtype=bool)
    dists = dists[mask]
    diameter = np.max(dists)
    if diameter < 1e-12:
        return 1e10
    log_prod = np.sum(np.log(dists))
    log_score = log_prod - n * (n - 1) * np.log(diameter)
    return -log_score

def _cartesian_to_polar(pts):
    """Convert Cartesian points to polar (r, theta)."""
    x = pts[:, 0]
    y = pts[:, 1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return np.column_stack([r, theta]).flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    # Regular octagon as baseline
    angles = 2 * np.pi * np.arange(8) / 8
    octagon = np.column_stack([np.cos(angles), np.sin(angles)])
    best_pts = octagon.copy()
    best_score = _score(octagon)

    # Convert to polar for structured optimization
    best_polar = _cartesian_to_polar(octagon)

    # Phase 1: Differential Evolution in polar coordinates (global exploration)
    # Bounds: r in [0.1, 3.0], theta in [0, 2*pi]
    bounds = [(0.1, 3.0)] * 8 + [(0, 2 * np.pi)] * 8

    try:
        if time.time() < deadline - 60:
            de_result = differential_evolution(
                _neg_log_score_from_polar,
                bounds=bounds,
                seed=42,
                maxiter=200,
                tol=1e-8,
                popsize=15,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube',
                updating='immediate'
            )
            if de_result.fun < _neg_log_score_from_polar(best_polar):
                candidate_polar = de_result.x
                # Convert to Cartesian
                polar = candidate_polar.reshape(8, 2)
                radii = np.abs(polar[:, 0])
                thetas = polar[:, 1]
                candidate_cart = np.column_stack([radii * np.cos(thetas), radii * np.sin(thetas)])
                cand_score = _score(candidate_cart)
                if cand_score > best_score:
                    best_pts = candidate_cart
                    best_score = cand_score
                    best_polar = candidate_polar
    except Exception:
        pass

    # Phase 2: L-BFGS-B local refinement from best polar configuration
    if time.time() < deadline - 20:
        try:
            res = minimize(
                _neg_log_score_from_polar,
                best_polar,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < _neg_log_score_from_polar(best_polar):
                candidate_polar = res.x
                polar = candidate_polar.reshape(8, 2)
                radii = np.abs(polar[:, 0])
                thetas = polar[:, 1]
                candidate_cart = np.column_stack([radii * np.cos(thetas), radii * np.sin(thetas)])
                cand_score = _score(candidate_cart)
                if cand_score > best_score:
                    best_pts = candidate_cart
                    best_score = cand_score
                    best_polar = candidate_polar
        except Exception:
            pass

    # Phase 3: L-BFGS-B in Cartesian coordinates (different parameterization)
    if time.time() < deadline - 10:
        try:
            cart_flat = best_pts.flatten()
            res = minimize(
                _neg_log_score_from_cartesian,
                cart_flat,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < _neg_log_score_from_cartesian(cart_flat):
                candidate_cart = res.x.reshape(8, 2)
                cand_score = _score(candidate_cart)
                if cand_score > best_score:
                    best_pts = candidate_cart
                    best_score = cand_score
        except Exception:
            pass

    # Phase 4: Multiple perturbation + L-BFGS-B restarts in polar space
    if time.time() < deadline - 5:
        rng = np.random.default_rng(123)
        for i in range(20):
            if time.time() > deadline - 5:
                break
            # Perturb in polar space
            noise_r = rng.normal(0, 0.2, size=8)
            noise_t = rng.normal(0, 0.3, size=8)
            perturbed_polar = best_polar.copy()
            perturbed_polar[:8] = np.abs(perturbed_polar[:8] + noise_r)
            perturbed_polar[8:] = (perturbed_polar[8:] + noise_t) % (2 * np.pi)
            # Clamp
            perturbed_polar[:8] = np.clip(perturbed_polar[:8], 0.1, 3.0)

            try:
                res = minimize(
                    _neg_log_score_from_polar,
                    perturbed_polar,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < _neg_log_score_from_polar(best_polar):
                    candidate_polar = res.x
                    polar = candidate_polar.reshape(8, 2)
                    radii = np.abs(polar[:, 0])
                    thetas = polar[:, 1]
                    candidate_cart = np.column_stack([radii * np.cos(thetas), radii * np.sin(thetas)])
                    cand_score = _score(candidate_cart)
                    if cand_score > best_score:
                        best_pts = candidate_cart
                        best_score = cand_score
                        best_polar = candidate_polar
            except Exception:
                pass

    # Scale to diameter 2
    pts = np.array(best_pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(8, dtype=bool)
    diameter = np.max(dists[mask])
    if diameter > 1e-12:
        pts = pts * (2.0 / diameter)

    return [[float(p[0]), float(p[1])] for p in pts]
