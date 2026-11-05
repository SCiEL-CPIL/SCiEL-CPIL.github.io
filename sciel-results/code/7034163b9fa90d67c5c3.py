"""Two-square symmetric model + multi-phase polar/Cartesian optimization on exact objective."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 8

def _compute_score(pts):
    """Scale-invariant score: prod(ordered-pair distances) / diameter^(n(n-1))."""
    pts = np.asarray(pts, dtype=float)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(N, dtype=bool)
    d = dists[mask]
    if np.any(d < 1e-14):
        return -np.inf
    diameter = np.max(d)
    if diameter < 1e-14:
        return -np.inf
    log_prod = np.sum(np.log(d))
    log_score = log_prod - N * (N - 1) * np.log(diameter)
    return np.exp(log_score)

def _two_square_points(params):
    """Build 8 points from 4 parameters: two squares, one rotated by pi/4.
    params = [r1, r2, theta, phi]
    Square 1: radius r1, rotation theta, 4 points at theta + k*pi/2
    Square 2: radius r2, rotation phi, 4 points at phi + k*pi/2
    """
    r1 = abs(params[0])
    r2 = abs(params[1])
    theta = params[2]
    phi = params[3]
    k = np.arange(4)
    a1 = theta + k * (np.pi / 2)
    a2 = phi + k * (np.pi / 2)
    pts = np.zeros((8, 2))
    pts[:4, 0] = r1 * np.cos(a1)
    pts[:4, 1] = r1 * np.sin(a1)
    pts[4:, 0] = r2 * np.cos(a2)
    pts[4:, 1] = r2 * np.sin(a2)
    return pts

def _neg_log_score_two_square(params):
    """Negative log score for the two-square parameterization."""
    try:
        pts = _two_square_points(params)
        sc = _compute_score(pts)
        if sc <= 0 or not np.isfinite(sc):
            return 1e10
        return -np.log(sc)
    except Exception:
        return 1e10

def _neg_log_score_cartesian(x):
    """Negative log score for raw 16-coordinate parameterization."""
    try:
        pts = x.reshape(N, 2)
        sc = _compute_score(pts)
        if sc <= 0 or not np.isfinite(sc):
            return 1e10
        return -np.log(sc)
    except Exception:
        return 1e10

def _scale_to_diameter2(pts):
    pts = np.asarray(pts, dtype=float)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(N, dtype=bool)
    diameter = np.max(dists[mask])
    if diameter > 1e-14:
        pts = pts * (2.0 / diameter)
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    best_pts = None
    best_score = -np.inf

    def _update_best(pts, sc):
        nonlocal best_pts, best_score
        if sc > best_score and np.isfinite(sc):
            best_pts = np.array(pts, dtype=float).copy()
            best_score = sc

    # Initialize with regular octagon
    angles = 2 * np.pi * np.arange(N) / N
    octagon = np.column_stack([np.cos(angles), np.sin(angles)])
    _update_best(octagon, _compute_score(octagon))

    # ---- Phase A: Optimize the two-square symmetric model (4 DOF) ----
    # The two-square family is known to beat the regular octagon.
    # Try a grid of starting parameters around the symmetric point.
    two_square_inits = []
    # Symmetric: r1=1, r2=1, theta=0, phi=pi/4 (this IS the octagon)
    two_square_inits.append([1.0, 1.0, 0.0, np.pi / 4])
    # Perturb radii
    for r2 in [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3]:
        two_square_inits.append([1.0, r2, 0.0, np.pi / 4])
    # Perturb rotation
    for dphi in [0.05, 0.1, 0.2, -0.05, -0.1, -0.2]:
        two_square_inits.append([1.0, 1.0, 0.0, np.pi / 4 + dphi])
    # Combined perturbations
    rng = np.random.default_rng(42)
    for _ in range(20):
        r1 = rng.uniform(0.6, 1.4)
        r2 = rng.uniform(0.4, 1.5)
        theta = rng.uniform(-0.5, 0.5)
        phi = np.pi / 4 + rng.uniform(-0.5, 0.5)
        two_square_inits.append([r1, r2, theta, phi])

    for init in two_square_inits:
        if time.time() > deadline - 30:
            break
        try:
            result = minimize(
                _neg_log_score_two_square,
                init,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'gtol': 1e-14, 'ftol': 1e-14}
            )
            if result.fun < 1e8:
                pts = _two_square_points(result.x)
                sc = _compute_score(pts)
                _update_best(pts, sc)
        except Exception:
            pass

    # ---- Phase B: Nelder-Mead polish on two-square model ----
    if time.time() < deadline - 25:
        try:
            result = minimize(
                _neg_log_score_two_square,
                best_pts and _two_square_params_from_pts(best_pts) or [1.0, 1.0, 0.0, np.pi/4],
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            if result.fun < 1e8:
                pts = _two_square_points(result.x)
                sc = _compute_score(pts)
                _update_best(pts, sc)
        except Exception:
            pass

    # ---- Phase C: Full Cartesian refinement from best two-square result ----
    if best_pts is not None and time.time() < deadline - 20:
        try:
            result = minimize(
                _neg_log_score_cartesian,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 8000, 'gtol': 1e-14, 'ftol': 1e-14}
            )
            if result.fun < 1e8:
                pts = result.x.reshape(N, 2)
                sc = _compute_score(pts)
                _update_best(pts, sc)
        except Exception:
            pass

    # ---- Phase D: Nelder-Mead in Cartesian for fine polish ----
    if best_pts is not None and time.time() < deadline - 15:
        try:
            result = minimize(
                _neg_log_score_cartesian,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            if result.fun < 1e8:
                pts = result.x.reshape(N, 2)
                sc = _compute_score(pts)
                _update_best(pts, sc)
        except Exception:
            pass

    # ---- Phase E: Random restarts in Cartesian ----
    rng2 = np.random.default_rng(123)
    for restart in range(30):
        if time.time() > deadline - 10:
            break
        if best_pts is not None:
            noise = rng2.normal(0, 0.15, size=(N, 2))
            x0 = best_pts + noise
        else:
            x0 = rng2.uniform(-1, 1, size=(N, 2))
        try:
            result = minimize(
                _neg_log_score_cartesian,
                x0.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 5000, 'gtol': 1e-14, 'ftol': 1e-14}
            )
            if result.fun < 1e8:
                pts = result.x.reshape(N, 2)
                sc = _compute_score(pts)
                _update_best(pts, sc)
        except Exception:
            pass

    # ---- Final: scale to diameter 2 and return ----
    if best_pts is None:
        best_pts = octagon

    pts = _scale_to_diameter2(best_pts)
    return [[float(p[0]), float(p[1])] for p in pts]

def _two_square_params_from_pts(pts):
    """Extract approximate two-square parameters from 8 points."""
    # Sort by radius
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    idx = np.argsort(radii)
    sorted_pts = pts[idx]
    # First 4 are inner square, last 4 are outer
    inner = sorted_pts[:4]
    outer = sorted_pts[4:]
    r1 = np.mean(np.sqrt(np.sum(inner ** 2, axis=1)))
    r2 = np.mean(np.sqrt(np.sum(outer ** 2, axis=1)))
    # Estimate rotation from angles
    a1 = np.arctan2(inner[:, 1], inner[:, 0])
    a2 = np.arctan2(outer[:, 1], outer[:, 0])
    theta = np.mean(a1)
    phi = np.mean(a2)
    return [r1, r2, theta, phi]
# EVOLVE-BLOCK-END
