"""
Strategy: Complex-polynomial root representation.
Represent the 8 points as the roots of a monic degree-8 complex polynomial
P(z) = z^8 + c7 z^7 + ... + c0, where z = x + i y.
The log-discriminant of P is proportional to log(product of pairwise |zi - zj|^2),
which is exactly 2 * log(product of pairwise distances). So maximizing the
log-discriminant (up to a constant) maximizes the target score.

We optimize the 16 real coefficients (Re/Im of each ck) to maximize
log_discriminant - penalty * (smooth_max_modulus - 2)^2, then extract the
roots and polish in Cartesian space.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill


def _log_score(points):
    """Compute log of Delta/n^n where Delta is product of distances over ordered pairs, after scaling to diameter 2."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    mask = ~np.eye(n, dtype=bool)
    valid_dists = dists[mask]
    if np.any(valid_dists <= 0):
        return -np.inf
    diameter = np.max(dists)
    if diameter <= 0:
        return -np.inf
    n_pairs = n * (n - 1)
    log_delta = n_pairs * np.log(2.0) + np.sum(np.log(valid_dists)) - n_pairs * np.log(diameter)
    log_score = log_delta - n * np.log(n)
    return log_score


def _score(points):
    ls = _log_score(points)
    if not np.isfinite(ls):
        return -np.inf
    if ls > 50:
        return float('inf')
    return math.exp(ls)


def _neg_log_score(params):
    if time.time() > _DEADLINE:
        return 1e10
    pts = params.reshape(8, 2)
    for i in range(8):
        for j in range(i + 1, 8):
            d = math.sqrt((pts[i][0] - pts[j][0])**2 + (pts[i][1] - pts[j][1])**2)
            if d < 1e-10:
                return 1e10
    ls = _log_score(pts)
    if not np.isfinite(ls):
        return 1e10
    return -ls


def _regular_octagon():
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _roots_to_points(roots):
    """Convert complex roots to (x, y) array."""
    return np.column_stack([np.real(roots), np.imag(roots)])


def _log_discriminant_from_coeffs(coeffs):
    """
    Compute log-discriminant of monic polynomial with given coefficients.
    coeffs: array of 8 complex numbers [c7, c6, ..., c0] for z^8 + c7*z^7 + ... + c0.
    Returns log(|Disc|) where Disc = product_{i<j} (ri - rj)^2 (up to sign).
    """
    # Build full coefficient array for numpy roots
    full_coeffs = np.concatenate([[1.0], np.conjugate(coeffs)])  # monic
    # Actually, numpy.roots expects [a_n, a_{n-1}, ..., a_0]
    # Our polynomial: z^8 + c7 z^7 + c6 z^6 + ... + c0
    # So full_coeffs = [1, c7, c6, ..., c0]
    full_coeffs = np.concatenate([[1.0 + 0j], coeffs])
    
    try:
        roots = np.roots(full_coeffs)
    except Exception:
        return -np.inf
    
    if not np.all(np.isfinite(roots)):
        return -np.inf
    
    # Check for real roots (points must be in the plane, but complex roots are fine
    # since we interpret them as (x,y) points)
    # Compute pairwise squared distances
    n = len(roots)
    diff = roots[:, np.newaxis] - roots[np.newaxis, :]
    sq_dists = np.abs(diff)**2
    mask = ~np.eye(n, dtype=bool)
    valid_sq = sq_dists[mask]
    
    if np.any(valid_sq <= 1e-30):
        return -np.inf
    
    # log discriminant = sum_{i<j} 2*log(|ri-rj|) = sum_{i<j} log(|ri-rj|^2)
    log_disc = np.sum(np.log(valid_sq))
    
    if not np.isfinite(log_disc):
        return -np.inf
    
    return log_disc


def _neg_poly_objective(params, beta=30.0):
    """
    Objective in coefficient space.
    params: 16 real numbers = [Re(c7), Im(c7), Re(c6), Im(c6), ..., Re(c0), Im(c0)]
    We maximize log_discriminant - penalty * (max_modulus - 2)^2.
    """
    if time.time() > _DEADLINE:
        return 1e10
    
    coeffs = params[0::2] + 1j * params[1::2]  # 8 complex coefficients
    
    full_coeffs = np.concatenate([[1.0 + 0j], coeffs])
    
    try:
        roots = np.roots(full_coeffs)
    except Exception:
        return 1e10
    
    if not np.all(np.isfinite(roots)):
        return 1e10
    
    # Pairwise squared distances
    n = len(roots)
    diff = roots[:, np.newaxis] - roots[np.newaxis, :]
    sq_dists = np.abs(diff)**2
    mask = ~np.eye(n, dtype=bool)
    valid_sq = sq_dists[mask]
    
    if np.any(valid_sq <= 1e-30):
        return 1e10
    
    log_disc = np.sum(np.log(valid_sq))
    
    if not np.isfinite(log_disc):
        return 1e10
    
    # Smooth max modulus constraint
    moduli = np.abs(roots)
    max_mod = np.max(moduli)
    # Soft constraint: penalize if max_mod > 2
    shifted = beta * (moduli - max_mod)
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    smooth_max = max_mod + log_sum_exp / beta
    
    penalty = 50.0 * (smooth_max - 2.0)**2
    
    # Also penalize very small pairwise distances (coincident roots)
    min_sq = np.min(valid_sq)
    small_penalty = 10.0 * np.maximum(0, 0.01 - min_sq)**2
    
    obj = -log_disc + penalty + small_penalty
    
    if not np.isfinite(obj):
        return 1e10
    
    return obj


def _octagon_coeffs():
    """Get coefficients of the polynomial whose roots are the regular octagon (scaled to diameter 2)."""
    pts = _regular_octagon()
    # Scale to diameter 2: regular octagon on unit circle has diameter 2*cos(pi/8)
    # Actually, diameter of unit circle octagon = 2 (opposite points at distance 2)
    # Wait: regular octagon on unit circle: max distance = 2 (diameter of circle)
    # So it's already at diameter 2.
    z = pts[:, 0] + 1j * pts[:, 1]
    # Build polynomial with these roots
    coeffs = np.poly(z)  # returns [1, c7, c6, ..., c0]
    # coeffs[1:] are c7, c6, ..., c0
    c = coeffs[1:]
    # Flatten to real params: [Re(c7), Im(c7), ..., Re(c0), Im(c0)]
    params = np.zeros(16)
    params[0::2] = np.real(c)
    params[1::2] = np.imag(c)
    return params


def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 140.0

    best_points = _regular_octagon()
    best_log_score = _log_score(best_points)

    rng = np.random.default_rng(42)

    # --- Phase 1: Optimize in coefficient space ---
    # Initial parameter vectors
    init_params = []

    # Regular octagon
    p_oct = _octagon_coeffs()
    init_params.append(p_oct)

    # Perturbed octagon coefficients
    for _ in range(30):
        p = p_oct + rng.normal(0, 0.3, 16)
        init_params.append(p)

    # Two-concentric-squares configurations
    for r1 in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
        for r2 in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
            if abs(r1 - r2) < 0.15:
                continue
            for rot1 in [0.0, np.pi/8]:
                for rot2 in [0.0, np.pi/8]:
                    pts = []
                    for k in range(4):
                        a = rot1 + 2*np.pi*k/4
                        pts.append([r1*np.cos(a), r1*np.sin(a)])
                    for k in range(4):
                        a = rot2 + 2*np.pi*k/4
                        pts.append([r2*np.cos(a), r2*np.sin(a)])
                    pts = np.array(pts)
                    # Scale to diameter 2
                    d = np.sqrt(((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2).sum(axis=2))
                    max_d = np.max(d)
                    if max_d > 0:
                        pts = pts / max_d * 2.0
                    z = pts[:, 0] + 1j * pts[:, 1]
                    coeffs = np.poly(z)
                    c = coeffs[1:]
                    p = np.zeros(16)
                    p[0::2] = np.real(c)
                    p[1::2] = np.imag(c)
                    init_params.append(p)

    # Antipodal pair configurations
    for _ in range(20):
        base_angles = rng.uniform(0, np.pi/2, 4)
        radii = rng.uniform(0.4, 1.5, 4)
        pts = []
        for i in range(4):
            a = base_angles[i]
            r = radii[i]
            pts.append([r*np.cos(a), r*np.sin(a)])
            pts.append([-r*np.cos(a), -r*np.sin(a)])
        pts = np.array(pts)
        d = np.sqrt(((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2).sum(axis=2))
        max_d = np.max(d)
        if max_d > 0:
            pts = pts / max_d * 2.0
        z = pts[:, 0] + 1j * pts[:, 1]
        coeffs = np.poly(z)
        c = coeffs[1:]
        p = np.zeros(16)
        p[0::2] = np.real(c)
        p[1::2] = np.imag(c)
        init_params.append(p)

    # Random points
    for _ in range(20):
        radii = rng.uniform(0.1, 1.8, 8)
        angles_r = rng.uniform(0, 2*np.pi, 8)
        pts = np.column_stack([radii*np.cos(angles_r), radii*np.sin(angles_r)])
        d = np.sqrt(((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2).sum(axis=2))
        max_d = np.max(d)
        if max_d > 0:
            pts = pts / max_d * 2.0
        z = pts[:, 0] + 1j * pts[:, 1]
        coeffs = np.poly(z)
        c = coeffs[1:]
        p = np.zeros(16)
        p[0::2] = np.real(c)
        p[1::2] = np.imag(c)
        init_params.append(p)

    # Score initializations by their log-discriminant (proxy for quality)
    scored_inits = []
    for p in init_params:
        if time.time() > _DEADLINE - 50:
            break
        try:
            coeffs = p[0::2] + 1j * p[1::2]
            full_coeffs = np.concatenate([[1.0 + 0j], coeffs])
            roots = np.roots(full_coeffs)
            if not np.all(np.isfinite(roots)):
                continue
            n = len(roots)
            diff = roots[:, np.newaxis] - roots[np.newaxis, :]
            sq_dists = np.abs(diff)**2
            mask = ~np.eye(n, dtype=bool)
            valid_sq = sq_dists[mask]
            if np.any(valid_sq <= 1e-30):
                continue
            log_disc = np.sum(np.log(valid_sq))
            if np.isfinite(log_disc):
                scored_inits.append((log_disc, p))
        except Exception:
            pass

    scored_inits.sort(key=lambda x: x[0], reverse=True)

    # Optimize in coefficient space
    best_params = None
    best_ld_score = -np.inf

    for ld0, p0 in scored_inits[:25]:
        if time.time() > _DEADLINE - 30:
            break
        x0 = p0.copy()
        try:
            res = minimize(_neg_poly_objective, x0, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < 1e10:
                # Evaluate the actual log-discriminant at the solution
                coeffs = res.x[0::2] + 1j * res.x[1::2]
                full_coeffs = np.concatenate([[1.0 + 0j], coeffs])
                roots = np.roots(full_coeffs)
                if np.all(np.isfinite(roots)):
                    n = len(roots)
                    diff = roots[:, np.newaxis] - roots[np.newaxis, :]
                    sq_dists = np.abs(diff)**2
                    mask = ~np.eye(n, dtype=bool)
                    valid_sq = sq_dists[mask]
                    if np.all(valid_sq > 1e-30):
                        log_disc = np.sum(np.log(valid_sq))
                        if np.isfinite(log_disc) and log_disc > best_ld_score:
                            best_ld_score = log_disc
                            best_params = res.x.copy()
        except Exception:
            pass

    # Also try Nelder-Mead on coefficient space for top inits
    for ld0, p0 in scored_inits[:10]:
        if time.time() > _DEADLINE - 20:
            break
        x0 = p0.copy()
        try:
            res = minimize(_neg_poly_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < 1e10:
                coeffs = res.x[0::2] + 1j * res.x[1::2]
                full_coeffs = np.concatenate([[1.0 + 0j], coeffs])
                roots = np.roots(full_coeffs)
                if np.all(np.isfinite(roots)):
                    n = len(roots)
                    diff = roots[:, np.newaxis] - roots[np.newaxis, :]
                    sq_dists = np.abs(diff)**2
                    mask = ~np.eye(n, dtype=bool)
                    valid_sq = sq_dists[mask]
                    if np.all(valid_sq > 1e-30):
                        log_disc = np.sum(np.log(valid_sq))
                        if np.isfinite(log_disc) and log_disc > best_ld_score:
                            best_ld_score = log_disc
                            best_params = res.x.copy()
        except Exception:
            pass

    # --- Phase 2: Extract points from best coefficient solution ---
    candidate_points_list = []

    if best_params is not None:
        coeffs = best_params[0::2] + 1j * best_params[1::2]
        full_coeffs = np.concatenate([[1.0 + 0j], coeffs])
        roots = np.roots(full_coeffs)
        if np.all(np.isfinite(roots)):
            pts = _roots_to_points(roots)
            candidate_points_list.append(pts)

    # Also extract from top-5 coefficient-space solutions
    top_solutions = []
    for ld0, p0 in scored_inits[:20]:
        if time.time() > _DEADLINE - 15:
            break
        x0 = p0.copy()
        try:
            res = minimize(_neg_poly_objective, x0, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < 1e10:
                coeffs = res.x[0::2] + 1j * res.x[1::2]
                full_coeffs = np.concatenate([[1.0 + 0j], coeffs])
                roots = np.roots(full_coeffs)
                if np.all(np.isfinite(roots)):
                    n = len(roots)
                    diff = roots[:, np.newaxis] - roots[np.newaxis, :]
                    sq_dists = np.abs(diff)**2
                    mask = ~np.eye(n, dtype=bool)
                    valid_sq = sq_dists[mask]
                    if np.all(valid_sq > 1e-30):
                        log_disc = np.sum(np.log(valid_sq))
                        top_solutions.append((log_disc, roots))
        except Exception:
            pass

    top_solutions.sort(key=lambda x: x[0], reverse=True)
    for ld, roots in top_solutions[:5]:
        pts = _roots_to_points(roots)
        candidate_points_list.append(pts)

    # Add regular octagon and perturbed versions
    candidate_points_list.append(_regular_octagon())

    for _ in range(10):
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False) + rng.normal(0, 0.2, 8)
        r = 1.0 + rng.normal(0, 0.15, 8)
        r = np.abs(r) + 0.1
        pts = np.column_stack([r*np.cos(angles), r*np.sin(angles)])
        candidate_points_list.append(pts)

    # --- Phase 3: Polish in coordinate space ---
    scored_candidates = []
    for cp in candidate_points_list:
        if time.time() > _DEADLINE - 10:
            break
        try:
            ls = _log_score(cp)
            if np.isfinite(ls):
                scored_candidates.append((ls, cp))
        except Exception:
            pass

    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    # Update best
    if scored_candidates:
        ls_best, pts_best = scored_candidates[0]
        if ls_best > best_log_score:
            best_log_score = ls_best
            best_points = pts_best

    # Nelder-Mead polish on top candidates
    for ls, cp in scored_candidates[:8]:
        if time.time() > _DEADLINE - 5:
            break
        x0 = cp.flatten()
        try:
            res = minimize(_neg_log_score, x0, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
            if res.fun < 1e10:
                opt_pts = res.x.reshape(8, 2)
                ls_val = _log_score(opt_pts)
                if np.isfinite(ls_val) and ls_val > best_log_score:
                    best_log_score = ls_val
                    best_points = opt_pts
        except Exception:
            pass

    # Final polish on the best
    if time.time() < _DEADLINE - 2:
        try:
            res = minimize(_neg_log_score, best_points.flatten(), method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < 1e10:
                opt_pts = res.x.reshape(8, 2)
                ls_val = _log_score(opt_pts)
                if np.isfinite(ls_val) and ls_val > best_log_score:
                    best_log_score = ls_val
                    best_points = opt_pts
        except Exception:
            pass

    # Ensure coordinates are within bounds
    final_result = []
    for p in best_points:
        x = max(-1e6, min(1e6, float(p[0])))
        y = max(-1e6, min(1e6, float(p[1])))
        final_result.append([x, y])

    return final_result
