import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 13
DEADLINE = 115.0  # safety margin before 120s limit


def _triangle_areas(pts):
    """Compute all triangle areas for a set of points."""
    n = len(pts)
    areas = np.empty((n * (n - 1) * (n - 2)) // 6)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                areas[idx] = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                idx += 1
    return areas


def _convex_hull_area(pts):
    """Compute area of convex hull."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume = area
    except Exception:
        # Fallback: use shoelace on all points if hull fails
        order = np.argsort(np.arctan2(pts[:, 1], pts[:, 0]))
        p = pts[order]
        x, y = p[:, 0], p[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _score(pts):
    """Score = min triangle area / convex hull area."""
    areas = _triangle_areas(pts)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area


def _soft_min_objective(flat, tau=0.05):
    """
    Smooth surrogate: maximize soft-min of triangle areas / hull area.
    We minimize the negative of this.
    """
    pts = flat.reshape(-1, 2)
    areas = _triangle_areas(pts)
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    # soft_min approximates min(areas)
    # -tau * log(sum(exp(-a/tau)))
    scaled = -areas / tau
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -tau * log_sum
    return -(soft_min_val / hull_area)


def _exact_objective(flat):
    """Exact (nonsmooth) objective for final polish."""
    pts = flat.reshape(-1, 2)
    return -_score(pts)


def construct():
    start_time = time.time()
    
    # Initial guess: regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts0 = np.column_stack([np.cos(angles), np.sin(angles)])
    flat0 = pts0.flatten()
    
    best_score = _score(pts0)
    best_pts = pts0.copy()
    
    # Phase 1: L-BFGS-B with soft-min surrogate
    try:
        if time.time() - start_time < DEADLINE - 30:
            result = minimize(
                _soft_min_objective,
                flat0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success or result.fun < 1e10:
                pts1 = result.x.reshape(-1, 2)
                s1 = _score(pts1)
                if s1 > best_score:
                    best_score = s1
                    best_pts = pts1.copy()
    except Exception:
        pass
    
    # Phase 2: Try a few different tau values for soft-min
    for tau in [0.02, 0.01, 0.005]:
        try:
            if time.time() - start_time > DEADLINE - 20:
                break
            result = minimize(
                lambda x: _soft_min_objective(x, tau=tau),
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < 1e10:
                pts_t = result.x.reshape(-1, 2)
                s_t = _score(pts_t)
                if s_t > best_score:
                    best_score = s_t
                    best_pts = pts_t.copy()
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead polish on exact objective
    try:
        if time.time() - start_time < DEADLINE - 10:
            result = minimize(
                _exact_objective,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            pts_nm = result.x.reshape(-1, 2)
            s_nm = _score(pts_nm)
            if s_nm > best_score:
                best_score = s_nm
                best_pts = pts_nm.copy()
    except Exception:
        pass
    
    # Phase 4: Powell method for final precision
    try:
        if time.time() - start_time < DEADLINE - 5:
            result = minimize(
                _exact_objective,
                best_pts.flatten(),
                method='Powell',
                options={'maxiter': 500, 'ftol': 1e-12, 'xtol': 1e-12}
            )
            pts_pw = result.x.reshape(-1, 2)
            s_pw = _score(pts_pw)
            if s_pw > best_score:
                best_score = s_pw
                best_pts = pts_pw.copy()
    except Exception:
        pass
    
    # Final check
    if time.time() > DEADLINE:
        # Return best found so far
        pass
    
    return best_pts


# EVOLVE-BLOCK-START
# (The construct function is defined above; this block is for compatibility)
# EVOLVE-BLOCK-END
