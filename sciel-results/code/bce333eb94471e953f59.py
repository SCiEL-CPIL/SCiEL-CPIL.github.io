"""Heilbronn convex regions: maximize min triangle area / hull area for 13 points."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import itertools

N = 13
MAX_TIME = 145.0  # Safety margin before 150s kill

# Precompute triangle indices
_TRI_IDX = np.array(list(itertools.combinations(range(N), 3)))
_TRI_I = _TRI_IDX[:, 0]
_TRI_J = _TRI_IDX[:, 1]
_TRI_K = _TRI_IDX[:, 2]

def _hull_area(pts):
    """Calculate the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _min_tri_area(pts):
    """Calculate the minimum area of any triangle formed by three points."""
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _score(pts):
    """Calculate the score: min triangle area / hull area."""
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 0.0
    mta = _min_tri_area(pts)
    return mta / ha

def _fourier_to_points(coeffs):
    """
    Convert Fourier coefficients to 13 points on a convex curve.
    coeffs: [r0, a1, b1, a2, b2, ..., aM, bM]
    r0 is the mean radius, (a_k, b_k) are cosine/sine amplitudes for mode k.
    Points are placed at equally spaced angles 2*pi*k/13.
    """
    n = N
    r0 = coeffs[0]
    angles = 2 * np.pi * np.arange(n) / n
    
    radii = np.full(n, r0)
    for k in range(1, len(coeffs) // 2 + 1):
        a = coeffs[2 * k - 1]
        b = coeffs[2 * k]
        radii += a * np.cos(k * angles) + b * np.sin(k * angles)
    
    # Ensure positive radii (convexity requires r > 0 and appropriate curvature)
    # For safety, clip to a minimum
    radii = np.maximum(radii, 0.01)
    
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts

def _smooth_min(areas, tau):
    """Compute smooth minimum using log-sum-exp."""
    max_area = np.max(areas)
    log_sum_exp = np.log(np.sum(np.exp(-(areas - max_area) / tau)))
    return max_area - tau * log_sum_exp

def _fourier_objective(coeffs, tau=0.05):
    """
    Objective function for Fourier parameterization.
    Returns negative smooth min triangle area / hull area.
    """
    pts = _fourier_to_points(coeffs)
    
    # Check convexity: all cross products of consecutive edges should have same sign
    # For a convex polygon with vertices in CCW order, cross products should be positive
    n = len(pts)
    cross_prods = []
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        p2 = pts[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        cross_prods.append(cross)
    
    cross_prods = np.array(cross_prods)
    
    # For convexity, all cross products should be positive (CCW) or all negative (CW)
    # If mixed signs, the polygon is non-convex
    if np.any(cross_prods * cross_prods[0] < 0):
        # Non-convex: return large penalty
        return 1e6
    
    # Compute triangle areas
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    areas = 0.5 * np.abs(cross)
    
    if np.any(areas < 1e-10):
        return 1e6
    
    # Compute hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 1e6
    
    if hull_area < 1e-10:
        return 1e6
    
    # Smooth min using log-sum-exp
    smooth_min = _smooth_min(areas, tau)
    score = smooth_min / hull_area
    return -score

def _refine_with_lbfgs(pts, deadline):
    """
    Refine a point configuration using L-BFGS-B in Cartesian space.
    """
    temperatures = [0.05, 0.02, 0.01, 0.005]
    best_pts = pts.copy()
    best_score = _score(pts)
    
    for temp in temperatures:
        if time.time() > deadline - 2:
            break
        
        def obj(x):
            p = x.reshape(N, 2)
            ha = _hull_area(p)
            if ha < 1e-8:
                return 1e6
            p_i = p[_TRI_I]
            p_j = p[_TRI_J]
            p_k = p[_TRI_K]
            cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
                    (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
            areas = 0.5 * np.abs(cross)
            if np.any(areas < 1e-10):
                return 1e6
            smooth_min = _smooth_min(areas, temp)
            return -smooth_min / ha
        
        try:
            res = minimize(
                obj,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            opt_pts = res.x.reshape(N, 2)
            score = _score(opt_pts)
            
            if score > best_score:
                best_score = score
                best_pts = opt_pts
        except Exception:
            continue
    
    return best_pts, best_score

def _refine_with_nelder_mead(pts, deadline):
    """
    Refine a point configuration using Nelder-Mead in Cartesian space.
    """
    def obj(x):
        p = x.reshape(N, 2)
        ha = _hull_area(p)
        if ha < 1e-8:
            return 1e6
        p_i = p[_TRI_I]
        p_j = p[_TRI_J]
        p_k = p[_TRI_K]
        cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
                (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
        areas = 0.5 * np.abs(cross)
        if np.any(areas < 1e-10):
            return 1e6
        return -np.min(areas) / ha
    
    try:
        res = minimize(
            obj,
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        opt_pts = res.x.reshape(N, 2)
        score = _score(opt_pts)
        return opt_pts, score
    except Exception:
        return pts, _score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    best_score = -np.inf
    best_pts = None
    
    # Phase 1: Differential evolution in Fourier space
    # Use up to 3 modes (7 parameters: r0, a1, b1, a2, b2, a3, b3)
    n_modes = 3
    n_params = 1 + 2 * n_modes
    
    # Bounds: r0 in [0.5, 2.0], amplitudes in [-0.5, 0.5]
    bounds = [(0.5, 2.0)] + [(-0.5, 0.5)] * (2 * n_modes)
    
    def de_objective(coeffs):
        return _fourier_objective(coeffs, tau=0.1)
    
    # Run differential evolution with time limit
    max_iter_de = 80
    try:
        result = differential_evolution(
            de_objective,
            bounds,
            seed=42,
            maxiter=max_iter_de,
            tol=1e-6,
            polish=True,
            init='sobol',
            popsize=12,
            updating='deferred',
            callback=None
        )
        
        if result.x is not None:
            pts = _fourier_to_points(result.x)
            score = _score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30:
        if best_pts is None:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            best_pts = np.column_stack((np.cos(angles), np.sin(angles)))
            best_score = _score(best_pts)
        return best_pts
    
    # Phase 2: Try more Fourier modes with DE
    n_modes2 = 5
    n_params2 = 1 + 2 * n_modes2
    bounds2 = [(0.5, 2.0)] + [(-0.3, 0.3)] * (2 * n_modes2)
    
    def de_objective2(coeffs):
        return _fourier_objective(coeffs, tau=0.05)
    
    try:
        result2 = differential_evolution(
            de_objective2,
            bounds2,
            seed=123,
            maxiter=50,
            tol=1e-6,
            polish=True,
            init='sobol',
            popsize=10,
            updating='deferred'
        )
        
        if result2.x is not None:
            pts = _fourier_to_points(result2.x)
            score = _score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts
    except Exception:
        pass
    
    if time.time() > deadline - 20:
        if best_pts is not None:
            # Quick L-BFGS polish
            if time.time() < deadline - 5:
                opt_pts, opt_score = _refine_with_lbfgs(best_pts, deadline)
                if opt_score > best_score:
                    best_score = opt_score
                    best_pts = opt_pts
        else:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            best_pts = np.column_stack((np.cos(angles), np.sin(angles)))
            best_score = _score(best_pts)
        return best_pts
    
    # Phase 3: Refine best solution with L-BFGS-B
    if best_pts is not None:
        opt_pts, opt_score = _refine_with_lbfgs(best_pts, deadline)
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts
    
    # Phase 4: Try Nelder-Mead refinement
    if time.time() < deadline - 10 and best_pts is not None:
        nm_pts, nm_score = _refine_with_nelder_mead(best_pts, deadline)
        if nm_score > best_score:
            best_score = nm_score
            best_pts = nm_pts
    
    # Phase 5: Multi-start in Cartesian space if time permits
    if time.time() < deadline - 15:
        starts = []
        
        # Regular 13-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        starts.append(np.column_stack((np.cos(angles), np.sin(angles))))
        
        # Random starts
        for seed in [1, 2, 3]:
            rng = np.random.default_rng(seed)
            r = np.sqrt(rng.random(N))
            theta = rng.random(N) * 2 * np.pi
            pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
            starts.append(pts)
        
        for start in starts:
            if time.time() > deadline - 5:
                break
            opt_pts, opt_score = _refine_with_lbfgs(start, deadline)
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts
    
    # Fallback
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_pts
