"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from itertools import combinations

def _convex_hull_area(points):
    """Compute convex hull area of points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        pts = np.array(points)
        pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
        if len(pts) <= 1:
            return 0.0
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        hull = lower[:-1] + upper[:-1]
        if len(hull) < 3:
            return 0.0
        x = hull[:, 0]
        y = hull[:, 1]
        return 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

def _all_triangle_areas(points):
    """Compute all triangle areas among all triples."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * np.abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                areas.append(area)
    return np.array(areas)

def _score(points):
    """Compute the minimum triangle area divided by the convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-15:
        return 0.0
    min_area = np.min(_all_triangle_areas(points))
    return min_area / hull_area

def _soft_min_objective(params, beta=50.0):
    """
    Smooth approximation of the negative minimum triangle area divided by hull area.
    Uses log-sum-exp for soft-min.
    params: 28-dimensional vector of point coordinates.
    """
    points = params.reshape(-1, 2)
    areas = _all_triangle_areas(points)
    
    if len(areas) == 0:
        return 1e10
    
    # Soft-min: -log(sum(exp(-beta * areas))) / beta
    # Numerically stable version
    min_val = np.min(areas)
    shifted = areas - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_val - log_sum / beta
    
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-15:
        return 1e10
    
    return -soft_min / hull_area

def _hard_objective(params):
    """Hard objective: negative of the score."""
    points = params.reshape(-1, 2)
    return -_score(points)

def _decode_central_symmetric(params):
    """
    Decode central symmetric parameters to 14 points.
    params: 7 angles + 7 radii = 14 parameters
    For even N=14, we can use central symmetry: if point i is at angle theta_i, radius r_i,
    then point i+7 is at angle theta_i + pi, radius r_i.
    """
    n_half = N // 2  # 7
    angles = params[:n_half]
    radii = params[n_half:]
    
    # Generate first 7 points
    pts_half = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    # Generate second 7 points (centrally symmetric)
    angles_sym = angles + np.pi
    pts_sym = np.column_stack([radii * np.cos(angles_sym), radii * np.sin(angles_sym)])
    
    # Combine and sort by angle to maintain convex order
    all_pts = np.vstack([pts_half, pts_sym])
    all_angles = np.arctan2(all_pts[:, 1], all_pts[:, 0])
    sort_idx = np.argsort(all_angles)
    all_pts = all_pts[sort_idx]
    
    return all_pts

def _objective_central_symmetric(params):
    """Objective for central symmetric optimization."""
    points = _decode_central_symmetric(params)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-15:
        return 1e10
    min_area = np.min(_all_triangle_areas(points))
    return -min_area / hull_area

def _objective_central_symmetric_soft(params, beta=50.0):
    """Soft objective for central symmetric optimization."""
    points = _decode_central_symmetric(params)
    areas = _all_triangle_areas(points)
    
    if len(areas) == 0:
        return 1e10
    
    min_val = np.min(areas)
    shifted = areas - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_val - log_sum / beta
    
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-15:
        return 1e10
    
    return -soft_min / hull_area

def _generate_central_symmetric_inits(rng, n_inits=10):
    """Generate initial configurations for central symmetric optimization."""
    inits = []
    for _ in range(n_inits):
        angles = np.sort(rng.uniform(0, np.pi, N // 2))
        radii = rng.uniform(0.5, 1.0, N // 2)
        params = np.concatenate([angles, radii])
        inits.append(params)
    return inits

def construct():
    """
    Generate 14 points in a planar convex region normalized to unit area
    to maximize the minimum area over all triples.
    
    Uses central symmetry exploitation with differential evolution and L-BFGS-B.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit, keep 10s safety margin
    
    best_points = None
    best_score = -1
    
    rng = np.random.RandomState(42)
    
    # Strategy 1: Start with regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_points = np.column_stack([np.cos(angles), np.sin(angles)])
    hull_area = _convex_hull_area(initial_points)
    if hull_area > 1e-15:
        scale = np.sqrt(1.0 / hull_area)
        initial_points = initial_points * scale
    
    best_points = initial_points.copy()
    best_score = _score(initial_points)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_points
    
    # Strategy 2: Central symmetric optimization with Differential Evolution
    # Parameters: 7 angles (0 to pi) + 7 radii (0.1 to 2.0)
    bounds_cs = [(0, np.pi)] * (N // 2) + [(0.1, 2.0)] * (N // 2)
    
    if time.time() < deadline - 30:
        try:
            de_result = differential_evolution(
                _objective_central_symmetric_soft,
                bounds=bounds_cs,
                seed=42,
                maxiter=100,
                tol=1e-8,
                popsize=15,
                init='latinhypercube',
                polish=True,
                options={'maxiter': 200}
            )
            
            if de_result.fun < 0:
                opt_params = de_result.x
                opt_points = _decode_central_symmetric(opt_params)
                hull_area = _convex_hull_area(opt_points)
                if hull_area > 1e-15:
                    scale = np.sqrt(1.0 / hull_area)
                    opt_points = opt_points * scale
                
                opt_score = _score(opt_points)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
        except Exception:
            pass
    
    # Strategy 3: L-BFGS-B refinement from best central symmetric point
    if time.time() < deadline - 15:
        # Get current best as central symmetric params
        current_pts = best_points
        # Try to extract central symmetric params from best points
        # For now, use a few different starting points for L-BFGS-B
        for init_idx in range(5):
            if time.time() > deadline - 10:
                break
            
            if init_idx == 0:
                # From regular polygon
                angles = np.linspace(0, np.pi, N // 2)
                radii = np.ones(N // 2)
            elif init_idx == 1:
                angles = np.linspace(0.1, np.pi - 0.1, N // 2)
                radii = np.linspace(0.8, 1.2, N // 2)
            elif init_idx == 2:
                angles = rng.uniform(0, np.pi, N // 2)
                angles = np.sort(angles)
                radii = rng.uniform(0.7, 1.3, N // 2)
            elif init_idx == 3:
                angles = np.linspace(0, np.pi, N // 2)
                radii = np.ones(N // 2) * 1.1
            else:
                angles = np.linspace(0, np.pi, N // 2) + rng.normal(0, 0.05, N // 2)
                angles = np.sort(angles)
                radii = np.ones(N // 2) + rng.normal(0, 0.05, N // 2)
                radii = np.clip(radii, 0.3, 1.5)
            
            init_params = np.concatenate([angles, radii])
            
            try:
                lbfgs_result = minimize(
                    _objective_central_symmetric_soft,
                    init_params,
                    method='L-BFGS-B',
                    bounds=bounds_cs,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if lbfgs_result.fun < 0:
                    opt_params = lbfgs_result.x
                    opt_points = _decode_central_symmetric(opt_params)
                    hull_area = _convex_hull_area(opt_points)
                    if hull_area > 1e-15:
                        scale = np.sqrt(1.0 / hull_area)
                        opt_points = opt_points * scale
                    
                    opt_score = _score(opt_points)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = opt_points.copy()
            except Exception:
                pass
    
    # Strategy 4: Full 28-dim L-BFGS-B refinement from best point
    if time.time() < deadline - 10:
        current_params = best_points.flatten()
        try:
            lbfgs_full = minimize(
                _soft_min_objective,
                current_params,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if lbfgs_full.fun < 0:
                opt_points = lbfgs_full.x.reshape(-1, 2)
                hull_area = _convex_hull_area(opt_points)
                if hull_area > 1e-15:
                    scale = np.sqrt(1.0 / hull_area)
                    opt_points = opt_points * scale
                
                opt_score = _score(opt_points)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
        except Exception:
            pass
    
    # Final check: ensure we return valid points
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        hull_area = _convex_hull_area(best_points)
        if hull_area > 1e-15:
            scale = np.sqrt(1.0 / hull_area)
            best_points = best_points * scale
    
    return best_points
# EVOLVE-BLOCK-END
