"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from scipy.interpolate import CubicSpline
from itertools import combinations

N = 14

# Pre-compute all triangle index combinations for vectorized area calculation
# C(14, 3) = 364 triangles
tri_indices = np.array(list(combinations(range(N), 3)))  # shape (364, 3)

def _all_triangle_areas(points):
    """
    Compute all triangle areas among all triples in a vectorized manner.
    points: shape (N, 2)
    returns: array of shape (364,)
    """
    pi = points[tri_indices[:, 0]]
    pj = points[tri_indices[:, 1]]
    pk = points[tri_indices[:, 2]]
    
    # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _convex_hull_area(points):
    """Compute convex hull area of points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # Fallback: manual convex hull (Andrew's monotone chain)
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

def _score(points):
    """Compute the minimum triangle area divided by the convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-15:
        return 0.0
    min_area = np.min(_all_triangle_areas(points))
    return min_area / hull_area

def _soft_min_objective_cartesian(params, beta=50.0):
    """
    Soft-min objective for Cartesian points.
    params: 28-dimensional vector of point coordinates.
    """
    points = params.reshape(-1, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
        
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    
    # Soft-min: -log(sum(exp(-beta * areas))) / beta
    # Numerically stable version
    min_val = np.min(areas)
    shifted = areas - min_val
    # To avoid overflow in exp, we shift by min_val
    # sum(exp(-beta * (areas - min_val))) = exp(beta*min_val) * sum(exp(-beta*areas))
    # log(sum(exp(-beta*areas))) = beta*min_val + log(sum(exp(-beta*(areas-min_val))))
    # soft_min = min_val - (beta*min_val + log(sum(exp(-beta*(areas-min_val))))) / beta
    # soft_min = min_val - min_val - log(sum(exp(-beta*(areas-min_val)))) / beta
    # soft_min = - log(sum(exp(-beta*(areas-min_val)))) / beta
    # Wait, the standard soft min is -1/beta * log(sum(exp(-beta * x)))
    # Let's stick to the stable form:
    # log_sum = log(sum(exp(-beta * areas)))
    # max_a = max(areas) -> no, min is better for soft min of min
    # Actually, soft_min approximates min(x).
    # min(x) = -1/beta * log(sum(exp(-beta * x))) as beta -> inf
    # Stable:
    m = np.min(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - m))))
    soft_min = m - log_sum / beta
    
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-15:
        return 1e10
    
    return -soft_min / hull_area

def _decode_bspline(ctrl_params):
    """
    Decode 14-dim control point parameters to 14 points on a B-spline curve.
    ctrl_params: shape (14,) = [x0,y0, x1,y1, ..., x6,y6]
    """
    ctrl = ctrl_params.reshape(7, 2)
    
    try:
        # Create a closed cubic B-spline
        # For a closed curve with 7 control points, we need to handle the periodicity
        # CubicSpline with bc_type='periodic' requires the first and last points to be the same
        # So we append the first control point
        ctrl_closed = np.vstack([ctrl, ctrl[0]])
        
        # Parameter values 0..7 for 8 points (7 unique + repeat)
        t = np.arange(8)
        
        # Create cubic splines for x and y
        cs_x = CubicSpline(t, ctrl_closed[:, 0], bc_type='periodic')
        cs_y = CubicSpline(t, ctrl_closed[:, 1], bc_type='periodic')
        
        # Sample many points along the curve to get accurate arc length
        t_fine = np.linspace(0, 7, 2000)
        x_fine = cs_x(t_fine)
        y_fine = cs_y(t_fine)
        
        # Compute arc length
        dx = np.diff(x_fine)
        dy = np.diff(y_fine)
        ds = np.sqrt(dx**2 + dy**2)
        s = np.concatenate([[0], np.cumsum(ds)])
        
        # Total arc length
        total_s = s[-1]
        
        # Sample 14 points at equal arc-length intervals
        s_targets = np.linspace(0, total_s, N, endpoint=False)
        
        # Interpolate to get points at target arc lengths
        x_pts = np.interp(s_targets, s, x_fine)
        y_pts = np.interp(s_targets, s, y_fine)
        
        pts = np.column_stack([x_pts, y_pts])
        
        # Center the points to avoid translation invariance issues in optimization
        pts -= pts.mean(axis=0)
        
        return pts
        
    except Exception:
        # Fallback: just use the control points plus their antipodes
        # This gives 14 points (7 + 7)
        ctrl_centered = ctrl - ctrl.mean(axis=0)
        pts = np.vstack([ctrl_centered, -ctrl_centered[::-1]])
        return pts

def _objective_bspline(ctrl_params, beta=50.0):
    """
    Objective for B-spline control points.
    """
    pts = _decode_bspline(ctrl_params)
    if np.any(~np.isfinite(pts)):
        return 1e10
        
    areas = _all_triangle_areas(pts)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    
    min_val = np.min(areas)
    shifted = areas - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_val - log_sum / beta
    
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-15:
        return 1e10
    
    return -soft_min / hull_area

def _generate_bspline_inits():
    """Generate diverse initial control point configurations for B-spline."""
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Regular heptagon control points
    angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    ctrl = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(ctrl.flatten().copy())
    
    # 2. Elliptical control points
    for rx, ry in [(1.2, 0.8), (0.8, 1.2), (1.3, 0.7), (0.7, 1.3), (1.1, 0.9)]:
        angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        ctrl = np.column_stack([rx * np.cos(angles), ry * np.sin(angles)])
        inits.append(ctrl.flatten().copy())
    
    # 3. Perturbed regular heptagon
    for js in [0.05, 0.1, 0.15, 0.2]:
        angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        ctrl = np.column_stack([np.cos(angles), np.sin(angles)])
        ctrl += rng.normal(0, js, size=(7, 2))
        inits.append(ctrl.flatten().copy())
    
    # 4. Random convex-ish polygons
    for seed in range(10):
        rs = np.random.default_rng(seed * 13 + 7)
        angles = np.sort(rs.uniform(0, 2 * np.pi, 7))
        radii = rs.uniform(0.6, 1.4, 7)
        ctrl = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(ctrl.flatten().copy())
        
    return inits

def construct():
    """
    Generate 14 points in a planar convex region normalized to unit area
    to maximize the minimum area over all triples.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit, keep 10s safety margin
    
    best_points = None
    best_score = -1
    
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
    
    # Strategy 2: B-spline optimization with Differential Evolution
    # Parameters: 7 control points * 2 coords = 14 parameters
    # Bounds: [-2, 2] for each coordinate
    bounds_bs = [(-2.0, 2.0)] * 14
    
    if time.time() < deadline - 40:
        try:
            # Use a subset of inits for DE to speed up initialization
            de_inits = _generate_bspline_inits()
            # DE expects a list of arrays or a single array. 
            # We'll let DE generate its own population but seed it if possible.
            # For simplicity and speed, we just run DE with default init 'latinhypercube'
            de_result = differential_evolution(
                _objective_bspline,
                bounds=bounds_bs,
                seed=42,
                maxiter=200,
                tol=1e-8,
                popsize=15,
                init='latinhypercube',
                polish=True,
                options={'maxiter': 200}
            )
            
            if de_result.fun < 0:
                opt_params = de_result.x
                opt_points = _decode_bspline(opt_params)
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
    
    # Strategy 3: Cartesian L-BFGS-B refinement from best point
    # This allows points to move off the B-spline curve
    if time.time() < deadline - 15:
        current_params = best_points.flatten()
        try:
            lbfgs_full = minimize(
                _soft_min_objective_cartesian,
                current_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
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
            
    # Strategy 4: Nelder-Mead polish
    # NM can be effective for final tweaks, especially if L-BFGS-B got stuck in a local min due to non-smoothness
    if time.time() < deadline - 5:
        try:
            nm_result = minimize(
                _soft_min_objective_cartesian,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            if nm_result.fun < 0:
                opt_points = nm_result.x.reshape(-1, 2)
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
