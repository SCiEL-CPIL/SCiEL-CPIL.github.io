import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

def _triangle_areas(points):
    """Compute areas of all triangles formed by 3 points. Returns array of shape (286,)."""
    n = len(points)
    indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                indices.append((i, j, k))
    indices = np.array(indices)
    
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return areas

def _score(points):
    """Compute A_min / A_hull. Returns 0 if invalid."""
    try:
        if np.any(~np.isfinite(points)):
            return 0.0
        if len(points) != 13 or points.shape != (13, 2):
            return 0.0
        hull = ConvexHull(points)
        a_hull = hull.volume
        if a_hull <= 0:
            return 0.0
        areas = _triangle_areas(points)
        a_min = np.min(areas)
        if a_min <= 0:
            return 0.0
        return a_min / a_hull
    except Exception:
        return 0.0

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    deadline = time.time() + 580  # Leave 20s safety margin
    
    # Precompute triangle indices for fast access
    tri_indices = list(itertools.combinations(range(n), 3))
    tri_indices = np.array(tri_indices)
    n_tris = len(tri_indices)
    
    def _soft_min_objective(x, p_power):
        """
        Objective function for minimization.
        We want to maximize min(areas) / hull_area.
        This is equivalent to minimizing -min(areas) / hull_area.
        We use a soft minimum approximation for min(areas).
        soft_min = (sum(a_i^-p) / N)^(-1/p)
        """
        pts = x.reshape(n, 2)
        if np.any(~np.isfinite(pts)):
            return 1e10
            
        # Calculate areas
        p1 = pts[tri_indices[:, 0]]
        p2 = pts[tri_indices[:, 1]]
        p3 = pts[tri_indices[:, 2]]
        
        # Cross product magnitude is 2 * Area
        cross = p1[:, 0] * (p2[:, 1] - p3[:, 1]) + p2[:, 0] * (p3[:, 1] - p1[:, 1]) + p3[:, 0] * (p1[:, 1] - p2[:, 1])
        areas = 0.5 * np.abs(cross)
        
        # Avoid division by zero or tiny areas
        min_area = np.min(areas)
        if min_area < 1e-12:
            return 1e10
        
        # Calculate Hull Area
        try:
            hull = ConvexHull(pts)
            a_hull = hull.volume
            if a_hull < 1e-12:
                return 1e10
        except Exception:
            return 1e10
            
        # Soft minimum calculation
        # To improve numerical stability, we can scale areas or use log-sum-exp
        # Let's use the power mean directly, but be careful with overflow/underflow
        # For large p, a_i^-p can underflow. 
        # Alternative: -1/p * log( sum(exp(-p * log(a_i))) )
        # log(soft_min) = -1/p * log( sum(exp(-p * log(a_i))) )
        # log(a_i) is stable if a_i > 0.
        
        log_areas = np.log(areas)
        exp_terms = np.exp(-p_power * log_areas)
        
        # Check for overflow in exp_terms
        if np.any(~np.isfinite(exp_terms)):
             # Fallback to hard min if soft min calculation fails
             return -(min_area / a_hull)
             
        sum_exp = np.sum(exp_terms)
        if sum_exp == 0 or not np.isfinite(sum_exp):
             return -(min_area / a_hull)
             
        log_soft_min = -1.0 / p_power * np.log(sum_exp)
        soft_min_val = np.exp(log_soft_min)
        
        score = soft_min_val / a_hull
        return -score

    best_points = None
    best_score = 0.0
    
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    inits = []
    
    # 1. Regular polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    inits.append(np.column_stack((np.cos(angles), np.sin(angles))))
    
    # 2. Regular polygon + center (12+1)
    angles12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    poly12 = np.column_stack((np.cos(angles12), np.sin(angles12)))
    inits.append(np.vstack([poly12, [[0, 0]]]))
    
    # 3. Concentric rings (e.g., 6 outer, 6 inner, 1 center)
    # Outer ring
    ang_o = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    outer = np.column_stack((np.cos(ang_o), np.sin(ang_o)))
    # Inner ring
    ang_i = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    inner = 0.5 * np.column_stack((np.cos(ang_i), np.sin(ang_i)))
    center = np.array([[0, 0]])
    inits.append(np.vstack([outer, inner, center]))
    
    # 4. Random points
    for _ in range(10):
        inits.append(rng.uniform(-1, 1, size=(n, 2)))
        
    # 5. Halton sequence points (low discrepancy)
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
        
    halton_pts = np.zeros((n, 2))
    for i in range(n):
        halton_pts[i, 0] = halton(i + 1, 2)
        halton_pts[i, 1] = halton(i + 1, 3)
    # Scale to unit circle-ish
    halton_pts = (halton_pts - 0.5) * 2
    inits.append(halton_pts)

    # Optimization Strategy: Annealing the soft minimum power
    # Start with small p (smooth, global-ish) and increase p (sharper, local)
    p_schedule = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    current_best = None
    for init in inits:
        if time.time() > deadline - 60:
            break
            
        pts = init.copy()
        
        # Normalize initial points to unit circle area roughly?
        # The objective is scale invariant, but numerical stability might benefit.
        # Let's just run optimization.
        
        for p_power in p_schedule:
            if time.time() > deadline - 60:
                break
                
            try:
                res = minimize(
                    _soft_min_objective, 
                    pts.flatten(), 
                    args=(p_power,),
                    method='L-BFGS-B', 
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                new_pts = res.x.reshape(n, 2)
                
                # Validate and score
                if np.all(np.isfinite(new_pts)):
                    s = _score(new_pts)
                    if s > best_score:
                        best_score = s
                        best_points = new_pts.copy()
                        current_best = new_pts.copy()
                else:
                    # If optimization went to infinity, keep previous
                    pass
                    
                # Update current point for next annealing step
                if np.all(np.isfinite(new_pts)) and _score(new_pts) > 0:
                    pts = new_pts
            except Exception:
                continue

    # Final local polish with Nelder-Mead on the hard objective
    if best_points is not None and time.time() < deadline - 10:
        def _hard_obj(x):
            pts = x.reshape(n, 2)
            if np.any(~np.isfinite(pts)):
                return 1e10
            try:
                areas = _triangle_areas(pts)
                a_min = np.min(areas)
                if a_min < 1e-15:
                    return 1e10
                hull = ConvexHull(pts)
                a_hull = hull.volume
                if a_hull < 1e-15:
                    return 1e10
                return -(a_min / a_hull)
            except Exception:
                return 1e10

        try:
            res = minimize(
                _hard_obj, 
                best_points.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            final_pts = res.x.reshape(n, 2)
            if np.all(np.isfinite(final_pts)):
                s = _score(final_pts)
                if s > best_score:
                    best_score = s
                    best_points = final_pts
        except Exception:
            pass

    if best_points is None or _score(best_points) <= 0:
        # Fallback
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
