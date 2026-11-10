# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580.0  # 20s safety margin
    
    # Precompute triangle indices
    tri_indices = list(itertools.combinations(range(n), 3))
    tri_indices = np.array(tri_indices)
    num_tris = len(tri_indices)
    
    def compute_areas(points):
        """Compute areas of all 286 triangles."""
        # points: (n, 2)
        # For each triangle (i, j, k), area = 0.5 * |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
        # Vectorized:
        p1 = points[tri_indices[:, 0]]  # (num_tris, 2)
        p2 = points[tri_indices[:, 1]]
        p3 = points[tri_indices[:, 2]]
        
        # Cross product magnitude / 2
        areas = 0.5 * np.abs(
            p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
            p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
            p3[:, 0] * (p1[:, 1] - p2[:, 1])
        )
        return areas
    
    def compute_hull_area(points):
        """Compute convex hull area."""
        try:
            hull = ConvexHull(points)
            return hull.volume  # In 2D, volume is area
        except:
            return np.inf
    
    def compute_score(points):
        """Compute A_min / A_hull."""
        areas = compute_areas(points)
        a_min = np.min(areas)
        if a_min <= 0:
            return 0.0
        a_hull = compute_hull_area(points)
        if a_hull <= 0 or np.isinf(a_hull):
            return 0.0
        return a_min / a_hull
    
    def soft_min_objective(flat_pts, beta=50.0):
        """
        Objective to minimize: -log-sum-exp of triangle areas / hull area
        We want to maximize min(areas) / hull_area
        Equivalently, minimize -log(sum(exp(areas/beta))) / hull_area approximately
        
        Actually, let's use a different approach:
        We want to maximize t such that t <= area_i / hull_area for all i
        This is equivalent to maximizing t subject to t * hull_area <= area_i
        
        For smooth optimization, use log-sum-exp:
        min_area_smooth = (1/beta) * log(sum(exp(beta * areas)))  -- this is wrong direction
        
        Let's use: we want to maximize the minimum area relative to hull.
        f(x) = - (1/beta) * log(sum(exp(-beta * areas))) / hull_area
        As beta -> inf, this approaches -min(areas) / hull_area
        So minimizing f(x) maximizes min(areas)/hull_area
        
        But hull_area is also a function of x. This makes it complicated.
        
        Alternative: normalize points to unit area hull, then maximize min triangle area.
        Or: use the ratio directly in a smooth way.
        
        Let me try a different approach: 
        Objective = -min_area + penalty if hull_area is not normalized
        
        Actually, the simplest smooth approach:
        Let's define g(x) = -log(sum(exp(-beta * areas / hull_area)))
        This is approximately -log(num_tris * exp(-beta * min(areas/hull_area)))
        = -log(num_tris) + beta * min(areas/hull_area)
        So minimizing g(x) is approximately maximizing min(areas/hull_area)
        """
        pts = flat_pts.reshape(n, 2)
        areas = compute_areas(pts)
        hull_area = compute_hull_area(pts)
        if hull_area <= 0 or np.isinf(hull_area) or np.any(~np.isfinite(pts)):
            return 1e10
        
        # Ratios = areas / hull_area
        ratios = areas / hull_area
        
        # Log-sum-exp of -beta * ratios, then negative
        # This approximates -min(ratios) as beta -> inf
        max_val = np.max(-beta * ratios)
        if max_val > 700:
            # Overflow protection
            return 1e10
        log_sum = max_val + np.log(np.sum(np.exp(-beta * ratios - max_val)))
        return -log_sum  # Minimizing this maximizes min(ratios)
    
    def gradient_soft_min(flat_pts, beta=50.0):
        """Numerical gradient for the soft-min objective."""
        pts = flat_pts.reshape(n, 2)
        areas = compute_areas(pts)
        hull_area = compute_hull_area(pts)
        if hull_area <= 0 or np.isinf(hull_area):
            return np.zeros_like(flat_pts)
        
        ratios = areas / hull_area
        max_val = np.max(-beta * ratios)
        if max_val > 700:
            return np.zeros_like(flat_pts)
        exp_terms = np.exp(-beta * ratios - max_val)
        sum_exp = np.sum(exp_terms)
        
        # Gradient with respect to each point coordinate
        grad = np.zeros_like(flat_pts)
        # This would require computing gradients of areas w.r.t. points
        # For now, use numerical gradient
        eps = 1e-7
        for i in range(len(flat_pts)):
            orig = flat_pts[i]
            flat_pts[i] = orig + eps
            f_plus = soft_min_objective(flat_pts, beta)
            flat_pts[i] = orig - eps
            f_minus = soft_min_objective(flat_pts, beta)
            flat_pts[i] = orig
            grad[i] = (f_plus - f_minus) / (2 * eps)
        
        return grad
    
    # Initial point: use a regular grid or random
    best_score = 0.0
    best_points = None
    
    # Try multiple initializations
    rng = np.random.default_rng(seed=42)
    
    for init_iter in range(20):
        if time.time() > deadline:
            break
        
        if init_iter == 0:
            # Regular grid-like initialization
            points = np.array([
                [0.1, 0.1], [0.3, 0.1], [0.5, 0.1], [0.7, 0.1], [0.9, 0.1],
                [0.1, 0.3], [0.3, 0.3], [0.5, 0.3], [0.7, 0.3], [0.9, 0.3],
                [0.1, 0.5], [0.3, 0.5], [0.5, 0.5]
            ])
        else:
            # Random initialization
            points = rng.random((n, 2))
        
        # Normalize to unit area
        hull_area = compute_hull_area(points)
        if hull_area > 0 and np.isfinite(hull_area):
            points = points / np.sqrt(hull_area)
        
        flat_pts = points.flatten().copy()
        
        # L-BFGS-B optimization
        try:
            result = minimize(
                soft_min_objective,
                flat_pts,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < 1e9:
                opt_points = result.x.reshape(n, 2)
                # Check validity
                if np.all(np.isfinite(opt_points)):
                    score = compute_score(opt_points)
                    if score > best_score:
                        best_score = score
                        best_points = opt_points.copy()
        except Exception as e:
            pass
    
    if best_points is None:
        # Fallback to random
        points = rng.random((n, 2))
        return points
    
    return best_points

# EVOLVE-BLOCK-END
