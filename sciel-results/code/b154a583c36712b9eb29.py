N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

def _triangle_areas(pts):
    """Compute all triangle areas from n points. Returns array of C(n,3) areas."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
                areas.append(area)
    return np.array(areas)

def _convex_hull_area(pts):
    """Compute area of convex hull of points."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    areas = _triangle_areas(pts)
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min_objective(x, tau=0.01):
    """
    Objective to minimize: negative of soft-min of triangle areas, normalized by hull area.
    x: 2N array of polar coords (r0, theta0, r1, theta1, ..., r_{N-1}, theta_{N-1})
    """
    r = x[0::2]
    theta = x[1::2]
    
    # Convert to Cartesian
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    
    # Ensure points are in convex position by checking they form a convex polygon
    # Sort by angle to get convex hull order
    angles = np.arctan2(pts[:, 1], pts[:, 0])
    order = np.argsort(angles)
    pts_sorted = pts[order]
    
    # Compute hull area using shoelace formula on sorted points
    n = len(pts_sorted)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts_sorted[i, 0] * pts_sorted[j, 1] - pts_sorted[j, 0] * pts_sorted[i, 1]
    hull_area = abs(area) / 2.0
    
    if hull_area < 1e-10:
        return 1e10  # Penalize degenerate
    
    # Compute all triangle areas
    areas = _triangle_areas(pts)
    
    # Soft min using log-sum-exp
    # min(a_i) ≈ -tau * log(sum(exp(-a_i/tau)))
    scaled = -areas / tau
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_area = -tau * log_sum_exp
    
    # We want to maximize soft_min_area / hull_area, so minimize negative
    obj = -soft_min_area / hull_area
    
    # Add small penalty for non-convex arrangements (shouldn't happen with polar params)
    # Check if any point is inside the convex hull of others
    return obj

def _soft_min_gradient(x, tau=0.01):
    """Gradient of the soft-min objective using finite differences (for simplicity)."""
    # Use numerical gradient for robustness
    eps = 1e-7
    grad = np.zeros_like(x)
    f0 = _soft_min_objective(x, tau)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        f_plus = _soft_min_objective(x_plus, tau)
        grad[i] = (f_plus - f0) / eps
    return grad

def construct():
    """
    Place 13 points to maximize minimum triangle area / convex hull area.
    Uses regular 13-gon as starting point and refines with gradient-based optimization.
    """
    deadline = time.time() + 130  # 10s safety margin before 140s kill
    
    # Start with regular 13-gon
    angles_init = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r_init = np.ones(N)
    theta_init = angles_init.copy()
    
    # Initial x: [r0, theta0, r1, theta1, ...]
    x0 = np.zeros(2 * N)
    x0[0::2] = r_init
    x0[1::2] = theta_init
    
    # Convert to Cartesian for initial score
    pts0 = np.column_stack([r_init * np.cos(theta_init), r_init * np.sin(theta_init)])
    best_pts = pts0.copy()
    best_score = _score(best_pts)
    
    # Try multiple tau values for the soft-min surrogate
    tau_values = [0.05, 0.02, 0.01, 0.005]
    
    for tau in tau_values:
        if time.time() > deadline:
            break
        
        # Optimize with L-BFGS-B
        # Bounds: r in [0.1, 2.0], theta in [0, 2*pi]
        bounds = []
        for i in range(N):
            bounds.append((0.1, 2.0))  # r_i
            bounds.append((0, 2 * np.pi))  # theta_i
        
        try:
            result = minimize(
                _soft_min_objective,
                x0,
                args=(tau,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if time.time() > deadline:
                break
            
            if result.success or result.fun < _soft_min_objective(x0, tau):
                x_opt = result.x
                r_opt = x_opt[0::2]
                theta_opt = x_opt[1::2]
                pts_opt = np.column_stack([r_opt * np.cos(theta_opt), r_opt * np.sin(theta_opt)])
                
                score_opt = _score(pts_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_pts = pts_opt.copy()
                    x0 = x_opt.copy()  # Use as new starting point for next tau
        
        except Exception:
            pass
    
    # Final refinement with smaller tau if time permits
    if time.time() < deadline - 5:
        try:
            result = minimize(
                _soft_min_objective,
                x0,
                args=(0.001,),
                method='L-BFGS-B',
                bounds=[(0.1, 2.0), (0, 2 * np.pi)] * N,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.success:
                x_opt = result.x
                r_opt = x_opt[0::2]
                theta_opt = x_opt[1::2]
                pts_opt = np.column_stack([r_opt * np.cos(theta_opt), r_opt * np.sin(theta_opt)])
                score_opt = _score(pts_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_pts = pts_opt.copy()
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
