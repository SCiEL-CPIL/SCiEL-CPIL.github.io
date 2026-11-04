import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def construct():
    """
    Place 13 points in a planar convex region to maximize the minimum triangle area
    normalized by the convex hull area.
    
    Strategy:
    1. Use a radial function r(theta) = 1 + sum_{k=1}^{K} c_k cos(k*theta) + d_k sin(k*theta)
       to parameterize the points on a smooth curve.
    2. The points are P_j = r(theta_j) * (cos(theta_j), sin(theta_j)) for theta_j = 2*pi*j/N.
    3. This parameterization naturally keeps points in convex position if the radial
       perturbation is small enough (r'' + r > 0).
    4. We maximize the minimum triangle area / hull area using Nelder-Mead.
    
    To improve upon the seed:
    - Use more harmonics (K=4) for more flexibility.
    - Use a better initial guess (slightly perturbed regular polygon).
    - Use multiple restarts with different random perturbations.
    - Carefully tune the bounds to maintain convexity.
    """
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    K = 4  # Number of harmonics
    n_params = 2 * K
    
    def get_points(params):
        r = np.ones(N)
        for k in range(1, K + 1):
            c_k = params[2 * (k - 1)]
            d_k = params[2 * (k - 1) + 1]
            r += c_k * np.cos(k * angles) + d_k * np.sin(k * angles)
        
        # Ensure r is positive
        r = np.maximum(r, 0.1)
        
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        
        return np.column_stack((x, y))
    
    def compute_score(points):
        """Compute the score: min triangle area / hull area."""
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 0.0
        except Exception:
            return 0.0
        
        # Check if all points are on the hull
        if len(hull.vertices) < N:
            return 0.0
        
        # Compute minimum triangle area
        min_area = float('inf')
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    area = 0.5 * abs(
                        p1[0] * (p2[1] - p3[1]) +
                        p2[0] * (p3[1] - p1[1]) +
                        p3[0] * (p1[1] - p2[1])
                    )
                    if area < min_area:
                        min_area = area
        
        if min_area == float('inf'):
            return 0.0
        
        return min_area / hull_area
    
    def objective(params):
        pts = get_points(params)
        s = compute_score(pts)
        return -s
    
    # Regular polygon score as baseline
    regular_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    regular_score = compute_score(regular_pts)
    
    best_score = regular_score
    best_params = np.zeros(n_params)
    
    # Multiple restarts with different initial perturbations
    np.random.seed(42)
    n_restarts = 10
    
    for restart in range(n_restarts):
        if restart == 0:
            x0 = np.zeros(n_params)
        else:
            # Small random perturbations
            scale = 0.05 * (1 + restart * 0.1)
            x0 = np.random.uniform(-scale, scale, n_params)
        
        # Bounds to maintain convexity (r > 0 and r'' + r > 0)
        # For r = 1 + sum c_k cos(k*theta) + d_k sin(k*theta)
        # r'' + r = 1 + sum c_k(1 - k^2)cos(k*theta) + d_k(1 - k^2)sin(k*theta)
        # For k=1: coefficient is 0, so no constraint from k=1
        # For k=2: coefficient is -3, so |c_2| + |d_2| < 1/3
        # For k=3: coefficient is -8, so |c_3| + |d_3| < 1/8
        # For k=4: coefficient is -15, so |c_4| + |d_4| < 1/15
        bounds = []
        for k in range(1, K + 1):
            if k == 1:
                bounds.extend([(-0.3, 0.3), (-0.3, 0.3)])
            else:
                limit = 1.0 / (k * k - 1)
                bounds.extend([(-limit * 0.5, limit * 0.5), (-limit * 0.5, limit * 0.5)])
        
        try:
            res = minimize(
                objective, x0, method='Nelder-Mead',
                bounds=bounds,
                options={
                    'maxiter': 3000,
                    'xatol': 1e-7,
                    'fatol': 1e-7,
                    'adaptive': True
                }
            )
            
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x
        except Exception:
            continue
    
    best_points = get_points(best_params)
    
    # Final validation
    try:
        hull = ConvexHull(best_points)
        if len(hull.vertices) < N:
            return regular_pts
    except Exception:
        return regular_pts
    
    return best_points
