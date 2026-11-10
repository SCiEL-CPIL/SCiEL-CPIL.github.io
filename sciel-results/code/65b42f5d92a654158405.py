# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min / d_max)^2.
    
    Returns
        points: np.ndarray of shape (14, 3) containing the coordinates of the 14 points.
    """
    deadline = time.time() + 600  # 600 second limit
    
    def compute_score(pts):
        """Compute the metric: (d_min / d_max)^2"""
        if pts.shape != (14, 3):
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        d = pdist(pts)
        if len(d) == 0:
            return 0.0
        d_min = d.min()
        d_max = d.max()
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2
    
    def fibonacci_sphere(n, d=3):
        """Generate n points on a 3D sphere using Fibonacci lattice."""
        points = np.zeros((n, d))
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n):
            y = 1 - (i / (n - 1)) * 2
            r = np.sqrt(1 - y**2)
            theta = golden_angle * i
            x = r * np.cos(theta)
            z = r * np.sin(theta)
            points[i] = [x, y, z]
        return points
    
    # Initial configuration: Fibonacci sphere points
    n = 14
    d = 3
    best_points = fibonacci_sphere(n, d)
    best_score = compute_score(best_points)
    
    # Gradient-based optimization using L-BFGS
    from scipy.optimize import minimize
    
    def objective(params):
        """Objective function to minimize (negative of our metric)."""
        pts = params.reshape(n, d)
        score = compute_score(pts)
        return -score
    
    # Multiple random restarts with different initial configurations
    np.random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Fibonacci sphere
            x0 = fibonacci_sphere(n, d).flatten()
        elif restart == 1:
            # Scaled Fibonacci sphere
            x0 = (fibonacci_sphere(n, d) * 0.5).flatten()
        elif restart == 2:
            # Regular simplex-like in 3D (first 14 vertices of a higher-dim simplex projected)
            pts = np.zeros((n, d))
            for i in range(n):
                angle = 2 * np.pi * i / n
                pts[i, 0] = np.cos(angle)
                pts[i, 1] = np.sin(angle)
                pts[i, 2] = 0.3 * np.cos(2 * angle)
            x0 = pts.flatten()
        elif restart % 2 == 0:
            # Random points on sphere
            pts = np.random.randn(n, d)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            x0 = pts.flatten()
        else:
            # Random points in cube
            x0 = np.random.rand(n, d) * 2 - 1
        
        # Optimize
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_score:
                candidate = result.x.reshape(n, d)
                candidate_score = compute_score(candidate)
                
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate
        except Exception:
            continue
        
        # Local refinement with smaller steps
        if time.time() > deadline - 10:
            break
        
        try:
            result2 = minimize(
                objective,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result2.fun < best_score:
                candidate = result2.x.reshape(n, d)
                candidate_score = compute_score(candidate)
                
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate
        except Exception:
            continue
    
    # Final local optimization from best point
    if time.time() < deadline - 5:
        try:
            result_final = minimize(
                objective,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if result_final.fun < best_score:
                candidate = result_final.x.reshape(n, d)
                candidate_score = compute_score(candidate)
                
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate
        except Exception:
            pass
    
    # Ensure valid output
    if not np.all(np.isfinite(best_points)):
        best_points = fibonacci_sphere(n, d)
    
    return best_points


# EVOLVE-BLOCK-END
