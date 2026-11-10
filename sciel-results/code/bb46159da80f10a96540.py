# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    def get_score(pts):
        """Compute the actual score R = (d_min/d_max)^2"""
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def soft_min(dists, t=1.0):
        """Differentiable approximation of the minimum distance using log-sum-exp."""
        # We want to approximate min(dists)
        # log-sum-exp of -dists/t, then negate and scale
        a = -dists / t
        a_max = np.max(a)
        # Handle case where all a are very negative
        if not np.isfinite(a_max):
            return 0.0
        s = np.sum(np.exp(a - a_max))
        if s == 0:
            return 0.0
        return -t * (a_max + np.log(s / len(dists)))

    def soft_max(dists, t=1.0):
        """Differentiable approximation of the maximum distance using log-sum-exp."""
        a = dists / t
        a_max = np.max(a)
        if not np.isfinite(a_max):
            return 0.0
        s = np.sum(np.exp(a - a_max))
        if s == 0:
            return 0.0
        return t * (a_max + np.log(s / len(dists)))

    def objective(x, t_min=1.0, t_max=1.0):
        """
        Objective function to minimize: -log(soft_min/soft_max)
        which is equivalent to minimizing log(soft_max/soft_min)
        """
        pts = x.reshape(n, d)
        dists = pdist(pts)
        
        # Avoid log of zero or negative values
        sm = soft_min(dists, t_min)
        sx = soft_max(dists, t_max)
        
        if sm <= 0 or sx <= 0 or not np.isfinite(sm) or not np.isfinite(sx):
            return 1e10
        
        return np.log(sx / sm)

    def objective_grad(x, t_min=1.0, t_max=1.0):
        """
        Gradient of the objective function.
        This is a numerical gradient for simplicity, but could be analytic.
        """
        eps = 1e-7
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (objective(x_plus, t_min, t_max) - objective(x_minus, t_min, t_max)) / (2 * eps)
        return grad

    # Initial best solution
    np.random.seed(42)
    best_points = np.random.randn(n, d)
    best_score = get_score(best_points)

    # Strategy 1: Use a hexagonal lattice as a good starting point
    # Then optimize from there
    # Generate a hexagonal lattice pattern
    hex_points = []
    # Center point
    hex_points.append([0, 0])
    # First ring: 6 points
    for i in range(6):
        angle = i * np.pi / 3
        hex_points.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
    # Second ring: 12 points
    for i in range(12):
        angle = i * np.pi / 6 + np.pi / 6
        hex_points.append([2.0 * np.cos(angle), 2.0 * np.sin(angle)])
    
    # We have 1 + 6 + 12 = 19 points, we need 16
    # Take the first 16
    hex_points = np.array(hex_points[:16])
    
    # Center and scale
    hex_points = hex_points - np.mean(hex_points, axis=0)
    hex_points = hex_points / np.max(np.abs(hex_points))
    
    score_hex = get_score(hex_points)
    if score_hex > best_score:
        best_score = score_hex
        best_points = hex_points.copy()

    # Strategy 2: Use L-BFGS-B with smooth surrogate to optimize
    # We'll do multiple restarts with different initializations
    np.random.seed(42)
    
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 100:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Start from hex lattice
            x0 = hex_points.flatten()
        elif restart == 1:
            # Random initialization
            x0 = np.random.randn(n * d) * 2
        elif restart == 2:
            # Circle initialization
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
        elif restart == 3:
            # Square grid initialization
            grid_x = np.linspace(-1, 1, 4)
            grid_y = np.linspace(-1, 1, 4)
            xx, yy = np.meshgrid(grid_x, grid_y)
            x0 = np.column_stack([xx.flatten(), yy.flatten()]).flatten()
        else:
            # Perturbed hex lattice
            x0 = hex_points.flatten() + np.random.randn(n * d) * 0.1
        
        # Try different temperature values for the soft min/max
        t_values = [0.1, 0.5, 1.0]
        
        for t in t_values:
            if time.time() > deadline - 80:
                break
            
            try:
                result = minimize(
                    objective,
                    x0,
                    args=(t, t),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.success or result.fun < 1e9:
                    pts = result.x.reshape(n, d)
                    score = get_score(pts)
                    if score > best_score:
                        best_score = score
                        best_points = pts.copy()
            except Exception:
                pass

    # Strategy 3: Nelder-Mead polish on the best solution found
    if time.time() < deadline - 50:
        try:
            # Use a tighter tolerance for the final polish
            result_nm = minimize(
                lambda x: -get_score(x.reshape(n, d)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            
            pts_nm = result_nm.x.reshape(n, d)
            score_nm = get_score(pts_nm)
            if score_nm > best_score:
                best_score = score_nm
                best_points = pts_nm.copy()
        except Exception:
            pass

    # Strategy 4: One more round of L-BFGS-B with very tight tolerance
    if time.time() < deadline - 30:
        try:
            x0 = best_points.flatten()
            result_final = minimize(
                objective,
                x0,
                args=(0.01, 0.01),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            
            if result_final.success or result_final.fun < 1e9:
                pts_final = result_final.x.reshape(n, d)
                score_final = get_score(pts_final)
                if score_final > best_score:
                    best_score = score_final
                    best_points = pts_final.copy()
        except Exception:
            pass

    return best_points


# EVOLVE-BLOCK-END
