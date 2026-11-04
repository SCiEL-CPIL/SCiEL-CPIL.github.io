"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull

def _compute_score(points):
    """
    Compute the score: minimum triangle area / convex hull area.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    
    # Check for degenerate hull
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0

    # Compute all triangle areas
    # There are C(13, 3) = 286 triangles
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area of triangle formed by points i, j, k
                # 0.5 * | (x_j - x_i)(y_k - y_i) - (x_k - x_i)(y_j - y_i) |
                dx1 = points[j, 0] - points[i, 0]
                dy1 = points[j, 1] - points[i, 1]
                dx2 = points[k, 0] - points[i, 0]
                dy2 = points[k, 1] - points[i, 1]
                cross = abs(dx1 * dy2 - dx2 * dy1)
                area = 0.5 * cross
                if area < min_area:
                    min_area = area

    return min_area / hull_area

def _score(points):
    """
    Objective function for minimization (negative score).
    """
    return -_compute_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    best_points = None
    best_score = -np.inf

    # Strategy 1: Start with a regular 13-gon on the unit circle
    # This is a strong initial guess for convex regions.
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize to unit area? The task says "normalized to unit area" but the score 
    # is "minimum triangle area divided by the convex hull area". 
    # The scale of the points doesn't matter for the ratio, as both numerator and 
    # denominator scale by the square of the linear scale.
    # So we can work in any coordinate system.
    
    score0 = _compute_score(initial_points)
    best_score = score0
    best_points = initial_points.copy()

    # Strategy 2: Local optimization using scipy.optimize.minimize
    # We will try to maximize the minimum triangle area relative to the hull area.
    # Since the objective is non-smooth (min of linear functions), we can use
    # a soft-min approximation or just rely on the optimizer's ability to handle
    # non-smooth objectives (e.g., Nelder-Mead or Powell).
    
    # Let's use Powell's method which doesn't require gradients and can handle
    # non-smooth functions reasonably well.
    
    # To make the optimization more effective, we can fix the centroid to the origin
    # and perhaps fix the scale. However, since the score is scale-invariant,
    # we can just optimize the 2N coordinates.
    
    # We might get stuck in local minima. Let's try a few restarts with perturbed initial points.
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x0 = initial_points
        else:
            # Perturb the initial points
            noise_scale = 0.1 * (1.0 / (restart + 1))
            x0 = initial_points + noise_scale * np.random.randn(N, 2)
        
        try:
            # Use Powell's method
            result = scipy_minimize(_score, x0, method='Powell', options={'maxiter': 200, 'tol': 1e-6})
            
            if result.fun < -best_score:
                best_score = -result.fun
                best_points = result.x.reshape(N, 2)
        except Exception:
            pass

    # Strategy 3: Try a different initial configuration: Points on a circle with random radii
    # Or maybe points on a regular polygon but with some radial perturbation.
    
    # Let's also try a "grid-like" or "random" start and optimize from there.
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Random points in a disk
        r = np.sqrt(np.random.rand(N))
        theta = np.random.rand(N) * 2 * np.pi
        x0 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        
        try:
            result = scipy_minimize(_score, x0, method='Powell', options={'maxiter': 200, 'tol': 1e-6})
            
            if result.fun < -best_score:
                best_score = -result.fun
                best_points = result.x.reshape(N, 2)
        except Exception:
            pass

    # Final polish: If we have time, do a very fine-tuned optimization on the best point found
    if best_points is not None and time.time() < deadline - 5:
        try:
            result = scipy_minimize(_score, best_points, method='Powell', options={'maxiter': 500, 'tol': 1e-8})
            if result.fun < -best_score:
                best_score = -result.fun
                best_points = result.x.reshape(N, 2)
        except Exception:
            pass

    return best_points

# We need to import scipy.optimize.minimize
from scipy.optimize import minimize as scipy_minimize

# EVOLVE-BLOCK-END
