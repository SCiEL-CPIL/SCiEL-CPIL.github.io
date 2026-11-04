"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution

N = 13

# EVOLVE-BLOCK-START

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: compute area using shoelace formula if hull fails
        # This shouldn't happen with 13 points, but just in case
        return 1.0


def _score(points):
    """Compute the score: minimum triangle area divided by convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area


def _params_to_points(params):
    """Convert parameter vector to 13 points in the unit disk.
    
    params: array of shape (13, 2)
    Each pair (r, theta) defines a point in polar coordinates.
    We clamp r to [0.01, 0.99] to avoid boundary issues.
    """
    params = np.asarray(params)
    r = np.clip(params[:, 0], 0.01, 0.99)
    theta = params[:, 1]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])


def _objective(params):
    """Objective function for optimization: negative score."""
    points = _params_to_points(params)
    score = _score(points)
    return -score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Initial best: random points
    rng = np.random.RandomState(N)
    best_points = None
    best_score = -float('inf')
    
    # Try multiple random initializations and local refinements
    n_restarts = 8
    max_time_per_restart = (deadline - time.time()) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random initial parameters
        r_init = rng.uniform(0.3, 0.9, N)
        theta_init = rng.uniform(0, 2 * np.pi, N)
        # Sort by angle for better structure
        sort_idx = np.argsort(theta_init)
        r_init = r_init[sort_idx]
        theta_init = theta_init[sort_idx]
        
        initial_params = np.column_stack([r_init, theta_init])
        
        # Use differential evolution with limited population and iterations
        bounds = [(0.01, 0.99), (0, 2 * np.pi)] * N
        bounds = list(zip(*bounds))
        
        try:
            result = differential_evolution(
                _objective,
                bounds,
                maxiter=30,
                popsize=10,
                tol=1e-4,
                seed=N + restart,
                polish=True,
                update_interval=1
            )
            
            if time.time() > deadline - 2:
                break
            
            opt_params = result.x
            opt_points = _params_to_points(opt_params)
            opt_score = _score(opt_points)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
                
        except Exception:
            continue
    
    # If no valid result, return random points
    if best_points is None:
        best_points = np.array([[rng.random(), rng.random()] for _ in range(N)])
    
    return best_points

# EVOLVE-BLOCK-END
