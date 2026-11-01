"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution

def _score(points):
    """Calculate the score: min triangle area / convex hull area."""
    n = len(points)
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0
    
    min_area = float('inf')
    # Pre-extract coordinates for speed
    px = points[:, 0]
    py = points[:, 1]
    
    for i in range(n):
        xi, yi = px[i], py[i]
        for j in range(i + 1, n):
            xj, yj = px[j], py[j]
            dx_ij = xj - xi
            dy_ij = yj - yi
            for k in range(j + 1, n):
                xk, yk = px[k], py[k]
                dx_ik = xk - xi
                dy_ik = yk - yi
                cross = abs(dx_ij * dy_ik - dy_ij * dx_ik)
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
                    
    return min_area / hull_area

def _objective(x):
    points = x.reshape(N, 2)
    return -_score(points)

def construct():
    # Bounds for each coordinate
    bounds = [(-1.5, 1.5)] * (N * 2)
    
    # Use Differential Evolution for global optimization
    # popsize: 20 (multiple of N? N=14, 20 is fine)
    # maxiter: 150
    # seed: fixed for reproducibility in this context, but random in production?
    # The prompt implies a single run.
    
    result = differential_evolution(
        _objective, 
        bounds, 
        popsize=20, 
        maxiter=150, 
        tol=1e-8, 
        seed=12345,
        polish=True # Try to polish with L-BFGS at the end
    )
    
    best_points = result.x.reshape(N, 2)
    
    # Optional: A quick local search (hill climbing) to polish further?
    # DE with polish=True should do this.
    
    return best_points
# EVOLVE-BLOCK-END
