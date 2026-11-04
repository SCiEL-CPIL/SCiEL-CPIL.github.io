"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by the points.
    points: (N, 2) array
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Vectorized calculation of all triangle areas
    # i < j < k
    # We can compute cross products for all pairs, but simpler to loop for N=14
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(P_j - P_i, P_k - P_i)|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _objective(x):
    """
    Objective function for minimization.
    We want to MAXIMIZE the min triangle area, so we minimize its negative.
    """
    points = x.reshape((N, 2))
    # Ensure points are within bounds (DE handles this, but minimize might drift)
    # Actually, DE and Nelder-Mead with bounds handle this.
    area = _min_triangle_area(points)
    return -area

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin
    
    # Initial guess: Regular 14-gon inscribed in [0,1]x[0,1]
    # Or a grid. Let's use a regular polygon scaled to fit.
    # A regular 14-gon has better symmetry than a grid for "max min area".
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # Scale to fit in [0,1]
    # Center at 0.5, radius 0.5
    init_x = np.zeros((N, 2))
    init_x[:, 0] = 0.5 + 0.5 * np.cos(angles)
    init_x[:, 1] = 0.5 + 0.5 * np.sin(angles)
    
    # Clip to [0,1] just in case of floating point errors, though radius 0.5 fits perfectly in square if centered?
    # Wait, a circle of radius 0.5 fits in [0,1]x[0,1] only if it's a square of side 1.
    # The points will be within [0,1] if the polygon is inscribed in the square.
    # A regular polygon inscribed in a circle of radius 0.5 centered at (0.5, 0.5) fits in [0,1]x[0,1].
    
    bounds = [(0, 1)] * (2 * N)
    
    best_x = init_x.flatten()
    best_score = -_objective(best_x) # This is the min area
    
    # Try Differential Evolution for global search
    # DE can be slow. Let's use a small population.
    if time.time() - start_time < time_limit * 0.5:
        try:
            de_result = differential_evolution(
                _objective, 
                bounds, 
                maxiter=30, 
                popsize=5, 
                tol=1e-4, 
                seed=42,
                init='latinhypercube',
                polish=True # Let DE polish the result
            )
            if de_result.success or de_result.fun < _objective(best_x):
                best_x = de_result.x
                best_score = -de_result.fun
        except Exception:
            pass

    # Try Nelder-Mead for local refinement on the best found so far
    if time.time() - start_time < time_limit * 0.8:
        try:
            nm_result = minimize(
                _objective, 
                best_x, 
                method='Nelder-Mead', 
                bounds=bounds,
                options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if nm_result.fun < _objective(best_x):
                best_x = nm_result.x
                best_score = -nm_result.fun
        except Exception:
            pass
            
    # One more pass of Nelder-Mead if time permits
    if time.time() - start_time < time_limit * 0.95:
        try:
            nm_result2 = minimize(
                _objective, 
                best_x, 
                method='Nelder-Mead', 
                bounds=bounds,
                options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-7}
            )
            if nm_result2.fun < _objective(best_x):
                best_x = nm_result2.x
                best_score = -nm_result2.fun
        except Exception:
            pass

    points = best_x.reshape((N, 2))
    return points

# EVOLVE-BLOCK-END
