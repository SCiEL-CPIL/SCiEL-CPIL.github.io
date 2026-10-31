import numpy as np
import time
from scipy.optimize import differential_evolution
from itertools import combinations

N = 11

# Pre-compute triangle indices
# Points are indexed 0 to 10
tri_indices = list(combinations(range(N), 3))
tri_indices = np.array(tri_indices)
I = tri_indices[:, 0]
J = tri_indices[:, 1]
K = tri_indices[:, 2]

def _get_points(x):
    """
    Map 22-dim vector x in [0,1]^22 to 11 points in the unit equilateral triangle.
    x is flattened: [u0, v0, u1, v1, ..., u10, v10]
    """
    x = np.reshape(x, (N, 2))
    u = x[:, 0]
    v = x[:, 1] * (1 - u)
    w = 1 - u - v
    
    # Barycentric to Cartesian
    # A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    # P = u*A + v*B + w*C
    px = 0.5 * w + 1.0 * v
    py = (np.sqrt(3) / 2) * w
    
    return np.column_stack((px, py))

def _min_triangle_area(points):
    """
    Calculate the minimum area of triangles formed by the points.
    points: (N, 2) array
    """
    Pi = points[I]
    Pj = points[J]
    Pk = points[K]
    
    v1 = Pj - Pi
    v2 = Pk - Pi
    
    # Cross product in 2D
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def objective(x):
    """
    Objective function for minimization.
    Returns negative of the minimum triangle area.
    """
    points = _get_points(x)
    min_area = _min_triangle_area(points)
    return -min_area

def construct():
    start_time = time.time()
    timeout = 100.0  # Safety margin for 120s limit
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Run Differential Evolution
    # We can run multiple seeds if time permits, or just one long run
    best_x = None
    best_val = -np.inf
    
    # DE Parameters
    # popsize 20, maxiter 5000 is usually plenty for N=11
    # Let's try to be aggressive but safe
    for seed in [42, 123, 999]:
        if time.time() - start_time > timeout - 10.0:
            break
            
        try:
            res = differential_evolution(
                objective,
                bounds,
                maxiter=3000,
                popsize=20,
                tol=1e-8,
                seed=seed,
                polish=False, # We will polish manually
                init='latinhypercube',
                update_interval=1,
                callback=lambda xk, convergence: None # No callback needed for simple guard
            )
            if res.success or res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
            
    if best_x is None:
        # Fallback to random if DE failed
        x = np.random.rand(2 * N)
        best_x = x
        best_val = objective(x)
        
    # Local Polish using Nelder-Mead
    # This can significantly improve the precision
    if time.time() - start_time < timeout - 5.0:
        try:
            from scipy.optimize import minimize
            res_polish = minimize(
                objective,
                best_x,
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 1000, 'maxfev': 2000}
            )
            if res_polish.fun < best_val:
                best_val = res_polish.fun
                best_x = res_polish.x
        except Exception:
            pass

    # Convert best_x to points
    points = _get_points(best_x)
    
    # Ensure points are strictly inside (clamp if necessary, though mapping ensures it)
    # The mapping u in [0,1], v in [0, 1-u] ensures points are on or inside.
    # If a point is on the edge, area might be 0 if another point is on the same edge?
    # The mapping allows u=0, v=0, etc.
    # If we get a point exactly on a vertex, it's fine.
    # But if two points are identical, area is 0. DE minimizes this, so it will avoid it.
    
    return points
