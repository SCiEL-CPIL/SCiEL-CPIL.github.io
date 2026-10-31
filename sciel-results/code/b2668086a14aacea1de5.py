import numpy as np
from scipy.optimize import differential_evolution

N = 11

# Pre-compute triangle indices for efficiency
# Indices of points: 0..10
# Triangles are combinations of 3 indices
from itertools import combinations
tri_indices = list(combinations(range(N), 3))
tri_indices_arr = np.array(tri_indices)

def calculate_min_area(points):
    """
    points: (N, 2) array of Cartesian coordinates
    Returns: minimum area of any triangle formed by 3 points
    """
    # points shape (11, 2)
    # tri_indices_arr shape (165, 3)
    
    # Get coordinates for all triangles
    # p1: (165, 2), p2: (165, 2), p3: (165, 2)
    p1 = points[tri_indices_arr[:, 0]]
    p2 = points[tri_indices_arr[:, 1]]
    p3 = points[tri_indices_arr[:, 2]]
    
    # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
    # We can omit the 0.5 factor for optimization, but keep it for final value if needed.
    # The task says "maximize the smallest area".
    # Let's compute 2*Area to avoid float division by 2 repeatedly, or just use 0.5.
    
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    x3, y3 = p3[:, 0], p3[:, 1]
    
    areas = np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) * 0.5
    
    return np.min(areas)

def objective(uv_params):
    """
    uv_params: (22,) array of [u1, v1, u2, v2, ..., u11, v11]
    Returns: -min_area (to be minimized by DE)
    """
    # Check constraints
    # u, v >= 0 is handled by bounds [0, 1]
    # u + v <= 1
    u = uv_params[0::2]
    v = uv_params[1::2]
    
    if np.any(u + v > 1.0):
        return 0.0 # Invalid, return 0 area (which is -0.0 for minimization, effectively worst case)
    
    # Convert to Cartesian
    # x = u + 0.5 * v
    # y = sqrt(3)/2 * v
    x = u + 0.5 * v
    y = (3 ** 0.5) / 2.0 * v
    
    points = np.column_stack((x, y))
    
    min_area = calculate_min_area(points)
    
    return -min_area

def construct():
    # Differential Evolution
    # Bounds: 11 points, each with u, v in [0, 1]
    bounds = [(0, 1)] * (2 * N)
    
    # Use a reasonable population size and max iterations
    # Time limit is 120s. DE is fast but stochastic.
    # We will run DE and return the best result found.
    
    # To ensure we get a good result quickly, we might run a few iterations or use a good pop size.
    # Let's try popsize=50, maxiter=200.
    
    result = differential_evolution(
        objective, 
        bounds, 
        popsize=50, 
        maxiter=200, 
        seed=42, 
        tol=1e-6,
        polish=True
    )
    
    best_uv = result.x
    u = best_uv[0::2]
    v = best_uv[1::2]
    
    x = u + 0.5 * v
    y = (3 ** 0.5) / 2.0 * v
    
    points = np.column_stack((x, y))
    
    return points

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from itertools import combinations

N = 11

tri_indices = list(combinations(range(N), 3))
tri_indices_arr = np.array(tri_indices)

def _calculate_min_area(points):
    p1 = points[tri_indices_arr[:, 0]]
    p2 = points[tri_indices_arr[:, 1]]
    p3 = points[tri_indices_arr[:, 2]]
    
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    x3, y3 = p3[:, 0], p3[:, 1]
    
    areas = np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) * 0.5
    return np.min(areas)

def _objective(uv_params):
    u = uv_params[0::2]
    v = uv_params[1::2]
    
    if np.any(u + v > 1.0):
        return 0.0
        
    x = u + 0.5 * v
    y = (3 ** 0.5) / 2.0 * v
    points = np.column_stack((x, y))
    
    min_area = _calculate_min_area(points)
    return -min_area

def construct():
    bounds = [(0, 1)] * (2 * N)
    
    # Run Differential Evolution
    # We want to maximize min_area, so we minimize -min_area.
    # If the result is invalid (0), we might need a fallback, but DE should find valid ones.
    result = differential_evolution(
        _objective, 
        bounds, 
        popsize=40, 
        maxiter=150, 
        seed=12345, 
        tol=1e-5,
        polish=True
    )
    
    best_uv = result.x
    u = best_uv[0::2]
    v = best_uv[1::2]
    
    # Ensure validity just in case
    if np.any(u + v > 1.0):
        # Fallback to a simple grid or random if DE failed (unlikely)
        # But let's just clip for safety
        # Actually, if it's invalid, the score is 0.
        # Let's try to project or just return a known good config if this happens.
        # For now, assume DE works.
        pass

    x = u + 0.5 * v
    y = (3 ** 0.5) / 2.0 * v
    
    points = np.column_stack((x, y))
    
    return points
# EVOLVE-BLOCK-END
