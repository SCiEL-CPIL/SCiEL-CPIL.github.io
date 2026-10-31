"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

# Triangle vertices
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, np.sqrt(3)/2])

# Precompute triangle indices for min area calculation
# There are C(11, 3) = 165 triangles
TRIS = np.array(list(itertools.combinations(range(N), 3)))

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by the points.
    points: (N, 2) array
    """
    p1 = points[TRIS[:, 0]]
    p2 = points[TRIS[:, 1]]
    p3 = points[TRIS[:, 2]]
    
    # Cross product (z-component)
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1])
            
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _penalty(points):
    """
    Calculate penalty for points outside the triangle.
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # y >= 0
    pen1 = np.sum(np.maximum(0, -y)**2)
    
    # y <= sqrt(3) * min(x, 1-x)
    upper = np.sqrt(3) * np.minimum(x, 1 - x)
    # Handle case where x is outside [0,1] more robustly?
    # If x < 0, min(x, 1-x) is x (negative). y - x is positive if y>0.
    # If x > 1, min(x, 1-x) is 1-x (negative). y - (1-x) is positive.
    # This works.
    violation = np.maximum(0, y - upper)
    pen2 = np.sum(violation**2)
    
    return 10.0 * (pen1 + pen2)

def _objective(params):
    """
    Objective function for minimization.
    """
    points = params.reshape(N, 2)
    
    pen = _penalty(points)
    
    if pen > 1e-4:
        return 1000.0 + pen
    
    min_area = _min_triangle_area(points)
    
    return -min_area + pen

def construct():
    """
    Construct 11 points inside the equilateral triangle to maximize min triangle area.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_score = -np.inf
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Helper to generate random points in triangle
    def _random_points():
        u = rng.random(N)
        v = rng.random(N)
        valid = u + v < 1
        while not np.all(valid):
            u = rng.random(N)
            v = rng.random(N)
            valid = u + v < 1
        x = u + 0.5 * v
        y = (np.sqrt(3) / 2) * v
        return np.column_stack((x, y))

    # Initial best points from random
    best_points = _random_points()
    best_score = _min_triangle_area(best_points)
    
    num_starts = 0
    while time.time() < deadline:
        num_starts += 1
        # Generate initial points
        # Mix of random and slightly perturbed best points
        if num_starts % 5 == 0:
            init_points = best_points + rng.normal(0, 0.1, best_points.shape)
        else:
            init_points = _random_points()
            
        # Clip to valid range roughly to help optimizer
        # x in [0, 1], y in [0, sqrt(3)/2]
        init_points[:, 0] = np.clip(init_points[:, 0], 0.01, 0.99)
        init_points[:, 1] = np.clip(init_points[:, 1], 0.01, np.sqrt(3)/2 - 0.01)
        
        try:
            res = minimize(_objective, init_points.flatten(), method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-7})
        except Exception:
            continue
            
        res_points = res.x.reshape(N, 2)
        
        # Check validity
        pen = _penalty(res_points)
        if pen < 1e-3:
            min_area = _min_triangle_area(res_points)
            if min_area > best_score:
                best_score = min_area
                best_points = res_points
        else:
            # If invalid, try to project? Or just ignore.
            # Ignoring is safer.
            pass

    return best_points

# EVOLVE-BLOCK-END
