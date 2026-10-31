"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# Pre-compute triangle indices for efficiency
from itertools import combinations
_tri_indices = np.array(list(combinations(range(N), 3)))

def _min_triangle_area(points):
    """
    points: (N, 2) array of Cartesian coordinates
    Returns: minimum area of any triangle formed by 3 points
    """
    p1 = points[_tri_indices[:, 0]]
    p2 = points[_tri_indices[:, 1]]
    p3 = points[_tri_indices[:, 2]]
    
    # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _objective(x):
    """
    Objective function for optimization.
    x: (22,) array of 11 points' coordinates
    Returns: negative of the minimum triangle area (to maximize)
    """
    points = x.reshape(N, 2)
    
    # Check if all points are inside the equilateral triangle
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # For a point (px, py) to be inside:
    # 1. py >= 0
    # 2. py <= sqrt(3) * px  (left edge)
    # 3. py <= sqrt(3) * (1 - px)  (right edge)
    
    px = points[:, 0]
    py = points[:, 1]
    
    # Check constraints
    if np.any(py < -1e-10) or np.any(py > np.sqrt(3) * px + 1e-10) or \
       np.any(py > np.sqrt(3) * (1 - px) + 1e-10):
        return 1e10  # Large penalty for points outside
    
    return -_min_triangle_area(points)

# Initial configuration based on known good Heilbronn triangle solutions
def _initial_points():
    """
    Generate initial points based on known configurations.
    Uses a mix of boundary and interior points.
    """
    # Known optimal or near-optimal configurations for small N
    # For N=11, we'll use a structured initial guess
    
    # Start with points on the boundary and interior
    # Using a hexagonal-like packing pattern
    
    points = []
    
    # Add points near the three vertices
    points.append([0.05, 0.05])
    points.append([0.95, 0.05])
    points.append([0.5, 0.80])
    
    # Add points along the edges
    points.append([0.3, 0.05])
    points.append([0.7, 0.05])
    points.append([0.2, 0.35])
    points.append([0.8, 0.35])
    
    # Add interior points
    points.append([0.5, 0.25])
    points.append([0.35, 0.55])
    points.append([0.65, 0.55])
    points.append([0.5, 0.15])
    
    return np.array(points)

def _refine(points, max_iters=500, time_limit=60):
    """
    Refine points using L-BFGS-B optimization.
    """
    x0 = points.flatten()
    
    # Set bounds for each coordinate
    # x: [0, 1], y: [0, sqrt(3)/2]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, np.sqrt(3) / 2))
    
    start_time = time.time()
    
    result = minimize(_objective, x0, method='L-BFGS-B', 
                      bounds=bounds, options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-8})
    
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        # Return the current best if timeout
        pass
    
    return result.x.reshape(N, 2), result.fun

def construct():
    """
    Main function to construct 11 points inside the unit equilateral triangle
    that maximize the minimum triangle area.
    """
    start_time = time.time()
    deadline = 110  # 110 seconds to leave safety margin
    
    # Initial points
    points = _initial_points()
    best_points = points.copy()
    best_area = _min_triangle_area(points)
    
    # Multiple restarts with different initial configurations
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:  # Leave 30 seconds for final refinement
            break
        
        if restart == 0:
            current_points = best_points.copy()
        else:
            # Add some randomness to explore different regions
            rng = np.random.RandomState(restart * 123 + 456)
            current_points = best_points + rng.randn(N, 2) * 0.1
            # Project back into triangle
            for i in range(N):
                px, py = current_points[i]
                # Clamp to triangle
                if py < 0:
                    py = 0.01
                if px < 0:
                    px = 0.01
                if px > 1:
                    px = 0.99
                if py > np.sqrt(3) * px:
                    py = np.sqrt(3) * px * 0.99
                if py > np.sqrt(3) * (1 - px):
                    py = np.sqrt(3) * (1 - px) * 0.99
                current_points[i] = [px, py]
        
        refined_points, fun = _refine(current_points, max_iters=300, time_limit=15)
        
        area = -fun
        if area > best_area and not np.isinf(area):
            best_area = area
            best_points = refined_points
    
    # Final refinement with more iterations
    if time.time() < deadline - 10:
        final_points, fun = _refine(best_points, max_iters=1000, time_limit=deadline - time.time() - 5)
        area = -fun
        if area > best_area and not np.isinf(area):
            best_points = final_points
    
    return best_points
