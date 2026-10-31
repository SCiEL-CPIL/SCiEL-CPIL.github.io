"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools

def _triangle_area(p1, p2, p3):
    """Compute the area of triangle formed by points p1, p2, p3."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three points."""
    min_area = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _is_inside_triangle(p, vertices):
    """Check if point p is inside or on the boundary of triangle with given vertices."""
    # The triangle vertices are (0,0), (1,0), (0.5, sqrt(3)/2)
    x, y = p
    # Check if point is within the triangle using barycentric coordinates
    # or simple bounds
    if y < -1e-10 or y > np.sqrt(3) / 2 + 1e-10:
        return False
    if x < -1e-10 or x > 1 + 1e-10:
        return False
    # For the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2):
    # The left edge goes from (0,0) to (0.5, sqrt(3)/2): y = sqrt(3)*x => x >= y/sqrt(3)
    # The right edge goes from (1,0) to (0.5, sqrt(3)/2): y = -sqrt(3)*(x-1) => x <= 1 - y/sqrt(3)
    if y > 0:
        if x < y / np.sqrt(3) - 1e-10:
            return False
        if x > 1 - y / np.sqrt(3) + 1e-10:
            return False
    return True

def _objective(params):
    """
    Objective function to minimize the negative of the minimum triangle area.
    params: array of shape (2*N,) representing N points in 2D.
    """
    points = params.reshape(N, 2)
    
    # Check if all points are inside the triangle
    for p in points:
        if not _is_inside_triangle(p, None):
            return 1e6  # Large penalty for invalid points
    
    min_area = _min_triangle_area(points)
    return -min_area

def _penalty_objective(params):
    """
    Objective with penalty for points outside the triangle.
    """
    points = params.reshape(N, 2)
    penalty = 0.0
    for p in points:
        x, y = p
        # Distance from left edge: y - sqrt(3)*x (should be >= 0)
        d_left = max(0, y - np.sqrt(3) * x)
        # Distance from right edge: sqrt(3)*(1-x) - y (should be >= 0)
        d_right = max(0, np.sqrt(3) * (1 - x) - y)
        # Distance from bottom: y (should be >= 0)
        d_bottom = max(0, y)
        # Distance from top: sqrt(3)/2 - y (should be >= 0)
        d_top = max(0, np.sqrt(3) / 2 - y)
        penalty += d_left**2 + d_right**2 + d_bottom**2 + d_top**2
    
    min_area = _min_triangle_area(points)
    return -min_area + 100.0 * penalty

def _project_to_triangle(x, y):
    """Project point (x,y) onto the equilateral triangle."""
    # Clamp y
    y = max(0.0, min(y, np.sqrt(3) / 2))
    
    # For given y, x must satisfy: y/sqrt(3) <= x <= 1 - y/sqrt(3)
    x_min = y / np.sqrt(3)
    x_max = 1 - y / np.sqrt(3)
    
    if x < x_min:
        x = x_min
    elif x > x_max:
        x = x_max
    
    return x, y

def _project_points(params):
    """Project all points to be inside the triangle."""
    points = params.reshape(N, 2)
    projected = np.zeros_like(points)
    for i in range(N):
        x, y = _project_to_triangle(points[i][0], points[i][1])
        projected[i][0] = x
        projected[i][1] = y
    return projected.flatten()

def construct():
    """
    Construct 11 points inside an equilateral triangle to maximize the minimum
    triangle area.
    """
    best_params = None
    best_score = -float('inf')
    
    # Try multiple random initializations
    for seed in range(20):
        rng = np.random.RandomState(seed)
        
        # Initialize points randomly inside the triangle
        params = np.zeros(2 * N)
        for i in range(N):
            u, v = rng.random(), rng.random()
            while u + v > 1:
                u, v = rng.random(), rng.random()
            x = u + 0.5 * v
            y = (np.sqrt(3) / 2) * v
            params[2 * i] = x
            params[2 * i + 1] = y
        
        # Use gradient descent with projection
        lr = 0.01
        for iteration in range(500):
            # Compute gradient numerically
            grad = np.zeros(2 * N)
            eps = 1e-5
            for j in range(2 * N):
                params_plus = params.copy()
                params_plus[j] += eps
                f_plus = _penalty_objective(params_plus)
                
                params_minus = params.copy()
                params_minus[j] -= eps
                f_minus = _penalty_objective(params_minus)
                
                grad[j] = (f_plus - f_minus) / (2 * eps)
            
            # Gradient descent step
            params = params - lr * grad
            
            # Project to triangle
            params = _project_points(params)
            
            # Adaptive learning rate
            if iteration % 50 == 0:
                lr *= 0.95
        
        score = -_penalty_objective(params)
        if score > best_score:
            best_score = score
            best_params = params.copy()
    
    # Convert to array of points
    points = best_params.reshape(N, 2)
    
    # Final check: ensure all points are strictly inside
    for i in range(N):
        x, y = _project_to_triangle(points[i][0], points[i][1])
        points[i][0] = x
        points[i][1] = y
    
    return points
# EVOLVE-BLOCK-END
