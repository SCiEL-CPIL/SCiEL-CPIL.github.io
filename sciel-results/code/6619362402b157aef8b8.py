import random
import numpy as np
from scipy.optimize import differential_evolution
import time
import itertools

N = 11
# Vertices of the unit-area equilateral triangle
# Area = 0.5 * base * height = 0.5 * 1 * sqrt(3)/2 = sqrt(3)/4.
# Wait, the task says "unit-area triangle".
# The prompt says: "The notebook uses an equilateral triangle and divides by its area."
# And: "construct() -> array of shape (n, 2): n points inside the equilateral triangle (0,0), (1,0), (1/2, sqrt(3)/2)"
# The triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2) has area sqrt(3)/4.
# The evaluator measures the "raw value".
# "The search score is value/seed".
# The current program returns points in this specific triangle.
# So I should return points in this specific triangle.
# The "raw value" is likely the minimum area of the triangles formed by the points.
# If the evaluator divides by the area of the big triangle, it would be a normalized score.
# However, the prompt says "maximize the smallest area...".
# Let's assume the objective to maximize is the raw minimum area.

# Pre-compute triangle indices
indices = list(itertools.combinations(range(N), 3))
idx = np.array(indices)
i1 = idx[:, 0]
i2 = idx[:, 1]
i3 = idx[:, 2]

def calculate_min_area(points):
    """
    points: array of shape (N, 2)
    Returns the minimum area of any triangle formed by 3 points.
    """
    # points are already in Cartesian coordinates
    p1 = points[i1]
    p2 = points[i2]
    p3 = points[i3]
    
    # Vectorized area calculation
    # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
    dx12 = p2[:, 0] - p1[:, 0]
    dy12 = p2[:, 1] - p1[:, 1]
    dx13 = p3[:, 0] - p1[:, 0]
    dy13 = p3[:, 1] - p1[:, 1]
    
    cross = dx12 * dy13 - dx13 * dy12
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def objective(x):
    """
    x: flat array of length 2*N
    Represents 11 points in barycentric-like coords (u, v) for each point.
    P = (1-u-v)A + uB + vC
    A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    x_cart = u*B + v*C = (u + 0.5v, sqrt(3)/2 * v)
    
    Constraints: u >= 0, v >= 0, u+v <= 1
    """
    # Check time
    if time.time() - start_time > 110:
        # Return a very bad score to stop optimization? 
        # DE doesn't support dynamic stopping easily mid-function without exceptions or global flags.
        # We will rely on maxiter and popsize to be conservative.
        pass
        
    uv = x.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    
    # Check constraints
    # If any point is outside, return 0 (or negative infinity for maximization)
    if np.any(u < 0) or np.any(v < 0) or np.any(u + v > 1):
        return 0.0
        
    # Convert to Cartesian
    # x = u * 1 + v * 0.5
    # y = u * 0 + v * sqrt(3)/2
    sqrt3_2 = np.sqrt(3) / 2
    cart_points = np.column_stack((u + 0.5 * v, sqrt3_2 * v))
    
    min_area = calculate_min_area(cart_points)
    return min_area

def construct():
    global start_time
    start_time = time.time()
    
    # Define bounds for u and v
    # u in [0, 1], v in [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Differential Evolution parameters
    # We want to maximize objective, so we minimize -objective
    def neg_objective(x):
        return -objective(x)
        
    # Strategy: Run DE with a moderate population size and iterations
    # If time permits, we could do multiple restarts, but DE is already stochastic.
    # Let's try one robust run.
    
    try:
        result = differential_evolution(
            neg_objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15, 
            tol=1e-6,
            init='sobol', # More uniform initial population
            polish=True
        )
        
        best_x = result.x
        best_val = -result.fun
        
        # If DE failed or returned 0, try a few random restarts or a simpler local search
        if best_val <= 1e-9:
             # Fallback to random search if DE failed completely
             best_val = 0
             best_x = None
             for _ in range(1000):
                 if time.time() - start_time > 110: break
                 uv = np.random.rand(N, 2)
                 # Reject outside
                 mask = (uv[:, 0] + uv[:, 1] <= 1)
                 if not np.all(mask): continue
                 sqrt3_2 = np.sqrt(3) / 2
                 cart_points = np.column_stack((uv[:, 0] + 0.5 * uv[:, 1], sqrt3_2 * uv[:, 1]))
                 val = calculate_min_area(cart_points)
                 if val > best_val:
                     best_val = val
                     best_x = uv.flatten()
        
        if best_x is None:
             # Absolute fallback
             return np.array([[0.5, 0.0]] * N)
             
        # Convert best_x back to Cartesian
        uv = best_x.reshape(N, 2)
        sqrt3_2 = np.sqrt(3) / 2
        cart_points = np.column_stack((uv[:, 0] + 0.5 * uv[:, 1], sqrt3_2 * uv[:, 1]))
        
        return cart_points

    except Exception:
        # Fallback
        return np.array([[0.5, 0.0]] * N)

# EVOLVE-BLOCK-START
# The code above is the new design.
# We keep the N definition and construct function.
# We place the helper functions and constants outside or inside construct?
# The prompt asks to keep function names.
# I will put the logic in construct and helpers at module level for speed if needed, 
# but since construct is the entry point, module level is fine.

# To ensure the EVOLVE-BLOCK structure is respected, I will wrap the new code.
# The current program has:
# N = 11
# # EVOLVE-BLOCK-START
# ...
# # EVOLVE-BLOCK-END

# I will replace the content between the markers.
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time
import itertools

# Pre-compute triangle indices for 11 points
_N_GLOBAL = 11
_indices = list(itertools.combinations(range(_N_GLOBAL), 3))
_idx = np.array(_indices)
_i1 = _idx[:, 0]
_i2 = _idx[:, 1]
_i3 = _idx[:, 2]
_sqrt3_2 = np.sqrt(3) / 2

def _calculate_min_area(points):
    p1 = points[_i1]
    p2 = points[_i2]
    p3 = points[_i3]
    
    dx12 = p2[:, 0] - p1[:, 0]
    dy12 = p2[:, 1] - p1[:, 1]
    dx13 = p3[:, 0] - p1[:, 0]
    dy13 = p3[:, 1] - p1[:, 1]
    
    cross = dx12 * dy13 - dx13 * dy12
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _objective(x, start_time):
    uv = x.reshape(_N_GLOBAL, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    
    if np.any(u < 0) or np.any(v < 0) or np.any(u + v > 1):
        return 0.0
        
    cart_points = np.column_stack((u + 0.5 * v, _sqrt3_2 * v))
    return _calculate_min_area(cart_points)

def construct():
    start_time = time.time()
    bounds = [(0.0, 1.0)] * (2 * _N_GLOBAL)
    
    def neg_objective(x):
        return -_objective(x, start_time)
        
    best_val = 0
    best_x = None
    
    # Run Differential Evolution
    try:
        result = differential_evolution(
            neg_objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=20, 
            tol=1e-5,
            init='sobol',
            polish=True
        )
        
        if result.success or result.fun < 0:
            best_x = result.x
            best_val = -result.fun
    except Exception:
        pass
    
    # If DE didn't find a good solution or failed, do random restarts
    if best_val < 0.01: # Threshold for "good enough"
        rng = np.random.default_rng(42)
        for _ in range(2000):
            if time.time() - start_time > 110:
                break
            uv = rng.random((_N_GLOBAL, 2))
            # Simple rejection for u+v <= 1
            # To sample uniformly from the triangle in uv space, we can just reject
            # But random in square is not uniform in triangle.
            # Better: u = r1 * (1-r2), v = r2 * (1-r1) ... no, that's for uniform in triangle?
            # Actually, generating u,v in [0,1] and rejecting u+v>1 is fine for coverage, just less efficient.
            mask = (uv[:, 0] + uv[:, 1] <= 1)
            if not np.all(mask):
                continue
            
            cart_points = np.column_stack((uv[:, 0] + 0.5 * uv[:, 1], _sqrt3_2 * uv[:, 1]))
            val = _calculate_min_area(cart_points)
            if val > best_val:
                best_val = val
                best_x = uv.flatten()
                
    # Final fallback
    if best_x is None:
        # Return a valid configuration, e.g., center
        # Or just the seed program's output logic
        rng = random.Random(7)
        points = []
        while len(points) < _N_GLOBAL:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, _sqrt3_2 * v])
        return np.array(points)
        
    uv = best_x.reshape(_N_GLOBAL, 2)
    cart_points = np.column_stack((uv[:, 0] + 0.5 * uv[:, 1], _sqrt3_2 * uv[:, 1]))
    return cart_points

# EVOLVE-BLOCK-END
