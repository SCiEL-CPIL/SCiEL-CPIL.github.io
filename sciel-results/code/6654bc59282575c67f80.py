"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _is_valid(circles, eps=1e-7):
    """Check if circles are valid: inside unit square and non-overlapping."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Check bounds
    if np.any(x < -eps) or np.any(x > 1 + eps) or np.any(y < -eps) or np.any(y > 1 + eps):
        return False
    if np.any(r < -eps):
        return False
        
    # Check non-overlapping
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - eps:
                return False
    return True

def _sum_radii(circles):
    return np.sum(circles[:, 2])

def _init_hex(n, seed=0):
    """
    Initialize circles on a hexagonal lattice.
    This provides a high-density, symmetric starting point.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Hexagonal packing in a square:
    # Rows are offset. 
    # Let's estimate rows and cols to fit ~n circles.
    # For a square, roughly sqrt(n) rows.
    side = int(np.ceil(np.sqrt(n)))
    
    # We want to place circles in a grid that approximates hex packing.
    # Spacing s. Radius r = s/2.
    # Width = side * s. Height = (side-1) * s * sqrt(3)/2 + s.
    # We want to fit in [0,1]^2.
    # Let's just place them in a grid first, then optimize.
    # A simple grid is a valid starting point for the penalty method.
    # But hex is better. Let's generate a hex grid.
    
    points = []
    s = 0.15 # Initial spacing, will be scaled/optimized
    r_init = s / 2 * 0.9 # Slightly smaller to avoid initial overlap
    
    # Generate a large enough hex grid and pick the best n
    # Or just generate a grid of size side x side
    for i in range(side):
        for j in range(side):
            if len(points) >= n:
                break
            # Hex offset
            x = (j + 0.5 * (i % 2)) * s
            y = i * s * np.sqrt(3) / 2
            
            # Center in the unit square
            # We'll normalize later or just place in [0,1]
            points.append([x, y])
            
    # If we don't have enough, add random ones
    while len(points) < n:
        points.append([random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)])
        
    points = np.array(points[:n])
    
    # Normalize points to fit in [0.05, 0.95] roughly
    # Find bounds
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    
    # Scale to fit in [0.1, 0.9]
    scale_x = 0.8 / (max_x - min_x + 1e-9)
    scale_y = 0.8 / (max_y - min_y + 1e-9)
    scale = min(scale_x, scale_y)
    
    points[:, 0] = (points[:, 0] - min_x) * scale + 0.1
    points[:, 1] = (points[:, 1] - min_y) * scale + 0.1
    
    # Initial radii
    # Estimate based on local spacing
    # For simplicity, start with uniform small radii
    r = np.full(n, 0.05)
    
    circles = np.column_stack((points, r))
    return circles

def _penalty_objective(vec, penalty_weight):
    """
    Objective function to minimize.
    vec: [x1, y1, r1, x2, y2, r2, ...]
    """
    n = len(vec) // 3
    x = vec[0::3]
    y = vec[1::3]
    r = vec[2::3]
    
    # We want to maximize sum(r), so minimize -sum(r)
    obj = -np.sum(r)
    
    # Penalty for overlaps
    overlap_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            # Overlap if dist < r[i] + r[j]
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                overlap_penalty += overlap**2
                
    # Penalty for bounds
    bound_penalty = 0.0
    for i in range(n):
        # x bounds
        if x[i] < r[i]:
            bound_penalty += (r[i] - x[i])**2
        if x[i] > 1 - r[i]:
            bound_penalty += (x[i] - (1 - r[i]))**2
        # y bounds
        if y[i] < r[i]:
            bound_penalty += (r[i] - y[i])**2
        if y[i] > 1 - r[i]:
            bound_penalty += (y[i] - (1 - r[i]))**2
            
    return obj + penalty_weight * (overlap_penalty + bound_penalty)

def _penalty_gradients(vec, penalty_weight):
    """
    Analytical gradients for the penalty objective.
    """
    n = len(vec) // 3
    x = vec[0::3]
    y = vec[1::3]
    r = vec[2::3]
    
    grad_x = np.zeros(n)
    grad_y = np.zeros(n)
    grad_r = -np.ones(n) # Derivative of -sum(r)
    
    # Overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy) + 1e-12
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                # d(overlap^2)/dx_i = 2 * overlap * (-dx/dist)
                # d(overlap^2)/dx_j = 2 * overlap * (dx/dist)
                # d(overlap^2)/dr_i = 2 * overlap * 1
                # d(overlap^2)/dr_j = 2 * overlap * 1
                
                factor = 2 * penalty_weight * overlap / dist
                
                grad_x[i] -= factor * dx
                grad_x[j] += factor * dx
                grad_y[i] -= factor * dy
                grad_y[j] += factor * dy
                grad_r[i] += 2 * penalty_weight * overlap
                grad_r[j] += 2 * penalty_weight * overlap
                
    # Bounds
    for i in range(n):
        if x[i] < r[i]:
            # d(r - x)^2 / dx = -2(r-x)
            grad_x[i] += 2 * penalty_weight * (r[i] - x[i]) * -1
            # d(r - x)^2 / dr = 2(r-x)
            grad_r[i] += 2 * penalty_weight * (r[i] - x[i])
            
        if x[i] > 1 - r[i]:
            # d(x - (1-r))^2 / dx = 2(x - 1 + r)
            grad_x[i] += 2 * penalty_weight * (x[i] - (1 - r[i]))
            # d(x - (1-r))^2 / dr = -2(x - 1 + r)
            grad_r[i] += 2 * penalty_weight * (x[i] - (1 - r[i])) * -1
            
        if y[i] < r[i]:
            grad_y[i] += 2 * penalty_weight * (r[i] - y[i]) * -1
            grad_r[i] += 2 * penalty_weight * (r[i] - y[i])
            
        if y[i] > 1 - r[i]:
            grad_y[i] += 2 * penalty_weight * (y[i] - (1 - r[i]))
            grad_r[i] += 2 * penalty_weight * (y[i] - (1 - r[i])) * -1
            
    # Assemble gradient
    grad = np.zeros_like(vec)
    grad[0::3] = grad_x
    grad[1::3] = grad_y
    grad[2::3] = grad_r
    
    return grad

def _optimize_penalty(initial_circles, deadline, start_penalty=100.0):
    """
    Run penalty-based optimization.
    """
    vec = initial_circles.flatten()
    
    # Bounds for the vector
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(len(initial_circles)):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    current_penalty = start_penalty
    best_circles = initial_circles.copy()
    best_score = _sum_radii(initial_circles)
    
    # Iterate with increasing penalty weight
    while time.time() < deadline - 10:
        # Define objective and gradient for current penalty weight
        def obj(v):
            return _penalty_objective(v, current_penalty)
            
        def grad(v):
            return _penalty_gradients(v, current_penalty)
            
        # Optimize
        res = minimize(
            obj, 
            vec, 
            jac=grad, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        vec = res.x
        circles = vec.reshape(-1, 3)
        
        # Check validity
        if _is_valid(circles):
            score = _sum_radii(circles)
            if score > best_score:
                best_score = score
                best_circles = circles.copy()
                
        # Increase penalty weight to enforce constraints more strictly
        current_penalty *= 10.0
        
        # If penalty is very high and we haven't improved much, stop
        if current_penalty > 1e12:
            break
            
    return best_circles, best_score

def construct():
    deadline = time.time() + 140 # Safety margin
    
    best_circles = None
    best_score = -1
    
    # Multiple starts with different seeds for hex initialization
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for seed in seeds:
        if time.time() > deadline - 20:
            break
            
        init_circles = _init_hex(N, seed)
        
        # Ensure initial validity (shrink if needed)
        if not _is_valid(init_circles):
            init_circles[:, 2] *= 0.5
            if not _is_valid(init_circles):
                init_circles[:, 2] *= 0.5
                if not _is_valid(init_circles):
                    continue
        
        optimized_circles, score = _optimize_penalty(init_circles, deadline)
        
        if score > best_score:
            best_score = score
            best_circles = optimized_circles
            
    if best_circles is None:
        # Fallback to grid
        best_circles = _init_hex(N, 1)
        
    return best_circles
# EVOLVE-BLOCK-END
