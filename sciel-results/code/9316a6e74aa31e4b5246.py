"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _make_initial_grid():
    """Create a grid of 21 circles in a unit square (perimeter 4)."""
    # Use a 5x5 grid, take 21 of them
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append([x, y, r])
    return np.array(cells[:N])

def _compute_score(circles):
    """Compute sum of radii if valid, else 0."""
    n = len(circles)
    if n != N:
        return 0.0
    # Check bounding box
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    # All radii must be positive
    if np.any(rs <= 0):
        return 0.0
    
    # Bounding box: circles must be inside [min_x - r, max_x + r] etc.
    # Actually, the bounding rectangle of the circles has:
    # left = min(x_i - r_i), right = max(x_i + r_i)
    # bottom = min(y_i - r_i), top = max(y_i + r_i)
    left = np.min(xs - rs)
    right = np.max(xs + rs)
    bottom = np.min(ys - rs)
    top = np.max(ys + rs)
    
    width = right - left
    height = top - bottom
    perimeter = 2 * (width + height)
    
    if perimeter > 4 + 1e-9:
        return 0.0
    
    # Check disjoint interiors: for all pairs, distance >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-9:
                return 0.0
    
    return np.sum(rs)

def _penalized_objective(params):
    """
    Objective: minimize negative sum of radii + penalty for violations.
    params: [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
    """
    n = N
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    penalty = 0.0
    
    # Radii must be positive
    penalty += 1000 * np.sum(np.maximum(0, -rs)) ** 2
    
    # Perimeter constraint: 2*(width + height) <= 4
    left = np.min(xs - rs)
    right = np.max(xs + rs)
    bottom = np.min(ys - rs)
    top = np.max(ys + rs)
    width = right - left
    height = top - bottom
    perimeter = 2 * (width + height)
    if perimeter > 4:
        penalty += 100 * (perimeter - 4) ** 2
    
    # Disjointness: dist(i,j) >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            violation = rs[i] + rs[j] - dist
            if violation > 0:
                penalty += 50 * violation ** 2
    
    return -np.sum(rs) + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Start with grid
    best_circles = _make_initial_grid()
    best_score = _compute_score(best_circles)
    
    # Try multiple random restarts with local optimization
    random.seed(42)
    np.random.seed(42)
    
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Create perturbed initial guess
        if restart == 0:
            init = _make_initial_grid().flatten()
        else:
            # Random perturbation of grid
            base = _make_initial_grid()
            scale = 0.1 + 0.05 * restart
            init = base.flatten() + np.random.randn(3 * N) * scale
            # Ensure radii are positive
            init[2::3] = np.maximum(init[2::3], 0.01)
        
        # Run L-BFGS-B optimization
        try:
            result = minimize(
                _penalized_objective,
                init,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if time.time() > deadline:
                break
            
            # Reconstruct circles
            circles = result.x.reshape(N, 3)
            # Ensure radii are positive
            circles[:, 2] = np.maximum(circles[:, 2], 0.001)
            
            score = _compute_score(circles)
            if score > best_score:
                best_score = score
                best_circles = circles.copy()
                
        except Exception:
            continue
    
    # Final validation
    final_score = _compute_score(best_circles)
    if final_score <= 0:
        # Fallback to grid
        best_circles = _make_initial_grid()
    
    return best_circles
# EVOLVE-BLOCK-END
