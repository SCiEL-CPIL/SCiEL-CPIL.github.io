"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import random

def _make_initial_grid(n):
    """Create a simple grid-based initial configuration."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append([x, y, r])
    return np.array(cells[:n])


def _objective(params, n):
    """
    Objective function: minimize negative sum of radii + penalty for violations.
    params: array of shape (3n,) -> [x1,y1,r1, x2,y2,r2, ...]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Penalty for circles going outside [0,1]^2
    penalty_boundary = 0.0
    for i in range(n):
        if x[i] < r[i]:
            penalty_boundary += (r[i] - x[i]) ** 2
        if x[i] > 1.0 - r[i]:
            penalty_boundary += (x[i] - (1.0 - r[i])) ** 2
        if y[i] < r[i]:
            penalty_boundary += (r[i] - y[i]) ** 2
        if y[i] > 1.0 - r[i]:
            penalty_boundary += (y[i] - (1.0 - r[i])) ** 2
    
    # Penalty for overlapping circles
    penalty_overlap = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                penalty_overlap += (min_dist - dist) ** 2
    
    # Sum of radii (we want to maximize, so minimize negative)
    sum_r = np.sum(r)
    
    # Large penalty weight to enforce constraints
    penalty_weight = 100.0
    return -sum_r + penalty_weight * (penalty_boundary + penalty_overlap)


def _objective_grad(params, n):
    """
    Gradient of the objective function.
    """
    x = params[0::3].copy()
    y = params[1::3].copy()
    r = params[2::3].copy()
    
    grad = np.zeros_like(params)
    
    # Boundary penalties
    penalty_weight = 100.0
    for i in range(n):
        # x boundaries
        if x[i] < r[i]:
            grad[3*i] += 2 * penalty_weight * (r[i] - x[i])  # d/dx of (r-x)^2
            grad[3*i+2] -= 2 * penalty_weight * (r[i] - x[i])  # d/dr of (r-x)^2
        if x[i] > 1.0 - r[i]:
            grad[3*i] -= 2 * penalty_weight * (x[i] - (1.0 - r[i]))
            grad[3*i+2] += 2 * penalty_weight * (x[i] - (1.0 - r[i]))
        
        # y boundaries
        if y[i] < r[i]:
            grad[3*i+1] += 2 * penalty_weight * (r[i] - y[i])
            grad[3*i+2] -= 2 * penalty_weight * (r[i] - y[i])
        if y[i] > 1.0 - r[i]:
            grad[3*i+1] -= 2 * penalty_weight * (y[i] - (1.0 - r[i]))
            grad[3*i+2] += 2 * penalty_weight * (y[i] - (1.0 - r[i]))
    
    # Overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist and dist > 1e-10:
                diff = min_dist - dist
                factor = 2 * penalty_weight * diff / dist
                grad[3*i] += factor * dx
                grad[3*i+1] += factor * dy
                grad[3*i+2] += 2 * penalty_weight * diff
                grad[3*j] -= factor * dx
                grad[3*j+1] -= factor * dy
                grad[3*j+2] += 2 * penalty_weight * diff
            elif dist <= 1e-10:
                # Handle degenerate case
                grad[3*i+2] += 2 * penalty_weight * min_dist
                grad[3*j+2] += 2 * penalty_weight * min_dist
    
    # Gradient of -sum(r)
    grad[2::3] -= 1.0
    
    return grad


def _is_valid(circles, tol=1e-6):
    """Check if circles are valid (inside unit square, no overlaps)."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < 0:
            return False
        if x < -tol or x > 1.0 + tol or y < -tol or y > 1.0 + tol:
            return False
        if x - r < -tol or x + r > 1.0 + tol or y - r < -tol or y + r > 1.0 + tol:
            return False
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dx = x1 - x2
            dy = y1 - y2
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r1 + r2 - tol:
                return False
    return True


def _project_to_valid(circles):
    """Project circles to be inside unit square with positive radii."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        r = max(r, 1e-6)
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        circles[i] = [x, y, r]
    return circles


def construct():
    """
    Pack 26 circles with disjoint interiors inside [0,1]^2 and maximize the sum of their radii.
    Returns array of shape (n, 3): rows (x, y, radius) of n disjoint circles inside the unit square.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with grid configuration
    circles = _make_initial_grid(N)
    best_circles = circles.copy()
    best_score = np.sum(circles[:, 2])
    
    # Multi-start optimization
    n_restarts = 15
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Create initial configuration
        if restart == 0:
            params = circles.flatten().copy()
        else:
            # Perturb the best solution or use random initialization
            if restart < 5:
                # Small perturbation of best
                params = best_circles.flatten().copy()
                noise = rng.randn(3 * N) * 0.05
                params += noise
            else:
                # Larger perturbation or random
                params = best_circles.flatten().copy()
                noise = rng.randn(3 * N) * 0.2
                params += noise
        
        # Ensure radii are positive
        params[2::3] = np.maximum(params[2::3], 1e-4)
        
        # Run optimization
        try:
            result = minimize(
                _objective,
                params,
                args=(N,),
                method='L-BFGS-B',
                jac=_objective_grad,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
        except Exception:
            continue
        
        # Extract result
        new_params = result.x
        new_circles = new_params.reshape(N, 3)
        new_circles[:, 2] = np.maximum(new_circles[:, 2], 1e-6)
        new_circles = _project_to_valid(new_circles)
        
        # Check validity
        if _is_valid(new_circles):
            new_score = np.sum(new_circles[:, 2])
            if new_score > best_score:
                best_score = new_score
                best_circles = new_circles.copy()
        
        # Also try a local refinement step
        if time.time() > deadline - 10.0:
            break
        
    # Final projection to ensure validity
    best_circles = _project_to_valid(best_circles)
    
    # If still invalid, fall back to grid
    if not _is_valid(best_circles):
        best_circles = _make_initial_grid(N)
    
    return best_circles
# EVOLVE-BLOCK-END
