"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _is_valid(circles, tol=1e-9):
    """Check if circles are valid: disjoint interiors and inside [0,1]^2."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - tol:
                return False
    return True

def _objective(params, n, circle_indices):
    """
    Objective to minimize: negative sum of radii.
    params: [x1, y1, r1, x2, y2, r2, ...]
    We use a penalty method for constraints.
    """
    # Unpack
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Ensure radii are positive (soft)
    r = np.maximum(r, 1e-6)
    
    penalty = 0.0
    # Boundary constraints
    for i in range(n):
        if x[i] - r[i] < 0:
            penalty += (x[i] - r[i])**2 * 1000
        if x[i] + r[i] > 1:
            penalty += (x[i] + r[i] - 1)**2 * 1000
        if y[i] - r[i] < 0:
            penalty += (y[i] - r[i])**2 * 1000
        if y[i] + r[i] > 1:
            penalty += (y[i] + r[i] - 1)**2 * 1000
    
    # Disjointness constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                penalty += (min_dist - dist)**2 * 1000
    
    return -np.sum(r) + penalty

def _initial_grid(n):
    """Create an initial grid configuration."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side * 0.9  # slightly smaller to allow movement
                cells.append([x, y, r])
    return np.array(cells)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with grid
    best_circles = _initial_grid(N)
    best_value = -np.sum(best_circles[:, 2])
    
    # Try multiple random restarts with perturbation
    np.random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Perturb the grid
        perturb_scale = 0.1 / (restart + 1)
        circles = best_circles.copy()
        for i in range(N):
            circles[i, 0] += np.random.uniform(-perturb_scale, perturb_scale)
            circles[i, 1] += np.random.uniform(-perturb_scale, perturb_scale)
            circles[i, 2] = max(0.01, circles[i, 2] * (1 + np.random.uniform(-0.2, 0.2)))
        
        # Clamp to unit square
        for i in range(N):
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
        
        params = circles.flatten()
        
        try:
            res = minimize(
                _objective,
                params,
                args=(N, None),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (2 * N) + [(0.001, 0.5)] * N,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
        except Exception:
            continue
        
        if time.time() > deadline:
            break
        
        if res.success or res.fun < 1e6:
            new_params = res.x
            new_circles = new_params.reshape(N, 3)
            new_circles[:, 2] = np.maximum(new_circles[:, 2], 1e-6)
            
            # Clamp to valid range
            for i in range(N):
                r = new_circles[i, 2]
                new_circles[i, 0] = np.clip(new_circles[i, 0], r, 1 - r)
                new_circles[i, 1] = np.clip(new_circles[i, 1], r, 1 - r)
            
            if _is_valid(new_circles, tol=1e-8):
                new_value = np.sum(new_circles[:, 2])
                if new_value > best_value:
                    best_value = new_value
                    best_circles = new_circles
    
    # Final validation
    if not _is_valid(best_circles, tol=1e-7):
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side * 0.95
                    cells.append([x, y, r])
        best_circles = np.array(cells)
    
    return best_circles
# EVOLVE-BLOCK-END
