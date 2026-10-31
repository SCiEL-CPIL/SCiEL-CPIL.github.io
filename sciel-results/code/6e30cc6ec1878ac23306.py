"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
DEADLINE = 145.0  # safety margin before 150s kill

def _objective(params, penalty_scale=100.0):
    """Objective to minimize: -sum(radii) + penalty for violations."""
    x = params[:N]
    y = params[N:2*N]
    r = np.abs(params[2*N:]) + 1e-4  # ensure positive radii
    
    # Boundary violations
    boundary_viol = np.maximum(0, r - x) + np.maximum(0, r - (1 - x)) + \
                    np.maximum(0, r - y) + np.maximum(0, r - (1 - y))
    
    # Circle-circle collisions
    total_viol = boundary_viol.sum()
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                total_viol += overlap
    
    return -r.sum() + penalty_scale * total_viol

def _is_valid(circles, tol=1e-6):
    """Check if circles are valid (disjoint, inside unit square)."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0 or x < -tol or y < -tol or x + r > 1 + tol or y + r > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((circles[i][0]-circles[j][0])**2 + (circles[i][1]-circles[j][1])**2)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True

def _try_optimize(initial_params, penalty_scale=50.0, max_iter=5000):
    """Try to optimize from initial params, return (circles, score) or None."""
    result = minimize(_objective, initial_params, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'maxls': 40, 'ftol': 1e-10, 'gtol': 1e-8})
    
    x = result.x[:N]
    y = result.x[N:2*N]
    r = np.abs(result.x[2*N:]) + 1e-4
    
    circles = np.column_stack([x, y, r])
    
    # Post-process: clip to valid region
    for i in range(N):
        r_i = circles[i, 2]
        circles[i, 0] = np.clip(circles[i, 0], r_i, 1 - r_i)
        circles[i, 1] = np.clip(circles[i, 1], r_i, 1 - r_i)
    
    if _is_valid(circles):
        return circles, r.sum()
    return None

def _grid_initial():
    """Grid-based initial configuration."""
    side = 5  # 5x5 grid = 25 cells, need 26
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side * 0.95
            cells.append([cx, cy, r])
    # Add one more circle in a corner area
    cells.append([0.1, 0.1, 0.05])
    cells = cells[:N]
    params = np.array([c[0] for c in cells] + [c[1] for c in cells] + [c[2] for c in cells])
    return params

def _random_initial():
    """Random initial configuration with small circles."""
    x = np.random.uniform(0.1, 0.9, N)
    y = np.random.uniform(0.1, 0.9, N)
    r = np.random.uniform(0.02, 0.08, N)
    return np.concatenate([x, y, r])

def _structured_initial():
    """Structured initial config with larger circles in center, smaller at edges."""
    params_list = []
    # Central cluster
    for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
        x = 0.5 + 0.15 * np.cos(angle)
        y = 0.5 + 0.15 * np.sin(angle)
        r = 0.08
        params_list.extend([x, y, r])
    # Corner circles
    corners = [(0.12, 0.12), (0.88, 0.12), (0.12, 0.88), (0.88, 0.88)]
    for cx, cy in corners:
        params_list.extend([cx, cy, 0.1])
    # Edge midpoints
    edges = [(0.5, 0.08), (0.5, 0.92), (0.08, 0.5), (0.92, 0.5)]
    for ex, ey in edges:
        params_list.extend([ex, ey, 0.09])
    # Fill remaining
    while len(params_list) < 3*N:
        params_list.extend([0.5, 0.5, 0.05])
    return np.array(params_list[:3*N])

def construct():
    start_time = time.time()
    
    best_circles = None
    best_score = 0.0
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Grid-based
    initial_configs.append(_grid_initial())
    
    # Structured
    initial_configs.append(_structured_initial())
    
    # Multiple random restarts
    for _ in range(8):
        initial_configs.append(_random_initial())
    
    # Try each initial configuration with varying penalty scales
    penalty_scales = [30.0, 50.0, 80.0, 120.0]
    
    for init_params in initial_configs:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        for ps in penalty_scales:
            if time.time() - start_time > DEADLINE - 15:
                break
            
            try:
                result = _try_optimize(init_params, penalty_scale=ps, max_iter=3000)
                if result is not None:
                    circles, score = result
                    if score > best_score:
                        best_score = score
                        best_circles = circles.copy()
            except Exception:
                continue
    
    # Final refinement with lower penalty on best solution
    if best_circles is not None and time.time() - start_time < DEADLINE - 10:
        params = np.concatenate([best_circles[:, 0], best_circles[:, 1], best_circles[:, 2]])
        try:
            result = _try_optimize(params, penalty_scale=200.0, max_iter=5000)
            if result is not None:
                circles, score = result
                if score > best_score:
                    best_score = score
                    best_circles = circles.copy()
        except Exception:
            pass
    
    if best_circles is None:
        # Fallback to grid
        return _grid_initial().reshape(N, 3)
    
    return best_circles

# EVOLVE-BLOCK-END
