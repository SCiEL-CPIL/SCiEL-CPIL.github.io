"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial.distance import cdist

N = 21

# EVOLVE-BLOCK-START

def _pack_circles(n, max_time=100.0):
    """
    Pack n circles in a rectangle of perimeter <= 4 to maximize sum of radii.
    Returns array of shape (n, 3) with columns [x, y, r].
    """
    start_time = time.time()
    
    # Initial guess: grid layout
    cols = int(np.ceil(np.sqrt(n * 2)))  # wider than tall since perimeter constraint
    rows = int(np.ceil(n / cols))
    
    # For perimeter 4: 2*(w+h)=4 => w+h=2
    # Let's start with a square-ish rectangle
    w, h = 1.0, 1.0
    
    # Initial positions: grid in unit square
    xs = np.linspace(0.1, 0.9, cols)
    ys = np.linspace(0.1, 0.9, rows)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_x = grid_x.ravel()
    grid_y = grid_y.ravel()
    
    # Take first n
    if len(grid_x) >= n:
        init_x = grid_x[:n].copy()
        init_y = grid_y[:n].copy()
    else:
        # Pad with random positions
        init_x = np.concatenate([grid_x, np.random.rand(n - len(grid_x)) * 0.8 + 0.1])
        init_y = np.concatenate([grid_y, np.random.rand(n - len(grid_y)) * 0.8 + 0.1])
    
    init_r = np.full(n, 0.1)
    
    best_result = None
    best_value = -np.inf
    
    def _evaluate(x, y, r, w, h):
        """Check validity and compute objective."""
        if time.time() - start_time > max_time:
            return None, -np.inf
        
        # Check bounds
        if np.any(x < -r) or np.any(x > w + r) or np.any(y < -r) or np.any(y > h + r):
            return None, -np.inf
        
        # Check perimeter constraint
        # The bounding rectangle of all circles
        x_min = np.min(x - r)
        x_max = np.max(x + r)
        y_min = np.min(y - r)
        y_max = np.max(y + r)
        bw = x_max - x_min
        bh = y_max - y_min
        perimeter = 2 * (bw + bh)
        if perimeter > 4 + 1e-6:
            return None, -np.inf
        
        # Check non-overlap
        dist = cdist(np.column_stack([x, y]), np.column_stack([x, y]))
        min_separations = r[:, None] + r[None, :]
        # For i != j, dist[i,j] >= r_i + r_j
        for i in range(n):
            for j in range(i+1, n):
                if dist[i, j] < min_separations[i, j] - 1e-6:
                    return None, -np.inf
        
        value = np.sum(r)
        return np.column_stack([x, y, r]), value
    
    # We'll use a penalty-based optimization
    # Variables: x (n), y (n), r (n), w, h -> total 2n+2 variables
    def _objective(vars_):
        if time.time() - start_time > max_time:
            return 1e10
        x = vars_[:n]
        y = vars_[n:2*n]
        r = vars_[2*n:3*n]
        w = vars_[3*n]
        h = vars_[3*n+1]
        
        # Perimeter penalty
        x_min = np.min(x - r)
        x_max = np.max(x + r)
        y_min = np.min(y - r)
        y_max = np.max(y + r)
        bw = x_max - x_min
        bh = y_max - y_min
        perim = 2 * (bw + bh)
        perim_pen = max(0, perim - 4) ** 2 * 100
        
        # Boundary penalty (circles should be within [0,w] x [0,h] roughly)
        # Actually, we want circles to fit in the bounding box, so let's just use the bounding box
        # but add a small penalty if circles go outside [0, w] x [0, h]
        bound_pen = 0
        for i in range(n):
            bound_pen += max(0, -(x[i] - r[i]))**2
            bound_pen += max(0, (x[i] + r[i]) - w)**2
            bound_pen += max(0, -(y[i] - r[i]))**2
            bound_pen += max(0, (y[i] + r[i]) - h)**2
        
        # Overlap penalty
        overlap_pen = 0
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                min_d = r[i] + r[j]
                if d < min_d:
                    overlap_pen += (min_d - d)**2 * 50
        
        # Negative sum of radii (we minimize negative)
        return -np.sum(r) + perim_pen + bound_pen + overlap_pen
    
    # Try multiple random initializations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() - start_time > max_time - 5:
            break
        
        # Random initialization
        x0 = np.random.rand(n) * 0.8 + 0.1
        y0 = np.random.rand(n) * 0.8 + 0.1
        r0 = np.random.rand(n) * 0.1 + 0.05
        w0 = 1.0
        h0 = 1.0
        
        x_init = np.concatenate([x0, y0, r0, [w0, h0]])
        
        # Bounds
        bounds = []
        for i in range(n):
            bounds.append((0, 2.0))  # x
        for i in range(n):
            bounds.append((0, 2.0))  # y
        for i in range(n):
            bounds.append((0.01, 0.5))  # r
        bounds.append((0.5, 2.0))  # w
        bounds.append((0.5, 2.0))  # h
        
        try:
            result = minimize(
                _objective,
                x_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'maxfun': 10000}
            )
        except Exception:
            continue
        
        x_opt = result.x[:n]
        y_opt = result.x[n:2*n]
        r_opt = result.x[2*n:3*n]
        w_opt = result.x[3*n]
        h_opt = result.x[3*n+1]
        
        # Validate
        valid_result, value = _evaluate(x_opt, y_opt, r_opt, w_opt, h_opt)
        if valid_result is not None and value > best_value:
            best_value = value
            best_result = valid_result
    
    # Also try the grid initialization
    x_init = np.concatenate([init_x, init_y, init_r, [1.0, 1.0]])
    bounds = []
    for i in range(n):
        bounds.append((0, 2.0))
    for i in range(n):
        bounds.append((0, 2.0))
    for i in range(n):
        bounds.append((0.01, 0.5))
    bounds.append((0.5, 2.0))
    bounds.append((0.5, 2.0))
    
    try:
        result = minimize(
            _objective,
            x_init,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 10000}
        )
        x_opt = result.x[:n]
        y_opt = result.x[n:2*n]
        r_opt = result.x[2*n:3*n]
        w_opt = result.x[3*n]
        h_opt = result.x[3*n+1]
        
        valid_result, value = _evaluate(x_opt, y_opt, r_opt, w_opt, h_opt)
        if valid_result is not None and value > best_value:
            best_value = value
            best_result = valid_result
    except Exception:
        pass
    
    if best_result is None:
        # Fallback: return a simple valid configuration
        # Small circles in a grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result


def construct():
    """
    Returns array of shape (n, 3): n disjoint circles whose bounding rectangle has perimeter at most 4.
    """
    result = _pack_circles(N, max_time=110.0)
    return result

# EVOLVE-BLOCK-END
