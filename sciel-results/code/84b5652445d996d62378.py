"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _objective(params):
    """
    params: array of length 2*N + 2 = [x_0, y_0, r_0, x_1, y_1, r_1, ..., x_N-1, y_N-1, r_N-1, W, H]
    Actually let's use: [x_0, y_0, ..., x_{N-1}, y_{N-1}, r_0, ..., r_{N-1}, W, H]
    Total = 2*N + N + 2 = 3*N + 2 = 65
    
    Objective: minimize -sum(r_i) + penalty
    Constraints:
      - Perimeter: 2*(W + H) <= 4  -> W + H <= 2
      - Each circle inside: x_i - r_i >= 0, x_i + r_i <= W, y_i - r_i >= 0, y_i + r_i <= H
      - Pairwise: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    """
    n = N
    # Unpack
    xy = params[:2*n].reshape(n, 2)
    r = params[2*n:3*n]
    W = params[3*n]
    H = params[3*n+1]
    
    # Ensure positive radii and dimensions
    r = np.maximum(r, 1e-6)
    W = max(W, 1e-6)
    H = max(H, 1e-6)
    
    # Objective: negative sum of radii
    obj = -np.sum(r)
    
    # Penalty for perimeter violation
    perim = 2.0 * (W + H)
    if perim > 4.0:
        obj += 100.0 * (perim - 4.0)**2
    
    # Penalty for circles outside rectangle
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        # Left
        if x - ri < 0:
            obj += 50.0 * (x - ri)**2
        # Right
        if x + ri > W:
            obj += 50.0 * (x + ri - W)**2
        # Bottom
        if y - ri < 0:
            obj += 50.0 * (y - ri)**2
        # Top
        if y + ri > H:
            obj += 50.0 * (y + ri - H)**2
    
    # Penalty for overlapping circles
    for i in range(n):
        for j in range(i+1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist * min_dist:
                # Penalty proportional to overlap
                overlap = min_dist - np.sqrt(dist_sq + 1e-12)
                obj += 100.0 * overlap**2
    
    return obj


def _initial_layout_seed(seed):
    """Generate a random initial layout."""
    rng = np.random.RandomState(seed)
    
    # Random aspect ratio between 0.5 and 2.0
    W = rng.uniform(0.8, 1.6)
    H = 2.0 - W  # So W + H = 2, perimeter = 4
    
    # Place circles randomly
    params = np.zeros(3*N + 2)
    for i in range(N):
        r_i = rng.uniform(0.05, 0.25)
        x_i = rng.uniform(r_i, W - r_i)
        y_i = rng.uniform(r_i, H - r_i)
        params[2*i] = x_i
        params[2*i+1] = y_i
        params[2*N + i] = r_i
    params[3*N] = W
    params[3*N+1] = H
    
    return params


def _grid_initial():
    """Grid-based initial layout."""
    # Use a 5x5 grid, take 21 cells
    side = 5
    W = 1.0
    H = 1.0  # Square, perimeter = 4
    
    params = np.zeros(3*N + 2)
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N:
                break
            x = (i + 0.5) / side * W
            y = (j + 0.5) / side * H
            r = 0.5 / side * W - 0.005
            params[2*count] = x
            params[2*count+1] = y
            params[2*N + count] = r
            count += 1
    
    params[3*N] = W
    params[3*N+1] = H
    return params


def _validate(params):
    """Check if params represent a valid packing. Return (is_valid, sum_radii)."""
    n = N
    xy = params[:2*n].reshape(n, 2)
    r = params[2*n:3*n]
    W = params[3*n]
    H = params[3*n+1]
    
    # Check perimeter
    if 2.0 * (W + H) > 4.0 + 1e-6:
        return False, 0.0
    
    # Check positive dimensions and radii
    if W <= 0 or H <= 0:
        return False, 0.0
    if np.any(r <= 0):
        return False, 0.0
    
    # Check circles inside
    for i in range(n):
        if xy[i, 0] - r[i] < -1e-6 or xy[i, 0] + r[i] > W + 1e-6:
            return False, 0.0
        if xy[i, 1] - r[i] < -1e-6 or xy[i, 1] + r[i] > H + 1e-6:
            return False, 0.0
    
    # Check pairwise non-overlap
    for i in range(n):
        for j in range(i+1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - 1e-6:
                return False, 0.0
    
    return True, np.sum(r)


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s
    
    best_params = None
    best_sum = 0.0
    best_valid = False
    
    # Try grid initial
    try:
        p0 = _grid_initial()
        res = minimize(_objective, p0, method='L-BFGS-B', 
                      options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
        if _validate(res.x):
            s = np.sum(res.x[2*N:3*N])
            if s > best_sum:
                best_sum = s
                best_params = res.x.copy()
                best_valid = True
    except:
        pass
    
    if time.time() - start_time > deadline:
        if best_valid and best_params is not None:
            return _format_output(best_params)
        return _format_output(_grid_initial())
    
    # Try multiple random starts
    for seed in range(20):
        if time.time() - start_time > deadline:
            break
        try:
            p0 = _initial_layout_seed(seed * 12345 + 678)
            res = minimize(_objective, p0, method='L-BFGS-B',
                          options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-8})
            if _validate(res.x):
                s = np.sum(res.x[2*N:3*N])
                if s > best_sum:
                    best_sum = s
                    best_params = res.x.copy()
                    best_valid = True
        except:
            pass
    
    # If we have a valid solution, try to refine with more iterations
    if best_valid and best_params is not None:
        if time.time() - start_time < deadline - 10:
            try:
                res = minimize(_objective, best_params, method='L-BFGS-B',
                              options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10})
                if _validate(res.x):
                    s = np.sum(res.x[2*N:3*N])
                    if s > best_sum:
                        best_sum = s
                        best_params = res.x.copy()
            except:
                pass
    
    if best_valid and best_params is not None:
        return _format_output(best_params)
    
    # Fallback: grid
    return _format_output(_grid_initial())


def _format_output(params):
    """Convert params to (N, 3) array of [x, y, r]."""
    n = N
    xy = params[:2*n].reshape(n, 2)
    r = params[2*n:3*n]
    result = np.zeros((n, 3))
    result[:, 0] = xy[:, 0]
    result[:, 1] = xy[:, 1]
    result[:, 2] = r
    return result
