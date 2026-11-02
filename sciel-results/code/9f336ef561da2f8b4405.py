"""Improved circle packing in unit square n=32."""
import numpy as np
from scipy.optimize import minimize
import time

N = 32
DEADLINE = 145.0  # safety margin before 150s kill

def _check_validity(arr):
    """Check if circles are valid (disjoint, inside unit square)."""
    n = len(arr)
    for i in range(n):
        x, y, r = arr[i]
        if r < 0 or x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = arr[i, 0] - arr[j, 0]
            dy = arr[i, 1] - arr[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < arr[i, 2] + arr[j, 2] - 1e-9:
                return False
    return True

def _objective(params, N):
    """
    Smooth objective: maximize sum of radii.
    params: [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
    We minimize negative sum of radii plus penalties for violations.
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Penalty for boundary violations
    boundary_pen = 0.0
    for i in range(N):
        if x[i] - r[i] < 0:
            boundary_pen += (x[i] - r[i]) ** 2
        if x[i] + r[i] > 1:
            boundary_pen += (x[i] + r[i] - 1) ** 2
        if y[i] - r[i] < 0:
            boundary_pen += (y[i] - r[i]) ** 2
        if y[i] + r[i] > 1:
            boundary_pen += (y[i] + r[i] - 1) ** 2
    
    # Penalty for overlaps (smooth)
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy + 1e-12)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                overlap_pen += (min_dist - dist) ** 2
    
    # Objective: minimize -sum(r) + penalties
    obj = -np.sum(r) + 100.0 * boundary_pen + 100.0 * overlap_pen
    return obj

def _make_hex_init(N):
    """Hexagonal packing initialization."""
    side = int(np.ceil(np.sqrt(N)))
    pts = []
    r_base = 0.15
    dx = 2 * r_base
    dy = np.sqrt(3) * r_base
    row = 0
    while len(pts) < N:
        col = 0
        while len(pts) < N:
            x = 0.5 + col * dx + (row % 2) * dx / 2
            y = 0.5 + row * dy
            if 0 < x < 1 and 0 < y < 1:
                pts.append((x, y, r_base))
            col += 1
        row += 1
    pts = pts[:N]
    if len(pts) > 0:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        scale_x = 0.8 / (max_x - min_x + 1e-9) if max_x > min_x else 1.0
        scale_y = 0.8 / (max_y - min_y + 1e-9) if max_y > min_y else 1.0
        scale = min(scale_x, scale_y)
        pts = [((x - cx) * scale + 0.5, (y - cy) * scale + 0.5, r_base * scale) for x, y, r in pts]
    return np.array(pts)

def _make_grid_init(N):
    """Grid initialization with equal circles."""
    side = int(np.ceil(np.sqrt(N)))
    r = 0.5 / side - 0.01
    pts = []
    for i in range(side):
        for j in range(side):
            if len(pts) >= N:
                break
            pts.append(((i + 0.5) / side, (j + 0.5) / side, r))
    return np.array(pts[:N])

def _make_random_init(N, seed):
    """Random initialization."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.15, 0.85, N)
    y = rng.uniform(0.15, 0.85, N)
    r = rng.uniform(0.05, 0.12, N)
    return np.column_stack([x, y, r])

def _local_search(init, N, time_limit, max_iter=5000):
    """Run local search from initial configuration."""
    start_time = time.time()
    params = init.flatten().copy()
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    result = minimize(
        _objective,
        params,
        args=(N,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'maxfun': 100000, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    if result.success:
        best_params = result.x
        best_obj = result.fun
    else:
        best_params = params
        best_obj = _objective(params, N)
    
    elapsed = time.time() - start_time
    return best_params, best_obj, elapsed

def _polish(params, N, time_limit):
    """High-precision polish pass with tighter tolerance."""
    start_time = time.time()
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    result = minimize(
        _objective,
        params,
        args=(N,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 10000, 'maxfun': 200000, 'ftol': 1e-14, 'gtol': 1e-12}
    )
    
    if result.success:
        return result.x
    return params

def _params_to_array(params, N):
    """Convert flat params to (N, 3) array."""
    arr = np.zeros((N, 3))
    arr[:, 0] = params[0::3]
    arr[:, 1] = params[1::3]
    arr[:, 2] = params[2::3]
    return arr

def construct():
    """Construct 32 circles in unit square maximizing sum of radii."""
    start_time = time.time()
    
    inits = []
    
    hex_init = _make_hex_init(N)
    inits.append(hex_init)
    
    grid_init = _make_grid_init(N)
    inits.append(grid_init)
    
    for seed in range(20):
        rand_init = _make_random_init(N, seed)
        inits.append(rand_init)
    
    rng = np.random.default_rng(42)
    for i in range(5):
        perturbed = hex_init.copy()
        perturbed[:, 0] += rng.normal(0, 0.05, N)
        perturbed[:, 1] += rng.normal(0, 0.05, N)
        perturbed[:, 0] = np.clip(perturbed[:, 0], 0.1, 0.9)
        perturbed[:, 1] = np.clip(perturbed[:, 1], 0.1, 0.9)
        inits.append(perturbed)
    
    best_params = None
    best_score = -np.inf
    best_array = None
    
    for init in inits:
        elapsed_so_far = time.time() - start_time
        if elapsed_so_far > DEADLINE - 30:
            break
        
        remaining = DEADLINE - 30 - elapsed_so_far
        if remaining < 5:
            break
        
        try:
            params, obj, elapsed = _local_search(init, N, remaining, max_iter=3000)
            elapsed_so_far = time.time() - start_time
            
            if obj < best_score:
                best_score = obj
                best_params = params.copy()
                arr = _params_to_array(params, N)
                if _check_validity(arr):
                    best_array = arr.copy()
        except Exception:
            continue
    
    if best_params is not None:
        elapsed_so_far = time.time() - start_time
        if elapsed_so_far < DEADLINE - 10:
            try:
                polished = _polish(best_params, N, DEADLINE - 10)
                arr = _params_to_array(polished, N)
                if _check_validity(arr):
                    obj_polished = _objective(polished, N)
                    if obj_polished < best_score:
                        best_params = polished
                        best_score = obj_polished
                        best_array = arr.copy()
            except Exception:
                pass
    
    if best_array is None:
        if best_params is not None:
            arr = _params_to_array(best_params, N)
            if _check_validity(arr):
                best_array = arr
            else:
                best_array = _make_grid_init(N)
        else:
            best_array = _make_grid_init(N)
    
    if not _check_validity(best_array):
        best_array = best_array.copy()
        best_array[:, 2] *= 0.95
        if not _check_validity(best_array):
            best_array = _make_grid_init(N)
    
    return best_array

# EVOLVE-BLOCK-END
