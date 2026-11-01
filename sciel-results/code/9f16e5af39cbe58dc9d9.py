"""Circle packing in perimeter-4 rectangle: maximize sum of radii for 21 circles."""
import numpy as np
import time
from scipy.optimize import minimize
import random

N = 21
DEADLINE = 150.0  # seconds wall-clock limit (evaluator kills at 150s, use 145s safety)
SAFETY_MARGIN = 5.0


def _is_valid(arr, tol=1e-9):
    """Check if circles are valid: disjoint interiors, inside bounding box, perimeter <= 4."""
    n = arr.shape[0]
    if n != N:
        return False
    xs, ys, rs = arr[:, 0], arr[:, 1], arr[:, 2]
    
    if np.any(rs <= 0):
        return False
    
    # Bounding box extents
    x_ext_min = np.min(xs - rs)
    x_ext_max = np.max(xs + rs)
    y_ext_min = np.min(ys - rs)
    y_ext_max = np.max(ys + rs)
    
    perim = 2.0 * ((x_ext_max - x_ext_min) + (y_ext_max - y_ext_min))
    if perim > 4.0 + tol:
        return False
    
    # Disjoint interiors - vectorized
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = rs[i_idx] + rs[j_idx]
    if np.any(dist < min_dist - tol):
        return False
    
    return True


def _vectorized_objective(vars_):
    """Negative sum of radii, with penalty for constraint violations."""
    n = len(vars_) // 3
    x = vars_[::3]
    y = vars_[1::3]
    r = vars_[2::3]
    
    # Bounding box
    x_min = np.min(x - r)
    x_max = np.max(x + r)
    y_min = np.min(y - r)
    y_max = np.max(y + r)
    
    w = x_max - x_min
    h = y_max - y_min
    
    # Perimeter constraint: w + h <= 2
    perim_viol = max(0.0, (w + h) - 2.0)
    
    # Overlap constraints - vectorized
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist = np.sqrt(dx * dx + dy * dy)
    overlap = np.maximum(0.0, r[i_idx] + r[j_idx] - dist)
    overlap_viol = np.sum(overlap)
    
    penalty = 100.0 * (perim_viol ** 2 + overlap_viol ** 2)
    return -np.sum(r) + penalty


def _vectorized_jacobian(vars_):
    """Approximate Jacobian for SLSQP."""
    n = len(vars_) // 3
    eps = 1e-7
    jacobian = np.zeros_like(vars_)
    f0 = _vectorized_objective(vars_)
    for i in range(len(vars_)):
        v = vars_.copy()
        v[i] += eps
        f1 = _vectorized_objective(v)
        jacobian[i] = (f1 - f0) / eps
    return jacobian


def _fast_expansion_polish(arr, time_budget=10.0):
    """Polish by expanding radii where possible using vectorized checks."""
    start = time.time()
    best = arr.copy()
    best_val = np.sum(best[:, 2])
    n = best.shape[0]
    
    # Precompute pair indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    while time.time() - start < time_budget:
        improved = False
        for i in range(n):
            # Try to expand circle i
            dr = 0.005
            arr_try = best.copy()
            arr_try[i, 2] += dr
            
            # Check validity quickly
            xs, ys, rs = arr_try[:, 0], arr_try[:, 1], arr_try[:, 2]
            
            # Bounding box check
            x_min = np.min(xs - rs)
            x_max = np.max(xs + rs)
            y_min = np.min(ys - rs)
            y_max = np.max(ys + rs)
            if 2.0 * ((x_max - x_min) + (y_max - y_min)) > 4.0 + 1e-9:
                continue
            
            # Overlap check
            dx = xs[i_idx] - xs[j_idx]
            dy = ys[i_idx] - ys[j_idx]
            dist = np.sqrt(dx * dx + dy * dy)
            if np.any(dist < rs[i_idx] + rs[j_idx] - 1e-9):
                continue
            
            best = arr_try
            best_val = np.sum(best[:, 2])
            improved = True
        
        if not improved:
            break
    
    return best, best_val


def _slsqp_optimize(arr_init, time_budget=5.0):
    """Run SLSQP optimization on the initial configuration."""
    start = time.time()
    x0 = arr_init.flatten()
    
    # Bounds: positions in [0, 2], radii in [0.001, 1.0]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 2.0))  # x
        bounds.append((0.0, 2.0))  # y
        bounds.append((0.001, 1.0))  # r
    
    result = minimize(
        _vectorized_objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-8}
    )
    
    elapsed = time.time() - start
    if elapsed > time_budget:
        # Return the initial if optimization took too long
        return arr_init
    
    arr_opt = result.x.reshape(N, 3)
    return arr_opt


def _initial_hexagonal(w, h):
    """Create hexagonal packing initial guess."""
    arr = np.zeros((N, 3))
    # Hexagonal grid
    dx = min(w, h) / 5.0
    dy = dx * np.sqrt(3) / 2
    count = 0
    for row in range(10):
        for col in range(10):
            if count >= N:
                break
            x_off = dx / 2 if row % 2 == 1 else 0
            x = (col + 0.5) * dx + x_off
            y = (row + 0.5) * dy
            if x < w and y < h:
                arr[count, 0] = x
                arr[count, 1] = y
                arr[count, 2] = dx / 2 * 0.85
                count += 1
        if count >= N:
            break
    return arr


def _initial_grid(w, h):
    """Create grid initial guess."""
    arr = np.zeros((N, 3))
    side = 5
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N:
                break
            arr[count, 0] = (i + 0.5) * w / side
            arr[count, 1] = (j + 0.5) * h / side
            arr[count, 2] = min(w, h) / (2 * side) * 0.8
            count += 1
    return arr


def _initial_random(w, h):
    """Create random initial guess."""
    arr = np.random.rand(N, 3)
    arr[:, 0] *= w
    arr[:, 1] *= h
    arr[:, 2] = 0.03 + np.random.rand(N) * 0.04
    return arr


def _fallback_arr():
    """Return a known-valid fallback configuration."""
    side = 5
    positions = [(i, j) for i in range(side) for j in range(side)][:N]
    best_arr = np.zeros((N, 3))
    for idx, (i, j) in enumerate(positions):
        best_arr[idx, 0] = (i + 0.5) / side
        best_arr[idx, 1] = (j + 0.5) / side
        best_arr[idx, 2] = 0.5 / side - 0.02
    return best_arr


def construct():
    start_time = time.time()
    
    best_arr = None
    best_sum = 0.0
    
    # Try different aspect ratios
    aspects = [1.0, 1.2, 0.8, 1.5, 0.67, 1.33, 0.75, 1.1]
    
    for aspect in aspects:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - SAFETY_MARGIN - 10.0:
            break
        
        h = 2.0 / (aspect + 1.0)
        w = aspect * h
        
        # Generate initial configurations
        inits = [
            _initial_hexagonal(w, h),
            _initial_grid(w, h),
            _initial_random(w, h),
        ]
        
        for arr_init in inits:
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - SAFETY_MARGIN - 5.0:
                break
            
            # SLSQP optimization with time budget
            time_per_opt = min(4.0, (DEADLINE - SAFETY_MARGIN - 5.0 - elapsed) / 2.0)
            if time_per_opt < 0.5:
                break
            arr_opt = _slsqp_optimize(arr_init, time_budget=time_per_opt)
            
            # Fast expansion polish
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - SAFETY_MARGIN - 2.0:
                break
            polish_time = min(3.0, DEADLINE - SAFETY_MARGIN - 2.0 - elapsed)
            if polish_time < 0.3:
                break
            polished, val = _fast_expansion_polish(arr_opt, time_budget=polish_time)
            
            if val > best_sum:
                best_sum = val
                best_arr = polished.copy()
    
    # Final polish with remaining time
    if best_arr is not None:
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - SAFETY_MARGIN:
            remaining = DEADLINE - SAFETY_MARGIN - elapsed
            if remaining > 0.5:
                polished, val = _fast_expansion_polish(best_arr, time_budget=remaining)
                if val > best_sum:
                    best_sum = val
                    best_arr = polished
    
    if best_arr is None:
        best_arr = _fallback_arr()
    
    if not _is_valid(best_arr):
        best_arr = _fallback_arr()
    
    return best_arr
