"""Circle packing in perimeter-4 rectangle: maximize sum of radii for 21 circles."""
import numpy as np
import time
from scipy.optimize import minimize
import random

N = 21
DEADLINE = 120.0  # seconds wall-clock limit


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
    
    # Disjoint interiors
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - tol:
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
    
    # Overlap constraints
    overlap_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap_viol += max(0.0, r[i] + r[j] - dist)
    
    penalty = 100.0 * (perim_viol ** 2 + overlap_viol ** 2)
    return -np.sum(r) + penalty


def _constraint_refine(arr, max_iter=500, tol=1e-10):
    """Refine positions and radii to satisfy constraints more tightly."""
    n = arr.shape[0]
    best = arr.copy()
    best_val = np.sum(best[:, 2])
    
    for it in range(max_iter):
        # Try to increase a random radius slightly
        i = random.randint(0, n - 1)
        dr = random.uniform(0.001, 0.01)
        arr_new = best.copy()
        arr_new[i, 2] += dr
        
        if _is_valid(arr_new):
            best = arr_new
            best_val = np.sum(best[:, 2])
        else:
            # Try to move the circle to make room
            for _ in range(10):
                dx = random.uniform(-0.05, 0.05)
                dy = random.uniform(-0.05, 0.05)
                arr_try = best.copy()
                arr_try[i, 0] += dx
                arr_try[i, 1] += dy
                arr_try[i, 2] = best[i, 2] + dr
                if _is_valid(arr_try):
                    best = arr_try
                    best_val = np.sum(best[:, 2])
                    break
    return best, best_val


def _force_relax_unified(arr, max_iter=1000):
    """Force relaxation to handle constraint violations."""
    n = arr.shape[0]
    best = arr.copy()
    
    for it in range(max_iter):
        # Check for violations
        xs, ys, rs = best[:, 0], best[:, 1], best[:, 2]
        x_min = np.min(xs - rs)
        x_max = np.max(xs + rs)
        y_min = np.min(ys - rs)
        y_max = np.max(ys + rs)
        w = x_max - x_min
        h = y_max - y_min
        perim_viol = max(0.0, (w + h) - 2.0)
        
        overlap_viol = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                overlap_viol += max(0.0, rs[i] + rs[j] - dist)
        
        if perim_viol < 1e-6 and overlap_viol < 1e-6:
            break
        
        # Reduce radii slightly to fix violations
        best[:, 2] *= 0.99
        
    return best


def _expansion_relaxation_polish(arr, max_iter=2000):
    """Polish by expanding radii where possible."""
    n = arr.shape[0]
    best = arr.copy()
    best_val = np.sum(best[:, 2])
    
    for it in range(max_iter):
        # Try to expand each circle
        for i in range(n):
            # Max radius for circle i given current positions of others
            max_r = 0.0
            xs, ys, rs = best[:, 0], best[:, 1], best[:, 2]
            
            # Distance to boundaries
            x_min = np.min(xs - rs)
            x_max = np.max(xs + rs)
            y_min = np.min(ys - rs)
            y_max = np.max(ys + rs)
            
            # For now, just try small expansion
            dr = 0.005
            arr_try = best.copy()
            arr_try[i, 2] += dr
            
            if _is_valid(arr_try):
                best = arr_try
                best_val = np.sum(best[:, 2])
    
    return best, best_val


def _initial_hexagonal(w, h):
    """Create hexagonal packing initial guess."""
    arr = np.zeros((N, 3))
    # Hexagonal grid
    dx = 0.18
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
                arr[count, 2] = dx / 2 * 0.9
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
            arr[count, 2] = min(w, h) / (2 * side) * 0.85
            count += 1
    return arr


def _initial_random(w, h):
    """Create random initial guess."""
    arr = np.random.rand(N, 3)
    arr[:, 0] *= w
    arr[:, 1] *= h
    arr[:, 2] = 0.05 + np.random.rand(N) * 0.05
    return arr


def _polish_result(arr, time_budget=10.0):
    """Polish the result using multiple techniques."""
    start = time.time()
    best = arr.copy()
    best_val = np.sum(best[:, 2])
    
    # Step 1: Force relaxation
    if time.time() - start < time_budget * 0.2:
        relaxed = _force_relax_unified(best, max_iter=200)
        if np.sum(relaxed[:, 2]) > best_val:
            best = relaxed
            best_val = np.sum(best[:, 2])
    
    # Step 2: Constraint refinement
    if time.time() - start < time_budget * 0.5:
        refined, val = _constraint_refine(best, max_iter=300)
        if val > best_val:
            best = refined
            best_val = val
    
    # Step 3: Expansion polish
    if time.time() - start < time_budget * 0.9:
        polished, val = _expansion_relaxation_polish(best, max_iter=500)
        if val > best_val:
            best = polished
            best_val = val
    
    return best, best_val


def _max_radius_for_circle(arr, i):
    """Compute max radius for circle i given others."""
    n = arr.shape[0]
    x_i, y_i, r_i = arr[i]
    xs, ys, rs = arr[:, 0], arr[:, 1], arr[:, 2]
    
    max_r = 1.0  # large initial guess
    
    # Distance to other circles
    for j in range(n):
        if j == i:
            continue
        dx = x_i - xs[j]
        dy = y_i - ys[j]
        dist = np.sqrt(dx * dx + dy * dy)
        max_r = min(max_r, max(0.0, dist - rs[j]))
    
    return max_r


def _is_valid_move(arr, i, dx, dy, dr):
    """Check if a move is valid."""
    arr_try = arr.copy()
    arr_try[i, 0] += dx
    arr_try[i, 1] += dy
    arr_try[i, 2] = max(1e-6, arr_try[i, 2] + dr)
    return _is_valid(arr_try)


def construct():
    start_time = time.time()
    
    best_arr = None
    best_sum = 0.0
    
    # Try different aspect ratios
    aspects = [1.0, 1.2, 0.8, 1.5, 0.67, 2.0, 0.5, 1.33, 0.75, 1.1, 0.9]
    
    for aspect in aspects:
        if time.time() - start_time > DEADLINE - 5.0:
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
            if time.time() - start_time > DEADLINE - 5.0:
                break
            
            # Polish
            polished, val = _polish_result(arr_init, time_budget=5.0)
            if val > best_sum:
                best_sum = val
                best_arr = polished.copy()
    
    # Final polish with remaining time
    if best_arr is not None and time.time() - start_time < DEADLINE - 2.0:
        remaining = DEADLINE - 2.0 - (time.time() - start_time)
        polished, val = _polish_result(best_arr, time_budget=remaining)
        if val > best_sum:
            best_sum = val
            best_arr = polished
    
    if best_arr is None:
        # Fallback
        side = 5
        positions = [(i, j) for i in range(side) for j in range(side)][:N]
        best_arr = np.zeros((N, 3))
        for idx, (i, j) in enumerate(positions):
            best_arr[idx, 0] = (i + 0.5) / side
            best_arr[idx, 1] = (j + 0.5) / side
            best_arr[idx, 2] = 0.5 / side - 0.01
    
    if not _is_valid(best_arr):
        # Return a known-valid fallback
        side = 5
        positions = [(i, j) for i in range(side) for j in range(side)][:N]
        best_arr = np.zeros((N, 3))
        for idx, (i, j) in enumerate(positions):
            best_arr[idx, 0] = (i + 0.5) / side
            best_arr[idx, 1] = (j + 0.5) / side
            best_arr[idx, 2] = 0.5 / side - 0.01
    
    return best_arr
