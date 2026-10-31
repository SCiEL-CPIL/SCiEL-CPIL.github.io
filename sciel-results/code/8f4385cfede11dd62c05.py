"""Rewritten program: structural change to a physics-based (repulsion + attraction)
optimization with adaptive rectangle aspect ratio and a tight deadline guard."""

import numpy as np
from scipy.optimize import minimize
import time

N = 21
PERIM = 4.0
DEADLINE = 140.0  # wall-clock safety margin before the 150s kill


def _validity_penalty(c, w, h):
    """Penalty for overlaps and boundary violations.
    c: (N, 2) centers, radii: (N,)
    """
    x, y = c[:, 0], c[:, 1]
    r = c[:, 2] if c.shape[1] == 3 else None
    # We'll pass radii separately in the actual objective; this helper is for clarity.
    return 0.0


def _objective(params, w, h):
    """
    params: length 3N array: [x1, y1, r1, x2, y2, r2, ...]
    Returns a scalar to minimize. We want to maximize sum(r) subject to constraints.
    So we minimize: -sum(r) + lambda * penalty
    """
    c = params.reshape(N, 3)
    x, y, r = c[:, 0], c[:, 1], c[:, 2]
    
    # Boundary violations
    pen = 0.0
    # Circles must be inside [0, w] x [0, h]
    pen += np.sum(np.maximum(0, -x + r)) ** 2
    pen += np.sum(np.maximum(0, x - w + r)) ** 2
    pen += np.sum(np.maximum(0, -y + r)) ** 0
    pen += np.sum(np.maximum(0, y - h + r)) ** 2
    # Wait, let's be precise:
    # x - r >= 0  =>  violation if x - r < 0  =>  max(0, r - x)
    # x + r <= w  =>  violation if x + r > w  =>  max(0, x + r - w)
    # Similarly for y.
    pen = 0.0
    pen += np.sum(np.maximum(0, r - x)) ** 2
    pen += np.sum(np.maximum(0, x + r - w)) ** 2
    pen += np.sum(np.maximum(0, r - y)) ** 2
    pen += np.sum(np.maximum(0, y + r - h)) ** 2
    
    # Overlap violations
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                pen += (min_dist - dist) ** 2
    
    # Objective: minimize -sum(r) + penalty
    return -np.sum(r) + 10.0 * pen


def _sum_radii(params, w, h):
    """Return sum of radii if configuration is valid (low penalty), else -inf."""
    c = params.reshape(N, 3)
    x, y, r = c[:, 0], c[:, 1], c[:, 2]
    
    # Check boundary
    if np.any(r - x > 1e-8) or np.any(x + r - w > 1e-8) or np.any(r - y > 1e-8) or np.any(y + r - h > 1e-8):
        return -np.inf
    
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-8:
                return -np.inf
    
    return np.sum(r)


def construct():
    start_time = time.time()
    
    best_params = None
    best_value = -np.inf
    best_w, best_h = 1.0, 1.0
    
    # Try several aspect ratios
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.67, 1.33, 0.75, 1.25]
    
    for ar in aspect_ratios:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Perimeter = 2(w + h) = 4  =>  w + h = 2
        # h = w / ar  =>  w + w/ar = 2  =>  w(1 + 1/ar) = 2  =>  w = 2 * ar / (1 + ar)
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        # Initial guess: grid layout
        side = int(np.ceil(np.sqrt(N)))
        # Scale to fit in w x h
        # Place circles in a grid, with initial small radius
        init_params = np.zeros(3 * N)
        for k in range(N):
            i = k % side
            j = k // side
            x = (i + 0.5) * w / side
            y = (j + 0.5) * h / side
            r = 0.5 * min(w / side, h / side) * 0.9
            init_params[3 * k] = x
            init_params[3 * k + 1] = y
            init_params[3 * k + 2] = r
        
        # Optimize
        try:
            res = minimize(_objective, init_params, args=(w, h), method='L-BFGS-B',
                          options={'maxiter': 2000, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.fun < best_value:
                val = _sum_radii(res.x, w, h)
                if val > best_value:
                    best_value = val
                    best_params = res.x.copy()
                    best_w, best_h = w, h
        except Exception:
            pass
    
    # If no valid solution found, fall back to a simple grid
    if best_params is None:
        w, h = 1.0, 1.0
        side = int(np.ceil(np.sqrt(N)))
        best_params = np.zeros(3 * N)
        for k in range(N):
            i = k % side
            j = k // side
            x = (i + 0.5) * w / side
            y = (j + 0.5) * h / side
            r = 0.5 * min(w / side, h / side) * 0.9
            best_params[3 * k] = x
            best_params[3 * k + 1] = y
            best_params[3 * k + 2] = r
        best_w, best_h = w, h
    
    # Final polish: try to improve with more iterations if time permits
    if time.time() - start_time < DEADLINE - 10:
        try:
            res2 = minimize(_objective, best_params, args=(best_w, best_h), method='L-BFGS-B',
                          options={'maxiter': 5000, 'maxfun': 10000, 'ftol': 1e-14, 'gtol': 1e-10})
            val2 = _sum_radii(res2.x, best_w, best_h)
            if val2 > best_value:
                best_value = val2
                best_params = res2.x.copy()
        except Exception:
            pass
    
    # Convert to (N, 3) array
    result = best_params.reshape(N, 3)
    return result
