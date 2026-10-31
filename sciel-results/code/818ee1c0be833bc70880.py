"""Circle packing in perimeter-4 rectangle: maximize sum of radii for 21 circles."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def _evaluate(params, rect_w, rect_h):
    """
    params: array of shape (N, 3) with [x, y, r]
    Returns (score, valid).
    Score = sum(r). Valid if all circles are inside rectangle and non-overlapping.
    """
    x = params[:, 0]
    y = params[:, 1]
    r = params[:, 2]
    
    # Check boundaries
    if np.any(x - r < -1e-9) or np.any(x + r > rect_w + 1e-9):
        return 0.0, False
    if np.any(y - r < -1e-9) or np.any(y + r > rect_h + 1e-9):
        return 0.0, False
    if np.any(r < -1e-9):
        return 0.0, False
    
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist * min_dist - 1e-9:
                return 0.0, False
    
    return np.sum(r), True

def _grid_initialization(rect_w, rect_h, n_circles):
    """Create a grid of circles that fit in the rectangle."""
    # Try different grid aspects
    best_score = -1
    best_params = None
    
    # Try grids from 1xN to Nx1
    for cols in range(1, n_circles + 1):
        rows = int(np.ceil(n_circles / cols))
        total_cells = rows * cols
        
        # Scale grid to fit in rectangle
        cell_w = rect_w / cols
        cell_h = rect_h / rows
        
        # Radius is limited by half the smaller dimension of a cell
        max_r = 0.5 * min(cell_w, cell_h)
        
        if max_r <= 0:
            continue
        
        params = np.zeros((n_circles, 3))
        for i in range(n_circles):
            row = i // cols
            col = i % cols
            params[i, 0] = (col + 0.5) * cell_w
            params[i, 1] = (row + 0.5) * cell_h
            params[i, 2] = max_r * 0.95  # slight shrink for safety
        
        score, valid = _evaluate(params, rect_w, rect_h)
        if valid and score > best_score:
            best_score = score
            best_params = params.copy()
    
    return best_params if best_params is not None else None

def _optimize(rect_w, rect_h, initial_params):
    """Optimize circle positions and radii using SLSQP."""
    n_vars = N * 3
    x0 = initial_params.flatten()
    
    # Objective: maximize sum of radii -> minimize negative sum
    def objective(x):
        params = x.reshape(N, 3)
        # Small penalty for negative radii
        r = np.maximum(params[:, 2], 0)
        return -np.sum(r)
    
    # Constraints:
    # 1. Non-overlapping: dist(i,j) >= r_i + r_j
    # 2. Boundary: x_i >= r_i, x_i <= rect_w - r_i, y_i >= r_i, y_i <= rect_h - r_i
    # 3. r_i >= 0
    
    constraints = []
    
    # Overlap constraints (using squared distance for smoothness)
    for i in range(N):
        for j in range(i + 1, N):
            def overlap_constraint(x, i=i, j=j):
                params = x.reshape(N, 3)
                dx = params[i, 0] - params[j, 0]
                dy = params[i, 1] - params[j, 1]
                dist_sq = dx * dx + dy * dy
                min_dist = params[i, 2] + params[j, 2]
                # dist_sq >= min_dist^2 -> dist_sq - min_dist^2 >= 0
                return dist_sq - min_dist * min_dist
            constraints.append({'type': 'ineq', 'fun': overlap_constraint})
    
    # Boundary constraints
    for i in range(N):
        # x_i - r_i >= 0
        def left_boundary(x, i=i):
            params = x.reshape(N, 3)
            return params[i, 0] - params[i, 2]
        constraints.append({'type': 'ineq', 'fun': left_boundary})
        
        # rect_w - x_i - r_i >= 0
        def right_boundary(x, i=i):
            params = x.reshape(N, 3)
            return rect_w - params[i, 0] - params[i, 2]
        constraints.append({'type': 'ineq', 'fun': right_boundary})
        
        # y_i - r_i >= 0
        def bottom_boundary(x, i=i):
            params = x.reshape(N, 3)
            return params[i, 1] - params[i, 2]
        constraints.append({'type': 'ineq', 'fun': bottom_boundary})
        
        # rect_h - y_i - r_i >= 0
        def top_boundary(x, i=i):
            params = x.reshape(N, 3)
            return rect_h - params[i, 1] - params[i, 2]
        constraints.append({'type': 'ineq', 'fun': top_boundary})
        
        # r_i >= 0
        def radius_nonneg(x, i=i):
            params = x.reshape(N, 3)
            return params[i, 2]
        constraints.append({'type': 'ineq', 'fun': radius_nonneg})
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0, rect_w))      # x_i
        bounds.append((0, rect_h))      # y_i
        bounds.append((0, min(rect_w, rect_h) / 2))  # r_i
    
    try:
        result = minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        if result.success:
            params = result.x.reshape(N, 3)
            score, valid = _evaluate(params, rect_w, rect_h)
            if valid:
                return params, score
    except Exception:
        pass
    
    return initial_params, _evaluate(initial_params, rect_w, rect_h)[0]

def construct():
    start_time = time.time()
    
    # Perimeter = 2*(w+h) = 4 -> w+h = 2
    # Try different aspect ratios
    best_score = 0
    best_params = None
    
    # Candidate aspect ratios (w, h) with w+h=2
    aspect_ratios = [
        (1.0, 1.0),       # square
        (1.2, 0.8),
        (0.8, 1.2),
        (1.4, 0.6),
        (0.6, 1.4),
        (1.6, 0.4),
        (0.4, 1.6),
        (1.8, 0.2),
        (0.2, 1.8),
    ]
    
    for w, h in aspect_ratios:
        if time.time() > DEADLINE:
            break
        
        # Get initial grid
        init_params = _grid_initialization(w, h, N)
        if init_params is None:
            continue
        
        # Optimize
        params, score = _optimize(w, h, init_params)
        
        if score > best_score:
            best_score = score
            best_params = params.copy()
    
    # If no valid solution found, return a trivial valid one
    if best_params is None:
        # Fallback: tiny circles in a grid
        w, h = 1.0, 1.0
        best_params = _grid_initialization(w, h, N)
        if best_params is None:
            # Absolute fallback
            best_params = np.zeros((N, 3))
            for i in range(N):
                best_params[i, 0] = 0.5
                best_params[i, 1] = 0.5
                best_params[i, 2] = 0.01
            # Make them non-overlapping by spreading out
            for i in range(N):
                row = i // 5
                col = i % 5
                best_params[i, 0] = (col + 0.5) * 0.2
                best_params[i, 1] = (row + 0.5) * 0.2
                best_params[i, 2] = 0.09
    
    # Final validation and cleanup
    # Determine which rectangle this fits in
    # We need to find w,h such that 2*(w+h) <= 4 and all circles fit
    # Since we optimized for specific w,h, use those
    # But we don't track which w,h was used for best_params
    # Let's re-derive: try to find the tightest bounding box
    x = best_params[:, 0]
    y = best_params[:, 1]
    r = best_params[:, 2]
    
    min_x = np.min(x - r)
    max_x = np.max(x + r)
    min_y = np.min(y - r)
    max_y = np.max(y + r)
    
    # Translate to origin
    best_params[:, 0] -= min_x
    best_params[:, 1] -= min_y
    
    w = max_x - min_x
    h = max_y - min_y
    perimeter = 2 * (w + h)
    
    # If perimeter > 4, scale down
    if perimeter > 4.0:
        scale = 4.0 / perimeter
        best_params *= scale
        w *= scale
        h *= scale
    
    return best_params
