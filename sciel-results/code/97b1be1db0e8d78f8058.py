"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _make_initial(n_circles, aspect, rng):
    """Create initial configuration: positions in [0,1]x[0,1], radii ~0.1, aspect ratio."""
    w = 2.0 / (1.0 + aspect)  # width
    h = 2.0 - w               # height
    # Place circles in a rough grid with jitter
    side = int(np.ceil(np.sqrt(n_circles)))
    positions = []
    radii = []
    for i in range(n_circles):
        row = i // side
        col = i % side
        x = (col + 0.5 + rng.uniform(-0.1, 0.1)) / side
        y = (row + 0.5 + rng.uniform(-0.1, 0.1)) / side
        positions.append([x, y])
        radii.append(0.5 / side - 0.01)
    positions = np.array(positions)
    radii = np.array(radii)
    return positions, radii, w, h

def _objective(params, n_circles, penalty_weight=100.0):
    """Objective: minimize -sum(radii) + penalty for violations."""
    positions = params[:2 * n_circles].reshape(n_circles, 2)
    radii = params[2 * n_circles:3 * n_circles]
    aspect = params[3 * n_circles]
    
    w = 2.0 / (1.0 + aspect)
    h = 2.0 - w
    
    # Convert normalized positions to actual coordinates
    xs = positions[:, 0] * w
    ys = positions[:, 1] * h
    rs = radii * min(w, h)  # scale radii by the smaller dimension
    
    # Penalty for circles going out of bounds
    bound_pen = 0.0
    for i in range(n_circles):
        if xs[i] - rs[i] < 0:
            bound_pen += (xs[i] - rs[i]) ** 2
        if xs[i] + rs[i] > w:
            bound_pen += (xs[i] + rs[i] - w) ** 2
        if ys[i] - rs[i] < 0:
            bound_pen += (ys[i] - rs[i]) ** 2
        if ys[i] + rs[i] > h:
            bound_pen += (ys[i] + rs[i] - h) ** 2
    
    # Penalty for overlapping circles
    overlap_pen = 0.0
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = rs[i] + rs[j]
            if dist < min_dist:
                overlap_pen += (min_dist - dist) ** 2
    
    total_pen = bound_pen + overlap_pen
    return -np.sum(rs) + penalty_weight * total_pen

def _extract_result(params, n_circles):
    """Extract positions and radii from parameter vector."""
    positions = params[:2 * n_circles].reshape(n_circles, 2)
    radii = params[2 * n_circles:3 * n_circles]
    aspect = params[3 * n_circles]
    
    w = 2.0 / (1.0 + aspect)
    h = 2.0 - w
    
    xs = positions[:, 0] * w
    ys = positions[:, 1] * h
    rs = radii * min(w, h)
    
    return xs, ys, rs

def _validate(xs, ys, rs, w, h, tol=1e-6):
    """Check if configuration is valid."""
    n = len(rs)
    for i in range(n):
        if xs[i] - rs[i] < -tol or xs[i] + rs[i] > w + tol:
            return False
        if ys[i] - rs[i] < -tol or ys[i] + rs[i] > h + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    best_value = -np.inf
    best_params = None
    
    rng = np.random.RandomState(42)
    
    # Try multiple aspect ratios and initial configurations
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.67, 2.0, 0.5, 1.33]
    
    for trial in range(20):
        if time.time() > deadline:
            break
        
        aspect = aspect_ratios[trial % len(aspect_ratios)]
        positions, radii, w, h = _make_initial(N, aspect, rng)
        
        # Build initial parameter vector
        params = np.concatenate([
            positions.flatten(),
            radii,
            [aspect]
        ])
        
        # Run Nelder-Mead optimization
        result = minimize(
            _objective,
            params,
            args=(N,),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        
        if result.success or result.fun < _objective(params, N):
            xs, ys, rs = _extract_result(result.x, N)
            aspect_val = result.x[3 * N]
            w_val = 2.0 / (1.0 + aspect_val)
            h_val = 2.0 - w_val
            
            if _validate(xs, ys, rs, w_val, h_val):
                value = np.sum(rs)
                if value > best_value:
                    best_value = value
                    best_params = result.x.copy()
    
    if best_params is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.5 / side - 0.01
            cells.append([x, y, r])
        return np.array(cells)
    
    xs, ys, rs = _extract_result(best_params, N)
    aspect_val = best_params[3 * N]
    w_val = 2.0 / (1.0 + aspect_val)
    h_val = 2.0 - w_val
    
    # Final validation and adjustment
    if not _validate(xs, ys, rs, w_val, h_val):
        # Slightly shrink radii to ensure validity
        for i in range(N):
            rs[i] *= 0.99
        if not _validate(xs, ys, rs, w_val, h_val):
            side = int(np.ceil(np.sqrt(N)))
            cells = []
            for i in range(N):
                row = i // side
                col = i % side
                x = (col + 0.5) / side
                y = (row + 0.5) / side
                r = 0.5 / side - 0.01
                cells.append([x, y, r])
            return np.array(cells)
    
    # Scale back to normalized [0,1]x[0,1] for output
    # The evaluator expects circles in a bounding rectangle with perimeter <= 4
    # We need to output centers and radii in the actual rectangle
    result = np.column_stack([xs, ys, rs])
    return result
# EVOLVE-BLOCK-END
