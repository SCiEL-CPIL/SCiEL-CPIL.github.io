"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 150.0
START_TIME = None


def _compute_score(positions, radii, w, h):
    """Compute validity and score for a packing configuration.
    positions: (N,2), radii: (N,), w,h: rectangle dimensions with 2*(w+h)=4
    Returns (is_valid, sum_of_radii)
    """
    # Check perimeter constraint
    if 2.0 * (w + h) > 4.0 + 1e-9:
        return False, 0.0
    # Check circles inside rectangle
    for i in range(N):
        if positions[i, 0] - radii[i] < -1e-9 or positions[i, 1] - radii[i] < -1e-9:
            return False, 0.0
        if positions[i, 0] + radii[i] > w + 1e-9 or positions[i, 1] + radii[i] > h + 1e-9:
            return False, 0.0
    # Check no overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - 1e-9:
                return False, 0.0
    return True, np.sum(radii)


def _objective(x, w, h):
    """Penalty-based objective for L-BFGS-B.
    x: [x1,y1,r1, x2,y2,r2, ...] flattened
    Minimize negative sum of radii + penalties for violations.
    """
    n = N
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    # Main objective: maximize sum of radii (minimize negative)
    obj = -np.sum(rs)
    
    # Penalty for circles outside rectangle
    pen = 0.0
    for i in range(n):
        # Left wall
        viol = max(0.0, rs[i] - xs[i])
        pen += 100.0 * viol * viol
        # Right wall
        viol = max(0.0, xs[i] + rs[i] - w)
        pen += 100.0 * viol * viol
        # Bottom wall
        viol = max(0.0, rs[i] - ys[i])
        pen += 100.0 * viol * viol
        # Top wall
        viol = max(0.0, ys[i] + rs[i] - h)
        pen += 100.0 * viol * viol
    
    # Penalty for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = rs[i] + rs[j]
            viol = max(0.0, min_dist - dist)
            pen += 100.0 * viol * viol
    
    return obj + pen


def _objective_grad(x, w, h):
    """Gradient of the penalty objective."""
    n = N
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    grad = np.zeros_like(x)
    
    # Gradient of -sum(r)
    for i in range(n):
        grad[3*i + 2] = -1.0
    
    # Gradients of boundary penalties
    for i in range(n):
        # Left wall: max(0, r_i - x_i)^2
        viol = max(0.0, rs[i] - xs[i])
        if viol > 0:
            grad[3*i] += -200.0 * viol  # d/dx_i of (r_i - x_i)^2 = -2(r_i - x_i)
            grad[3*i + 2] += 200.0 * viol  # d/dr_i of (r_i - x_i)^2 = 2(r_i - x_i)
        
        # Right wall: max(0, x_i + r_i - w)^2
        viol = max(0.0, xs[i] + rs[i] - w)
        if viol > 0:
            grad[3*i] += 200.0 * viol
            grad[3*i + 2] += 200.0 * viol
        
        # Bottom wall: max(0, r_i - y_i)^2
        viol = max(0.0, rs[i] - ys[i])
        if viol > 0:
            grad[3*i + 1] += -200.0 * viol
            grad[3*i + 2] += 200.0 * viol
        
        # Top wall: max(0, y_i + r_i - h)^2
        viol = max(0.0, ys[i] + rs[i] - h)
        if viol > 0:
            grad[3*i + 1] += 200.0 * viol
            grad[3*i + 2] += 200.0 * viol
    
    # Gradients of overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx * dx + dy * dy
            min_dist = rs[i] + rs[j]
            if dist_sq > 1e-12:
                dist = np.sqrt(dist_sq)
                viol = max(0.0, min_dist - dist)
                if viol > 0:
                    # d/dx_i of (min_dist - dist)^2 = 2*(min_dist-dist)*(-dx/dist)
                    factor = 200.0 * viol / dist
                    grad[3*i] += factor * dx
                    grad[3*i + 1] += factor * dy
                    grad[3*j] -= factor * dx
                    grad[3*j + 1] -= factor * dy
                    grad[3*i + 2] += 200.0 * viol
                    grad[3*j + 2] += 200.0 * viol
    
    return grad


def _make_initial_grid(w, h, n_cols, n_rows):
    """Create an initial grid of circles in rectangle [0,w]x[0,h]."""
    positions = []
    radii = []
    for i in range(n_cols):
        for j in range(n_rows):
            if len(positions) >= N:
                break
            x = (i + 0.5) * w / n_cols
            y = (j + 0.5) * h / n_rows
            r = 0.5 * min(w / n_cols, h / n_rows) * 0.95
            positions.append([x, y])
            radii.append(r)
    while len(positions) < N:
        positions.append([w * 0.5, h * 0.5])
        radii.append(0.01)
    return np.array(positions[:N]), np.array(radii[:N])


def _repair(positions, radii, w, h):
    """Repair a configuration to be valid: project inside rectangle and resolve overlaps."""
    positions = positions.copy()
    radii = radii.copy()
    
    # Clamp radii to be positive
    radii = np.maximum(radii, 1e-6)
    
    # Project positions so circles stay inside rectangle
    for i in range(N):
        positions[i, 0] = np.clip(positions[i, 0], radii[i], w - radii[i])
        positions[i, 1] = np.clip(positions[i, 1], radii[i], h - radii[i])
    
    # Iteratively resolve overlaps by shrinking radii
    for iteration in range(50):
        max_overlap = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    max_overlap = max(max_overlap, overlap)
                    # Shrink the smaller circle
                    if radii[i] <= radii[j]:
                        shrink = overlap * 0.5 + 1e-6
                        radii[i] = max(radii[i] - shrink, 1e-6)
                    else:
                        shrink = overlap * 0.5 + 1e-6
                        radii[j] = max(radii[j] - shrink, 1e-6)
        if max_overlap < 1e-10:
            break
    
    # Final check and clamp
    for i in range(N):
        radii[i] = max(radii[i], 1e-6)
        positions[i, 0] = np.clip(positions[i, 0], radii[i], w - radii[i])
        positions[i, 1] = np.clip(positions[i, 1], radii[i], h - radii[i])
    
    return positions, radii


def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    best_positions = None
    best_radii = None
    best_score = 0.0
    best_w = None
    best_h = None
    
    # Try different aspect ratios and grid configurations
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.4]
    
    # Grid configurations: (n_cols, n_rows)
    grid_configs = [
        (5, 5),  # 25 cells, use 21
        (4, 6),  # 24 cells
        (6, 4),  # 24 cells
        (3, 7),  # 21 cells
        (7, 3),  # 21 cells
        (4, 5),  # 20 cells, add 1
        (5, 4),  # 20 cells, add 1
    ]
    
    for aspect in aspect_ratios:
        if time.time() - START_TIME > DEADLINE - 30:
            break
        
        # Perimeter 4: 2*(w+h) = 4, so w+h = 2
        # w/h = aspect
        h = 2.0 / (1.0 + aspect)
        w = aspect * h
        
        for n_cols, n_rows in grid_configs:
            if time.time() - START_TIME > DEADLINE - 30:
                break
            
            positions, radii = _make_initial_grid(w, h, n_cols, n_rows)
            
            # Build initial x vector
            x0 = np.zeros(3 * N)
            for i in range(N):
                x0[3*i] = positions[i, 0]
                x0[3*i + 1] = positions[i, 1]
                x0[3*i + 2] = radii[i]
            
            # Bounds: x in [0, w], y in [0, h], r in [0, min(w,h)/2]
            r_max = min(w, h) / 2.0
            bounds = []
            for i in range(N):
                bounds.append((1e-6, w - 1e-6))
                bounds.append((1e-6, h - 1e-6))
                bounds.append((1e-6, r_max))
            
            # Run L-BFGS-B
            try:
                result = minimize(
                    _objective,
                    x0,
                    args=(w, h),
                    method='L-BFGS-B',
                    jac=_objective_grad,
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < _objective(x0, w, h):
                    x_opt = result.x
                    pos = np.column_stack([x_opt[0::3], x_opt[1::3]])
                    rad = x_opt[2::3]
                    
                    # Repair to ensure validity
                    pos_rep, rad_rep = _repair(pos, rad, w, h)
                    
                    valid, score = _compute_score(pos_rep, rad_rep, w, h)
                    if valid and score > best_score:
                        best_score = score
                        best_positions = pos_rep
                        best_radii = rad_rep
                        best_w = w
                        best_h = h
                        
            except Exception:
                pass
    
    # If no valid result found, use a simple grid fallback
    if best_positions is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        result = np.array(cells[:N])
        return result
    
    # Build output array: (x, y, r)
    output = np.column_stack([best_positions, best_radii])
    return output
# EVOLVE-BLOCK-END
