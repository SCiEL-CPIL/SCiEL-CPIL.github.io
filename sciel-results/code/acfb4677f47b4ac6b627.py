"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

DEADLINE = time.time() + 140.0  # safety margin before 150s kill


def _softplus(x, beta=1.0):
    """Numerically stable softplus."""
    return np.where(x > 1.0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x)))


def _hex_init(n, w, h, rng):
    """Generate hexagonal packing initialization in rectangle w x h."""
    # Start with hexagonal grid
    # For n circles, estimate grid size
    # Hexagonal packing: rows of alternating offsets
    # Try to fit n circles in w x h
    
    # Estimate radius from area
    area = w * h
    r_est = np.sqrt(area / (n * np.pi * 1.15))  # 1.15 for hex packing efficiency
    
    # Build hex grid
    circles = []
    dx = 2 * r_est
    dy = np.sqrt(3) * r_est
    
    # Center the grid
    # Try different grid configurations
    best_circles = None
    best_score = -1
    
    for rows in range(3, 10):
        cols_needed = int(np.ceil(n / rows))
        # Hex grid: alternate rows offset
        positions = []
        for row in range(rows):
            for col in range(cols_needed + 1):
                x = col * dx + (dx / 2 if row % 2 == 1 else 0)
                y = row * dy
                positions.append((x, y))
                if len(positions) >= n:
                    break
            if len(positions) >= n:
                break
        
        if len(positions) < n:
            continue
        
        positions = positions[:n]
        xs = np.array([p[0] for p in positions])
        ys = np.array([p[1] for p in positions])
        
        # Scale to fit in rectangle
        x_range = xs.max() - xs.min()
        y_range = ys.max() - ys.min()
        
        if x_range < 1e-10 or y_range < 1e-10:
            continue
        
        scale_x = w / (x_range + 2 * r_est)
        scale_y = h / (y_range + 2 * r_est)
        scale = min(scale_x, scale_y)
        
        # Center
        cx = (xs.max() + xs.min()) / 2
        cy = (ys.max() + ys.min()) / 2
        
        xs = (xs - cx) * scale + w / 2
        ys = (ys - cy) * scale + h / 2
        
        r = r_est * scale
        
        # Check if all circles fit
        valid = True
        for i in range(n):
            if xs[i] < r or xs[i] > w - r or ys[i] < r or ys[i] > h - r:
                valid = False
                break
        
        if valid:
            # Score: sum of radii (all same r here)
            score = n * r
            if score > best_score:
                best_score = score
                best_circles = np.column_stack([xs, ys, np.full(n, r)])
    
    if best_circles is None:
        # Fallback: uniform grid
        side = int(np.ceil(np.sqrt(n)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= n:
                    break
                x = (i + 0.5) * w / side
                y = (j + 0.5) * h / side
                r = min(w, h) / (2 * side) - 1e-3
                cells.append((x, y, r))
        best_circles = np.array(cells[:n])
    
    return best_circles


def _objective_and_grad(x, w, h, beta):
    """
    Objective: minimize -sum(r) + penalty for violations
    x = [x1, y1, r1, x2, y2, r2, ...]
    """
    n = len(x) // 3
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    # Negative sum of radii
    obj = -np.sum(rs)
    
    # Gradient
    grad = np.zeros_like(x)
    grad[2::3] = -1.0
    
    # Wall constraints: x_i >= r_i, x_i <= w - r_i, y_i >= r_i, y_i <= h - r_i
    # Use softplus for smoothness
    violations = np.maximum(0, rs - xs) + np.maximum(0, rs - (w - xs)) + \
                 np.maximum(0, rs - ys) + np.maximum(0, rs - (h - ys))
    obj += beta * np.sum(_softplus(violations, 1.0))
    
    # Pairwise constraints: dist(i,j) >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d = np.sqrt(dx * dx + dy * dy) + 1e-10
            min_d = rs[i] + rs[j]
            violation = min_d - d
            if violation > 0:
                sp = _softplus(violation, 1.0)
                obj += beta * sp
                # Gradient of softplus(v) * beta where v = min_d - d
                # d(sp)/dv = sigmoid(v)
                sig = 1.0 / (1.0 + np.exp(-violation))
                # d(v)/d(xs[i]) = -dx/d, etc.
                grad[0::3][i] += beta * sig * (-dx / d)
                grad[1::3][i] += beta * sig * (-dy / d)
                grad[2::3][i] += beta * sig
                grad[0::3][j] += beta * sig * (dx / d)
                grad[1::3][j] += beta * sig * (dy / d)
                grad[2::3][j] += beta * sig
    
    return obj, grad


def _is_valid(x, w, h, tol=1e-6):
    """Check if packing is valid."""
    n = len(x) // 3
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    # Wall constraints
    if np.any(xs < -tol) or np.any(xs > w + tol) or np.any(ys < -tol) or np.any(ys > h + tol):
        return False
    if np.any(rs < -tol):
        return False
    for i in range(n):
        if xs[i] < rs[i] - tol or xs[i] > w - rs[i] + tol:
            return False
        if ys[i] < rs[i] - tol or ys[i] > h - rs[i] + tol:
            return False
    
    # Pairwise
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d = np.sqrt(dx * dx + dy * dy)
            if d < rs[i] + rs[j] - tol:
                return False
    
    return True


def _polish(x0, w, h, max_iter=200):
    """Multi-phase polish with decreasing beta."""
    x = x0.copy()
    best_x = x.copy()
    best_val = -np.inf
    
    # Phase 1: beta = 1.0
    for beta in [1.0, 0.5, 0.1, 0.01]:
        res = minimize(_objective_and_grad, x, args=(w, h, beta),
                       method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
        x = res.x
        val = -_objective_and_grad(x, w, h, 0.0)[0]  # sum of radii
        if val > best_val:
            best_val = val
            best_x = x.copy()
    
    return best_x, best_val


def construct():
    """Construct 21 circles in a perimeter-4 rectangle, maximizing sum of radii."""
    start_time = time.time()
    best_result = None
    best_score = -np.inf
    
    # Explore different aspect ratios
    # Perimeter = 2(w + h) = 4, so w + h = 2
    # Try various aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.4, 2.5, 0.33, 3.0]
    
    rng = np.random.RandomState(42)
    
    for ar in aspect_ratios:
        if time.time() > DEADLINE - 10:
            break
        
        w = 2.0 * ar / (1 + ar)
        h = 2.0 / (1 + ar)
        
        # Generate initial guess
        x0 = _hex_init(N, w, h, rng)
        
        # Polish
        try:
            x_opt, val = _polish(x0, w, h)
            
            # Check validity
            if _is_valid(x_opt, w, h):
                score = np.sum(x_opt[2::3])
                if score > best_score:
                    best_score = score
                    best_result = x_opt.copy()
        except Exception:
            continue
    
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
        # Scale to fit perimeter 4: square has side 1
        # Already in unit square, perimeter = 4
    
    return best_result
# EVOLVE-BLOCK-END
