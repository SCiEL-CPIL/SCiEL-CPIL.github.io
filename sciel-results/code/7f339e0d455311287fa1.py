"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _check_valid(circles, tol=1e-7):
    """Check if circles are valid (disjoint, inside unit square)."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0 or r > 1:
            return False
        if x - r < -tol or x + r > 1 + tol:
            return False
        if y - r < -tol or y + r > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < ri + rj - tol:
                return False
    return True

def _relax_and_grow(circles, max_iter=500, growth_rate=0.002, shrink_rate=0.001):
    """
    Iteratively relax overlaps and grow radii.
    circles: (N, 3) array
    """
    circles = circles.copy()
    n = len(circles)
    
    for it in range(max_iter):
        # Check for overlaps and push apart
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                dx = xj - xi
                dy = yj - yi
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = ri + rj
                if dist < min_dist - 1e-10:
                    if dist < 1e-10:
                        angle = random.random() * 2 * np.pi
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                        dist = 1.0
                    overlap = min_dist - dist
                    ux = dx / dist
                    uy = dy / dist
                    push = overlap / 2.0
                    circles[i, 0] -= push * ux
                    circles[i, 1] -= push * uy
                    circles[j, 0] += push * ux
                    circles[j, 1] += push * uy
        
        # Keep circles inside the square
        for i in range(n):
            r = circles[i, 2]
            circles[i, 0] = np.clip(circles[i, 0], r, 1 - r)
            circles[i, 1] = np.clip(circles[i, 1], r, 1 - r)
        
        # Try to grow radii
        if it % 10 == 0:
            can_grow = True
            for i in range(n):
                for j in range(i+1, n):
                    xi, yi, ri = circles[i]
                    xj, yj, rj = circles[j]
                    dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                    if dist < ri + rj + growth_rate:
                        can_grow = False
                        break
                if not can_grow:
                    break
            for i in range(n):
                r = circles[i, 2]
                if circles[i, 0] - r < growth_rate or 1 - circles[i, 0] - r < growth_rate:
                    can_grow = False
                    break
                if circles[i, 1] - r < growth_rate or 1 - circles[i, 1] - r < growth_rate:
                    can_grow = False
                    break
            
            if can_grow:
                circles[:, 2] += growth_rate
            else:
                circles[:, 2] = np.maximum(circles[:, 2] - shrink_rate, 0.01)
    
    return circles

def _grid_init(n):
    """Grid initialization."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 0.01
            cells.append([x, y, r])
            count += 1
    return np.array(cells[:n])

def _hex_init(n):
    """Hexagonal initialization."""
    circles = []
    r = 0.04
    spacing = 2 * r * 1.01
    dx = spacing
    dy = spacing * np.sqrt(3) / 2
    row = 0
    while len(circles) < n:
        for col in range(int(1/dx) + 1):
            if len(circles) >= n:
                break
            x = (col + 0.5) * dx
            y = (row + 0.5) * dy
            if 0 <= x <= 1 and 0 <= y <= 1:
                circles.append([x, y, r])
        row += 1
    return np.array(circles[:n])

def _random_init(n, seed=42):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    circles = []
    for i in range(n):
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        r = 0.03
        circles.append([x, y, r])
    return np.array(circles)

def _soft_penalty_objective_and_grad(z, n_circles, penalty_weight):
    """
    z: (3*N,) vector = [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    Objective: -sum(r_i) + penalty_weight * sum(violations^2)
    where violations include circle-circle overlaps and boundary violations.
    Returns (obj, grad)
    """
    N = n_circles
    z = z.reshape(N, 3)
    xs = z[:, 0]
    ys = z[:, 1]
    rs = z[:, 2]
    
    obj = -np.sum(rs)
    grad = np.zeros_like(z)
    grad[:, 2] = -1.0  # derivative of -sum(r) w.r.t. r
    
    # Circle-circle constraints: dist(i,j) >= r_i + r_j
    # violation_ij = max(0, r_i + r_j - dist_ij)
    # We use smooth approximation: violation^2 where violation = r_i + r_j - dist_ij
    # For smoothness, we penalize (r_i + r_j - dist_ij)^2 when positive
    
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[j] - xs[i]
            dy = ys[j] - ys[i]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq + 1e-12)
            
            # Required distance
            req = rs[i] + rs[j]
            # Violation (positive means overlap)
            viol = req - dist
            
            if viol > 0:
                # Gradient of (viol^2) w.r.t. positions and radii
                # d(viol^2)/d(x_i) = 2*viol * d(viol)/d(x_i)
                # d(viol)/d(x_i) = d(req - dist)/d(x_i) = -d(dist)/d(x_i) = (x_i - x_j)/dist = -dx/dist
                # d(viol)/d(y_i) = -dy/dist
                # d(viol)/d(r_i) = 1
                # d(viol)/d(r_j) = 1
                
                factor = 2.0 * viol / dist
                # Gradient w.r.t. x_i: 2*viol * (-dx/dist) = -2*viol*dx/dist
                grad[i, 0] += penalty_weight * factor * (-dx)
                grad[i, 1] += penalty_weight * factor * (-dy)
                # Gradient w.r.t. x_j: 2*viol * (dx/dist) = 2*viol*dx/dist
                grad[j, 0] += penalty_weight * factor * dx
                grad[j, 1] += penalty_weight * factor * dy
                # Gradient w.r.t. r_i and r_j
                grad[i, 2] += penalty_weight * 2.0 * viol
                grad[j, 2] += penalty_weight * 2.0 * viol
                
                obj += penalty_weight * viol * viol
    
    # Boundary constraints:
    # x_i >= r_i, x_i <= 1 - r_i
    # y_i >= r_i, y_i <= 1 - r_i
    
    for i in range(N):
        # Left boundary: x_i - r_i >= 0
        viol_left = rs[i] - xs[i]
        if viol_left > 0:
            grad[i, 0] += penalty_weight * 2.0 * viol_left
            grad[i, 2] += penalty_weight * 2.0 * viol_left
            obj += penalty_weight * viol_left * viol_left
        
        # Right boundary: 1 - x_i - r_i >= 0
        viol_right = xs[i] + rs[i] - 1.0
        if viol_right > 0:
            grad[i, 0] += penalty_weight * 2.0 * viol_right
            grad[i, 2] += penalty_weight * 2.0 * viol_right
            obj += penalty_weight * viol_right * viol_right
        
        # Bottom boundary: y_i - r_i >= 0
        viol_bottom = rs[i] - ys[i]
        if viol_bottom > 0:
            grad[i, 1] += penalty_weight * 2.0 * viol_bottom
            grad[i, 2] += penalty_weight * 2.0 * viol_bottom
            obj += penalty_weight * viol_bottom * viol_bottom
        
        # Top boundary: 1 - y_i - r_i >= 0
        viol_top = ys[i] + rs[i] - 1.0
        if viol_top > 0:
            grad[i, 1] += penalty_weight * 2.0 * viol_top
            grad[i, 2] += penalty_weight * 2.0 * viol_top
            obj += penalty_weight * viol_top * viol_top
    
    return obj, grad

def _gradient_optimize(circles, max_iter=200, penalty_start=10.0, penalty_end=1000.0, tol=1e-8):
    """
    Use L-BFGS-B to optimize circle positions and radii.
    """
    n_circles = len(circles)
    z0 = circles.flatten()
    
    # Bounds for each variable
    # For each circle: x in [0,1], y in [0,1], r in [0.001, 0.5]
    bounds = []
    for i in range(n_circles):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.001, 0.5))  # r
    
    # Start with moderate penalty, increase over iterations
    current_penalty = penalty_start
    
    for it in range(max_iter):
        if time.time() > _deadline:
            break
        
        # Define objective and gradient for current penalty
        def obj_grad(z):
            return _soft_penalty_objective_and_grad(z, n_circles, current_penalty)
        
        result = minimize(
            obj_grad,
            z0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        z0 = result.x
        current_penalty *= 2.0  # Increase penalty
        if current_penalty > penalty_end:
            current_penalty = penalty_end
        
        # Check if solution is valid
        circles_test = z0.reshape(n_circles, 3)
        if _check_valid(circles_test, tol=1e-6):
            # Check for improvement
            score = np.sum(circles_test[:, 2])
            if it > 0:
                prev_score = np.sum(circles[:, 2])
                if score > prev_score + 1e-8:
                    circles = circles_test.copy()
    
    # Final result
    circles_final = z0.reshape(n_circles, 3)
    if _check_valid(circles_final, tol=1e-6):
        return circles_final
    else:
        return circles

def construct():
    global _deadline
    start_time = time.time()
    _deadline = start_time + 130  # Safety margin
    
    best_circles = None
    best_score = -np.inf
    
    # Try multiple initializations
    inits = [
        ("grid", _grid_init(N)),
        ("hex", _hex_init(N)),
        ("random1", _random_init(N, seed=42)),
        ("random2", _random_init(N, seed=123)),
        ("random3", _random_init(N, seed=999)),
    ]
    
    for init_name, init_circles in inits:
        if time.time() > _deadline:
            break
        
        # First: relax and grow to get a feasible starting point
        circles = _relax_and_grow(init_circles, max_iter=200, growth_rate=0.001, shrink_rate=0.0005)
        
        # Then: gradient-based optimization
        if time.time() < _deadline:
            circles_opt = _gradient_optimize(circles, max_iter=30)
            
            if _check_valid(circles_opt, tol=1e-6):
                score = np.sum(circles_opt[:, 2])
                if score > best_score:
                    best_score = score
                    best_circles = circles_opt.copy()
        
        # Also try the relaxed version
        if _check_valid(circles, tol=1e-6):
            score = np.sum(circles[:, 2])
            if score > best_score:
                best_score = score
                best_circles = circles.copy()
    
    if best_circles is None:
        best_circles = _grid_init(N)
    
    return best_circles
# EVOLVE-BLOCK-END
