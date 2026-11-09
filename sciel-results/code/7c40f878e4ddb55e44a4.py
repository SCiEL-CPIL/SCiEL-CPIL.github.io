"""Rewritten program: local-search circle packing in the unit square."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 26
DEADLINE = 115.0  # safety margin before the 120 s kill


def _max_radius(x, y, other_idx, xs, ys, rs):
    """Largest radius at (x, y) given other circles and square walls."""
    r = min(x, 1.0 - x, y, 1.0 - y)
    for j in other_idx:
        d = math.hypot(x - xs[j], y - ys[j])
        r = min(r, d - rs[j])
    return max(r, 1e-9)


def _score(xs, ys, rs):
    return float(np.sum(rs))


def _validate(xs, ys, rs, tol=1e-7):
    if len(xs) != N:
        return False
    for i in range(N):
        if xs[i] < -tol or xs[i] > 1.0 + tol or ys[i] < -tol or ys[i] > 1.0 + tol:
            return False
        if rs[i] < 0:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            d = math.hypot(xs[i] - xs[j], ys[i] - ys[j])
            if d < rs[i] + rs[j] - tol:
                return False
    return True


def _clamp(x, y, r):
    x = min(max(x, r), 1.0 - r)
    y = min(max(y, r), 1.0 - r)
    return x, y


def _local_search(xs, ys, rs, deadline):
    """Iterative improvement: move each circle to maximize its own radius."""
    best_score = _score(xs, ys, rs)
    n = len(xs)
    while time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            old_r = rs[i]
            # Try moving circle i in small steps; pick best radius
            # Use a few random perturbations plus a local grid search
            best_r = old_r
            best_x, best_y = xs[i], ys[i]
            # Random perturbations
            for _ in range(20):
                dx = np.random.uniform(-0.05, 0.05)
                dy = np.random.uniform(-0.05, 0.05)
                nx, ny = xs[i] + dx, ys[i] + dy
                # Clamp to feasible region given current radius
                nx, ny = _clamp(nx, ny, old_r)
                other = [j for j in range(n) if j != i]
                new_r = _max_radius(nx, ny, other, xs, ys, rs)
                if new_r > best_r + 1e-12:
                    best_r = new_r
                    best_x, best_y = nx, ny
            if best_r > old_r + 1e-12:
                xs[i], ys[i], rs[i] = best_x, best_y, best_r
                improved = True
                new_score = _score(xs, ys, rs)
                if new_score > best_score:
                    best_score = new_score
        if not improved:
            # Try larger random jumps to escape local optima
            for _ in range(50):
                if time.time() >= deadline:
                    break
                i = np.random.randint(n)
                old_x, old_y, old_r = xs[i], ys[i], rs[i]
                nx = np.random.uniform(0.05, 0.95)
                ny = np.random.uniform(0.05, 0.95)
                nx, ny = _clamp(nx, ny, old_r)
                other = [j for j in range(n) if j != i]
                new_r = _max_radius(nx, ny, other, xs, ys, rs)
                if new_r > old_r + 1e-12:
                    xs[i], ys[i], rs[i] = nx, ny, new_r
                    improved = True
                    new_score = _score(xs, ys, rs)
                    if new_score > best_score:
                        best_score = new_score
            if not improved:
                break
    return xs, ys, rs, best_score


def _soft_min(vals, beta=50.0):
    """Smooth approximation of min using log-sum-exp."""
    m = np.min(vals)
    return m - math.log(np.sum(np.exp(-beta * (vals - m)))) / beta


def _soft_min_grad(vals, beta=50.0):
    """Gradient of soft_min w.r.t. vals."""
    m = np.min(vals)
    exps = np.exp(-beta * (vals - m))
    weights = exps / np.sum(exps)
    return -weights


def _polish_smooth(xs, ys, rs, deadline):
    """Polish using scipy L-BFGS-B with a soft-min surrogate for the radius objective.
    
    We maximize sum of radii, where each radius is constrained by walls and other circles.
    Use a smooth penalty formulation: minimize -sum(r_i) + penalty for constraint violations.
    """
    n = len(xs)
    best_score = _score(xs, ys, rs)
    
    # Pack into a single vector: [x0, y0, r0, x1, y1, r1, ...]
    def unpack(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        return x, y, r
    
    def objective(v):
        x, y, r = unpack(v)
        # Penalty for going outside the unit square
        pen = 0.0
        for i in range(n):
            pen += max(0, r[i] - x[i])**2
            pen += max(0, r[i] - (1.0 - x[i]))**2
            pen += max(0, r[i] - y[i])**2
            pen += max(0, r[i] - (1.0 - y[i]))**2
        # Penalty for circle-circle overlaps
        for i in range(n):
            for j in range(i + 1, n):
                d = math.hypot(x[i] - x[j], y[i] - y[j])
                overlap = r[i] + r[j] - d
                if overlap > 0:
                    pen += overlap**2
        # We want to maximize sum(r), so minimize -sum(r) + penalty
        return -np.sum(r) + 100.0 * pen
    
    def gradient(v):
        x, y, r = unpack(v)
        grad = np.zeros_like(v)
        
        # Gradient of -sum(r)
        grad[2::3] = -1.0
        
        # Gradient of wall penalties
        for i in range(n):
            # r - x < 0 means x < r (left wall violation)
            if r[i] - x[i] > 0:
                grad[2 * i] += -200.0 * (r[i] - x[i])  # d/dx of (r-x)^2 = -2(r-x)
                grad[2 * i + 2] += 200.0 * (r[i] - x[i])   # d/dr of (r-x)^2 = 2(r-x)
            # r - (1-x) < 0 means x > 1-r (right wall violation)
            if r[i] - (1.0 - x[i]) > 0:
                grad[2 * i] += 200.0 * (r[i] - (1.0 - x[i]))   # d/dx of (r-(1-x))^2 = 2(r-(1-x))
                grad[2 * i + 2] += 200.0 * (r[i] - (1.0 - x[i]))
            # r - y < 0 means y < r (bottom wall violation)
            if r[i] - y[i] > 0:
                grad[2 * i + 1] += -200.0 * (r[i] - y[i])
                grad[2 * i + 2] += 200.0 * (r[i] - y[i])
            # r - (1-y) < 0 means y > 1-r (top wall violation)
            if r[i] - (1.0 - y[i]) > 0:
                grad[2 * i + 1] += 200.0 * (r[i] - (1.0 - y[i]))
                grad[2 * i + 2] += 200.0 * (r[i] - (1.0 - y[i]))
        
        # Gradient of circle-circle overlap penalties
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = math.hypot(dx, dy)
                if d < 1e-12:
                    continue
                overlap = r[i] + r[j] - d
                if overlap > 0:
                    # d/dx_i of (r_i + r_j - d)^2 = 2*overlap * (dx/d)
                    grad[2 * i] += 200.0 * overlap * (dx / d)
                    grad[2 * i + 1] += 200.0 * overlap * (dy / d)
                    grad[2 * i + 2] += 200.0 * overlap
                    # d/dx_j
                    grad[2 * j] += -200.0 * overlap * (dx / d)
                    grad[2 * j + 1] += -200.0 * overlap * (dy / d)
                    grad[2 * j + 2] += 200.0 * overlap
        
        return grad
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((1e-9, 0.5))  # r
    
    v0 = np.concatenate([xs, ys, rs])
    
    try:
        result = minimize(
            objective,
            v0,
            jac=gradient,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000}
        )
        
        if result.success or result.fun < objective(v0):
            x_new, y_new, r_new = unpack(result.x)
            # Validate and clamp
            valid = True
            for i in range(n):
                if x_new[i] < -1e-6 or x_new[i] > 1.0 + 1e-6:
                    valid = False
                if y_new[i] < -1e-6 or y_new[i] > 1.0 + 1e-6:
                    valid = False
                if r_new[i] < 0:
                    valid = False
            
            if valid:
                # Clamp to ensure feasibility
                for i in range(n):
                    x_new[i] = min(max(x_new[i], r_new[i]), 1.0 - r_new[i])
                    y_new[i] = min(max(y_new[i], r_new[i]), 1.0 - r_new[i])
                
                # Check circle-circle constraints
                for i in range(n):
                    for j in range(i + 1, n):
                        d = math.hypot(x_new[i] - x_new[j], y_new[i] - y_new[j])
                        if d < r_new[i] + r_new[j] - 1e-7:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    score_new = _score(x_new, y_new, r_new)
                    if score_new > best_score:
                        xs, ys, rs = x_new, y_new, r_new
                        best_score = score_new
    except Exception:
        pass
    
    # Final local search polish
    xs, ys, rs, best_score = _local_search(xs, ys, rs, deadline)
    
    return xs, ys, rs, best_score


def _initial_grid():
    """Create an initial grid-based packing."""
    side = int(math.ceil(math.sqrt(N)))
    xs, ys, rs = [], [], []
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            xs.append(x)
            ys.append(y)
            rs.append(r)
            count += 1
    return np.array(xs), np.array(ys), np.array(rs)


def _initial_hexagonal():
    """Create an initial hexagonal (close-packed) packing."""
    # Hexagonal packing has higher density than square grid
    # For 26 circles, use a hexagonal arrangement
    xs, ys, rs = [], [], []
    
    # Hexagonal lattice: rows offset by half spacing
    # Try different configurations
    best_config = None
    best_score = -1
    
    for rows in range(3, 8):
        cols = int(math.ceil(N / rows))
        if cols * rows < N:
            continue
        
        # Hexagonal spacing
        spacing = 1.0 / (cols + 0.5)  # account for offset
        r = spacing * 0.5 * 0.95  # slightly smaller to avoid overlap
        
        count = 0
        for row in range(rows):
            for col in range(cols):
                if count >= N:
                    break
                # Offset odd rows by half spacing
                x_offset = spacing * 0.5 if row % 2 == 1 else 0
                x = (col + 0.5) * spacing + x_offset
                y = (row + 0.5) * spacing
                x = min(max(x, r), 1.0 - r)
                y = min(max(y, r), 1.0 - r)
                xs.append(x)
                ys.append(y)
                rs.append(r)
                count += 1
        
        if count == N:
            # Check for overlaps
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    d = math.hypot(xs[i] - xs[j], ys[i] - ys[j])
                    if d < rs[i] + rs[j] - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                score = sum(rs)
                if score > best_score:
                    best_score = score
                    best_config = (np.array(xs[:]), np.array(ys[:]), np.array(rs[:]))
    
    if best_config is not None:
        return best_config
    
    # Fallback to grid
    return _initial_grid()


def _initial_random():
    """Create a random initial packing with small circles."""
    xs, ys, rs = [], [], []
    for _ in range(N):
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        r = 0.02
        xs.append(x)
        ys.append(y)
        rs.append(r)
    return np.array(xs), np.array(ys), np.array(rs)


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    # Try multiple restarts with different initializations
    best_xs, best_ys, best_rs = None, None, None
    best_score = -1.0

    # Try hexagonal initialization (denser start)
    xs, ys, rs = _initial_hexagonal()
    # Run local search then smooth polish
    xs, ys, rs, score = _local_search(xs, ys, rs, deadline)
    if score > best_score:
        best_score = score
        best_xs, best_ys, best_rs = xs.copy(), ys.copy(), rs.copy()
    
    # Apply smooth polish if time remains
    if time.time() < deadline - 10.0:
        xs, ys, rs, score = _polish_smooth(xs, ys, rs, deadline)
        if score > best_score:
            best_score = score
            best_xs, best_ys, best_rs = xs.copy(), ys.copy(), rs.copy()

    # Start with grid initialization
    if time.time() < deadline - 10.0:
        xs, ys, rs = _initial_grid()
        xs, ys, rs, score = _local_search(xs, ys, rs, deadline)
        if score > best_score:
            best_score = score
            best_xs, best_ys, best_rs = xs.copy(), ys.copy(), rs.copy()
        
        # Apply smooth polish if time remains
        if time.time() < deadline - 10.0:
            xs, ys, rs, score = _polish_smooth(xs, ys, rs, deadline)
            if score > best_score:
                best_score = score
                best_xs, best_ys, best_rs = xs.copy(), ys.copy(), rs.copy()

    # Try several random restarts
    restart_count = 0
    while time.time() < deadline - 15.0:
        xs, ys, rs = _initial_random()
        xs, ys, rs, score = _local_search(xs, ys, rs, deadline)
        if score > best_score:
            best_score = score
            best_xs, best_ys, best_rs = xs.copy(), ys.copy(), rs.copy()
        
        # Apply smooth polish to best result if time remains
        if time.time() < deadline - 10.0 and restart_count % 2 == 0:
            xs, ys, rs, score = _polish_smooth(xs, ys, rs, deadline)
            if score > best_score:
                best_score = score
                best_xs, best_ys, best_rs = xs.copy(), ys.copy(), rs.copy()
        
        restart_count += 1

    # Final validation
    if best_xs is not None:
        if not _validate(best_xs, best_ys, best_rs):
            # Fallback to grid if validation fails
            xs, ys, rs = _initial_grid()
            best_xs, best_ys, best_rs = xs, ys, rs

    result = np.column_stack([best_xs, best_ys, best_rs])
    return result
