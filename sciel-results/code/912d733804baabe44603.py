"""Circle packing in unit square - 26 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 26
TIME_LIMIT = 115.0  # safety margin before 120s kill


def _check_validity(circles, tol=1e-7):
    """Check if circles are valid (inside unit square, disjoint)."""
    n = len(circles)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Check boundaries
    if np.any(r <= 0):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _softplus(x, beta=100.0):
    """Smooth approximation of max(0, x)."""
    return np.log1p(np.exp(-beta * x)) / beta


def _objective_and_grad(params):
    """
    params: [x_1, y_1, r_1, x_2, y_2, r_2, ...] length 3N
    Minimize negative sum of radii + penalties for constraint violations.
    """
    n = N
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Objective: maximize sum of radii
    obj = -np.sum(r)
    
    # Gradient of objective
    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = -np.ones(n)
    
    # Penalty for boundary violations
    # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
    beta = 50.0
    # x - r < 0
    viol_xl = -(x - r)  # positive when violated
    pen = np.sum(_softplus(viol_xl, beta))
    obj += 10.0 * pen
    gx -= 10.0 * (1.0 / (1.0 + np.exp(-beta * viol_xl)))
    gr += 10.0 * (1.0 / (1.0 + np.exp(-beta * viol_xl)))
    
    # x + r > 1
    viol_xr = (x + r) - 1.0
    pen = np.sum(_softplus(viol_xr, beta))
    obj += 10.0 * pen
    gx += 10.0 * (1.0 / (1.0 + np.exp(-beta * viol_xr)))
    gr += 10.0 * (1.0 / (1.0 + np.exp(-beta * viol_xr)))
    
    # y - r < 0
    viol_yl = -(y - r)
    pen = np.sum(_softplus(viol_yl, beta))
    obj += 10.0 * pen
    gy -= 10.0 * (1.0 / (1.0 + np.exp(-beta * viol_yl)))
    gr += 10.0 * (1.0 / (1.0 + np.exp(-beta * viol_yl)))
    
    # y + r > 1
    viol_yr = (y + r) - 1.0
    pen = np.sum(_softplus(viol_yr, beta))
    obj += 10.0 * pen
    gy += 10.0 * (1.0 / (1.0 + np.exp(-beta * viol_yr)))
    gr += 10.0 * (1.0 / (1.0 + np.exp(-beta * viol_yr)))
    
    # Penalty for overlap: dist(i,j) >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy) + 1e-12
            viol = (r[i] + r[j]) - dist
            s = _softplus(viol, beta)
            obj += 5.0 * s
            grad_s = 1.0 / (1.0 + np.exp(-beta * viol))
            gx[i] += 5.0 * grad_s * (dx / dist)
            gx[j] -= 5.0 * grad_s * (dx / dist)
            gy[i] += 5.0 * grad_s * (dy / dist)
            gy[j] -= 5.0 * grad_s * (dy / dist)
            gr[i] += 5.0 * grad_s
            gr[j] += 5.0 * grad_s
    
    grad = np.empty(3 * n)
    grad[0::3] = gx
    grad[1::3] = gy
    grad[2::3] = gr
    
    return obj, grad


def _repair(circles):
    """Repair minor constraint violations by shrinking radii and adjusting positions."""
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    
    # Clamp to boundaries
    for i in range(n):
        r[i] = min(r[i], x[i], 1.0 - x[i], y[i], 1.0 - y[i])
        r[i] = max(r[i], 1e-6)
    
    # Iteratively resolve overlaps
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    # Shrink the larger circle
                    overlap = min_dist - dist
                    if r[i] >= r[j]:
                        r[i] = max(r[i] - overlap / 2, 1e-6)
                    else:
                        r[j] = max(r[j] - overlap / 2, 1e-6)
                    changed = True
        # Re-clamp to boundaries
        for i in range(n):
            r[i] = min(r[i], x[i], 1.0 - x[i], y[i], 1.0 - y[i])
            r[i] = max(r[i], 1e-6)
        if not changed:
            break
    
    return np.column_stack([x, y, r])


def _grid_seed():
    """Generate a grid-based initial seed."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            cr = 0.5 / side - 1e-4
            cells.append([cx, cy, cr])
    return np.array(cells[:N])


def _random_seed(rng):
    """Generate a random initial seed."""
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = rng.uniform(0.02, 0.1, N)
    return np.column_stack([x, y, r])


def construct():
    start_time = time.time()
    best_circles = _grid_seed()
    best_score = np.sum(best_circles[:, 2])
    
    rng = np.random.default_rng(42)
    
    bounds = [(0, 1), (0, 1), (0.001, 0.5)] * N
    
    # Multi-start optimization
    n_starts = 0
    while time.time() - start_time < TIME_LIMIT - 5:
        if n_starts == 0:
            seed = best_circles
        elif n_starts % 3 == 0:
            seed = _grid_seed()
            # Add some perturbation
            seed = seed + rng.uniform(-0.02, 0.02, seed.shape)
            seed[:, 0] = np.clip(seed[:, 0], 0.01, 0.99)
            seed[:, 1] = np.clip(seed[:, 1], 0.01, 0.99)
            seed[:, 2] = np.clip(seed[:, 2], 0.001, 0.4)
        else:
            seed = _random_seed(rng)
        
        x0 = seed.flatten()
        
        try:
            res = minimize(
                _objective_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if res.success or res.fun < 0:
                opt = res.x.reshape(N, 3)
                # Repair
                opt = _repair(opt)
                if _check_validity(opt):
                    score = np.sum(opt[:, 2])
                    if score > best_score:
                        best_score = score
                        best_circles = opt.copy()
        except Exception:
            pass
        
        n_starts += 1
        # Check time
        if time.time() - start_time > TIME_LIMIT - 10:
            break
    
    # Final polish: try to improve the best solution
    if time.time() - start_time < TIME_LIMIT - 3:
        try:
            x0 = best_circles.flatten()
            res = minimize(
                _objective_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
            )
            if res.success:
                opt = res.x.reshape(N, 3)
                opt = _repair(opt)
                if _check_validity(opt):
                    score = np.sum(opt[:, 2])
                    if score > best_score:
                        best_score = score
                        best_circles = opt.copy()
        except Exception:
            pass
    
    return best_circles


# EVOLVE-BLOCK-END
