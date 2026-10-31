"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _validate(x, w, h):
    """Check if a configuration is valid."""
    n = len(x)
    if np.any(x[:, 0] - x[:, 2] < -1e-7) or np.any(x[:, 0] + x[:, 2] > w + 1e-7):
        return False
    if np.any(x[:, 1] - x[:, 2] < -1e-7) or np.any(x[:, 1] + x[:, 2] > h + 1e-7):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i, 0] - x[j, 0]
            dy = x[i, 1] - x[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < x[i, 2] + x[j, 2] - 1e-7:
                return False
    return True

def _objective(x, w, h):
    n = len(x)
    cx, cy, r = x[:, 0], x[:, 1], x[:, 2]
    penalty = 0.0
    penalty += np.sum(np.maximum(0, r - cx)) ** 2
    penalty += np.sum(np.maximum(0, r - (w - cx))) ** 2
    penalty += np.sum(np.maximum(0, r - cy)) ** 2
    penalty += np.sum(np.maximum(0, r - (h - cy))) ** 2
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                penalty += overlap ** 2
    return -np.sum(r) + penalty * 100.0

def _grad_objective(x, w, h):
    n = len(x)
    cx, cy, r = x[:, 0], x[:, 1], x[:, 2]
    grad = np.zeros_like(x)
    grad[:, 2] = -1.0
    for i in range(n):
        if r[i] > cx[i]:
            grad[i, 0] += 200 * (r[i] - cx[i])
            grad[i, 2] += 200 * (cx[i] - r[i])
        if r[i] > w - cx[i]:
            grad[i, 0] -= 200 * (r[i] - (w - cx[i]))
            grad[i, 2] += 200 * ((w - cx[i]) - r[i])
        if r[i] > cy[i]:
            grad[i, 1] += 200 * (r[i] - cy[i])
            grad[i, 2] += 200 * (cy[i] - r[i])
        if r[i] > h - cy[i]:
            grad[i, 1] -= 200 * (r[i] - (h - cy[i]))
            grad[i, 2] += 200 * ((h - cy[i]) - r[i])
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                factor = 200 * overlap
                if dist > 1e-10:
                    grad[i, 0] += factor * dx / dist
                    grad[i, 1] += factor * dy / dist
                    grad[j, 0] -= factor * dx / dist
                    grad[j, 1] -= factor * dy / dist
                grad[i, 2] += factor
                grad[j, 2] += factor
    return grad

def _local_optimize(x0, w, h, max_iter=5000, tol=1e-12):
    n = len(x0)
    bounds_flat = []
    for i in range(n):
        bounds_flat.append((0, w))
        bounds_flat.append((0, h))
        bounds_flat.append((0, min(w, h)))
    x_flat = x0.flatten()
    result = minimize(
        _objective, x_flat, args=(w, h),
        jac=_grad_objective,
        method='L-BFGS-B',
        bounds=bounds_flat,
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-12, 'maxls': 50}
    )
    x_opt = result.x.reshape(n, 3)
    return x_opt

def construct():
    start_time = time.time()
    deadline = start_time + 138.0
    
    best_x = None
    best_sum = 0.0
    
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 2.0, 0.67, 1.33]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        w = 2.0 * ar / (ar + 1.0)
        h = 2.0 / (ar + 1.0)
        
        for seed in range(4):
            if time.time() > deadline:
                break
            
            rng = np.random.RandomState(seed * 100 + int(ar * 10))
            n_grid = 10
            margin = 0.05
            xs = np.linspace(margin, w - margin, n_grid)
            ys = np.linspace(margin, h - margin, n_grid)
            
            x = np.zeros((N, 3))
            for i in range(N):
                gx = xs[rng.randint(n_grid)]
                gy = ys[rng.randint(n_grid)]
                gx += rng.uniform(-0.02, 0.02)
                gy += rng.uniform(-0.02, 0.02)
                gx = np.clip(gx, 0.01, w - 0.01)
                gy = np.clip(gy, 0.01, h - 0.01)
                r_est = 0.5 * min(w, h) / np.sqrt(N) * 0.7
                x[i] = [gx, gy, r_est]
            
            x_opt = _local_optimize(x, w, h, max_iter=3000, tol=1e-12)
            
            if _validate(x_opt, w, h):
                sum_r = np.sum(x_opt[:, 2])
                if sum_r > best_sum:
                    best_sum = sum_r
                    best_x = x_opt.copy()
    
    if best_x is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_x
# EVOLVE-BLOCK-END
