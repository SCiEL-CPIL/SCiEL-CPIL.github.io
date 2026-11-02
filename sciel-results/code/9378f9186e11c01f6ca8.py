"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
_START_TIME = None
_DEADLINE = 140.0


def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def _remaining():
    return _DEADLINE - _elapsed()


def _compute_obj_grad(params, n, lam=10.0):
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:]
    r_safe = np.maximum(r, 1e-8)
    
    f = -np.sum(r_safe)
    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = np.zeros(n)
    
    # Boundary
    for i in range(n):
        ri = r_safe[i]
        xi = x[i]
        yi = y[i]
        
        if xi < ri:
            v = ri - xi
            f += lam * v * v
            gx[i] += lam * 2 * v * (-1)
            gr[i] += lam * 2 * v
        elif xi > 1.0 - ri:
            v = xi - (1.0 - ri)
            f += lam * v * v
            gx[i] += lam * 2 * v
            gr[i] += lam * 2 * v * (-1)
        
        if yi < ri:
            v = ri - yi
            f += lam * v * v
            gy[i] += lam * 2 * v * (-1)
            gr[i] += lam * 2 * v
        elif yi > 1.0 - ri:
            v = yi - (1.0 - ri)
            f += lam * v * v
            gy[i] += lam * 2 * v
            gr[i] += lam * 2 * v * (-1)
    
    # Overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx*dx + dy*dy
            if d2 < 1e-12:
                d = 1e-6
                dxn = 1.0
                dyn = 0.0
            else:
                d = np.sqrt(d2)
                dxn = dx / d
                dyn = dy / d
            
            min_d = r_safe[i] + r_safe[j]
            if d < min_d:
                v = min_d - d
                f += lam * v * v
                
                gx[i] += lam * 2 * v * dxn
                gy[i] += lam * 2 * v * dyn
                gx[j] += lam * 2 * v * (-dxn)
                gy[j] += lam * 2 * v * (-dyn)
                gr[i] += lam * 2 * v
                gr[j] += lam * 2 * v
    
    g = np.concatenate([gx, gy, gr])
    return f, g


def _hex_initial(n=N):
    positions = []
    radius = 0.065
    dx = 2 * radius
    dy = radius * np.sqrt(3)
    row = 0
    while len(positions) < n:
        col = 0
        while len(positions) < n:
            x = radius + col * dx + (row % 2) * radius
            y = radius + row * dy
            if 0 <= x <= 1 and 0 <= y <= 1:
                positions.append((x, y))
            col += 1
        row += 1
    positions = positions[:n]
    x = np.array([p[0] for p in positions])
    y = np.array([p[1] for p in positions])
    r = np.full(n, radius)
    return np.concatenate([x, y, r])


def _random_initial(n=N, seed=42):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.15, 0.85, n)
    y = rng.uniform(0.15, 0.85, n)
    r = np.full(n, 0.04)
    return np.concatenate([x, y, r])


def _check_validity(result):
    x, y, r = result[:, 0], result[:, 1], result[:, 2]
    n = len(x)
    for i in range(n):
        if r[i] <= 1e-9:
            return False
        if x[i] < r[i] - 1e-8 or x[i] > 1.0 - r[i] + 1e-8:
            return False
        if y[i] < r[i] - 1e-8 or y[i] > 1.0 - r[i] + 1e-8:
            return False
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < r[i] + r[j] - 1e-8:
                return False
    return True


def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    n = N
    best_result = None
    best_score = -np.inf
    
    inits = [
        _hex_initial(n),
        _random_initial(n, seed=1),
        _random_initial(n, seed=2),
        _random_initial(n, seed=3),
        _random_initial(n, seed=7),
        _random_initial(n, seed=13),
    ]
    
    bounds = [(0, 1)] * n + [(0, 1)] * n + [(1e-6, 0.5)] * n
    
    for idx, init in enumerate(inits):
        if _remaining() < 15:
            break
        
        try:
            def obj(p):
                return _compute_obj_grad(p, n)[0]
            
            def grad(p):
                return _compute_obj_grad(p, n)[1]
            
            result = minimize(
                obj, init, jac=grad, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 40}
            )
            
            x = result.x[:n]
            y = result.x[n:2*n]
            r = np.maximum(result.x[2*n:], 1e-6)
            candidate = np.column_stack([x, y, r])
            
            if _check_validity(candidate):
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_result = candidate
        except Exception:
            continue
    
    if best_result is None:
        side = int(np.ceil(np.sqrt(n)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:n])
    
    return best_result
