import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    max_time = 110  # seconds

    n_vars = 1 + 3 * N
    
    def unpack(v):
        W = v[0]
        H = 2.0 - W
        circles = v[1:].reshape(N, 3)
        return W, H, circles

    def penalty_and_obj(v, lam):
        W = v[0]
        H = 2.0 - W
        if W <= 0.01 or H <= 0.01:
            return 1e6, np.zeros(n_vars)
            
        circles = v[1:].reshape(N, 3)
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Objective: maximize sum of radii
        obj = -np.sum(r)
        
        pen = 0.0
        
        # Boundary constraints
        pen += np.sum(np.maximum(0, r - x)**2)
        pen += np.sum(np.maximum(0, x - (W - r))**2)
        pen += np.sum(np.maximum(0, r - y)**2)
        pen += np.sum(np.maximum(0, y - (H - r))**2)
        
        # Overlap constraints
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        d = np.sqrt(dx**2 + dy**2)
        
        min_d = r[:, np.newaxis] + r[np.newaxis, :]
        
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        violations = np.maximum(0, min_d - d)
        violations = violations[mask]
        pen += np.sum(violations**2)
        
        total = obj + lam * pen
        return total, np.zeros(n_vars)

    def objective(v):
        total, _ = penalty_and_obj(v, current_lam)
        return total

    best_val = -np.inf
    best_v = None
    
    # Initializations
    inits = [
        _init_grid,
        _init_random,
        _init_hex
    ]
    
    for init_fn in inits:
        if time.time() - start_time > max_time:
            break
            
        v0 = init_fn()
        if v0 is None: continue
        
        current_lam = 1.0
        v = v0.copy()
        
        # Annealing penalty
        for lam_step in range(8):
            current_lam *= 5.0
            res = minimize(objective, v, method='SLSQP', 
                           bounds=[(0.1, 1.9)] + [(0.01, 1.0)] * (3*N),
                           options={'maxiter': 30, 'ftol': 1e-5})
            v = res.x
            if time.time() - start_time > max_time:
                break
                
        # Validate
        W, H, circles = unpack(v)
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
        
        valid = True
        if W <= 0.01 or H <= 0.01: valid = False
        
        if valid:
            if np.any(x < r - 1e-6) or np.any(x > W - r + 1e-6) or \
               np.any(y < r - 1e-6) or np.any(y > H - r + 1e-6):
                valid = False
        
        if valid:
            dx = x[:, np.newaxis] - x[np.newaxis, :]
            dy = y[:, np.newaxis] - y[np.newaxis, :]
            d = np.sqrt(dx**2 + dy**2)
            min_d = r[:, np.newaxis] + r[np.newaxis, :]
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            if np.any((min_d - d)[mask] > 1e-6):
                valid = False
                
        if valid:
            val = np.sum(r)
            if val > best_val:
                best_val = val
                best_v = v.copy()
                
    if best_v is None:
        return _fallback_grid()
        
    W, H, circles = unpack(best_v)
    return circles

def _init_grid():
    N = 21
    W = 1.0
    H = 1.0
    x_grid = np.linspace(0.1, 0.9, 5)
    y_grid = np.linspace(0.1, 0.9, 5)
    pts = []
    for i in range(5):
        for j in range(5):
            pts.append([x_grid[i], y_grid[j], 0.05])
            if len(pts) == N: break
        if len(pts) == N: break
    v = np.zeros(1 + 3*N)
    v[0] = W
    v[1:] = np.array(pts).flatten()
    return v

def _init_random():
    N = 21
    W = 1.0
    H = 1.0
    v = np.zeros(1 + 3*N)
    v[0] = W
    for i in range(N):
        v[1+3*i] = np.random.uniform(0.1, 0.9)
        v[2+3*i] = np.random.uniform(0.1, 0.9)
        v[3+3*i] = 0.05
    return v

def _init_hex():
    N = 21
    W = 1.0
    H = 1.0
    pts = []
    r = 0.05
    dx = 2*r * 1.1
    dy = 2*r * np.sqrt(3) * 1.1
    x0 = 0.5
    y0 = 0.5
    for row in range(5):
        for col in range(5):
            x = x0 + col*dx + (row%2)*dx/2
            y = y0 + row*dy
            if 0.1 <= x <= 0.9 and 0.1 <= y <= 0.9:
                pts.append([x, y, r])
            if len(pts) == N: break
        if len(pts) == N: break
    while len(pts) < N:
        pts.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9), r])
    v = np.zeros(1 + 3*N)
    v[0] = W
    v[1:] = np.array(pts[:N]).flatten()
    return v

def _fallback_grid():
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])

