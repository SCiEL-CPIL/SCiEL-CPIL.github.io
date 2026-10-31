import numpy as np

N = 26

import numpy as np
from scipy.optimize import minimize

N = 26

def _make_layouts():
    layouts = []
    
    # 1. Staggered Grid (Hexagonal packing approximation)
    # 5 rows, 6 cols = 30, take 26
    x, y, r = [], [], []
    for i in range(5):
        for j in range(6):
            if len(x) >= N: break
            x_base = (j + 0.5) / 6.0
            y_base = (i + 0.5) / 5.0
            if i % 2 == 1:
                x_base += 0.5 / 6.0
            x_base = np.clip(x_base, 0.05, 0.95)
            y_base = np.clip(y_base, 0.05, 0.95)
            x.append(x_base)
            y.append(y_base)
            r.append(0.08)
    layouts.append((np.array(x[:N]), np.array(y[:N]), np.array(r[:N])))
    
    # 2. Square Grid
    side = 6 # 36 cells
    x, y, r = [], [], []
    for i in range(side):
        for j in range(side):
            if len(x) >= N: break
            x.append((j + 0.5) / side)
            y.append((i + 0.5) / side)
            r.append(0.08)
    layouts.append((np.array(x[:N]), np.array(y[:N]), np.array(r[:N])))
    
    # 3. Random
    rng = np.random.default_rng(42)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = np.full(N, 0.03)
    layouts.append((x, y, r))
    
    return layouts

def _optimize(x0, y0, r0):
    z0 = np.concatenate([x0, y0, r0])
    
    def objective(z):
        r = z[2::3]
        return -np.sum(r)
        
    def constraint_non_overlap(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                req_sq = (r[i] + r[j])**2
                cons.append(dist_sq - req_sq)
        return np.array(cons)

    def constraint_walls(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        cons = []
        for i in range(N):
            cons.append(x[i] - r[i])
            cons.append(1 - x[i] - r[i])
            cons.append(y[i] - r[i])
            cons.append(1 - y[i] - r[i])
        return np.array(cons)

    constraints = [
        {'type': 'ineq', 'fun': constraint_non_overlap},
        {'type': 'ineq', 'fun': constraint_walls}
    ]
    
    bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
    
    try:
        res = minimize(objective, z0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 300, 'ftol': 1e-8})
        
        if res.fun < -1e10: # res.fun is -sum(r), so if sum(r) > 1e10, it's invalid? No, max sum is ~2.6
             # Actually, if it fails, res.fun might be nan or inf
             pass
             
        x = res.x[0::3]
        y = res.x[1::3]
        r = res.x[2::3]
        
        # Check validity
        valid = True
        for i in range(N):
            if x[i] < -1e-6 or x[i] > 1 + 1e-6 or y[i] < -1e-6 or y[i] > 1 + 1e-6 or r[i] < -1e-6:
                valid = False
                break
        
        if valid:
            dx = x[None, :] - x[:, None]
            dy = y[None, :] - y[:, None]
            dist = np.sqrt(dx*dx + dy*dy)
            req = r[None, :] + r[:, None]
            if np.any(dist < req - 1e-6):
                valid = False
                
        if valid:
            return np.column_stack((x, y, r))
            
    except Exception:
        pass
        
    return None

def construct():
    layouts = _make_layouts()
    best_result = None
    best_score = -1e10
    
    for x0, y0, r0 in layouts:
        result = _optimize(x0, y0, r0)
        if result is not None:
            score = np.sum(result[:, 2])
            if score > best_score:
                best_score = score
                best_result = result
                
    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
