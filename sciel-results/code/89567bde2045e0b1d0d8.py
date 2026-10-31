import numpy as np
from scipy.optimize import minimize
import time

N = 21
MAX_TIME = 140.0

def construct():
    start_time = time.time()
    best_val = -np.inf
    best_x = None

    # Variables: [w, x_1, y_1, r_1, ..., x_N, y_N, r_N]
    # h = 2 - w
    
    def func(x):
        w = x[0]
        h = 2.0 - w
        if w < 0.01 or h < 0.01:
            return 1e10
        xs = x[1::3]
        ys = x[2::3]
        rs = x[3::3]
        return -np.sum(rs)
        
    def make_constraints():
        cons = []
        # Boundaries
        for i in range(N):
            idx_x = 1 + 3*i
            idx_y = 2 + 3*i
            idx_r = 3 + 3*i
            # x >= r
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[idx_x] - x[idx_r]})
            # x <= w - r
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[0] - x[idx_x] - x[idx_r]})
            # y >= r
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[idx_y] - x[idx_r]})
            # y <= h - r
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: (2.0 - x[0]) - x[idx_y] - x[idx_r]})
            # r > 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[idx_r] - 1e-6})
            
        # Distances with small epsilon gap
        eps = 1e-5
        for i in range(N):
            for j in range(i + 1, N):
                idx_xi = 1 + 3*i
                idx_yi = 2 + 3*i
                idx_ri = 3 + 3*i
                idx_xj = 1 + 3*j
                idx_yj = 2 + 3*j
                idx_rj = 3 + 3*j
                cons.append({
                    'type': 'ineq', 
                    'fun': lambda x, i=i, j=j: (x[idx_xi] - x[idx_xj])**2 + (x[idx_yi] - x[idx_yj])**2 - (x[idx_ri] + x[idx_rj] + eps)**2
                })
        return cons

    cons = make_constraints()
    
    # Bounds
    bounds = [(0.1, 1.9)]
    for _ in range(N):
        bounds.append((0, 2.0)) # x
        bounds.append((0, 2.0)) # y
        bounds.append((0, 1.0)) # r

    def run_optimization(x0):
        try:
            res = minimize(func, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 100, 'ftol': 1e-8})
            if res.success:
                return res.x, -res.fun
            else:
                # Even if not fully converged, if it's better than current best, we might take it?
                # No, better to be safe.
                return None, -np.inf
        except Exception:
            return None, -np.inf

    # Initial guesses
    starts = []
    
    # 1. Grid 3x7 in 1x1
    w0 = 1.0
    h0 = 1.0
    x0 = np.zeros(1 + 3*N)
    x0[0] = w0
    rows = 3
    cols = 7
    dx = w0 / cols
    dy = h0 / rows
    r0 = min(dx, dy) / 2 * 0.95
    
    idx = 1
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= N: break
            x0[idx] = (j + 0.5) * dx
            x0[idx+1] = (i + 0.5) * dy
            x0[idx+2] = r0
            idx += 3
            count += 1
    starts.append(x0)
    
    # 2. Grid 4x5 + 1 in 1x1 (approx)
    # 4 rows, 5 cols = 20. Add 1 in center?
    w0 = 1.0
    h0 = 1.0
    x0 = np.zeros(1 + 3*N)
    x0[0] = w0
    rows = 4
    cols = 5
    dx = w0 / cols
    dy = h0 / rows
    r0 = min(dx, dy) / 2 * 0.95
    
    idx = 1
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= N: break
            x0[idx] = (j + 0.5) * dx
            x0[idx+1] = (i + 0.5) * dy
            x0[idx+2] = r0
            idx += 3
            count += 1
    # The 21st circle is at 0,0,0 which is invalid. 
    # Let's place it at center of the rectangle, maybe overlapping initially, optimizer will fix.
    x0[idx] = 0.5
    x0[idx+1] = 0.5
    x0[idx+2] = r0
    starts.append(x0)

    # 3. Random starts
    np.random.seed(42)
    for _ in range(15):
        x_rand = np.zeros(1 + 3*N)
        x_rand[0] = np.random.uniform(0.5, 1.5)
        for i in range(N):
            x_rand[1 + 3*i] = np.random.uniform(0.1, 1.9)
            x_rand[2 + 3*i] = np.random.uniform(0.1, 1.9)
            x_rand[3 + 3*i] = np.random.uniform(0.05, 0.15)
        starts.append(x_rand)

    for x_init in starts:
        if time.time() - start_time > MAX_TIME - 15:
            break
        res_x, val = run_optimization(x_init)
        if val > best_val:
            best_val = val
            best_x = res_x

    if best_x is None:
        # Fallback
        w = 1.0
        h = 1.0
        rows = 3
        cols = 7
        dx = w / cols
        dy = h / rows
        r = min(dx, dy) / 2 - 1e-4
        result = []
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= N: break
                result.append([(j + 0.5) * dx, (i + 0.5) * dy, r])
                idx += 1
        return np.array(result)
        
    w = best_x[0]
    h = 2.0 - w
    xs = best_x[1::3]
    ys = best_x[2::3]
    rs = best_x[3::3]
    
    result = []
    for i in range(N):
        x = np.clip(xs[i], rs[i], w - rs[i])
        y = np.clip(ys[i], rs[i], h - rs[i])
        r = rs[i]
        # Shrink slightly to ensure disjoint interiors if they are touching
        r_final = r * 0.999 
        result.append([x, y, r_final])
        
    return np.array(result)
