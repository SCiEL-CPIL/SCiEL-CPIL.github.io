import numpy as np
import time
from scipy.optimize import minimize

N = 21
_START_TIME = time.time()
_TIME_LIMIT = 110.0

def _smooth_hinge(x):
    """Smooth approximation of max(0, x)"""
    return 0.5 * (x + np.sqrt(x**2 + 1e-8))

def _objective(v):
    # v = [w, h, x1..xN, y1..yN, r1..rN]
    w = v[0]
    h = v[1]
    xs = v[2:2+N]
    ys = v[2+N:2+2*N]
    rs = v[2+2*N:2+3*N]
    
    # Perimeter penalty
    perim_viol = _smooth_hinge(2 * (w + h) - 4)
    
    # Boundary penalties
    # x in [r, w-r], y in [r, h-r]
    # Violation if x < r or x > w-r
    pen_bound = 0.0
    for i in range(N):
        pen_bound += _smooth_hinge(rs[i] - xs[i])
        pen_bound += _smooth_hinge(xs[i] + rs[i] - w)
        pen_bound += _smooth_hinge(rs[i] - ys[i])
        pen_bound += _smooth_hinge(ys[i] + rs[i] - h)
        
    # Overlap penalties
    pen_overlap = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            pen_overlap += _smooth_hinge(rs[i] + rs[j] - dist)
            
    # Objective: Max sum(r) -> Min -sum(r) + Penalty
    val = -np.sum(rs)
    val += 1000.0 * (perim_viol + pen_bound + pen_overlap)
    
    return val

def construct():
    global _START_TIME
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Return a simple valid grid if out of time
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    best_val = -np.inf
    best_sol = None
    
    # Bounds
    # w, h in [0.1, 1.9] (since w+h <= 2)
    # x, y in [0, 2]
    # r in [0, 1]
    bounds = [(0.1, 1.9), (0.1, 1.9)]
    for _ in range(N):
        bounds.append((0.0, 2.0)) # x
    for _ in range(N):
        bounds.append((0.0, 2.0)) # y
    for _ in range(N):
        bounds.append((0.0, 1.0)) # r
        
    # Initializations
    # 1. Grid in 1x1
    # 2. Random
    
    for restart in range(50):
        if time.time() - _START_TIME > _TIME_LIMIT:
            break
            
        # Initialize
        if restart % 2 == 0:
            # Grid init
            side = int(np.ceil(np.sqrt(N)))
            xs = np.array([(i + 0.5) / side for i in range(side) for j in range(side)][:N])
            ys = np.array([(j + 0.5) / side for i in range(side) for j in range(side)][:N])
            rs = np.full(N, 0.5 / side * 0.9)
            w = 1.0
            h = 1.0
        else:
            # Random init
            w = np.random.uniform(0.8, 1.2)
            h = np.random.uniform(0.8, 1.2)
            # Ensure w+h <= 2 roughly
            if w + h > 2:
                scale = 2.0 / (w + h)
                w *= scale
                h *= scale
                
            xs = np.random.uniform(0.1, w-0.1, N)
            ys = np.random.uniform(0.1, h-0.1, N)
            rs = np.random.uniform(0.05, 0.2, N)
            
        v0 = np.concatenate(([w, h], xs, ys, rs))
        
        # Optimize
        try:
            res = minimize(_objective, v0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-6})
        except Exception:
            continue
            
        if res.success or res.fun < best_val:
            v_opt = res.x
            # Check validity
            w_opt = v_opt[0]
            h_opt = v_opt[1]
            xs_opt = v_opt[2:2+N]
            ys_opt = v_opt[2+N:2+2*N]
            rs_opt = v_opt[2+2*N:2+3*N]
            
            # Strict check
            if 2*(w_opt + h_opt) > 4 + 1e-5: continue
            valid = True
            for i in range(N):
                if xs_opt[i] < rs_opt[i] - 1e-5 or xs_opt[i] + rs_opt[i] > w_opt + 1e-5:
                    valid = False; break
                if ys_opt[i] < rs_opt[i] - 1e-5 or ys_opt[i] + rs_opt[i] > h_opt + 1e-5:
                    valid = False; break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dx = xs_opt[i] - xs_opt[j]
                        dy = ys_opt[i] - ys_opt[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < rs_opt[i] + rs_opt[j] - 1e-5:
                            valid = False; break
                    if not valid: break
                    
            if valid:
                score = np.sum(rs_opt)
                if score > best_val:
                    best_val = score
                    best_sol = np.column_stack((xs_opt, ys_opt, rs_opt))
    
    if best_sol is not None:
        return best_sol
    else:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
