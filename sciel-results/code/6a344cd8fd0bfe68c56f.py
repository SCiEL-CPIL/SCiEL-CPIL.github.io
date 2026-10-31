import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 115.0  # Safety margin before 120s kill

def _vectorized_objective(vars_):
    # vars_ is shape (3*N,)
    # x = vars_[0::3], y = vars_[1::3], r = vars_[2::3]
    r = vars_[2::3]
    return -np.sum(r)

def _vectorized_cons_inside(vars_, w, h):
    x = vars_[0::3]
    y = vars_[1::3]
    r = vars_[2::3]
    # Constraints: x-r >= 0, w-x-r >= 0, y-r >= 0, h-y-r >= 0
    return np.concatenate([
        x - r,
        w - x - r,
        y - r,
        h - y - r
    ])

def _vectorized_cons_non_overlap(vars_):
    x = vars_[0::3]
    y = vars_[1::3]
    r = vars_[2::3]
    # Compute all pairwise distances
    # i < j
    n = len(x)
    cons = np.zeros((n*(n-1)//2,))
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            cons[idx] = dist - r[i] - r[j]
            idx += 1
    return cons

def _vectorized_cons_positive_r(vars_):
    r = vars_[2::3]
    return r

def _is_valid(x, y, r, w, h, tol=1e-6):
    if np.any(r < -tol):
        return False
    if np.any(x < r - tol) or np.any(x > w - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > h - r + tol):
        return False
    # Check overlaps
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _polish_result(x0, w, h, start_time, deadline):
    """
    Refine the solution with tighter tolerances and more iterations.
    """
    def objective(vars_):
        return _vectorized_objective(vars_)
    
    def cons_inside(vars_):
        return _vectorized_cons_inside(vars_, w, h)
    
    def cons_non_overlap(vars_):
        return _vectorized_cons_non_overlap(vars_)
    
    def cons_positive_r(vars_):
        return _vectorized_cons_positive_r(vars_)
    
    constraints = [
        {'type': 'ineq', 'fun': cons_inside},
        {'type': 'ineq', 'fun': cons_non_overlap},
        {'type': 'ineq', 'fun': cons_positive_r}
    ]
    
    # Try to polish with SLSQP, tighter ftol
    try:
        result = minimize(
            objective, x0, method='SLSQP',
            constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-10}
        )
        
        if result.success or result.fun < -np.sum(x0[2::3]): # If it improved
            x_res = result.x[0::3]
            y_res = result.x[1::3]
            r_res = result.x[2::3]
            
            # Clamp small negative radii to 0
            r_res[r_res < 0] = 0
            
            # Validate
            if _is_valid(x_res, y_res, r_res, w, h):
                return np.column_stack((x_res, y_res, r_res))
            else:
                # If polish made it invalid, try to repair or return original
                # Simple repair: scale down radii slightly if overlapping
                # For now, return original if polish fails validity
                return None
        else:
            return None
    except Exception:
        return None

def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    
    np.random.seed(42)
    
    # Aspect ratios
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.5]
    
    for ar in aspect_ratios:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        w = 2.0 / (1.0 + ar)
        h = 2.0 - w
        if h <= 0.1:
            continue
        
        max_restarts = 4
        
        for restart in range(max_restarts):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Initialization
            cols = int(np.sqrt(N * w / h))
            if cols < 2: cols = 2
            rows = int(np.ceil(N / cols))
            
            while cols * rows < N:
                if cols <= rows:
                    cols += 1
                else:
                    rows += 1
            
            positions = np.zeros((N, 2))
            radii = np.zeros(N)
            
            idx = 0
            for r_idx in range(rows):
                for c_idx in range(cols):
                    if idx >= N:
                        break
                    x = (c_idx + 0.5) * w / cols
                    y = (r_idx + 0.5) * h / rows
                    
                    # Jitter for diversity
                    if restart > 0:
                        jitter = 0.15 * min(w/cols, h/rows)
                        x += np.random.uniform(-jitter, jitter)
                        y += np.random.uniform(-jitter, jitter)
                    
                    x = np.clip(x, 0.01, w - 0.01)
                    y = np.clip(y, 0.01, h - 0.01)
                    
                    positions[idx] = [x, y]
                    idx += 1
            
            # Initial radii
            min_dist = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((positions[i,0]-positions[j,0])**2 + (positions[i,1]-positions[j,1])**2)
                    min_dist = min(min_dist, d)
            for i in range(N):
                min_dist = min(min_dist, positions[i,0], w-positions[i,0], positions[i,1], h-positions[i,1])
            
            r0 = 0.4 * min_dist
            radii = np.full(N, r0)
            
            x0 = np.zeros(3*N)
            for i in range(N):
                x0[3*i] = positions[i, 0]
                x0[3*i+1] = positions[i, 1]
                x0[3*i+2] = radii[i]
            
            # Optimization
            def objective(vars_):
                return _vectorized_objective(vars_)
            
            def cons_inside(vars_):
                return _vectorized_cons_inside(vars_, w, h)
            
            def cons_non_overlap(vars_):
                return _vectorized_cons_non_overlap(vars_)
            
            def cons_positive_r(vars_):
                return _vectorized_cons_positive_r(vars_)
            
            constraints = [
                {'type': 'ineq', 'fun': cons_inside},
                {'type': 'ineq', 'fun': cons_non_overlap},
                {'type': 'ineq', 'fun': cons_positive_r}
            ]
            
            try:
                result = minimize(
                    objective, x0, method='SLSQP',
                    constraints=constraints,
                    options={'maxiter': 150, 'ftol': 1e-8}
                )
                
                x_res = result.x[0::3]
                y_res = result.x[1::3]
                r_res = result.x[2::3]
                
                # Check validity
                if _is_valid(x_res, y_res, r_res, w, h):
                    score = np.sum(r_res)
                    
                    # Polish phase
                    if time.time() - start_time < DEADLINE - 15:
                        polished = _polish_result(result.x, w, h, start_time, DEADLINE)
                        if polished is not None:
                            x_res, y_res, r_res = polished[:,0], polished[:,1], polished[:,2]
                            score = np.sum(r_res)
                    
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack((x_res, y_res, r_res)).copy()
                        
            except Exception:
                continue

    # Fallback
    if best_result is None or best_score < 0:
        w = 1.0
        h = 1.0
        r = 0.095
        positions = []
        radii = []
        count = 0
        for i in range(5):
            for j in range(5):
                if count >= N:
                    break
                x = (i + 0.5) * w / 5
                y = (j + 0.5) * h / 5
                positions.append([x, y])
                radii.append(r)
                count += 1
        best_result = np.column_stack((np.array(positions), np.array(radii)))
        best_score = np.sum(radii)
    
    return best_result
