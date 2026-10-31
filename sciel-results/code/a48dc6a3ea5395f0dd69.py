import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 115.0  # Safety margin before 120s kill

def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    
    np.random.seed(42)
    
    # Aspect ratios to try. We will use a unified optimization that also
    # optimizes the aspect ratio jointly with positions and radii.
    # But first, we do a coarse sweep to find a good region, then refine.
    
    # Phase 1: Coarse sweep with fixed aspect ratios (fast, gives good initial guesses)
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8]
    
    for ar in aspect_ratios:
        if time.time() - start_time > DEADLINE - 20:
            break
            
        w = 2.0 / (1.0 + ar)
        h = 2.0 - w
        if h <= 0.1 or w <= 0.1:
            continue
            
        # Try 2 restarts per aspect ratio
        for restart in range(2):
            if time.time() - start_time > DEADLINE - 10:
                break
                
            # Generate initial positions using a grid-like layout
            cols = int(np.sqrt(N * w / h))
            if cols < 2: cols = 2
            rows = int(np.ceil(N / cols))
            while cols * rows < N:
                if cols <= rows:
                    cols += 1
                else:
                    rows += 1
            
            positions = np.zeros((N, 2))
            idx = 0
            for r_idx in range(rows):
                for c_idx in range(cols):
                    if idx >= N:
                        break
                    x = (c_idx + 0.5) * w / cols
                    y = (r_idx + 0.5) * h / rows
                    if restart > 0:
                        jitter = 0.1 * min(w/cols, h/rows)
                        x += np.random.uniform(-jitter, jitter)
                        y += np.random.uniform(-jitter, jitter)
                    x = np.clip(x, 0.01, w - 0.01)
                    y = np.clip(y, 0.01, h - 0.01)
                    positions[idx] = [x, y]
                    idx += 1
            
            # Estimate initial radius
            min_dist = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((positions[i,0]-positions[j,0])**2 + (positions[i,1]-positions[j,1])**2)
                    min_dist = min(min_dist, d)
            for i in range(N):
                min_dist = min(min_dist, positions[i,0], w-positions[i,0], positions[i,1], h-positions[i,1])
            r0 = 0.4 * min_dist
            radii = np.full(N, r0)
            
            # Run optimization with fixed aspect ratio
            x0 = np.zeros(3*N)
            for i in range(N):
                x0[3*i] = positions[i, 0]
                x0[3*i+1] = positions[i, 1]
                x0[3*i+2] = radii[i]
            
            def objective(vars_):
                return -np.sum(vars_[2::3])
            
            def cons_inside(vars_):
                x = vars_[0::3]
                y = vars_[1::3]
                r = vars_[2::3]
                return np.concatenate([
                    x - r,
                    w - x - r,
                    y - r,
                    h - y - r
                ])
            
            def cons_non_overlap(vars_):
                x = vars_[0::3]
                y = vars_[1::3]
                r = vars_[2::3]
                cons = []
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        cons.append(dist - r[i] - r[j])
                return np.array(cons)
            
            def cons_positive_r(vars_):
                return vars_[2::3]
            
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
                
                # Validate
                valid = True
                if np.any(r_res < 0):
                    valid = False
                else:
                    if np.any(x_res < r_res - 1e-6) or np.any(x_res > w - r_res + 1e-6):
                        valid = False
                    if np.any(y_res < r_res - 1e-6) or np.any(y_res > h - r_res + 1e-6):
                        valid = False
                    else:
                        for i in range(N):
                            for j in range(i+1, N):
                                dist = np.sqrt((x_res[i]-x_res[j])**2 + (y_res[i]-y_res[j])**2)
                                if dist < r_res[i] + r_res[j] - 1e-6:
                                    valid = False
                                    break
                            if not valid:
                                break
                
                if valid:
                    score = np.sum(r_res)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack((x_res, y_res, r_res)).copy()
                        
            except Exception:
                continue
    
    # Phase 2: Unified optimization where aspect ratio is also a variable
    # Variables: [x1, y1, r1, ..., xN, yN, rN, ar] where ar = h/w
    # Constraints:
    #   w = 2/(1+ar), h = 2 - w
    #   x_i - r_i >= 0
    #   w - x_i - r_i >= 0
    #   y_i - r_i >= 0
    #   h - y_i - r_i >= 0
    #   dist(i,j) - r_i - r_j >= 0
    #   r_i >= 0
    #   ar in [0.5, 2.0] (bounds)
    
    if best_result is not None:
        x_init = best_result[:, 0]
        y_init = best_result[:, 1]
        r_init = best_result[:, 2]
        # Infer aspect ratio from best result
        # We need to know w and h. Let's estimate from the bounding box
        x_min, x_max = np.min(x_init), np.max(x_init)
        y_min, y_max = np.min(y_init), np.max(y_init)
        # This is tricky because circles don't necessarily touch all walls.
        # Let's just use ar=1.0 as a reasonable default for the unified phase
        ar_init = 1.0
    else:
        x_init = np.random.uniform(0.1, 0.9, N)
        y_init = np.random.uniform(0.1, 0.9, N)
        r_init = np.full(N, 0.05)
        ar_init = 1.0
    
    # Unified optimization
    if time.time() - start_time < DEADLINE - 5:
        # Variables: 3*N + 1 (for ar)
        x0_unified = np.zeros(3*N + 1)
        for i in range(N):
            x0_unified[3*i] = x_init[i]
            x0_unified[3*i+1] = y_init[i]
            x0_unified[3*i+2] = r_init[i]
        x0_unified[3*N] = ar_init
        
        def objective_unified(vars_):
            return -np.sum(vars_[2::3][:N])  # sum of radii
        
        def get_wh(ar):
            ar = np.clip(ar, 0.3, 3.0)  # Keep aspect ratio reasonable
            w = 2.0 / (1.0 + ar)
            h = 2.0 - w
            return w, h
        
        def cons_unified(vars_):
            x = vars_[0::3][:N]
            y = vars_[1::3][:N]
            r = vars_[2::3][:N]
            ar = vars_[3*N]
            w, h = get_wh(ar)
            
            cons = []
            # Inside rectangle
            cons.extend(list(x - r))
            cons.extend(list(w - x - r))
            cons.extend(list(y - r))
            cons.extend(list(h - y - r))
            # Non-overlap
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    cons.append(dist - r[i] - r[j])
            # Positive radii
            cons.extend(list(r))
            # Aspect ratio bounds (as inequalities)
            cons.append(ar - 0.3)       # ar >= 0.3
            cons.append(3.0 - ar)       # ar <= 3.0
            
            return np.array(cons)
        
        constraints_unified = [
            {'type': 'ineq', 'fun': cons_unified}
        ]
        
        try:
            result_unified = minimize(
                objective_unified, x0_unified, method='SLSQP',
                constraints=constraints_unified,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            x_res = result_unified.x[0::3][:N]
            y_res = result_unified.x[1::3][:N]
            r_res = result_unified.x[2::3][:N]
            ar_res = result_unified.x[3*N]
            w_res, h_res = get_wh(ar_res)
            
            # Validate
            valid = True
            if np.any(r_res < 0):
                valid = False
            else:
                if np.any(x_res < r_res - 1e-6) or np.any(x_res > w_res - r_res + 1e-6):
                    valid = False
                if np.any(y_res < r_res - 1e-6) or np.any(y_res > h_res - r_res + 1e-6):
                    valid = False
                else:
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt((x_res[i]-x_res[j])**2 + (y_res[i]-y_res[j])**2)
                            if dist < r_res[i] + r_res[j] - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
            
            if valid:
                score = np.sum(r_res)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack((x_res, y_res, r_res)).copy()
                    
        except Exception:
            pass
    
    # Fallback: A simple valid grid packing
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
