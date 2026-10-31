"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import scipy.optimize as opt
import time

N = 21

# EVOLVE-BLOCK-START

def _seed_result():
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])

def _is_valid(v, w, h):
    """Check if the configuration is valid (no overlaps, within bounds)."""
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    
    # Check bounds
    if np.any(x < r - 1e-6) or np.any(x > w - r + 1e-6):
        return False
    if np.any(y < r - 1e-6) or np.any(y > h - r + 1e-6):
        return False
        
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (x[i] - x[j])**2 + (y[i] - y[j])**2
            r_sum = r[i] + r[j]
            if d2 < r_sum**2 - 1e-6:
                return False
    return True

def _format_result(v):
    """Format vector to (N, 3) array."""
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    # Shift to ensure min x, min y are 0? 
    # The evaluator checks bounding box perimeter.
    # Shifting doesn't change perimeter.
    # However, if we shift, we might violate "disjoint interiors" if we shift into negative? 
    # No, shifting the whole set is fine.
    # Let's shift so min(x-r) = 0 and min(y-r) = 0.
    min_x = np.min(x - r)
    min_y = np.min(y - r)
    x = x - min_x
    y = y - min_y
    
    # Check perimeter
    max_x = np.max(x + r)
    max_y = np.max(y + r)
    w = max_x
    h = max_y
    perim = 2 * (w + h)
    
    # If perimeter > 4, we need to scale down?
    # The optimizer constrained x,y,r to be within a box of w0, h0 with w0+h0=2.
    # So the resulting bounding box should have perimeter <= 4.
    # Due to numerical errors, it might be slightly over.
    if perim > 4.0:
        scale = 4.0 / perim
        x *= scale
        y *= scale
        r *= scale
        
    return np.column_stack((x, y, r))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = -1.0
    best_v = None
    
    # Define objective and constraints for SLSQP
    # Variables: v = [x0, y0, r0, x1, y1, r1, ...]
    
    def objective(v):
        r = v[2::3]
        return -np.sum(r)
    
    def make_constraints(w, h):
        cons = []
        # Boundary constraints
        for i in range(N):
            xi_idx = i * 3
            yi_idx = i * 3 + 1
            ri_idx = i * 3 + 2
            # x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=xi_idx, r=ri_idx: v[i] - v[r]})
            # w - x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=xi_idx, r=ri_idx, w=w: w - v[i] - v[r]})
            # y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=yi_idx, r=ri_idx: v[i] - v[r]})
            # h - y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=yi_idx, r=ri_idx, h=h: h - v[i] - v[r]})
            
        # Overlap constraints
        for i in range(N):
            for j in range(i + 1, N):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i, j=j: 
                             (v[i*3] - v[j*3])**2 + (v[i*3+1] - v[j*3+1])**2 - (v[i*3+2] + v[j*3+2])**2})
                             
        # Radius positivity
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i*3+2] - 1e-4})
            
        return cons
    
    # Initial configurations
    configs = []
    
    # 1. Square 1x1, 5x5 grid
    w, h = 1.0, 1.0
    side = 5
    v = []
    for i in range(side):
        for j in range(side):
            x = (j + 0.5) * (w / side)
            y = (i + 0.5) * (h / side)
            r = 0.5 / side - 0.001
            v.extend([x, y, r])
    v = np.array(v[:3*N])
    configs.append((w, h, v))
    
    # 2. Rectangular 1.4x0.6, 3x7 grid
    w, h = 1.4, 0.6
    rows, cols = 3, 7
    v = []
    for i in range(rows):
        for j in range(cols):
            x = (j + 0.5) * (w / cols)
            y = (i + 0.5) * (h / rows)
            r = min(w/cols, h/rows) * 0.5 - 0.001
            v.extend([x, y, r])
    v = np.array(v[:3*N])
    configs.append((w, h, v))
    
    # 3. Hexagonal 1.44x0.56
    w, h = 1.44, 0.56
    # 3 rows of 7
    # Row 0: 7 circles
    # Row 1: 7 circles (shifted by r)
    # Row 2: 7 circles
    r_est = 0.09
    v = []
    for row in range(3):
        for col in range(7):
            x = col * 2 * r_est + r_est + (r_est if row % 2 == 1 else 0)
            y = row * np.sqrt(3) * r_est + r_est
            v.extend([x, y, r_est])
    v = np.array(v[:3*N])
    configs.append((w, h, v))
    
    # 4. Another aspect ratio 1.2x0.8
    w, h = 1.2, 0.8
    # 4 rows, 6 cols (24 spots, use 21)
    rows, cols = 4, 6
    v = []
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= N: break
            x = (j + 0.5) * (w / cols)
            y = (i + 0.5) * (h / rows)
            r = min(w/cols, h/rows) * 0.5 - 0.001
            v.extend([x, y, r])
            count += 1
    v = np.array(v[:3*N])
    configs.append((w, h, v))
    
    for w, h, init_v in configs:
        if time.time() > deadline:
            break
            
        try:
            res = opt.minimize(objective, init_v, method='SLSQP', 
                               constraints=make_constraints(w, h), 
                               options={'maxiter': 100, 'ftol': 1e-5})
            
            if res.success:
                v = res.x
                if _is_valid(v, w, h):
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        best_v = v
        except Exception:
            pass

    if best_v is None:
        return _seed_result()
        
    return _format_result(best_v)

# EVOLVE-BLOCK-END

