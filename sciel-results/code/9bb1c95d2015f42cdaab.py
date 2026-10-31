"""Circle packing in perimeter-4 rectangle - SLSQP with free aspect ratio."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 140.0  # seconds, safety margin before 150s kill

def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - t0)
    
    def make_constraints(w, h):
        """Create constraint functions for given rectangle dimensions."""
        def boundary(x):
            # x = [x0,y0,r0, x1,y1,r1, ..., x20,y20,r20]
            # r_i <= x_i
            c1 = x[2::3] - x[0::3]  # r_i - x_i <= 0
            c2 = x[0::3] + x[2::3] - w  # x_i + r_i - w <= 0
            c3 = x[2::3] - x[1::3]  # r_i - y_i <= 0
            c4 = x[1::3] + x[2::3] - h  # y_i + r_i - h <= 0
            return np.concatenate([c1, c2, c3, c4])
        
        def non_overlap(x):
            xs = x[0::3]
            ys = x[1::3]
            rs = x[2::3]
            cons = []
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist_sq = dx*dx + dy*dy
                    min_dist = rs[i] + rs[j]
                    cons.append(dist_sq - min_dist*min_dist)
            return np.array(cons)
        
        return boundary, non_overlap
    
    def optimize_from_initial(x0, w0, h0):
        """Run SLSQP from initial state, return best feasible (positions, radii, w, h, value)."""
        # Variables: [x0, y0, r0, x1, y1, r1, ..., x20, y20, r20, w, h]
        def objective(v):
            rs = v[2::3][:N]
            return -np.sum(rs)
        
        # Constraints (inequality: >= 0 form for scipy: c >= 0)
        # We'll use scipy's form: fun(x) >= 0
        def constraints_func(v):
            xs = v[0::3][:N]
            ys = v[1::3][:N]
            rs = v[2::3][:N]
            w = v[3*N]
            h = v[3*N + 1]
            
            # Boundary constraints (>= 0): x_i - r_i >= 0, w - x_i - r_i >= 0, etc.
            c_bound = np.concatenate([
                xs - rs,          # x_i >= r_i
                w - xs - rs,      # w >= x_i + r_i
                ys - rs,          # y_i >= r_i
                h - ys - rs,      # h >= y_i + r_i
            ])
            
            # Non-overlap: dist_sq - (r_i+r_j)^2 >= 0
            cons_no = []
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist_sq = dx*dx + dy*dy
                    min_dist = rs[i] + rs[j]
                    cons_no.append(dist_sq - min_dist*min_dist)
            c_no = np.array(cons_no)
            
            # Perimeter: 4 - 2(w+h) >= 0
            c_perim = np.array([4.0 - 2.0*(w + h)])
            
            # Min radius
            c_rmin = rs - 0.005
            
            return np.concatenate([c_bound, c_no, c_perim, c_rmin])
        
        # Bounds for variables
        r_min = 0.005
        r_max = 1.0
        bounds = []
        for i in range(N):
            bounds.append((r_min, w0))      # x_i
            bounds.append((r_min, h0))      # y_i
            bounds.append((r_min, r_max))   # r_i
        bounds.append((0.5, 2.0))  # w
        bounds.append((0.5, 2.0))  # h
        
        # Initial feasibility check and adjustment
        x_init = np.array(x0)
        # Scale radii down slightly to ensure feasibility
        for i in range(N):
            idx_r = 2*i + 2
            idx_x = 2*i
            idx_y = 2*i + 1
            x_init[idx_r] = min(x_init[idx_r], 0.95 * min(x_init[idx_x], w0 - x_init[idx_x], x_init[idx_y], h0 - x_init[idx_y]))
            x_init[idx_r] = max(x_init[idx_r], r_min)
        
        result = minimize(
            objective, x_init, method='SLSQP',
            bounds=bounds,
            constraints=[{'type': 'ineq', 'fun': constraints_func}],
            options={'maxiter': 300, 'ftol': 1e-12, 'disp': False}
        )
        
        if result.success and result.fun < 0:
            v = result.x
            value = -result.fun
            return v, value
        return None, 0.0
    
    best_value = 0.0
    best_result = None
    
    # Restart 1: Grid-based initial with square aspect
    w0, h0 = 1.0, 1.0
    side = int(np.ceil(np.sqrt(N)))
    x0 = []
    for i in range(N):
        col = i % side
        row = i // side
        cx = (col + 0.5) / side * w0
        cy = (row + 0.5) / side * h0
        r = 0.45 / side
        x0.extend([cx, cy, r])
    x0.extend([w0, h0])
    
    v, val = optimize_from_initial(x0, w0, h0)
    if v is not None and val > best_value:
        best_value = val
        best_result = v
    print(f"Restart 1 (square grid): value={best_value:.6f}, time={time.time()-t0:.1f}s")
    
    if time_left() < 10:
        pass  # Not enough time for more restarts
    
    # Restart 2: Different aspect ratio - wider
    w0, h0 = 1.3, 0.7
    x0 = []
    cols = 7
    rows = 3
    for i in range(N):
        col = i % cols
        row = i // cols
        cx = (col + 0.5) / cols * w0
        cy = (row + 0.5) / rows * h0
        r = min(w0, h0) * 0.12
        x0.extend([cx, cy, r])
    x0.extend([w0, h0])
    
    if time_left() > 10:
        v, val = optimize_from_initial(x0, w0, h0)
        if v is not None and val > best_value:
            best_value = val
            best_result = v
        print(f"Restart 2 (wide): value={best_value:.6f}, time={time.time()-t0:.1f}s")
    
    # Restart 3: Taller aspect
    w0, h0 = 0.7, 1.3
    x0 = []
    cols = 3
    rows = 7
    for i in range(N):
        col = i % cols
        row = i // cols
        cx = (col + 0.5) / cols * w0
        cy = (row + 0.5) / rows * h0
        r = min(w0, h0) * 0.12
        x0.extend([cx, cy, r])
    x0.extend([w0, h0])
    
    if time_left() > 10:
        v, val = optimize_from_initial(x0, w0, h0)
        if v is not None and val > best_value:
            best_value = val
            best_result = v
        print(f"Restart 3 (tall): value={best_value:.6f}, time={time.time()-t0:.1f}s")
    
    # Restart 4: Random initialization with moderate aspect
    if time_left() > 15:
        np.random.seed(42)
        w0, h0 = 1.1, 0.9
        x0 = []
        for i in range(N):
            cx = np.random.uniform(0.1, w0 - 0.1)
            cy = np.random.uniform(0.1, h0 - 0.1)
            r = np.random.uniform(0.05, 0.12)
            x0.extend([cx, cy, r])
        x0.extend([w0, h0])
        
        v, val = optimize_from_initial(x0, w0, h0)
        if v is not None and val > best_value:
            best_value = val
            best_result = v
        print(f"Restart 4 (random): value={best_value:.6f}, time={time.time()-t0:.1f}s")
    
    # Restart 5: Hexagonal-like packing
    if time_left() > 15:
        w0, h0 = 1.2, 0.8
        x0 = []
        # Hexagonal arrangement
        dx = 0.17
        dy = dx * np.sqrt(3) / 2
        row_idx = 0
        for i in range(N):
            col = i % 7
            row = i // 7
            cx = (col + 0.5) * dx + (0.5 * dx if row % 2 == 1 else 0)
            cy = (row + 0.5) * dy
            cx = min(cx, w0 - 0.05)
            cy = min(cy, h0 - 0.05)
            r = 0.08
            x0.extend([cx, cy, r])
        x0.extend([w0, h0])
        
        v, val = optimize_from_initial(x0, w0, h0)
        if v is not None and val > best_value:
            best_value = val
            best_result = v
        print(f"Restart 5 (hex): value={best_value:.6f}, time={time.time()-t0:.1f}s")
    
    # Final polish: if we have a good result, try to refine it
    if best_result is not None and time_left() > 20:
        v = best_result.copy()
        w_cur = v[3*N]
        h_cur = v[3*N + 1]
        
        def objective(v):
            rs = v[2::3][:N]
            return -np.sum(rs)
        
        def constraints_func(v):
            xs = v[0::3][:N]
            ys = v[1::3][:N]
            rs = v[2::3][:N]
            w = v[3*N]
            h = v[3*N + 1]
            
            c_bound = np.concatenate([
                xs - rs,
                w - xs - rs,
                ys - rs,
                h - ys - rs,
            ])
            
            cons_no = []
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist_sq = dx*dx + dy*dy
                    min_dist = rs[i] + rs[j]
                    cons_no.append(dist_sq - min_dist*min_dist)
            c_no = np.array(cons_no)
            
            c_perim = np.array([4.0 - 2.0*(w + h)])
            c_rmin = rs - 0.001
            
            return np.concatenate([c_bound, c_no, c_perim, c_rmin])
        
        r_min = 0.001
        r_max = 1.0
        bounds = []
        for i in range(N):
            bounds.append((r_min, w_cur))
            bounds.append((r_min, h_cur))
            bounds.append((r_min, r_max))
        bounds.append((0.5, 2.0))
        bounds.append((0.5, 2.0))
        
        result = minimize(
            objective, v, method='SLSQP',
            bounds=bounds,
            constraints=[{'type': 'ineq', 'fun': constraints_func}],
            options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
        )
        
        if result.success and result.fun < 0 and -result.fun > best_value:
            best_value = -result.fun
            best_result = result.x
        print(f"Polish: value={best_value:.6f}, time={time.time()-t0:.1f}s")
    
    # Extract final result
    if best_result is not None:
        v = best_result
        xs = v[0::3][:N]
        ys = v[1::3][:N]
        rs = v[2::3][:N]
        w = v[3*N]
        h = v[3*N + 1]
        
        # Verify feasibility and adjust if needed
        # Scale down radii slightly if any constraint is violated
        for iteration in range(50):
            feasible = True
            for i in range(N):
                if xs[i] < rs[i] - 1e-10 or xs[i] + rs[i] > w + 1e-10:
                    feasible = False
                    break
                if ys[i] < rs[i] - 1e-10 or ys[i] + rs[i] > h + 1e-10:
                    feasible = False
                    break
            if not feasible:
                break
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < rs[i] + rs[j] - 1e-10:
                        feasible = False
                        break
                if not feasible:
                    break
            if feasible:
                break
            # Slightly shrink radii
            rs = rs * 0.999
        
        # Final perimeter check
        perim = 2*(w + h)
        if perim > 4.0 + 1e-9:
            scale = 4.0 / perim
            w *= scale
            h *= scale
            xs = xs * scale
            ys = ys * scale
            rs = rs * scale
        
        result_arr = np.column_stack([xs, ys, rs])
        return result_arr
    else:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
