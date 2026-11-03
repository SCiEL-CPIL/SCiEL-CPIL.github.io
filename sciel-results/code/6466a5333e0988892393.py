import numpy as np
from scipy.optimize import linprog, minimize
import time

N = 21
DEADLINE = 120.0

def construct():
    start_time = time.time()
    best_score = 0.0
    best_params = None
    
    # Aspect ratios to try
    # Based on island finding, near-square is good, but we should check a range.
    # Perimeter = 2(w+h) = 4 => w+h=2.
    # w in [0.5, 1.5]
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 1.33, 0.75, 0.66, 1.5, 2.0, 0.5]
    
    # Sort by likelihood of success (near square first)
    aspect_ratios.sort(key=lambda w: abs(w - 1.0))

    for w in aspect_ratios:
        if time.time() - start_time > DEADLINE - 20:
            break
            
        h = 2.0 - w
        if h <= 0.1 or w <= 0.1:
            continue
            
        # Initialize a feasible grid
        # Try to fit N circles in w x h
        # Estimate cols/rows
        aspect = w / h
        if aspect > 1:
            cols = int(np.ceil(np.sqrt(N * aspect)))
        else:
            cols = int(np.ceil(np.sqrt(N / aspect)))
        cols = max(1, min(cols, N))
        rows = int(np.ceil(N / cols))
        
        cell_w = w / cols
        cell_h = h / rows
        
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        r0 = np.zeros(N)
        
        for i in range(N):
            c = i % cols
            r_idx = i // cols
            x0[i] = (c + 0.5) * cell_w
            y0[i] = (r_idx + 0.5) * cell_h
            r0[i] = 0.4 * min(cell_w, cell_h) # Start small to ensure feasibility
            
        # Ensure initial feasibility
        # If any overlap, reduce radii
        for _ in range(10):
            feasible = True
            for i in range(N):
                if r0[i] > x0[i] or r0[i] > w - x0[i] or r0[i] > y0[i] or r0[i] > h - y0[i]:
                    r0[i] = 0.9 * min(x0[i], w - x0[i], y0[i], h - y0[i])
                    feasible = False
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((x0[i]-x0[j])**2 + (y0[i]-y0[j])**2)
                    if r0[i] + r0[j] > dist:
                        # Reduce radii proportionally
                        scale = 0.9 * dist / (r0[i] + r0[j])
                        r0[i] *= scale
                        r0[j] *= scale
                        feasible = False
            if feasible:
                break

        # SCR Loop
        v = np.concatenate(([w], x0, y0, r0))
        # Indices
        idx_w = 0
        idx_x = slice(1, 1+N)
        idx_y = slice(1+N, 1+2*N)
        idx_r = slice(1+2*N, 1+3*N)
        
        # Objective: Max sum(r) => Min -sum(r)
        c = np.zeros(3*N + 1)
        c[idx_r] = -1.0
        
        # Bounds
        bounds = [(w, w)] # w is fixed in this sub-problem
        bounds += [(0, w)] * N
        bounds += [(0, h)] * N
        bounds += [(0, 1.0)] * N
        
        # Linear Constraints A_ub @ v <= b_ub
        # We will rebuild A_ub and b_ub in each iteration
        
        for iter in range(100):
            if time.time() - start_time > DEADLINE - 10:
                break
                
            # Build Constraints
            A_ub_rows = []
            b_ub_rows = []
            
            # 1. Box Constraints
            # r_i - x_i <= 0
            # x_i + r_i <= w
            # r_i - y_i <= 0
            # y_i + r_i <= h
            
            for i in range(N):
                # r_i - x_i <= 0
                row = np.zeros(3*N + 1)
                row[idx_x][i] = -1.0
                row[idx_r][i] = 1.0
                A_ub_rows.append(row)
                b_ub_rows.append(0.0)
                
                # x_i + r_i <= w
                row = np.zeros(3*N + 1)
                row[idx_x][i] = 1.0
                row[idx_r][i] = 1.0
                A_ub_rows.append(row)
                b_ub_rows.append(w)
                
                # r_i - y_i <= 0
                row = np.zeros(3*N + 1)
                row[idx_y][i] = -1.0
                row[idx_r][i] = 1.0
                A_ub_rows.append(row)
                b_ub_rows.append(0.0)
                
                # y_i + r_i <= h
                row = np.zeros(3*N + 1)
                row[idx_y][i] = 1.0
                row[idx_r][i] = 1.0
                A_ub_rows.append(row)
                b_ub_rows.append(h)
                
            # 2. Overlap Constraints (Linearized)
            # g_ij(v) >= 0
            # Linearize at v
            for i in range(N):
                for j in range(i+1, N):
                    xi, yi, ri = v[idx_x][i], v[idx_y][i], v[idx_r][i]
                    xj, yj, rj = v[idx_x][j], v[idx_y][j], v[idx_r][j]
                    
                    dx = xi - xj
                    dy = yi - yj
                    dr = ri + rj
                    
                    g = dx*dx + dy*dy - dr*dr
                    # Gradient of g w.r.t v
                    # dg/dxi = 2*dx
                    # dg/dxj = -2*dx
                    # dg/dyi = 2*dy
                    # dg/dyj = -2*dy
                    # dg/dri = -2*dr
                    # dg/drj = -2*dr
                    
                    # Constraint: g + grad * (v_new - v) >= 0
                    # grad * v_new >= -g + grad * v
                    # -grad * v_new <= g - grad * v
                    
                    grad = np.zeros(3*N + 1)
                    grad[idx_x][i] = 2*dx
                    grad[idx_x][j] = -2*dx
                    grad[idx_y][i] = 2*dy
                    grad[idx_y][j] = -2*dy
                    grad[idx_r][i] = -2*dr
                    grad[idx_r][j] = -2*dr
                    
                    rhs = g - np.dot(grad, v)
                    
                    # Add -grad * v_new <= rhs
                    A_ub_rows.append(-grad)
                    b_ub_rows.append(rhs)
                    
            A_ub = np.array(A_ub_rows)
            b_ub = np.array(b_ub_rows)
            
            try:
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if not res.success:
                    break
                    
                v_new = res.x
                # Check feasibility of v_new for original non-linear constraints
                feasible = True
                x_new = v_new[idx_x]
                y_new = v_new[idx_y]
                r_new = v_new[idx_r]
                
                # Box check
                if np.any(r_new > x_new + 1e-8) or np.any(r_new > w - x_new + 1e-8) or \
                   np.any(r_new > y_new + 1e-8) or np.any(r_new > h - y_new + 1e-8):
                    feasible = False
                    
                if feasible:
                    for i in range(N):
                        for j in range(i+1, N):
                            dist_sq = (x_new[i]-x_new[j])**2 + (y_new[i]-y_new[j])**2
                            rad_sum = r_new[i] + r_new[j]
                            if dist_sq < rad_sum**2 - 1e-8:
                                feasible = False
                                break
                        if not feasible: break
                            
                if feasible:
                    v = v_new
                    # Check for convergence
                    if np.linalg.norm(v_new - v) < 1e-6:
                        break
                else:
                    # Try step back
                    # v_try = v + alpha * (v_new - v)
                    alpha = 0.5
                    for _ in range(5):
                        v_try = v + alpha * (v_new - v)
                        x_try = v_try[idx_x]
                        y_try = v_try[idx_y]
                        r_try = v_try[idx_r]
                        
                        feasible_try = True
                        if np.any(r_try > x_try + 1e-8) or np.any(r_try > w - x_try + 1e-8) or \
                           np.any(r_try > y_try + 1e-8) or np.any(r_try > h - y_try + 1e-8):
                            feasible_try = False
                            
                        if feasible_try:
                            for i in range(N):
                                for j in range(i+1, N):
                                    dist_sq = (x_try[i]-x_try[j])**2 + (y_try[i]-y_try[j])**2
                                    rad_sum = r_try[i] + r_try[j]
                                    if dist_sq < rad_sum**2 - 1e-8:
                                        feasible_try = False
                                        break
                                if not feasible_try: break
                                        
                        if feasible_try:
                            v = v_try
                            break
                        else:
                            alpha *= 0.5
                    else:
                        # No feasible step found, break
                        break
                        
            except Exception as e:
                break
                
        # After SCR loop, evaluate best
        x_best = v[idx_x]
        y_best = v[idx_y]
        r_best = v[idx_r]
        
        # Final Validation
        valid = True
        if np.any(r_best > x_best + 1e-6) or np.any(r_best > w - x_best + 1e-6) or \
           np.any(r_best > y_best + 1e-6) or np.any(r_best > h - y_best + 1e-6):
            valid = False
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((x_best[i]-x_best[j])**2 + (y_best[i]-y_best[j])**2)
                    if r_best[i] + r_best[j] > dist + 1e-6:
                        valid = False
                        break
                if not valid: break
                
        if valid:
            score = np.sum(r_best)
            if score > best_score:
                best_score = score
                best_params = (x_best.copy(), y_best.copy(), r_best.copy())

    if best_params is None:
        # Fallback
        side = 5
        w = 1.0
        h = 1.0
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                x0 = cx * w
                y0 = cy * h
                r0 = 0.5 / side - 1e-3
                cells.append([x0, y0, r0])
        np.random.shuffle(cells)
        x = np.array([c[0] for c in cells[:N]])
        y = np.array([c[1] for c in cells[:N]])
        r = np.array([c[2] for c in cells[:N]])
        return np.column_stack((x, y, r))
        
    x, y, r = best_params
    return np.column_stack((x, y, r))
