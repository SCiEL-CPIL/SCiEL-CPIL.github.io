import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def generate_hex_init(w_init):
        """Generate hexagonal close-packing initialization"""
        h_init = 2.0 - w_init
        xs, ys, rs = [], [], []
        # Try different hex grid orientations and spacings
        best_local = None
        best_local_val = -1e10
        
        for row_shift in [0.0, 0.5]:
            for spacing in np.linspace(0.1, 0.5, 8):
                xs_t, ys_t, rs_t = [], [], []
                row = 0
                while len(xs_t) < N:
                    col = 0
                    while len(xs_t) < N:
                        cx = (col + row_shift * (row % 2)) * spacing + spacing * 0.5
                        cy = row * spacing * 0.866 + spacing * 0.5
                        if cx < w_init - spacing * 0.3 and cy < h_init - spacing * 0.3:
                            cx = np.clip(cx, spacing * 0.3, w_init - spacing * 0.3)
                            cy = np.clip(cy, spacing * 0.3, h_init - spacing * 0.3)
                            r = min(cx, w_init - cx, cy, h_init - cy) * 0.95
                            xs_t.append(cx)
                            ys_t.append(cy)
                            rs_t.append(r)
                        col += 1
                    row += 1
                    if row > 30:
                        break
                
                if len(xs_t) >= N:
                    xs_t = xs_t[:N]
                    ys_t = ys_t[:N]
                    rs_t = rs_t[:N]
                    # Check for overlaps
                    valid = True
                    for i in range(N):
                        for j in range(i+1, N):
                            d = np.sqrt((xs_t[i]-xs_t[j])**2 + (ys_t[i]-ys_t[j])**2)
                            if d < rs_t[i] + rs_t[j] - 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        val = sum(rs_t)
                        if val > best_local_val:
                            best_local_val = val
                            best_local = (np.array(xs_t), np.array(ys_t), np.array(rs_t))
        
        if best_local is None:
            # Fallback to grid
            side = int(np.ceil(np.sqrt(N)))
            xs_t, ys_t, rs_t = [], [], []
            for i in range(side):
                for j in range(side):
                    if len(xs_t) >= N:
                        break
                    cx = (i + 0.5) / side * w_init
                    cy = (j + 0.5) / side * h_init
                    r = min(cx, w_init - cx, cy, h_init - cy) * 0.85
                    xs_t.append(cx)
                    ys_t.append(cy)
                    rs_t.append(r)
            while len(xs_t) < N:
                xs_t.append(w_init/2)
                ys_t.append(h_init/2)
                rs_t.append(0.05)
            best_local = (np.array(xs_t[:N]), np.array(ys_t[:N]), np.array(rs_t[:N]))
        
        return best_local

    def compute_grad(v):
        """Compute gradient of objective (negative sum of radii + penalties)"""
        w = v[0]
        h = 2.0 - w
        xs = v[1:N+1]
        ys = v[N+1:2*N+1]
        rs = v[2*N+1:3*N+1]
        
        gw = 0.0
        gxs = np.zeros(N)
        gys = np.zeros(N)
        grs = np.zeros(N)
        
        # Objective: -sum(rs) + penalty
        # grad of -sum(rs) wrt rs is -1
        grs -= 1.0
        
        # Boundary penalties: sum(max(0, r - x)^2 + max(0, r - (w-x))^2 + max(0, r - y)^2 + max(0, r - (h-y))^2)
        # d/dx_i of max(0, r_i - x_i)^2 = -2*max(0, r_i - x_i)
        # d/dx_i of max(0, r_i - (w-x_i))^2 = 2*max(0, r_i - (w-x_i))
        # d/dw of max(0, r_i - (w-x_i))^2 = -2*max(0, r_i - (w-x_i))
        
        for i in range(N):
            p1 = max(0.0, rs[i] - xs[i])
            p2 = max(0.0, rs[i] - (w - xs[i]))
            p3 = max(0.0, rs[i] - ys[i])
            p4 = max(0.0, rs[i] - (h - ys[i]))
            
            gxs[i] += -2.0 * p1 + 2.0 * p2
            gys[i] += -2.0 * p3 + 2.0 * p4
            grs[i] += 2.0 * (p1 + p2 + p3 + p4)
            
            gw += -2.0 * p2 + 2.0 * p4
        
        # Overlap penalties: for each pair, if d < r_i + r_j, penalty = (r_i + r_j - d)^2
        # d = sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
        # d/dx_i of (r_i+r_j-d)^2 = -2*(r_i+r_j-d) * (x_i-x_j)/d
        # d/dr_i of (r_i+r_j-d)^2 = 2*(r_i+r_j-d)
        
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = rs[i] + rs[j]
                if d < min_d and d > 1e-12:
                    diff = min_d - d
                    # Gradient wrt x_i
                    gxs[i] += -2.0 * diff * dx / d
                    gxs[j] += 2.0 * diff * dx / d
                    # Gradient wrt y_i
                    gys[i] += -2.0 * diff * dy / d
                    gys[j] += 2.0 * diff * dy / d
                    # Gradient wrt r_i and r_j
                    grs[i] += 2.0 * diff
                    grs[j] += 2.0 * diff
                elif d <= 1e-12:
                    # Degenerate case
                    gxs[i] += 1.0
                    gxs[j] -= 1.0
        
        return np.concatenate(([gw], gxs, gys, grs))

    def compute_obj(v):
        """Compute objective value"""
        w = v[0]
        if w <= 0 or w >= 2:
            return 1e10
        h = 2.0 - w
        xs = v[1:N+1]
        ys = v[N+1:2*N+1]
        rs = v[2*N+1:3*N+1]
        
        if np.any(rs < 0):
            return 1e10
        
        pen = 0.0
        pen += np.sum(np.maximum(0, rs - xs)**2)
        pen += np.sum(np.maximum(0, rs - (w - xs))**2)
        pen += np.sum(np.maximum(0, rs - ys)**2)
        pen += np.sum(np.maximum(0, rs - (h - ys))**2)
        
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = rs[i] + rs[j]
                if d < min_d:
                    pen += (min_d - d)**2
        
        return -np.sum(rs) + 100.0 * pen

    def gradient_descent(v0, max_iters=5000, lr_init=0.01):
        """Simple gradient descent with adaptive learning rate"""
        v = v0.copy()
        obj = compute_obj(v)
        best_v = v.copy()
        best_obj = obj
        
        lr = lr_init
        prev_obj = obj
        
        for it in range(max_iters):
            if elapsed() > deadline - 10:
                break
            
            grad = compute_grad(v)
            
            # Adaptive learning rate
            if obj > prev_obj:
                lr *= 0.5
            elif it > 0 and obj < prev_obj - 1e-6:
                lr *= 1.1
            lr = np.clip(lr, 1e-6, 0.1)
            
            v_new = v - lr * grad
            # Keep w in valid range
            v_new[0] = np.clip(v_new[0], 0.1, 1.9)
            # Keep radii non-negative
            v_new[2*N+1:3*N+1] = np.maximum(0, v_new[2*N+1:3*N+1])
            
            new_obj = compute_obj(v_new)
            
            if new_obj < obj:
                v = v_new
                obj = new_obj
                if obj < best_obj:
                    best_obj = obj
                    best_v = v.copy()
            else:
                # Try smaller step
                lr *= 0.5
                if lr < 1e-8:
                    break
            
            prev_obj = obj
        
        return best_v, best_obj

    def polish_solution(x_sol):
        """Final high-precision polish pass"""
        w_best = x_sol[0]
        h_best = 2.0 - w_best
        xs_best = x_sol[1:N+1].copy()
        ys_best = x_sol[N+1:2*N+1].copy()
        rs_best = x_sol[2*N+1:3*N+1].copy()
        
        # Multiple passes to resolve collisions and ensure validity
        for _ in range(100):
            changed = False
            
            # Ensure within bounds
            for i in range(N):
                max_r = min(xs_best[i], w_best - xs_best[i], ys_best[i], h_best - ys_best[i])
                if rs_best[i] > max_r:
                    rs_best[i] = max_r
                    changed = True
            
            # Resolve overlaps iteratively
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs_best[i] - xs_best[j]
                    dy = ys_best[i] - ys_best[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    min_d = rs_best[i] + rs_best[j]
                    if d < min_d - 1e-12:
                        # Shrink both radii proportionally
                        if d > 1e-10:
                            factor = d / (min_d * 1.001)
                            rs_best[i] *= factor
                            rs_best[j] *= factor
                        else:
                            # Move apart if very close
                            if abs(dx) < 1e-10 and abs(dy) < 1e-10:
                                xs_best[i] += 0.01
                                xs_best[j] -= 0.01
                        changed = True
            
            if not changed:
                break
        
        # Final validation
        for i in range(N):
            rs_best[i] = min(rs_best[i], xs_best[i], w_best - xs_best[i], ys_best[i], h_best - ys_best[i])
            if rs_best[i] < 0:
                rs_best[i] = 0
        
        return np.column_stack((xs_best, ys_best, rs_best))

    def slsqp_refine(x0):
        """Use SLSQP for final refinement with explicit constraints"""
        w0 = x0[0]
        xs0 = x0[1:N+1]
        ys0 = x0[N+1:2*N+1]
        rs0 = x0[2*N+1:3*N+1]
        
        def obj(v):
            w = v[0]
            h = 2.0 - w
            xs = v[1:N+1]
            ys = v[N+1:2*N+1]
            rs = v[2*N+1:3*N+1]
            return -np.sum(rs)
        
        # Constraints: boundary and non-overlap
        cons = []
        # Boundary constraints
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[1+i] - v[2*N+1+i]})  # x_i >= r_i
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[0] - v[1+i] - v[2*N+1+i]})  # w - x_i >= r_i
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+1+i] - v[2*N+1+i]})  # y_i >= r_i
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 2.0 - v[0] - v[N+1+i] - v[2*N+1+i]})  # h - y_i >= r_i
        
        # Non-overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: 
                        np.sqrt((v[1+i]-v[1+j])**2 + (v[N+1+i]-v[N+1+j])**2) - v[2*N+1+i] - v[2*N+1+j]
                })
        
        # Non-negative radii
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*N+1+i]})
        
        bounds = [(0.1, 1.9)] + [(0, 2)] * (2*N) + [(0, 1)] * N
        
        try:
            res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons,
                          options={'maxiter': 500, 'ftol': 1e-12})
            if res.success and res.fun < compute_obj(x0):
                return res.x
        except:
            pass
        return x0

    # Try diverse aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4, 0.5, 1.5]
    
    best_val = 1e10
    best_x = None
    
    for w_init in aspect_ratios:
        if elapsed() > deadline - 20:
            break
        
        # Generate hexagonal initialization
        xs, ys, rs = generate_hex_init(w_init)
        x0 = np.concatenate(([w_init], xs, ys, rs))
        
        # Gradient descent
        x_gd, obj_gd = gradient_descent(x0, max_iters=3000, lr_init=0.005)
        
        if obj_gd < best_val:
            best_val = obj_gd
            best_x = x_gd.copy()
        
        # SLSQP refinement if time permits
        if elapsed() < deadline - 15:
            x_slsqp = slsqp_refine(x_gd)
            obj_slsqp = compute_obj(x_slsqp)
            if obj_slsqp < best_val:
                best_val = obj_slsqp
                best_x = x_slsqp.copy()
    
    if best_x is None:
        # Fallback
        w_init = 1.0
        xs, ys, rs = generate_hex_init(w_init)
        best_x = np.concatenate(([w_init], xs, ys, rs))
    
    # Final polish
    result = polish_solution(best_x)
    
    # One more SLSQP pass on polished solution if time permits
    if elapsed() < deadline - 5:
        w_polished = best_x[0]
        x_polished = np.concatenate(([w_polished], result[:, 0], result[:, 1], result[:, 2]))
        x_final = slsqp_refine(x_polished)
        obj_final = compute_obj(x_final)
        if obj_final < best_val:
            best_val = obj_final
            best_x = x_final.copy()
            result = polish_solution(best_x)
    
    return result
