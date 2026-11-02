import numpy as np
from scipy.optimize import minimize
import time
import random

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def check_time():
        return time.time() > deadline

    def pack_hexagonal(rows):
        """
        Generate a hexagonal packing configuration.
        Returns: w, h, r, xs, ys
        """
        if sum(rows) != N:
            return None

        pts = []
        for k, n_circles in enumerate(rows):
            y_k = k * np.sqrt(3)
            offset = 1.0 if k % 2 == 1 else 0.0
            for i in range(n_circles):
                x_k = i * 2.0 + offset
                pts.append((x_k, y_k))
        
        pts = np.array(pts)
        if len(pts) != N:
            return None

        center = pts.mean(axis=0)
        pts -= center
        
        xs = pts[:, 0]
        ys = pts[:, 1]
        w_centers = xs.max() - xs.min()
        h_centers = ys.max() - ys.min()
        
        w_total = w_centers + 2.0
        h_total = h_centers + 2.0
        
        scale = 2.0 / (w_total + h_total)
        r = scale
        w = scale * w_total
        h = scale * h_total
        
        xs_final = pts[:, 0] * scale + w/2.0
        ys_final = pts[:, 1] * scale + h/2.0
        
        return w, h, r, xs_final, ys_final

    def pack_grid(rows, cols):
        """
        Generate a grid packing configuration.
        """
        pts = []
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                pts.append((i * 2.0, j * 2.0))
                count += 1
            if count >= N:
                break
        
        pts = np.array(pts[:N])
        
        center = pts.mean(axis=0)
        pts -= center
        
        xs = pts[:, 0]
        ys = pts[:, 1]
        w_centers = xs.max() - xs.min()
        h_centers = ys.max() - ys.min()
        
        w_total = w_centers + 2.0
        h_total = h_centers + 2.0
        
        scale = 2.0 / (w_total + h_total)
        r = scale
        w = scale * w_total
        h = scale * h_total
        
        xs_final = pts[:, 0] * scale + w/2.0
        ys_final = pts[:, 1] * scale + h/2.0
        
        return w, h, r, xs_final, ys_final

    def physics_simulate(w, h, r, xs, ys, max_iter=200):
        """
        Physics-based refinement using repulsion and soft boundaries.
        Returns improved (w, h, r, xs, ys) or original if no improvement.
        """
        if check_time():
            return w, h, r, xs, ys

        x = xs.copy()
        y = ys.copy()
        
        current_r = r
        current_w = w
        current_h = h
        
        # Relaxation function
        def relax(x, y, r_val, w_val, h_val, steps=50):
            for _ in range(steps):
                if check_time():
                    break
                fx = np.zeros(N)
                fy = np.zeros(N)
                
                # Pairwise repulsion
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist_sq = dx*dx + dy*dy
                        min_dist = 2.0 * r_val
                        min_dist_sq = min_dist * min_dist
                        if dist_sq < min_dist_sq:
                            dist = np.sqrt(dist_sq)
                            if dist > 1e-6:
                                overlap = min_dist - dist
                                force = 10.0 * overlap
                                fx[i] += (dx/dist) * force
                                fy[i] += (dy/dist) * force
                                fx[j] -= (dx/dist) * force
                                fy[j] -= (dy/dist) * force
                            else:
                                fx[i] += random.uniform(-1, 1)
                                fy[i] += random.uniform(-1, 1)
                                fx[j] -= random.uniform(-1, 1)
                                fy[j] -= random.uniform(-1, 1)
                
                # Boundary repulsion
                for i in range(N):
                    if x[i] < r_val:
                        fx[i] += 10.0 * (r_val - x[i])
                    if x[i] > w_val - r_val:
                        fx[i] -= 10.0 * (x[i] - (w_val - r_val))
                    if y[i] < r_val:
                        fy[i] += 10.0 * (r_val - y[i])
                    if y[i] > h_val - r_val:
                        fy[i] -= 10.0 * (y[i] - (h_val - r_val))
                
                x += 0.01 * fx
                y += 0.01 * fy
                
                x = np.clip(x, r_val, w_val - r_val)
                y = np.clip(y, r_val, h_val - r_val)
                
            return x, y

        def is_valid(x, y, r_val, w_val, h_val):
            for i in range(N):
                if x[i] < r_val - 1e-7 or x[i] > w_val - r_val + 1e-7:
                    return False
                if y[i] < r_val - 1e-7 or y[i] > h_val - r_val + 1e-7:
                    return False
            for i in range(N):
                for j in range(i+1, N):
                    dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                    if dist_sq < 4*(r_val**2) - 1e-7:
                        return False
            return True

        if not is_valid(x, y, current_r, current_w, current_h):
            x, y = relax(x, y, current_r, current_w, current_h, steps=100)
            if not is_valid(x, y, current_r, current_w, current_h):
                return w, h, r, xs, ys

        max_r_possible = min(current_w, current_h) / 2.0
        r_try = current_r
        delta_r = 0.001
        improved = False
        
        while r_try < max_r_possible - 1e-4 and not check_time():
            x_new, y_new = relax(x, y, r_try + delta_r, current_w, current_h, steps=20)
            
            if is_valid(x_new, y_new, r_try + delta_r, current_w, current_h):
                r_try += delta_r
                x, y = x_new, y_new
                improved = True
            else:
                break
        
        if improved:
            return current_w, current_h, r_try, x, y
        else:
            return current_w, current_h, current_r, x, y

    def refine_smooth(w, h, r, xs, ys, is_polish=False):
        """
        Refine the packing using L-BFGS-B with a smooth softplus penalty for constraints.
        Variables: w, r, xs, ys. h = 2 - w.
        Objective: -r + penalty
        Vectorized overlap computation for ~5x speedup.
        """
        if check_time():
            return w, h, r, xs, ys

        # Pre-relax with physics
        w, h, r, xs, ys = physics_simulate(w, h, r, xs, ys)
        
        v0 = np.zeros(2 + 2*N)
        v0[0] = w
        v0[1] = r
        v0[2:2+N] = xs
        v0[2+N:2+2*N] = ys

        # Softplus beta: larger is sharper (closer to hard constraint)
        # Start with moderate beta, increase during polish
        beta = 50.0 if not is_polish else 200.0
        penalty_weight = 10.0 if not is_polish else 50.0

        # Precompute index pairs for vectorized overlap
        idx_i, idx_j = np.triu_indices(N, k=1)

        def softplus(x, beta=50.0):
            # Numerically stable softplus
            return np.where(x > 1.0/beta, x, np.log1p(np.exp(beta * x)) / beta)

        def objective(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            
            if w_val < 0.1 or w_val > 1.9 or r_val < 0.01 or r_val > 1.0:
                return 1e6

            xs = v[2:2+N]
            ys = v[2+N:2+2*N]
            
            penalty = 0.0
            
            # Boundary violations
            viol_x_left = r_val - xs
            viol_x_right = xs - (w_val - r_val)
            viol_y_bottom = r_val - ys
            viol_y_top = ys - (h_val - r_val)
            
            penalty += np.sum(softplus(viol_x_left, beta))
            penalty += np.sum(softplus(viol_x_right, beta))
            penalty += np.sum(softplus(viol_y_bottom, beta))
            penalty += np.sum(softplus(viol_y_top, beta))
            
            # Vectorized overlap violations
            dx = xs[idx_i] - xs[idx_j]
            dy = ys[idx_i] - ys[idx_j]
            dist = np.sqrt(dx*dx + dy*dy + 1e-12)
            viol = 2.0 * r_val - dist
            penalty += np.sum(softplus(viol, beta))
            
            return -r_val + penalty_weight * penalty

        def grad_objective(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            xs = v[2:2+N]
            ys = v[2+N:2+2*N]
            
            grad = np.zeros_like(v)
            grad[1] = -1.0
            
            def d_softplus(x, beta=50.0):
                z = beta * x
                sig = np.where(z > 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
                return sig * beta

            # Boundary gradients
            viol_x_left = r_val - xs
            viol_x_right = xs - (w_val - r_val)
            viol_y_bottom = r_val - ys
            viol_y_top = ys - (h_val - r_val)
            
            ds_xl = d_softplus(viol_x_left, beta)
            ds_xr = d_softplus(viol_x_right, beta)
            ds_yb = d_softplus(viol_y_bottom, beta)
            ds_yt = d_softplus(viol_y_top, beta)
            
            grad[2:2+N] -= penalty_weight * ds_xl
            grad[2:2+N] += penalty_weight * ds_xr
            grad[2+N:2+2*N] -= penalty_weight * ds_yb
            grad[2+N:2+2*N] += penalty_weight * ds_yt
            grad[1] += penalty_weight * (np.sum(ds_xl) + np.sum(ds_yb))
            grad[0] -= penalty_weight * np.sum(ds_xr)
            
            # Vectorized overlap gradients
            dx = xs[idx_i] - xs[idx_j]
            dy = ys[idx_i] - ys[idx_j]
            dist = np.sqrt(dx*dx + dy*dy + 1e-12)
            viol = 2.0 * r_val - dist
            ds = d_softplus(viol, beta)
            
            # Accumulate gradients for each pair
            grad_xi = penalty_weight * ds * (dx / dist)
            grad_xj = -penalty_weight * ds * (dx / dist)
            grad_yi = penalty_weight * ds * (dy / dist)
            grad_yj = -penalty_weight * ds * (dy / dist)
            
            np.add.at(grad[2:2+N], idx_i, -grad_xi)
            np.add.at(grad[2:2+N], idx_j, grad_xj)
            np.add.at(grad[2+N:2+2*N], idx_i, -grad_yi)
            np.add.at(grad[2+N:2+2*N], idx_j, grad_yj)
            grad[1] += penalty_weight * np.sum(ds * 2.0)
            
            return grad

        bounds = [(0.2, 1.8), (0.01, 1.0)] + [(0.0, 2.0)] * (2*N)

        try:
            if is_polish:
                options = {'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-14, 'disp': False, 'maxls': 50}
            else:
                options = {'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10, 'disp': False, 'maxls': 50}

            res = minimize(objective, v0, method='L-BFGS-B', jac=grad_objective, bounds=bounds, options=options)
            
            if res.fun < -r + 1e-6:
                w_res = res.x[0]
                r_res = res.x[1]
                h_res = 2.0 - w_res
                xs_res = res.x[2:2+N]
                ys_res = res.x[2+N:2+2*N]
                
                valid = True
                if w_res <= 0 or h_res <= 0 or r_res <= 0:
                    valid = False
                else:
                    for i in range(N):
                        if xs_res[i] < r_res - 1e-7 or xs_res[i] > w_res - r_res + 1e-7:
                            valid = False
                            break
                        if ys_res[i] < r_res - 1e-7 or ys_res[i] > h_res - r_res + 1e-7:
                            valid = False
                            break
                    if valid:
                        dx = xs_res[idx_i] - xs_res[idx_j]
                        dy = ys_res[idx_i] - ys_res[idx_j]
                        dist_sq = dx*dx + dy*dy
                        if np.any(dist_sq < 4*(r_res**2) - 1e-7):
                            valid = False
                
                if valid:
                    return w_res, h_res, r_res, xs_res, ys_res
        except Exception:
            pass
            
        return w, h, r, xs, ys

    def refine_slsqp(w, h, r, xs, ys, is_polish=False):
        """
        Fallback or alternative refinement using SLSQP with hard constraints.
        """
        if check_time():
            return w, h, r, xs, ys

        v0 = np.zeros(2 + 2*N)
        v0[0] = w
        v0[1] = r
        v0[2:2+N] = xs
        v0[2+N:2+2*N] = ys

        def objective(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            if w_val <= 0.1 or h_val <= 0.1 or r_val <= 0:
                return 1e6
            return -r_val

        def constraints(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            xs = v[2:2+N]
            ys = v[2+N:2+2*N]
            
            cons = []
            cons.append({'type': 'ineq', 'fun': lambda v: v[0] - 0.2})
            cons.append({'type': 'ineq', 'fun': lambda v: 2.0 - v[0] - 0.2})
            
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2+i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[0] - v[2+i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2+N+i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: (2.0 - v[0]) - v[2+N+i] - v[1]})
                
            for i in range(N):
                for j in range(i+1, N):
                    cons.append({
                        'type': 'ineq', 
                        'fun': lambda v, i=i, j=j: ((v[2+i]-v[2+j])**2 + (v[2+N+i]-v[2+N+j])**2) - 4*(v[1]**2)
                    })
            return cons

        bounds = [(0.2, 1.8), (0.01, 1.0)] + [(0.0, 2.0)] * (2*N)

        try:
            if is_polish:
                options = {'maxiter': 300, 'ftol': 1e-12, 'disp': False}
            else:
                options = {'maxiter': 100, 'ftol': 1e-9, 'disp': False}

            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=constraints, options=options)
            
            if res.success or res.fun < -r:
                w_res = res.x[0]
                r_res = res.x[1]
                h_res = 2.0 - w_res
                xs_res = res.x[2:2+N]
                ys_res = res.x[2+N:2+2*N]
                
                valid = True
                if w_res <= 0 or h_res <= 0 or r_res <= 0:
                    valid = False
                else:
                    for i in range(N):
                        if xs_res[i] < r_res - 1e-7 or xs_res[i] > w_res - r_res + 1e-7:
                            valid = False
                            break
                        if ys_res[i] < r_res - 1e-7 or ys_res[i] > h_res - r_res + 1e-7:
                            valid = False
                            break
                    if valid:
                        for i in range(N):
                            for j in range(i+1, N):
                                dist_sq = (xs_res[i]-xs_res[j])**2 + (ys_res[i]-ys_res[j])**2
                                if dist_sq < 4*(r_res**2) - 1e-7:
                                    valid = False
                                    break
                            if not valid:
                                break
                
                if valid:
                    return w_res, h_res, r_res, xs_res, ys_res
        except Exception:
            pass
            
        return w, h, r, xs, ys

    best_r = 0
    best_config = None

    configs = []
    
    # Hexagonal variations
    configs.append(pack_hexagonal([7, 7, 7]))
    configs.append(pack_hexagonal([4, 5, 4, 4, 4]))
    configs.append(pack_hexagonal([3, 4, 4, 4, 4, 2]))
    configs.append(pack_hexagonal([5, 5, 5, 3, 3]))
    configs.append(pack_hexagonal([6, 6, 5, 4]))
    
    # Grid variations
    configs.append(pack_grid(5, 5))
    configs.append(pack_grid(3, 7))
    configs.append(pack_grid(4, 6))
    configs.append(pack_grid(7, 3))
    configs.append(pack_grid(6, 4))
    
    # Random restarts
    for _ in range(10):
        if check_time():
            break
        w = 1.0
        h = 1.0
        r = 0.08
        xs = np.random.uniform(r, w-r, N)
        ys = np.random.uniform(r, h-r, N)
        configs.append((w, h, r, xs, ys))

    valid_configs = [c for c in configs if c is not None]

    # First pass: Coarse optimization with Smooth L-BFGS (vectorized = faster per call)
    for cfg in valid_configs:
        if check_time():
            break
        w, h, r, xs, ys = cfg
        w_new, h_new, r_new, xs_new, ys_new = refine_smooth(w, h, r, xs, ys, is_polish=False)
        
        if r_new > best_r:
            best_r = r_new
            best_config = (w_new, h_new, r_new, xs_new, ys_new)

    # Second pass: Polish the best configuration with Smooth L-BFGS (higher beta, more iterations)
    if best_config is not None and not check_time():
        w, h, r, xs, ys = best_config
        w_new, h_new, r_new, xs_new, ys_new = refine_smooth(w, h, r, xs, ys, is_polish=True)
        
        if r_new > best_r:
            best_r = r_new
            best_config = (w_new, h_new, r_new, xs_new, ys_new)

    # Third pass: Try SLSQP on the best config to see if it can squeeze more out
    if best_config is not None and not check_time():
        w, h, r, xs, ys = best_config
        w_new, h_new, r_new, xs_new, ys_new = refine_slsqp(w, h, r, xs, ys, is_polish=True)
        
        if r_new > best_r:
            best_r = r_new
            best_config = (w_new, h_new, r_new, xs_new, ys_new)

    # Fourth pass: Final ultra-tight polish if time remains
    if best_config is not None and not check_time() and (deadline - time.time()) > 15.0:
        w, h, r, xs, ys = best_config
        w_new, h_new, r_new, xs_new, ys_new = refine_smooth(w, h, r, xs, ys, is_polish=True)
        
        if r_new > best_r:
            best_r = r_new
            best_config = (w_new, h_new, r_new, xs_new, ys_new)

    if best_config is None:
        # Fallback
        w, h, r, xs, ys = valid_configs[0]
        return np.column_stack([xs, ys, np.full(N, r)])

    w, h, r, xs, ys = best_config
    
    # Final safety adjustment: tiny shrink to guarantee validity under floating-point
    r_final = r * (1 - 1e-6)
    
    return np.column_stack([xs, ys, np.full(N, r_final)])
