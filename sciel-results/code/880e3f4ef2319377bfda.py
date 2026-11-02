import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def check_time():
        return time.time() > deadline

    def unpack(v):
        w = v[0]
        r = v[1]
        xs = v[2:2+N]
        ys = v[2+N:2+2*N]
        return w, r, xs, ys

    def validate(w, r, xs, ys, tol=1e-7):
        h = 2.0 - w
        if w <= tol or h <= tol or r <= tol:
            return False
        for i in range(N):
            if xs[i] < r - tol or xs[i] > w - r + tol:
                return False
            if ys[i] < r - tol or ys[i] > h - r + tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2
                if dist_sq < 4*(r**2) - tol:
                    return False
        return True

    def pack_hexagonal(rows):
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
        return w, r, xs_final, ys_final

    def pack_grid(rows, cols):
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
        return w, r, xs_final, ys_final

    def refine_slsqp(w, r, xs, ys, is_polish=False):
        if check_time():
            return w, r, xs, ys
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
            xs_v = v[2:2+N]
            ys_v = v[2+N:2+2*N]
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
        if is_polish:
            options = {'maxiter': 300, 'ftol': 1e-12, 'disp': False}
        else:
            options = {'maxiter': 150, 'ftol': 1e-9, 'disp': False}

        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=constraints, options=options)
            if res.success or res.fun < -r:
                w_res, r_res, xs_res, ys_res = unpack(res.x)
                if validate(w_res, r_res, xs_res, ys_res):
                    return w_res, r_res, xs_res, ys_res
        except Exception:
            pass
        return w, r, xs, ys

    def refine_lbfgs(w, r, xs, ys, beta=50.0, max_iter=200):
        if check_time():
            return w, r, xs, ys
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
            xs_v = v[2:2+N]
            ys_v = v[2+N:2+2*N]
            penalty = 0.0
            # Boundary penalties
            for i in range(N):
                if xs_v[i] < r_val: penalty += (r_val - xs_v[i])**2
                if xs_v[i] > w_val - r_val: penalty += (xs_v[i] - (w_val - r_val))**2
                if ys_v[i] < r_val: penalty += (r_val - ys_v[i])**2
                if ys_v[i] > h_val - r_val: penalty += (ys_v[i] - (h_val - r_val))**2
            # Overlap penalties (soft)
            for i in range(N):
                for j in range(i+1, N):
                    dist_sq = (xs_v[i]-xs_v[j])**2 + (ys_v[i]-ys_v[j])**2
                    target = 4*(r_val**2)
                    if dist_sq < target:
                        penalty += (target - dist_sq)**2
            return -r_val + beta * penalty

        def jac(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            if w_val <= 0.1 or h_val <= 0.1 or r_val <= 0:
                return np.zeros_like(v)
            xs_v = v[2:2+N]
            ys_v = v[2+N:2+2*N]
            grad = np.zeros_like(v)
            grad[1] = -1.0 # d(-r)/dr
            
            # Boundary gradients
            for i in range(N):
                # x_i - r
                if xs_v[i] < r_val:
                    diff = r_val - xs_v[i]
                    grad[1] += -2 * beta * diff # d(-r + beta*(r-x)^2)/dr = -1 - 2*beta*(r-x) ... wait.
                    # f = -r + beta*(r-x)^2
                    # df/dr = -1 + 2*beta*(r-x)
                    # df/dx = -2*beta*(r-x)
                    grad[1] += 2 * beta * diff
                    grad[2+i] += -2 * beta * diff
                
                if xs_v[i] > w_val - r_val:
                    diff = xs_v[i] - (w_val - r_val)
                    # f = -r + beta*(x - (w-r))^2
                    # df/dr = -1 - 2*beta*(x - (w-r))
                    # df/dx = 2*beta*(x - (w-r))
                    # df/dw = -2*beta*(x - (w-r))
                    grad[1] += -2 * beta * diff
                    grad[2+i] += 2 * beta * diff
                    grad[0] += -2 * beta * diff

                if ys_v[i] < r_val:
                    diff = r_val - ys_v[i]
                    grad[1] += 2 * beta * diff
                    grad[2+N+i] += -2 * beta * diff

                if ys_v[i] > h_val - r_val:
                    diff = ys_v[i] - (h_val - r_val)
                    # h = 2-w, so h-r = 2-w-r
                    # x - (2-w-r)
                    # df/dr = -1 - 2*beta*(x - (2-w-r))
                    # df/dy = 2*beta*(x - (2-w-r))
                    # df/dw = -2*beta*(x - (2-w-r)) * (-1) = 2*beta*diff
                    grad[1] += -2 * beta * diff
                    grad[2+N+i] += 2 * beta * diff
                    grad[0] += 2 * beta * diff

            # Overlap gradients
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs_v[i] - xs_v[j]
                    dy = ys_v[i] - ys_v[j]
                    dist_sq = dx*dx + dy*dy
                    target = 4*(r_val**2)
                    if dist_sq < target:
                        diff = target - dist_sq
                        # f = beta * (target - dist_sq)^2
                        # df/dr = 2*beta*diff * d(target)/dr = 2*beta*diff * 8r
                        # df/dxi = 2*beta*diff * d(-dist_sq)/dxi = 2*beta*diff * (-2dx) = -4*beta*diff*dx
                        # df/dxj = 2*beta*diff * d(-dist_sq)/dxj = 2*beta*diff * (2dx) = 4*beta*diff*dx
                        grad[1] += 16 * beta * diff * r_val
                        grad[2+i] += -4 * beta * diff * dx
                        grad[2+j] += 4 * beta * diff * dx
                        grad[2+N+i] += -4 * beta * diff * dy
                        grad[2+N+j] += 4 * beta * diff * dy
            return grad

        bounds = [(0.2, 1.8), (0.01, 1.0)] + [(0.0, 2.0)] * (2*N)
        try:
            res = minimize(objective, v0, method='L-BFGS-B', bounds=bounds, jac=jac, 
                           options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False})
            w_res, r_res, xs_res, ys_res = unpack(res.x)
            if validate(w_res, r_res, xs_res, ys_res):
                return w_res, r_res, xs_res, ys_res
        except Exception:
            pass
        return w, r, xs, ys

    best_r = 0
    best_config = None

    # List of initial configurations
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
        configs.append((w, r, xs, ys))

    valid_configs = [c for c in configs if c is not None]

    # First pass: Coarse optimization with SLSQP
    refined_configs = []
    for cfg in valid_configs:
        if check_time():
            break
        w, r, xs, ys = cfg
        w_new, r_new, xs_new, ys_new = refine_slsqp(w, r, xs, ys, is_polish=False)
        if r_new > best_r:
            best_r = r_new
            best_config = (w_new, r_new, xs_new, ys_new)
        refined_configs.append((w_new, r_new, xs_new, ys_new))
        
    # Second pass: Refine with L-BFGS-B (soft constraints)
    for cfg in refined_configs:
        if check_time():
            break
        w, r, xs, ys = cfg
        w_new, r_new, xs_new, ys_new = refine_lbfgs(w, r, xs, ys, beta=50.0, max_iter=100)
        if r_new > best_r:
            best_r = r_new
            best_config = (w_new, r_new, xs_new, ys_new)
            
    # Third pass: Final polish with SLSQP (tighter tolerance)
    if best_config is not None and not check_time():
        w, r, xs, ys = best_config
        w_new, r_new, xs_new, ys_new = refine_slsqp(w, r, xs, ys, is_polish=True)
        if r_new > best_r:
            best_r = r_new
            best_config = (w_new, r_new, xs_new, ys_new)
            
    # If we still have time, do one more L-BFGS polish on the best
    if best_config is not None and not check_time():
        w, r, xs, ys = best_config
        w_new, r_new, xs_new, ys_new = refine_lbfgs(w, r, xs, ys, beta=100.0, max_iter=100)
        if r_new > best_r:
            best_r = r_new
            best_config = (w_new, r_new, xs_new, ys_new)

    if best_config is not None:
        w, r, xs, ys = best_config
        result = np.column_stack((xs, ys, np.full(N, r)))
        return result
    else:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
