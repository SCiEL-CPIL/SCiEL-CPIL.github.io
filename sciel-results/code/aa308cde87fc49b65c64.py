import numpy as np
from scipy.optimize import minimize
import time
import random

N = 21
PAIRS = [(i, j) for i in range(N) for j in range(i+1, N)]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def check_time():
        return time.time() > deadline

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
        return w, h, r, xs_final, ys_final

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
        return w, h, r, xs_final, ys_final

    def physics_simulate(x0, y0, w0, h0, r0, max_steps=3000, dt=0.002, cooling_rate=0.9995):
        if check_time():
            return w0, h0, r0, x0.copy(), y0.copy()
        sigma = 2.0 * r0
        eps = 1.0
        boundary_strength = 50.0
        x = x0.copy().astype(float)
        y = y0.copy().astype(float)
        vx = np.zeros(N)
        vy = np.zeros(N)
        temperature = 0.01
        damping = 0.99
        for step in range(max_steps):
            if check_time():
                break
            fx = np.zeros(N)
            fy = np.zeros(N)
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx*dx + dy*dy
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-10:
                        dx = random.uniform(-1e-6, 1e-6)
                        dy = random.uniform(-1e-6, 1e-6)
                        dist_sq = dx*dx + dy*dy
                        dist = np.sqrt(dist_sq)
                    sr6 = (sigma/dist)**6
                    sr12 = sr6*sr6
                    force_mag = 24*eps*(2*sr12 - sr6) / dist
                    fx[i] += force_mag * dx
                    fy[i] += force_mag * dy
                    fx[j] -= force_mag * dx
                    fy[j] -= force_mag * dy
            for i in range(N):
                if x[i] < r0:
                    penetration = r0 - x[i]
                    fx[i] += boundary_strength * penetration
                if x[i] > w0 - r0:
                    penetration = x[i] - (w0 - r0)
                    fx[i] -= boundary_strength * penetration
                if y[i] < r0:
                    penetration = r0 - y[i]
                    fy[i] += boundary_strength * penetration
                if y[i] > h0 - r0:
                    penetration = y[i] - (h0 - r0)
                    fy[i] -= boundary_strength * penetration
            vx = vx * damping + fx * dt
            vy = vy * damping + fy * dt
            x += vx * dt
            y += vy * dt
            temperature *= cooling_rate
            if step % 100 == 0 and temperature > 1e-4:
                x += np.random.normal(0, temperature*0.01, N)
                y += np.random.normal(0, temperature*0.01, N)
            x = np.clip(x, -0.5, w0 + 0.5)
            y = np.clip(y, -0.5, h0 + 0.5)
        xs_min = x.min() - r0
        xs_max = x.max() + r0
        ys_min = y.min() - r0
        ys_max = y.max() + r0
        w_actual = xs_max - xs_min
        h_actual = ys_max - ys_min
        if w_actual + h_actual > 0:
            scale = 2.0 / (w_actual + h_actual)
            w_final = scale * w_actual
            h_final = scale * h_actual
            r_final = scale * r0
            x_final = (x - xs_min) * scale
            y_final = (y - ys_min) * scale
        else:
            w_final = w0
            h_final = h0
            r_final = r0
            x_final = x.copy()
            y_final = y.copy()
        return w_final, h_final, r_final, x_final, y_final

    def refine_smooth(w, h, r, xs, ys, max_iter=500):
        if check_time():
            return w, h, r, xs, ys
        v0 = np.zeros(2 + 2*N)
        v0[0] = w
        v0[1] = r
        v0[2:2+N] = xs
        v0[2+N:2+2*N] = ys

        def penalty_obj(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            xs_arr = v[2:2+N]
            ys_arr = v[2+N:2+2*N]
            if w_val < 0.1 or h_val < 0.1 or r_val < 0.001:
                return 1e6
            penalty = 0.0
            lam = 50.0
            for i in range(N):
                if xs_arr[i] < r_val:
                    d = r_val - xs_arr[i]
                    penalty += lam * d * d
                if xs_arr[i] > w_val - r_val:
                    d = xs_arr[i] - (w_val - r_val)
                    penalty += lam * d * d
                if ys_arr[i] < r_val:
                    d = r_val - ys_arr[i]
                    penalty += lam * d * d
                if ys_arr[i] > h_val - r_val:
                    d = ys_arr[i] - (h_val - r_val)
                    penalty += lam * d * d
            for i, j in PAIRS:
                dx = xs_arr[i] - xs_arr[j]
                dy = ys_arr[i] - ys_arr[j]
                dist_sq = dx*dx + dy*dy
                min_dist = 2.0 * r_val
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq)
                    d = min_dist - dist
                    penalty += lam * d * d
            return -r_val + penalty

        bounds = [(0.1, 1.9), (0.001, 1.0)] + [(-0.5, 2.5)] * (2*N)
        try:
            res = minimize(penalty_obj, v0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False})
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
                    for i, j in PAIRS:
                        dist = np.sqrt((xs_res[i]-xs_res[j])**2 + (ys_res[i]-ys_res[j])**2)
                        if dist < 2*r_res - 1e-7:
                            valid = False
                            break
            if valid and r_res > r:
                return w_res, h_res, r_res, xs_res, ys_res
        except Exception:
            pass
        return w, h, r, xs, ys

    def refine_slsqp(w, h, r, xs, ys, max_iter=200, final_polish=False):
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
            xs_arr = v[2:2+N]
            ys_arr = v[2+N:2+2*N]
            cons = []
            cons.append({'type': 'ineq', 'fun': lambda v: v[0] - 0.2})
            cons.append({'type': 'ineq', 'fun': lambda v: 2.0 - v[0] - 0.2})
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2+i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[0] - v[2+i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2+N+i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: (2.0 - v[0]) - v[2+N+i] - v[1]})
            # Squared distance constraints (smoother)
            for i, j in PAIRS:
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: ((v[2+i]-v[2+j])**2 + (v[2+N+i]-v[2+N+j])**2) - 4*(v[1]**2)
                })
            return cons

        if final_polish:
            bounds = [(0.15, 1.85), (0.005, 1.0)] + [(-1.0, 3.0)] * (2*N)
            ftol = 1e-12
        else:
            bounds = [(0.2, 1.8), (0.01, 1.0)] + [(0.0, 2.0)] * (2*N)
            ftol = 1e-9

        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=constraints,
                           options={'maxiter': max_iter, 'ftol': ftol, 'disp': False})
            if res.success or res.fun < 0:
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
                        for i, j in PAIRS:
                            dist_sq = (xs_res[i]-xs_res[j])**2 + (ys_res[i]-ys_res[j])**2
                            if dist_sq < 4*(r_res**2) - 1e-7:
                                valid = False
                                break
                if valid:
                    return w_res, h_res, r_res, xs_res, ys_res
        except Exception:
            pass
        return w, h, r, xs, ys

    best_r = 0
    best_config = None

    configs = []
    configs.append(pack_hexagonal([7, 7, 7]))
    configs.append(pack_hexagonal([4, 5, 4, 4, 4]))
    configs.append(pack_hexagonal([3, 4, 4, 4, 4, 2]))
    configs.append(pack_hexagonal([5, 5, 5, 3, 3]))
    configs.append(pack_hexagonal([6, 6, 5, 4]))
    configs.append(pack_grid(5, 5))
    configs.append(pack_grid(3, 7))
    configs.append(pack_grid(4, 6))
    configs.append(pack_grid(7, 3))
    configs.append(pack_grid(6, 4))

    for _ in range(6):
        if check_time():
            break
        w = 1.0
        h = 1.0
        r = 0.08
        xs = np.random.uniform(r, w-r, N)
        ys = np.random.uniform(r, h-r, N)
        configs.append((w, h, r, xs, ys))

    valid_configs = [c for c in configs if c is not None]

    for cfg in valid_configs:
        if check_time():
            break
        w, h, r, xs, ys = cfg

        w_phys, h_phys, r_phys, xs_phys, ys_phys = physics_simulate(xs, ys, w, h, r)
        if r_phys > best_r:
            best_r = r_phys
            best_config = (w_phys, h_phys, r_phys, xs_phys, ys_phys)

        if not check_time():
            w_smooth, h_smooth, r_smooth, xs_smooth, ys_smooth = refine_smooth(w_phys, h_phys, r_phys, xs_phys, ys_phys)
            if r_smooth > best_r:
                best_r = r_smooth
                best_config = (w_smooth, h_smooth, r_smooth, xs_smooth, ys_smooth)

        if not check_time():
            cur_w, cur_h, cur_r, cur_xs, cur_ys = best_config if best_config is not None else (w_phys, h_phys, r_phys, xs_phys, ys_phys)
            w_slsqp, h_slsqp, r_slsqp, xs_slsqp, ys_slsqp = refine_slsqp(cur_w, cur_h, cur_r, cur_xs, cur_ys, max_iter=200)
            if r_slsqp > best_r:
                best_r = r_slsqp
                best_config = (w_slsqp, h_slsqp, r_slsqp, xs_slsqp, ys_slsqp)

    if best_config is not None and not check_time():
        w, h, r, xs, ys = best_config
        w_s, h_s, r_s, xs_s, ys_s = refine_smooth(w, h, r, xs, ys, max_iter=600)
        if r_s > r:
            w, h, r, xs, ys = w_s, h_s, r_s, xs_s, ys_s
        w_final, h_final, r_final, xs_final, ys_final = refine_slsqp(w, h, r, xs, ys, max_iter=500, final_polish=True)
        if r_final > best_r:
            best_r = r_final
            best_config = (w_final, h_final, r_final, xs_final, ys_final)

        if not check_time():
            w_s2, h_s2, r_s2, xs_s2, ys_s2 = refine_smooth(w_final, h_final, r_final, xs_final, ys_final, max_iter=400)
            if r_s2 > best_r:
                best_r = r_s2
                best_config = (w_s2, h_s2, r_s2, xs_s2, ys_s2)

    if best_config is None:
        w, h, r, xs, ys = valid_configs[0]
        return np.column_stack([xs, ys, np.full(N, r)])

    w, h, r, xs, ys = best_config
    r_final = r * (1 - 1e-6)
    return np.column_stack([xs, ys, np.full(N, r_final)])
