import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s kill

    # ---- Helpers ----
    def to_flat(centers, sides):
        x = np.empty(3 * n)
        x[0::3] = centers[:, 0]
        x[1::3] = centers[:, 1]
        x[2::3] = sides
        return x

    def from_flat(x):
        cx = x[0::3].copy()
        cy = x[1::3].copy()
        s = np.maximum(x[2::3].copy(), 1e-8)
        centers = np.stack([cx, cy], axis=1)
        return centers, s

    # Smooth penalty: boundary + overlap (soft via smooth min)
    def smooth_penalty(x, beta=100.0):
        cx = x[0::3]
        cy = x[1::3]
        s = np.maximum(x[2::3], 1e-8)
        pen = 0.0

        # Boundary
        tol_b = 1e-7
        for i in range(n):
            lo = -tol_b
            hi = 1.0 + tol_b
            # x-direction
            vl = (cx[i] - s[i]/2) - lo
            vh = hi - (cx[i] + s[i]/2)
            if vl < 0:
                pen += 500.0 * vl
            if vh < 0:
                pen += 500.0 * vh
            # y-direction
            vl = (cy[i] - s[i]/2) - lo
            vh = hi - (cy[i] + s[i]/2)
            if vl < 0:
                pen += 500.0 * vl
            if vh < 0:
                pen += 500.0 * vh

        # Overlap: for each pair, penetration in x and y
        # Overlap if dx < (s_i+s_j)/2 and dy < (s_i+s_j)/2
        # Penalty: product of penetrations, scaled
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j]) / 2
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                px = t - dx  # positive if x-overlap
                py = t - dy  # positive if y-overlap
                # Smooth: use max(0, px) * max(0, py) with softplus
                # For differentiability, use: (px + sqrt(px^2+eps))/2 * (py + sqrt(py^2+eps))/2
                eps = 1e-8
                spx = (px + np.sqrt(px*px + eps)) / 2.0
                spy = (py + np.sqrt(py*py + eps)) / 2.0
                pen += 200.0 * spx * spy

        return -np.sum(s) + pen

    def objective(x):
        return smooth_penalty(x)

    def objective_grad(x):
        # Numerical gradient (central difference)
        eps = 1e-6
        grad = np.empty_like(x)
        f0 = objective(x)
        for i in range(len(x)):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            grad[i] = (objective(xp) - objective(xm)) / (2*eps)
        return grad

    def clamp_and_fix(centers, sides):
        s = np.maximum(sides, 1e-8).copy()
        cx = centers[:, 0].copy()
        cy = centers[:, 1].copy()
        for i in range(n):
            cx[i] = np.clip(cx[i], s[i]/2, 1.0 - s[i]/2)
            cy[i] = np.clip(cy[i], s[i]/2, 1.0 - s[i]/2)
        centers_fixed = np.stack([cx, cy], axis=1)

        tol = 1e-7
        for i in range(n):
            if centers_fixed[i, 0] - s[i]/2 < -tol or centers_fixed[i, 0] + s[i]/2 > 1.0 + tol:
                return False, centers_fixed, s, 0.0
            if centers_fixed[i, 1] - s[i]/2 < -tol or centers_fixed[i, 1] + s[i]/2 > 1.0 + tol:
                return False, centers_fixed, s, 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers_fixed[i, 0] - centers_fixed[j, 0])
                dy = abs(centers_fixed[i, 1] - centers_fixed[j, 1])
                t = (s[i] + s[j])/2 - tol
                if dx < t and dy < t:
                    return False, centers_fixed, s, 0.0
        score = np.sum(s)
        return True, centers_fixed, s, score

    def local_search(x_flat, max_iter=3000, method='Nelder-Mead', tol_x=1e-7, tol_f=1e-7):
        best_x = x_flat.copy()
        best_f = objective(x_flat)
        for _ in range(max_iter):
            if time.time() > deadline - 3:
                break
            if method == 'Nelder-Mead':
                res = minimize(objective, best_x, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': tol_x, 'fatol': tol_f})
            else:
                res = minimize(objective, best_x, method='L-BFGS-B',
                               jac=objective_grad,
                               options={'maxiter': 300, 'gtol': 1e-8, 'ftol': 1e-9})
            f_new = res.fun
            if f_new < best_f - 1e-10:
                best_f = f_new
                best_x = res.x
            else:
                # Try small perturbation
                if np.random.rand() < 0.25:
                    best_x = best_x + np.random.normal(0, 0.005, len(x_flat))
                else:
                    break
        return best_x, -best_f if best_f < 0 else best_f

    def multi_start_local_search(centers_init, sides_init, max_iter=1500):
        c = centers_init.copy()
        s = np.maximum(sides_init, 1e-8).copy()
        for i in range(n):
            c[i, 0] = np.clip(c[i, 0], s[i]/2, 1.0 - s[i]/2)
            c[i, 1] = np.clip(c[i, 1], s[i]/2, 1.0 - s[i]/2)
        x_flat = to_flat(c, s)
        # Try both methods
        x_res1, f1 = local_search(x_flat, max_iter=max_iter, method='Nelder-Mead', tol_x=1e-8, tol_f=1e-9)
        x_res2, f2 = local_search(x_res1, max_iter=max_iter//2, method='L-BFGS-B', tol_x=1e-9, tol_f=1e-10)
        return x_res2

    # ---- Initial layouts ----
    layouts = []

    # Layout 1: 2-4-4 (original)
    c1 = np.array([
        [0.25, 0.25], [0.75, 0.25],
        [0.125, 0.625], [0.375, 0.625], [0.625, 0.625], [0.875, 0.625],
        [0.125, 0.875], [0.375, 0.875], [0.625, 0.875], [0.875, 0.875]
    ])
    s1 = np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
    layouts.append((c1, s1))

    # Layout 2: 2-3-5
    c2 = np.array([
        [0.25, 0.25], [0.75, 0.25],
        [0.2, 0.65], [0.5, 0.65], [0.8, 0.65],
        [0.1, 0.9], [0.3, 0.9], [0.5, 0.9], [0.7, 0.9], [0.9, 0.9]
    ])
    s2 = np.array([0.5, 0.5, 0.3, 0.3, 0.3, 0.18, 0.18, 0.18, 0.18, 0.18])
    layouts.append((c2, s2))

    # Layout 3: 3-3-4
    c3 = np.array([
        [0.167, 0.2], [0.5, 0.2], [0.833, 0.2],
        [0.2, 0.55], [0.5, 0.55], [0.8, 0.55],
        [0.15, 0.85], [0.4, 0.85], [0.65, 0.85], [0.9, 0.85]
    ])
    s3 = np.array([0.33, 0.33, 0.33, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2])
    layouts.append((c3, s3))

    # Layout 4: 4-3-3
    c4 = np.array([
        [0.125, 0.25], [0.375, 0.25], [0.625, 0.25], [0.875, 0.25],
        [0.2, 0.65], [0.5, 0.65], [0.8, 0.65],
        [0.2, 0.9], [0.5, 0.9], [0.8, 0.9]
    ])
    s4 = np.array([0.25, 0.25, 0.25, 0.25, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2])
    layouts.append((c4, s4))

    # Random layouts
    np.random.seed(42)
    for trial in range(6):
        c_rand = np.random.uniform(0.15, 0.85, (n, 2))
        s_rand = np.random.uniform(0.1, 0.35, n)
        layouts.append((c_rand, s_rand))

    # ---- Optimization loop ----
    best_centers, best_sides = c1.copy(), s1.copy()
    best_score = 0.0

    for idx, (c_init, s_init) in enumerate(layouts):
        if time.time() > deadline - 15:
            break
        t_start = time.time()
        x_res = multi_start_local_search(c_init, s_init, max_iter=1200)
        centers_res, sides_res = from_flat(x_res)
        valid, c_final, s_final, score_final = clamp_and_fix(centers_res, sides_res)
        if valid and score_final > best_score:
            best_score = score_final
            best_centers = c_final
            best_sides = s_final
        t_elapsed = time.time() - t_start

    # Final high-precision polish
    if time.time() < deadline - 30:
        x_flat = to_flat(best_centers, best_sides)
        x_res, f_res = local_search(x_flat, max_iter=5000, method='Nelder-Mead', tol_x=1e-9, tol_f=1e-10)
        x_res, f_res = local_search(x_res, max_iter=2000, method='L-BFGS-B', tol_x=1e-10, tol_f=1e-11)
        centers_res, sides_res = from_flat(x_res)
        valid, c_final, s_final, score_final = clamp_and_fix(centers_res, sides_res)
        if valid and score_final > best_score:
            best_score = score_final
            best_centers = c_final
            best_sides = s_final

    # Final validation
    valid, best_centers, best_sides, final_score = clamp_and_fix(best_centers, best_sides)
    if not valid:
        return c1, s1

    return best_centers, best_sides
