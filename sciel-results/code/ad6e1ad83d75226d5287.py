# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = start_time + 620  # leave 10s margin before 630s kill

    def is_feasible(c, s):
        """Check if configuration (centers, sides) is feasible."""
        if not np.all(np.isfinite(c)) or not np.all(np.isfinite(s)):
            return False
        if np.any(s <= 0):
            return False
        for i in range(n):
            cx, cy = c[i]
            si = s[i]
            if cx - si / 2 < -tol or cx + si / 2 > 1 + tol:
                return False
            if cy - si / 2 < -tol or cy + si / 2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                t = (s[i] + s[j]) / 2 - tol
                if abs(c[i][0] - c[j][0]) < t and abs(c[i][1] - c[j][1]) < t:
                    return False
        return True

    def objective(c, s):
        return np.sum(s)

    def flatten(c, s):
        return np.concatenate([c.flatten(), s])

    def unflatten(x):
        c = x[:2 * n].reshape(n, 2)
        s = x[2 * n:]
        return c, s

    def smooth_penalty(x, weight=1e6, beta=80.0):
        """Smooth penalty using soft-min for boundary and soft-max for overlap."""
        c, s = unflatten(x)
        pen = 0.0
        for i in range(n):
            cx, cy = c[i]
            si = s[i]
            if si <= 0:
                pen += weight * (1.0 - si)
            # Boundary violations (smooth via soft-min)
            for val in [-tol - (cx - si / 2),
                        (cx + si / 2) - (1 + tol),
                        -tol - (cy - si / 2),
                        (cy + si / 2) - (1 + tol)]:
                if val > -1e-5:
                    pen += weight * (1.0 / beta) * np.log1p(np.exp(beta * val))
        # Overlap violations using smooth max of the two axis violations
        for i in range(n):
            for j in range(i + 1, n):
                t = (s[i] + s[j]) / 2 - tol
                dx = c[i][0] - c[j][0]
                dy = c[i][1] - c[j][1]
                a = t - abs(dx)
                b = t - abs(dy)
                if a > -1e-5 or b > -1e-5:
                    m = max(a, b)
                    if m > 0:
                        sm = m + (1.0 / beta) * np.log(np.exp(beta * (a - m)) + np.exp(beta * (b - m)))
                        pen += weight * sm
        return pen

    def combined_objective(x, weight=1e6):
        c, s = unflatten(x)
        return -np.sum(s) + smooth_penalty(x, weight)

    def make_2big_8small():
        centers = np.array([
            [0.25, 0.25], [0.75, 0.25],
            [0.125, 0.875], [0.375, 0.875], [0.625, 0.875], [0.875, 0.875],
            [0.125, 0.625], [0.375, 0.625], [0.625, 0.625], [0.875, 0.625],
        ], dtype=np.float64)
        sides = np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        return centers, sides

    def make_3big_7small():
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        s_big = 1.0 / 3
        for i in range(3):
            centers[i] = [(i + 0.5) * s_big, 1.0 - s_big / 2]
            sides[i] = s_big
        s_small = 0.2
        for i in range(4):
            centers[3 + i] = [(i + 0.5) * s_small, 0.55]
            sides[3 + i] = s_small
        for i in range(3):
            centers[7 + i] = [(i + 0.5) * s_small + 0.1, 0.25]
            sides[7 + i] = s_small
        return centers, sides

    def make_grid():
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        return centers, sides

    def make_4big_6small():
        """4 squares of side 0.4 in corners, 6 small squares in middle."""
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        s_big = 0.4
        corners = [[s_big/2, s_big/2], [1-s_big/2, s_big/2], [s_big/2, 1-s_big/2], [1-s_big/2, 1-s_big/2]]
        for i, (cx, cy) in enumerate(corners):
            centers[i] = [cx, cy]
            sides[i] = s_big
        # 6 small squares in the middle region
        # Middle region is [0.4, 0.6] x [0.4, 0.6], size 0.2x0.2
        # We can fit 2x3 grid of small squares with side 0.1
        s_small = 0.1
        for i in range(6):
            col = i % 2
            row = i // 2
            cx = 0.4 + 0.05 + col * 0.2
            cy = 0.4 + 0.05 + row * 0.1
            centers[4 + i] = [cx, cy]
            sides[4 + i] = s_small
        return centers, sides

    def nelder_mead_optimize(x0, max_iter=3000, time_limit=25):
        start = time.time()

        def obj(x):
            if time.time() - start > time_limit:
                return 1e10
            return combined_objective(x)

        try:
            result = minimize(obj, x0, method='Nelder-Mead',
                              options={'maxiter': max_iter, 'xatol': 1e-9, 'fatol': 1e-9,
                                       'adaptive': True})
            return result.x
        except Exception:
            return x0

    def lbfgsb_optimize(x0, max_iter=500, time_limit=25):
        start = time.time()
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
            bounds.append((0.01, 1.0))

        def obj(x):
            if time.time() - start > time_limit:
                return 1e10
            return combined_objective(x)

        try:
            result = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': max_iter, 'ftol': 1e-9, 'gtol': 1e-7,
                                       'maxfun': 5000})
            return result.x
        except Exception:
            return x0

    def coordinate_ascent(c, s, max_rounds=400):
        """
        Coordinate ascent: for each square, try to increase its side length
        by repositioning within feasible space.
        """
        c = c.copy()
        s = s.copy()
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            order = np.random.permutation(n)
            for i in order:
                lo_s = s[i]
                hi_s = 1.0 + tol

                max_s_boundary = min(2 * c[i][0], 2 * (1 - c[i][0]),
                                     2 * c[i][1], 2 * (1 - c[i][1])) + 2 * tol
                hi_s = min(hi_s, max_s_boundary)
                if hi_s <= lo_s + 1e-12:
                    continue

                best_s = lo_s
                best_c = c[i].copy()

                for _ in range(50):
                    mid_s = (lo_s + hi_s) / 2
                    if mid_s <= best_s + 1e-13:
                        break

                    found = False
                    new_cx, new_cy = c[i][0], c[i][1]

                    ok = True
                    for j in range(n):
                        if j == i:
                            continue
                        t = (mid_s + s[j]) / 2 - tol
                        if abs(new_cx - c[j][0]) < t and abs(new_cy - c[j][1]) < t:
                            ok = False
                            break
                    if ok and new_cx - mid_s / 2 >= -tol and new_cx + mid_s / 2 <= 1 + tol \
                            and new_cy - mid_s / 2 >= -tol and new_cy + mid_s / 2 <= 1 + tol:
                        found = True
                    else:
                        # Systematic grid search
                        grid_points = []
                        for gx in [mid_s / 2, 1 - mid_s / 2, 0.25, 0.5, 0.75, 0.1, 0.9, 0.15, 0.85]:
                            for gy in [mid_s / 2, 1 - mid_s / 2, 0.25, 0.5, 0.75, 0.1, 0.9, 0.15, 0.85]:
                                if mid_s / 2 - 1e-9 <= gx <= 1 - mid_s / 2 + 1e-9 and \
                                   mid_s / 2 - 1e-9 <= gy <= 1 - mid_s / 2 + 1e-9:
                                    grid_points.append((gx, gy))
                        # Add corner-adjacent points
                        for gx in [mid_s / 2, 1 - mid_s / 2]:
                            for gy in [mid_s / 2, 1 - mid_s / 2]:
                                if (gx, gy) not in grid_points:
                                    grid_points.append((gx, gy))

                        for new_cx, new_cy in grid_points:
                            ok = True
                            for j in range(n):
                                if j == i:
                                    continue
                                t = (mid_s + s[j]) / 2 - tol
                                if abs(new_cx - c[j][0]) < t and abs(new_cy - c[j][1]) < t:
                                    ok = False
                                    break
                            if ok:
                                found = True
                                break

                        if not found:
                            for _ in range(200):
                                new_cx = np.random.uniform(mid_s / 2, 1 - mid_s / 2)
                                new_cy = np.random.uniform(mid_s / 2, 1 - mid_s / 2)
                                ok = True
                                for j in range(n):
                                    if j == i:
                                        continue
                                    t = (mid_s + s[j]) / 2 - tol
                                    if abs(new_cx - c[j][0]) < t and abs(new_cy - c[j][1]) < t:
                                        ok = False
                                        break
                                if ok:
                                    found = True
                                    break
                        if not found:
                            for dx in [0, 1]:
                                for dy in [0, 1]:
                                    new_cx = mid_s / 2 + dx * (1 - mid_s)
                                    new_cy = mid_s / 2 + dy * (1 - mid_s)
                                    ok = True
                                    for j in range(n):
                                        if j == i:
                                            continue
                                        t = (mid_s + s[j]) / 2 - tol
                                        if abs(new_cx - c[j][0]) < t and abs(new_cy - c[j][1]) < t:
                                            ok = False
                                            break
                                    if ok:
                                        found = True
                                        break
                                if found:
                                    break

                    if found:
                        best_s = mid_s
                        best_c = np.array([new_cx, new_cy])
                        lo_s = mid_s
                    else:
                        hi_s = mid_s

                if best_s > s[i] + 1e-10:
                    s[i] = best_s
                    c[i] = best_c
                    improved = True

            if not improved:
                break

        return c, s

    def local_search(c, s, max_iters=1500):
        """More aggressive local search with simulated annealing."""
        c = c.copy()
        s = s.copy()
        best_val = objective(c, s)
        best_c = c.copy()
        best_s = s.copy()
        temp = 0.05
        cooling = 0.995

        for it in range(max_iters):
            if time.time() > deadline:
                break
            temp *= cooling

            i = np.random.randint(n)
            old_c = c[i].copy()
            old_s = s[i]

            ds = np.random.uniform(-0.08, 0.12)
            new_s = max(0.01, min(1.0, old_s + ds))
            dcx = np.random.uniform(-0.15, 0.15)
            dcy = np.random.uniform(-0.15, 0.15)
            new_cx = np.clip(old_c[0] + dcx, new_s / 2, 1 - new_s / 2)
            new_cy = np.clip(old_c[1] + dcy, new_s / 2, 1 - new_s / 2)

            c[i] = [new_cx, new_cy]
            s[i] = new_s

            if is_feasible(c, s):
                new_val = objective(c, s)
                delta = new_val - best_val
                if delta > 0 or np.random.random() < np.exp(delta / max(temp, 1e-8)):
                    best_val = max(best_val, new_val)
                    best_c = c.copy()
                    best_s = s.copy()
                else:
                    c[i] = old_c
                    s[i] = old_s
            else:
                c[i] = old_c
                s[i] = old_s

        return best_c, best_s

    # Main optimization
    best_c = None
    best_s = None
    best_val = -1

    def try_config(centers, sides, label=""):
        nonlocal best_c, best_s, best_val
        if time.time() > deadline:
            return
        c_opt, s_opt = coordinate_ascent(centers, sides, max_rounds=200)
        if is_feasible(c_opt, s_opt):
            val = objective(c_opt, s_opt)
            if val > best_val:
                best_val = val
                best_c = c_opt.copy()
                best_s = s_opt.copy()
        x0 = flatten(c_opt, s_opt)
        x_nm = nelder_mead_optimize(x0, max_iter=3000, time_limit=20)
        c_nm, s_nm = unflatten(x_nm)
        if is_feasible(c_nm, s_nm):
            val = objective(c_nm, s_nm)
            if val > best_val:
                best_val = val
                best_c = c_nm.copy()
                best_s = s_nm.copy()
        x_lb = lbfgsb_optimize(x_nm, max_iter=400, time_limit=15)
        c_lb, s_lb = unflatten(x_lb)
        if is_feasible(c_lb, s_lb):
            val = objective(c_lb, s_lb)
            if val > best_val:
                best_val = val
                best_c = c_lb.copy()
                best_s = s_lb.copy()
        c_ca, s_ca = coordinate_ascent(c_lb, s_lb, max_rounds=150)
        if is_feasible(c_ca, s_ca):
            val = objective(c_ca, s_ca)
            if val > best_val:
                best_val = val
                best_c = c_ca.copy()
                best_s = s_ca.copy()

    # Try multiple initial configurations
    configs = [
        make_2big_8small(),
        make_3big_7small(),
        make_grid(),
        make_4big_6small(),
    ]

    for centers_init, sides_init in configs:
        try_config(centers_init, sides_init)

    # Random starts with optimization
    np.random.seed(42)
    for start in range(15):
        if time.time() > deadline - 60:
            break
        s_rand = np.random.uniform(0.08, 0.35, size=n)
        c_rand = np.zeros((n, 2))
        feasible = True
        for i in range(n):
            placed = False
            for _ in range(300):
                cx = np.random.uniform(s_rand[i] / 2, 1 - s_rand[i] / 2)
                cy = np.random.uniform(s_rand[i] / 2, 1 - s_rand[i] / 2)
                c_rand[i] = [cx, cy]
                ok = True
                for j in range(i):
                    t = (s_rand[i] + s_rand[j]) / 2 - tol
                    if abs(cx - c_rand[j][0]) < t and abs(cy - c_rand[j][1]) < t:
                        ok = False
                        break
                if ok:
                    placed = True
                    break
            if not placed:
                feasible = False
                break
        if not feasible or not is_feasible(c_rand, s_rand):
            continue
        try_config(c_rand, s_rand)

    # Final polish on the best solution
    if best_c is not None and best_s is not None:
        # High-precision Nelder-Mead polish
        x_best = flatten(best_c, best_s)
        x_polish = nelder_mead_optimize(x_best, max_iter=5000, time_limit=30)
        c_polish, s_polish = unflatten(x_polish)
        if is_feasible(c_polish, s_polish):
            val = objective(c_polish, s_polish)
            if val > best_val:
                best_val = val
                best_c = c_polish.copy()
                best_s = s_polish.copy()

        # L-BFGS-B polish
        x_best2 = flatten(best_c, best_s)
        x_polish2 = lbfgsb_optimize(x_best2, max_iter=600, time_limit=25)
        c_polish2, s_polish2 = unflatten(x_polish2)
        if is_feasible(c_polish2, s_polish2):
            val = objective(c_polish2, s_polish2)
            if val > best_val:
                best_val = val
                best_c = c_polish2.copy()
                best_s = s_polish2.copy()

        # Coordinate ascent final polish
        c_final, s_final = coordinate_ascent(best_c, best_s, max_rounds=300)
        if is_feasible(c_final, s_final):
            val = objective(c_final, s_final)
            if val > best_val:
                best_val = val
                best_c = c_final.copy()
                best_s = s_final.copy()

        # Local search final polish
        if time.time() < deadline - 30:
            c_ls, s_ls = local_search(best_c, best_s, max_iters=1000)
            if is_feasible(c_ls, s_ls):
                val = objective(c_ls, s_ls)
                if val > best_val:
                    best_val = val
                    best_c = c_ls.copy()
                    best_s = s_ls.copy()

    if best_c is None:
        best_c, best_s = make_2big_8small()

    if not is_feasible(best_c, best_s):
        best_c, best_s = make_2big_8small()

    best_s = np.maximum(best_s, 1e-9)

    return best_c, best_s


# EVOLVE-BLOCK-END
