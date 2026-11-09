# EVOLVE-BLOCK-START
import numpy as np
import time

TOL = 1e-7


def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares with pairwise disjoint interiors inside the
    unit square [0,1]^2 so as to MAXIMIZE the sum of their side lengths.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    """
    t0 = time.time()
    DEADLINE = t0 + 590.0
    n = 10

    def _is_feasible(centers, sides, tol=TOL):
        cx = centers[:, 0]
        cy = centers[:, 1]
        s = sides
        if np.any(~np.isfinite(centers)) or np.any(~np.isfinite(s)):
            return False
        if np.any(s <= 0):
            return False
        for i in range(n):
            if cx[i] - s[i] / 2.0 < -tol or cx[i] + s[i] / 2.0 > 1.0 + tol:
                return False
            if cy[i] - s[i] / 2.0 < -tol or cy[i] + s[i] / 2.0 > 1.0 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                t = (s[i] + s[j]) / 2.0 - tol
                if abs(cx[i] - cx[j]) < t and abs(cy[i] - cy[j]) < t:
                    return False
        return True

    def _penalty(centers, sides, tol=TOL):
        cx = centers[:, 0]
        cy = centers[:, 1]
        s = sides
        pen = 0.0
        for i in range(n):
            lo = min(cx[i] - s[i] / 2.0, cy[i] - s[i] / 2.0)
            hi = max(cx[i] + s[i] / 2.0, cy[i] + s[i] / 2.0)
            if lo < -tol:
                pen += (lo + tol) ** 2
            if hi > 1.0 + tol:
                pen += (hi - 1.0 - tol) ** 2
        for i in range(n):
            for j in range(i + 1, n):
                t = (s[i] + s[j]) / 2.0 - tol
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                if dx < t and dy < t:
                    pen += (t - dx) ** 2 + (t - dy) ** 2
        return pen

    def _objective(params):
        s = np.clip(params[:n], 1e-4, 0.99)
        cx = np.clip(params[n:2 * n], 0.0, 1.0)
        cy = np.clip(params[2 * n:3 * n], 0.0, 1.0)
        centers = np.stack([cx, cy], axis=1)
        pen = _penalty(centers, s)
        return -np.sum(s) + 100.0 * pen

    def _local_search(centers, sides, max_iters=2000, step_init=0.02, step_min=1e-6):
        best_c = centers.copy()
        best_s = sides.copy()
        best_score = np.sum(sides)
        step = step_init
        no_improve = 0

        for it in range(max_iters):
            if time.time() > DEADLINE - 30:
                break
            if no_improve > 100:
                break
            improved = False
            for _ in range(5):
                trial_c = best_c.copy()
                trial_s = best_s.copy()
                idx = np.random.randint(0, n, size=np.random.randint(1, 4))
                for i in idx:
                    trial_c[i, 0] += np.random.uniform(-step, step)
                    trial_c[i, 1] += np.random.uniform(-step, step)
                    trial_s[i] += np.random.uniform(-step * 0.5, step * 0.5)
                trial_s = np.clip(trial_s, 1e-4, 0.99)
                trial_c[:, 0] = np.clip(trial_c[:, 0], 0.0, 1.0)
                trial_c[:, 1] = np.clip(trial_c[:, 1], 0.0, 1.0)

                if _is_feasible(trial_c, trial_s):
                    new_score = np.sum(trial_s)
                    if new_score > best_score + 1e-10:
                        best_c = trial_c
                        best_s = trial_s
                        best_score = new_score
                        improved = True
                        no_improve = 0
                    else:
                        no_improve += 1

            if not improved:
                step *= 0.9
                if step < step_min:
                    break

        return best_c, best_s, best_score

    def _coordinate_descent(centers, sides, max_iters=5000):
        best_c = centers.copy()
        best_s = sides.copy()
        best_score = np.sum(sides)
        step = 0.01

        for it in range(max_iters):
            if time.time() > DEADLINE - 30:
                break
            improved = False
            for i in range(n):
                for delta in [step, -step]:
                    trial_s = best_s.copy()
                    trial_s[i] += delta
                    if trial_s[i] < 1e-4 or trial_s[i] > 0.99:
                        continue
                    if _is_feasible(best_c, trial_s):
                        new_score = np.sum(trial_s)
                        if new_score > best_score + 1e-10:
                            best_s = trial_s
                            best_score = new_score
                            improved = True
                for delta in [step, -step]:
                    trial_c = best_c.copy()
                    trial_c[i, 0] += delta
                    if trial_c[i, 0] < 0 or trial_c[i, 0] > 1:
                        continue
                    if _is_feasible(trial_c, best_s):
                        new_score = np.sum(best_s)
                        if new_score > best_score + 1e-10:
                            best_c = trial_c
                            improved = True
                for delta in [step, -step]:
                    trial_c = best_c.copy()
                    trial_c[i, 1] += delta
                    if trial_c[i, 1] < 0 or trial_c[i, 1] > 1:
                        continue
                    if _is_feasible(trial_c, best_s):
                        new_score = np.sum(best_s)
                        if new_score > best_score + 1e-10:
                            best_c = trial_c
                            improved = True

            if not improved:
                step *= 0.9
                if step < 1e-6:
                    break

        return best_c, best_s, best_score

    # ---- Analytic starting layouts ----

    # Layout A: 4x4 grid, 10 squares of side 1/4
    centersA = np.zeros((n, 2))
    sidesA = np.zeros(n)
    for i in range(n):
        col = i % 4
        row = i // 4
        side = 1.0 / 4.0
        centersA[i, 0] = (col + 0.5) * side
        centersA[i, 1] = (row + 0.5) * side
        sidesA[i] = side

    # Layout B: 2 rows of 4 (side 1/4) + 2 rows of 2 (side 1/2)
    centersB = np.zeros((n, 2))
    sidesB = np.zeros(n)
    for i in range(4):
        col = i
        side = 1.0 / 4.0
        centersB[i, 0] = (col + 0.5) * side
        centersB[i, 1] = 0.5 * side
        sidesB[i] = side
    for i in range(4):
        col = i
        side = 1.0 / 4.0
        centersB[4 + i, 0] = (col + 0.5) * side
        centersB[4 + i, 1] = 1.5 * side
        sidesB[4 + i] = side
    for i in range(2):
        col = i
        side = 1.0 / 2.0
        centersB[8 + i, 0] = (col + 0.5) * side
        centersB[8 + i, 1] = 0.5 + 0.5 * side
        sidesB[8 + i] = side

    # Layout D: 5 cols x 2 rows, 10 squares of side 1/5
    centersD = np.zeros((n, 2))
    sidesD = np.zeros(n)
    for i in range(n):
        col = i % 5
        row = i // 5
        side = 1.0 / 5.0
        centersD[i, 0] = (col + 0.5) * side
        centersD[i, 1] = (row + 0.5) * side
        sidesD[i] = side

    # Layout F: Heterogeneous packing with larger squares
    centersF = np.zeros((n, 2))
    sidesF = np.zeros(n)
    sidesF[0] = 0.4
    centersF[0, 0] = 0.2
    centersF[0, 1] = 0.2
    sidesF[1] = 0.4
    centersF[1, 0] = 0.8
    centersF[1, 1] = 0.2
    sidesF[2] = 0.3
    centersF[2, 0] = 0.5
    centersF[2, 1] = 0.5
    sidesF[3] = 0.3
    centersF[3, 0] = 0.5
    centersF[3, 1] = 0.8
    for i in range(6):
        sidesF[4 + i] = 0.2
        centersF[4 + i, 0] = np.random.uniform(0.1, 0.9)
        centersF[4 + i, 1] = np.random.uniform(0.1, 0.9)

    # Layout G: Another heterogeneous packing
    centersG = np.zeros((n, 2))
    sidesG = np.zeros(n)
    sidesG[0] = 0.5
    centersG[0, 0] = 0.25
    centersG[0, 1] = 0.25
    sidesG[1] = 0.3
    centersG[1, 0] = 0.7
    centersG[1, 1] = 0.3
    sidesG[2] = 0.3
    centersG[2, 0] = 0.7
    centersG[2, 1] = 0.7
    for i in range(7):
        sidesG[3 + i] = 0.2
        centersG[3 + i, 0] = np.random.uniform(0.1, 0.9)
        centersG[3 + i, 1] = np.random.uniform(0.1, 0.9)

    # Collect all feasible initial layouts
    initial_layouts = []
    for c, s in [(centersA, sidesA), (centersB, sidesB), (centersD, sidesD), (centersF, sidesF), (centersG, sidesG)]:
        if _is_feasible(c, s):
            initial_layouts.append((c.copy(), s.copy()))

    # If no feasible layout, use a very conservative one
    if not initial_layouts:
        c = np.zeros((n, 2))
        s = np.zeros(n)
        for i in range(n):
            col = i % 5
            row = i // 5
            side = 1.0 / 10.0
            c[i, 0] = (col + 0.5) * side
            c[i, 1] = (row + 0.5) * side
            s[i] = side
        initial_layouts.append((c, s))

    best_centers = initial_layouts[0][0].copy()
    best_sides = initial_layouts[0][1].copy()
    best_score = np.sum(best_sides)

    # Run local search from each initial layout
    rng = np.random.default_rng(42)

    for c_init, s_init in initial_layouts:
        if time.time() > DEADLINE - 60:
            break

        # Local search
        c_ls, s_ls, score_ls = _local_search(c_init, s_init, max_iters=3000, step_init=0.03)
        if score_ls > best_score:
            best_score = score_ls
            best_centers = c_ls
            best_sides = s_ls

        # Coordinate descent
        c_cd, s_cd, score_cd = _coordinate_descent(c_ls, s_ls, max_iters=8000)
        if score_cd > best_score:
            best_score = score_cd
            best_centers = c_cd
            best_sides = s_cd

        # More local search from CD result
        c_ls2, s_ls2, score_ls2 = _local_search(c_cd, s_cd, max_iters=2000, step_init=0.01)
        if score_ls2 > best_score:
            best_score = score_ls2
            best_centers = c_ls2
            best_sides = s_ls2

    # Try scipy minimize from best result as a final polish
    if time.time() < DEADLINE - 30:
        try:
            from scipy.optimize import minimize
            x0 = np.concatenate([best_sides, best_centers[:, 0], best_centers[:, 1]])
            bounds = [(1e-4, 0.99)] * n + [(0.0, 1.0)] * (2 * n)
            res = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < 0:
                s_opt = np.clip(res.x[:n], 1e-4, 0.99)
                cx_opt = np.clip(res.x[n:2 * n], 0.0, 1.0)
                cy_opt = np.clip(res.x[2 * n:3 * n], 0.0, 1.0)
                c_opt = np.stack([cx_opt, cy_opt], axis=1)
                if _is_feasible(c_opt, s_opt):
                    score = np.sum(s_opt)
                    if score > best_score:
                        best_score = score
                        best_centers = c_opt
                        best_sides = s_opt
        except Exception:
            pass

    # Final check
    if not _is_feasible(best_centers, best_sides):
        best_centers = initial_layouts[0][0]
        best_sides = initial_layouts[0][1]

    return best_centers.astype(np.float64), best_sides.astype(np.float64)


# EVOLVE-BLOCK-END
