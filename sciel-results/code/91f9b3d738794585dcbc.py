# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    deadline = time.time() + 620  # leave 10s margin before 630s kill

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

    def penalty(x, weight=1e6):
        c, s = unflatten(x)
        pen = 0.0
        for i in range(n):
            cx, cy = c[i]
            si = s[i]
            if si <= 0:
                pen += weight * (1.0 - si)
            if cx - si / 2 < -tol:
                pen += weight * (-tol - (cx - si / 2))
            if cx + si / 2 > 1 + tol:
                pen += weight * ((cx + si / 2) - (1 + tol))
            if cy - si / 2 < -tol:
                pen += weight * (-tol - (cy - si / 2))
            if cy + si / 2 > 1 + tol:
                pen += weight * ((cy + si / 2) - (1 + tol))
        for i in range(n):
            for j in range(i + 1, n):
                t = (s[i] + s[j]) / 2 - tol
                dx = c[i][0] - c[j][0]
                dy = c[i][1] - c[j][1]
                if abs(dx) < t and abs(dy) < t:
                    pen += weight * ((t - abs(dx)) + (t - abs(dy)))
        return pen

    def combined_objective(x, weight=1e6):
        c, s = unflatten(x)
        return -np.sum(s) + penalty(x, weight)

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

    def nelder_mead_optimize(x0, max_iter=3000, time_limit=25):
        start = time.time()

        def obj(x):
            if time.time() - start > time_limit:
                return 1e10
            return combined_objective(x)

        try:
            result = minimize(obj, x0, method='Nelder-Mead',
                              options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8})
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
                              options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6})
            return result.x
        except Exception:
            return x0

    def coordinate_ascent(c, s, max_rounds=300):
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

                for _ in range(40):
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
                        for _ in range(100):
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

    def local_search(c, s, max_iters=1000):
        """More aggressive local search."""
        c = c.copy()
        s = s.copy()
        best_val = objective(c, s)

        for it in range(max_iters):
            if time.time() > deadline:
                break

            i = np.random.randint(n)
            old_c = c[i].copy()
            old_s = s[i]

            ds = np.random.uniform(-0.05, 0.1)
            new_s = max(0.01, min(1.0, old_s + ds))
            dcx = np.random.uniform(-0.1, 0.1)
            dcy = np.random.uniform(-0.1, 0.1)
            new_cx = np.clip(old_c[0] + dcx, new_s / 2, 1 - new_s / 2)
            new_cy = np.clip(old_c[1] + dcy, new_s / 2, 1 - new_s / 2)

            c[i] = [new_cx, new_cy]
            s[i] = new_s

            if is_feasible(c, s):
                new_val = objective(c, s)
                if new_val > best_val:
                    best_val = new_val
                else:
                    c[i] = old_c
                    s[i] = old_s
            else:
                c[i] = old_c
                s[i] = old_s

        return c, s

    def guillotine_search():
        """
        Search over guillotine cut trees.
        """
        best_c_g = None
        best_s_g = None
        best_val_g = -1

        # Define a few tree topologies as lists of (split_dir, left_leaf_indices, right_leaf_indices)
        # This is complex to represent generally. 
        # Instead, we will use a recursive function that optimizes split ratios for a fixed structure.
        
        # Structure 1: 2 big, 8 small (matches make_2big_8small)
        # This is already handled by the initial guess.
        
        # Let's try a generic random guillotine search.
        # We will generate random binary trees and optimize.
        
        def eval_guillotine(splits, assignment):
            """
            splits: list of (direction, ratio) for internal nodes.
            assignment: list of square indices for leaves.
            Returns centers and sides.
            """
            # This is a placeholder for the actual logic.
            # For now, we rely on the existing heuristics.
            return None, None

        return best_c_g, best_s_g, best_val_g

    # Main optimization
    best_c = None
    best_s = None
    best_val = -1

    # Start with analytical construction
    centers_init, sides_init = make_2big_8small()

    # First pass: coordinate ascent from analytical start
    c_opt, s_opt = coordinate_ascent(centers_init, sides_init, max_rounds=300)
    if is_feasible(c_opt, s_opt):
        val = objective(c_opt, s_opt)
        if val > best_val:
            best_val = val
            best_c = c_opt.copy()
            best_s = s_opt.copy()

    # Second pass: Nelder-Mead from the coordinate-ascent result
    x0 = flatten(c_opt, s_opt)
    x_nm = nelder_mead_optimize(x0, max_iter=3000, time_limit=20)
    c_nm, s_nm = unflatten(x_nm)
    if is_feasible(c_nm, s_nm):
        val = objective(c_nm, s_nm)
        if val > best_val:
            best_val = val
            best_c = c_nm.copy()
            best_s = s_nm.copy()

    # Third pass: L-BFGS-B from Nelder-Mead result
    x_lb = lbfgsb_optimize(x_nm, max_iter=500, time_limit=20)
    c_lb, s_lb = unflatten(x_lb)
    if is_feasible(c_lb, s_lb):
        val = objective(c_lb, s_lb)
        if val > best_val:
            best_val = val
            best_c = c_lb.copy()
            best_s = s_lb.copy()

    # Fourth pass: coordinate ascent from L-BFGS-B result for fine-tuning
    c_ca, s_ca = coordinate_ascent(c_lb, s_lb, max_rounds=200)
    if is_feasible(c_ca, s_ca):
        val = objective(c_ca, s_ca)
        if val > best_val:
            best_val = val
            best_c = c_ca.copy()
            best_s = s_ca.copy()

    # Try 3big_7small configuration
    centers_3b, sides_3b = make_3big_7small()
    c_3b, s_3b = coordinate_ascent(centers_3b, sides_3b, max_rounds=200)
    if is_feasible(c_3b, s_3b):
        val = objective(c_3b, s_3b)
        if val > best_val:
            best_val = val
            best_c = c_3b.copy()
            best_s = s_3b.copy()
        x_3b = flatten(c_3b, s_3b)
        x_3b_nm = nelder_mead_optimize(x_3b, max_iter=2000, time_limit=15)
        c_3b_nm, s_3b_nm = unflatten(x_3b_nm)
        if is_feasible(c_3b_nm, s_3b_nm):
            val = objective(c_3b_nm, s_3b_nm)
            if val > best_val:
                best_val = val
                best_c = c_3b_nm.copy()
                best_s = s_3b_nm.copy()

    # Try grid configuration
    centers_grid, sides_grid = make_grid()
    c_grid, s_grid = coordinate_ascent(centers_grid, sides_grid, max_rounds=200)
    if is_feasible(c_grid, s_grid):
        val = objective(c_grid, s_grid)
        if val > best_val:
            best_val = val
            best_c = c_grid.copy()
            best_s = s_grid.copy()

    # Random starts with Nelder-Mead
    np.random.seed(42)
    for start in range(20):
        if time.time() > deadline:
            break
        # Generate random feasible config
        s_rand = np.random.uniform(0.05, 0.35, size=n)
        c_rand = np.zeros((n, 2))
        feasible = True
        for i in range(n):
            placed = False
            for _ in range(200):
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
        x_rand = flatten(c_rand, s_rand)
        x_rand_nm = nelder_mead_optimize(x_rand, max_iter=2000, time_limit=15)
        c_rand_nm, s_rand_nm = unflatten(x_rand_nm)
        if is_feasible(c_rand_nm, s_rand_nm):
            val = objective(c_rand_nm, s_rand_nm)
            if val > best_val:
                best_val = val
                best_c = c_rand_nm.copy()
                best_s = s_rand_nm.copy()
        else:
            # Fallback: coordinate ascent from random start
            c_rand_ca, s_rand_ca = coordinate_ascent(c_rand, s_rand, max_rounds=100)
            if is_feasible(c_rand_ca, s_rand_ca):
                val = objective(c_rand_ca, s_rand_ca)
                if val > best_val:
                    best_val = val
                    best_c = c_rand_ca.copy()
                    best_s = s_rand_ca.copy()

    # Final polish: high-precision Nelder-Mead on the best solution
    if best_c is not None and best_s is not None:
        x_best = flatten(best_c, best_s)
        x_polish = nelder_mead_optimize(x_best, max_iter=5000, time_limit=30)
        c_polish, s_polish = unflatten(x_polish)
        if is_feasible(c_polish, s_polish):
            val = objective(c_polish, s_polish)
            if val > best_val:
                best_val = val
                best_c = c_polish.copy()
                best_s = s_polish.copy()

    # Final coordinate ascent polish
    if best_c is not None and best_s is not None:
        c_final, s_final = coordinate_ascent(best_c, best_s, max_rounds=200)
        if is_feasible(c_final, s_final):
            val = objective(c_final, s_final)
            if val > best_val:
                best_val = val
                best_c = c_final.copy()
                best_s = s_final.copy()

    if best_c is None:
        best_c = centers_init.copy()
        best_s = sides_init.copy()

    if not is_feasible(best_c, best_s):
        best_c = centers_init.copy()
        best_s = sides_init.copy()

    best_s = np.maximum(best_s, 1e-9)

    return best_c, best_s


# EVOLVE-BLOCK-END
