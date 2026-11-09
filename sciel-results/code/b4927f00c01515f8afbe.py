# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

TOL = 1e-7


def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares with pairwise disjoint interiors inside the
    unit square [0,1]^2 so as to MAXIMIZE the sum of their side lengths.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    """
    import time
    t0 = time.time()
    DEADLINE = t0 + 580.0  # safety margin before 600s limit
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
        """Return total penalty (0 if feasible)."""
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
        """Custom gradient-free local search with adaptive step size."""
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
            # Try random perturbations
            for _ in range(5):
                trial_c = best_c.copy()
                trial_s = best_s.copy()
                # Perturb a random subset
                idx = np.random.randint(0, n, size=np.random.randint(1, 4))
                for i in idx:
                    trial_c[i, 0] += np.random.uniform(-step, step)
                    trial_c[i, 1] += np.random.uniform(-step, step)
                    trial_s[i] += np.random.uniform(-step * 0.5, step * 0.5)
                # Clip
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
        """Coordinate descent: optimize one parameter at a time."""
        best_c = centers.copy()
        best_s = sides.copy()
        best_score = np.sum(sides)
        step = 0.01

        for it in range(max_iters):
            if time.time() > DEADLINE - 30:
                break
            improved = False
            for i in range(n):
                # Try adjusting side length
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
                # Try adjusting cx
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
                # Try adjusting cy
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
    # Layout 1: 3x3 grid of 1/3 squares + one square of side 1.0 at top
    # 9 squares in 3x3 grid (rows 0,1,2), 1 square at top (row 3, side 1.0)
    centers1 = np.zeros((n, 2))
    sides1 = np.zeros(n)
    for i in range(9):
        col = i % 3
        row = i // 3
        side = 1.0 / 3.0
        centers1[i, 0] = (col + 0.5) * side
        centers1[i, 1] = (row + 0.5) * side
        sides1[i] = side
    # Square 9: side 1.0, centered at (0.5, 0.5) — but this overlaps!
    # Need to place it properly. If 3x3 grid occupies y in [0, 1], no room.
    # Instead: 3x3 grid of 1/4 squares (side 0.25) in bottom 3/4, top 1/4 has one square of side 0.25? No.
    # Better: 4x4 grid of 1/4 squares, use 10 of them. Sum = 10*0.25 = 2.5.
    # Or: 3 cols x 4 rows, but with different sizes.
    # Let me try: bottom 3 rows have 3 squares each of side 1/3. Top row has 1 square of side 1/3? That's 10 total, sum=10/3=3.33.
    # But can we do better? What if top square is bigger?
    # If bottom 3 rows are 3 squares of side 1/3 each (occupying y in [0,1]), no room for top.
    # Alternative: 4 cols x 3 rows = 12 cells, use 10. All side 1/4. Sum=2.5.
    # Better: 2 rows of 4 squares (side 1/4) + 2 rows of 2 squares (side 1/2). Sum = 8*0.25 + 2*0.5 = 3.0.
    # Even better: 3 rows. Row 0: 4 squares side 1/4. Row 1: 4 squares side 1/4. Row 2: 2 squares side 1/2. Sum=3.0.
    # Or: Row 0: 5 squares side 1/5. Row 1: 5 squares side 1/5. Sum=2.0. Worse.
    # Let me try a hybrid: 3 cols. Rows 0,1,2: 3 squares each of side 1/3. That's 9. Row 3: 1 square.
    # If row 3 square has side s, it must fit in y in [1, 1+s] but that's outside [0,1].
    # So all 4 rows must fit in [0,1]. If 4 rows of height h, 4h<=1, h<=1/4.
    # So max side is 1/4 if 4 rows. Sum=10*1/4=2.5.
    # If 3 rows, max side 1/3. 9 squares of 1/3 + 1 square. The 10th square must fit in the remaining space.
    # 3 rows of 1/3 height = 1.0 total. No room for 10th square unless it's in the same row.
    # So: 3 rows, each with 3 squares of side 1/3, plus 1 more square in one of the rows?
    # Row 0: 3 squares of side 1/3 (full width). No room for 4th in row 0.
    # Unless we make some squares smaller. E.g., row 0: 3 squares, but one is smaller to fit another.
    # This is getting complex. Let me just use multiple analytical starts and let the optimizer work.

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
    # Row 0: 4 squares, side 1/4
    for i in range(4):
        col = i
        side = 1.0 / 4.0
        centersB[i, 0] = (col + 0.5) * side
        centersB[i, 1] = 0.5 * side
        sidesB[i] = side
    # Row 1: 4 squares, side 1/4
    for i in range(4):
        col = i
        side = 1.0 / 4.0
        centersB[4 + i, 0] = (col + 0.5) * side
        centersB[4 + i, 1] = 1.5 * side
        sidesB[4 + i] = side
    # Row 2: 2 squares, side 1/2
    for i in range(2):
        col = i
        side = 1.0 / 2.0
        centersB[8 + i, 0] = (col + 0.5) * side
        centersB[8 + i, 1] = 0.5 + 0.5 * side
        sidesB[8 + i] = side

    # Layout C: 3x3 grid of 1/3 + 1 square of side 1/3 in a corner of one cell
    centersC = np.zeros((n, 2))
    sidesC = np.zeros(n)
    for i in range(9):
        col = i % 3
        row = i // 3
        side = 1.0 / 3.0
        centersC[i, 0] = (col + 0.5) * side
        centersC[i, 1] = (row + 0.5) * side
        sidesC[i] = side
    # 10th square: place in top-right area, side 1/6
    side10 = 1.0 / 6.0
    centersC[9, 0] = 1.0 - side10 / 2.0
    centersC[9, 1] = 1.0 - side10 / 2.0
    sidesC[9] = side10
    # But this overlaps with the top-right 1/3 square! Need to adjust.
    # Actually, the top-right 1/3 square is at center (5/6, 5/6) with side 1/3, so it spans [4/6, 6/6] x [4/6, 6/6].
    # A square of side 1/6 at top-right would be at [5/6, 6/6] x [5/6, 6/6], which is inside the 1/3 square. Overlap!
    # So this layout is infeasible. Let me use a different approach.

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

    # Layout E: Mixed sizes. 4 squares of side 1/4 in bottom row, 4 of side 1/4 in second row, 2 of side 1/2 in top row
    centersE = centersB.copy()
    sidesE = sidesB.copy()

    # Layout F: 3 squares of side 1/3 in bottom row, 3 in middle, 3 in top, 1 in corner
    # This is similar to C but infeasible. Skip.

    # Layout G: 2 squares of side 1/2 in bottom row, 2 in second row, 2 in third row, 2 in fourth row, 2 in fifth row? No, 5 rows of 1/2 height = 2.5 > 1.
    # 2 rows of 1/2, 2 rows of 1/4, 2 rows of 1/4? 2*0.5 + 2*0.25 + 2*0.25 = 1.0 + 0.5 + 0.5 = 2.0 > 1.
    # Let me just use the feasible layouts.

    # Collect all feasible initial layouts
    initial_layouts = []
    for c, s in [(centersA, sidesA), (centersB, sidesB), (centersD, sidesD), (centersE, sidesE)]:
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
        c_cd, s_cd, score_cd = _coordinate_descent(c_ls, s_ls, max_iters=5000)
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
        x0 = np.concatenate([best_sides, best_centers[:, 0], best_centers[:, 1]])
        bounds = [(1e-4, 0.99)] * n + [(0.0, 1.0)] * (2 * n)
        try:
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
        # Fallback to the first feasible layout
        best_centers = initial_layouts[0][0]
        best_sides = initial_layouts[0][1]

    return best_centers.astype(np.float64), best_sides.astype(np.float64)


# EVOLVE-BLOCK-END
