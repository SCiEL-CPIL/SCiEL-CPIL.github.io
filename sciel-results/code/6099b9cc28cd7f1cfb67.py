"""Circle packing in perimeter-4 rectangle: ADMM + SLSQP + Advanced Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_value = 0.0
    best_result = None

    def _is_valid(cx, cy, r, W, H, tol=1e-8):
        """Check if packing is valid."""
        if W <= 0.05 or H <= 0.05:
            return False
        if np.any(cx - r < -tol) or np.any(cy - r < -tol):
            return False
        if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                    return False
        return True

    def check_validity(z, tol=1e-9):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        if w <= 0.05 or h <= 0.05:
            return False, 0.0
        for i in range(N):
            if r[i] <= 0:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum ** 2 - tol:
                    return False, 0.0
        val = float(np.sum(r))
        return True, val

    def _admm_pack(z0, time_budget, rho=50.0, max_iter=2000):
        """
        ADMM-based circle packing optimizer.
        
        Variables:
        - x, y: circle centers (N each)
        - r: radii (N)
        - w: rectangle width (h = 2 - w)
        
        Constraints:
        - Wall: x_i >= r_i, x_i + r_i <= w, y_i >= r_i, y_i + r_i <= h
        - Circle-circle: ||p_i - p_j|| >= r_i + r_j
        
        ADMM split:
        - u: positions (x, y) for all circles
        - v: radii for all circles  
        - Auxiliary: s_ij for each pair (i,j) representing separation
        - Dual variables for coupling
        
        Simplified: We use a proximal ADMM where we alternate between:
        1. Update positions given radii (project onto wall constraints + pairwise)
        2. Update radii given positions (maximize sum subject to constraints)
        3. Update dual variables
        """
        x = z0[0::3][:N].copy()
        y = z0[1::3][:N].copy()
        r = z0[2::3][:N].copy()
        w = z0[-1]
        h = 2.0 - w

        # Precompute pair indices
        pair_idx = []
        for i in range(N):
            for j in range(i + 1, N):
                pair_idx.append((i, j))
        n_pairs = len(pair_idx)

        # Dual variables for pairwise constraints
        # For each pair, lambda_ij >= 0 is the dual for ||p_i - p_j||^2 >= (r_i + r_j)^2
        lam = np.zeros(n_pairs)

        # Wall duals (simplified: just project)
        best_val = float(np.sum(r))
        best_x, best_y, best_r, best_w = x.copy(), y.copy(), r.copy(), w

        for it in range(max_iter):
            if time.time() > time_budget:
                break

            # --- Step 1: Update positions (x, y) given r, w ---
            # Minimize: 0.5*||u - u_prev + lam/rho||^2 subject to constraints
            # This is a convex QP with quadratic constraints -> use iterative projection
            
            x_new = x.copy()
            y_new = y.copy()

            # Initialize with current positions
            for inner_it in range(50):
                # Wall projection
                x_new = np.clip(x_new, r, w - r)
                y_new = np.clip(y_new, r, h - r)

                # Pairwise projection (Gauss-Seidel style)
                moved = False
                for pi, (i, j) in enumerate(pair_idx):
                    dx = x_new[i] - x_new[j]
                    dy = y_new[i] - y_new[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    r_sum = r[i] + r[j]

                    if d < r_sum - 1e-12:
                        # Push apart
                        if d < 1e-12:
                            # Random direction
                            angle = np.random.uniform(0, 2 * np.pi)
                            dx_dir, dy_dir = np.cos(angle), np.sin(angle)
                        else:
                            dx_dir, dy_dir = dx / d, dy / d

                        push = (r_sum - d) / 2.0 + 1e-6

                        x_new[i] += dx_dir * push
                        y_new[i] += dy_dir * push
                        x_new[j] -= dx_dir * push
                        y_new[j] -= dy_dir * push

                        # Re-project to walls
                        x_new[i] = np.clip(x_new[i], r[i], w - r[i])
                        y_new[i] = np.clip(y_new[i], r[i], h - r[i])
                        x_new[j] = np.clip(x_new[j], r[j], w - r[j])
                        y_new[j] = np.clip(y_new[j], r[j], h - r[j])
                        moved = True

                if not moved:
                    break

            # --- Step 2: Update radii given positions ---
            # Maximize sum(r) subject to:
            # r_i <= min(x_i, w-x_i, y_i, h-y_i) for all i
            # r_i + r_j <= ||p_i - p_j|| for all pairs (i,j)
            # r_i >= 0

            # This is a linear program! But we solve it greedily/iteratively.
            # Use a simple iterative method: start from current r, increase where possible.

            r_new = r.copy()
            for _ in range(10):
                improved = False
                for i in range(N):
                    # Max radius for circle i given others
                    r_wall = min(x_new[i], w - x_new[i], y_new[i], h - y_new[i])
                    r_max = r_wall
                    for pi, (a, b) in enumerate(pair_idx):
                        if a == i:
                            j = b
                        elif b == i:
                            j = a
                        else:
                            continue
                        dx = x_new[i] - x_new[j]
                        dy = y_new[i] - y_new[j]
                        d = np.sqrt(dx * dx + dy * dy)
                        r_allowed = d - r_new[j]
                        if r_allowed < r_max:
                            r_max = r_allowed
                    r_max = max(0.0, r_max)

                    if r_max > r_new[i] + 1e-8:
                        r_new[i] = r_max
                        improved = True

                if not improved:
                    break

            # --- Step 3: Update dual variables ---
            for pi, (i, j) in enumerate(pair_idx):
                dx = x_new[i] - x_new[j]
                dy = y_new[i] - y_new[j]
                d_sq = dx * dx + dy * dy
                r_sum = r_new[i] + r_new[j]
                violation = max(0.0, r_sum * r_sum - d_sq)
                lam[pi] = max(0.0, lam[pi] + rho * violation)

            # --- Step 4: Update w (aspect ratio) ---
            # Try to optimize w to maximize total radius
            # For now, keep w fixed or do a small search
            best_w = w
            best_total = float(np.sum(r_new))
            for w_trial in [w - 0.05, w - 0.02, w, w + 0.02, w + 0.05]:
                if w_trial < 0.3 or w_trial > 1.7:
                    continue
                h_trial = 2.0 - w_trial
                if h_trial < 0.3:
                    continue

                # Check if current packing fits in new rectangle
                valid = True
                for i in range(N):
                    if x_new[i] < r_new[i] - 1e-6 or x_new[i] > w_trial - r_new[i] + 1e-6:
                        valid = False
                        break
                    if y_new[i] < r_new[i] - 1e-6 or y_new[i] > h_trial - r_new[i] + 1e-6:
                        valid = False
                        break
                if valid:
                    # Try to grow radii in new rectangle
                    r_trial = r_new.copy()
                    for _ in range(5):
                        improved = False
                        for i in range(N):
                            r_wall = min(x_new[i], w_trial - x_new[i], y_new[i], h_trial - y_new[i])
                            r_max = r_wall
                            for pi, (a, b) in enumerate(pair_idx):
                                if a == i:
                                    j = b
                                elif b == i:
                                    j = a
                                else:
                                    continue
                                dx = x_new[i] - x_new[j]
                                dy = y_new[i] - y_new[j]
                                d = np.sqrt(dx * dx + dy * dy)
                                r_allowed = d - r_trial[j]
                                if r_allowed < r_max:
                                    r_max = r_allowed
                            r_max = max(0.0, r_max)
                            if r_max > r_trial[i] + 1e-8:
                                r_trial[i] = r_max
                                improved = True
                        if not improved:
                            break

                    total = float(np.sum(r_trial))
                    if total > best_total:
                        best_total = total
                        best_w = w_trial

            w = best_w
            h = 2.0 - w

            # Update
            x, y, r = x_new, y_new, r_new
            val = float(np.sum(r))
            if val > best_val:
                best_val = val
                best_x, best_y, best_r, best_w = x.copy(), y.copy(), r.copy(), w

            # Convergence check
            if it > 100 and abs(val - best_val) < 1e-10:
                break

        # Reconstruct z
        z_out = np.empty(3 * N + 1)
        z_out[0::3][:N] = best_x
        z_out[1::3][:N] = best_y
        z_out[2::3][:N] = best_r
        z_out[-1] = best_w

        if _is_valid(best_x, best_y, best_r, best_w, 2.0 - best_w):
            return z_out
        return None

    def _vectorized_objective(z, penalty_weight=1000.0):
        """Vectorized objective with smooth penalties for speed."""
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w < 0.2 or h < 0.2 or np.any(r < 0):
            return 1e6

        obj = -np.sum(r)

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(N, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 0.0))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        overlap_penalty = penalty_weight * np.sum(overlap * overlap)

        obj += overlap_penalty

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)

        boundary_penalty = penalty_weight * (
            np.sum(left_viol * left_viol) +
            np.sum(right_viol * right_viol) +
            np.sum(bottom_viol * bottom_viol) +
            np.sum(top_viol * top_viol)
        )

        obj += boundary_penalty

        if w < 0.5:
            obj += penalty_weight * (0.5 - w) ** 2
        elif w > 1.5:
            obj += penalty_weight * (w - 1.5) ** 2

        return obj

    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.45
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        rng = np.random.default_rng(42)
        for _ in range(10):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.05, 0.12)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    def _force_relax(z, max_iter=500):
        """Force-directed relaxation to resolve initial overlaps."""
        z = z.copy().astype(float)
        n = N
        dt = 0.005
        min_dt = 0.0001

        for it in range(max_iter):
            x = z[0::3][:n]
            y = z[1::3][:n]
            r = z[2::3][:n]
            w = z[-1]
            h = 2.0 - w

            if w < 0.1 or h < 0.1:
                break

            fx = np.zeros(n)
            fy = np.zeros(n)

            left_viol = np.maximum(r - x, 0.0)
            right_viol = np.maximum(x + r - w, 0.0)
            bottom_viol = np.maximum(r - y, 0.0)
            top_viol = np.maximum(y + r - h, 0.0)
            fx -= left_viol * 100
            fx += right_viol * 100
            fy -= bottom_viol * 100
            fy += top_viol * 100

            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx * dx + dy * dy
            r_sum = r[:, None] + r[None, :]

            iu = np.triu_indices(n, k=1)
            d_sq_pairs = d_sq[iu]
            r_sum_pairs = r_sum[iu]

            d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
            overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
            max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

            if max_overlap > 1e-8:
                f_pairs = overlap * 15.0
                dx_pairs = dx[iu]
                dy_pairs = dy[iu]
                ux = dx_pairs / d_pairs
                uy = dy_pairs / d_pairs

                for idx in range(len(iu[0])):
                    i_idx = iu[0][idx]
                    j_idx = iu[1][idx]
                    fx[i_idx] += ux[idx] * f_pairs[idx]
                    fy[i_idx] += uy[idx] * f_pairs[idx]
                    fx[j_idx] -= ux[idx] * f_pairs[idx]
                    fy[j_idx] -= uy[idx] * f_pairs[idx]
            else:
                break

            force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
            if force_mag > 1.0:
                dt = max(min_dt, dt * 0.5)
            elif force_mag < 0.1:
                dt = min(0.01, dt * 1.1)

            z[0::3][:n] += dt * fx
            z[1::3][:n] += dt * fy

            for i in range(n):
                z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
                z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

        return z

    def _expansion_relaxation_polish(z, time_budget):
        """Iteratively expand circles and shift positions to create space."""
        z = z.copy().astype(float)
        n = N
        x = z[0::3][:n].copy()
        y = z[1::3][:n].copy()
        r = z[2::3][:n].copy()
        w = z[-1]
        h = 2.0 - w

        def try_expand_circle(i):
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)

            if r_max > r[i] + 1e-6:
                lo = r[i]
                hi = r_max
                for _ in range(25):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-7:
                    r[i] = lo
                    return True
            return False

        def try_move_circle(i):
            dirs = [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (0.7071, 0.7071), (-0.7071, 0.7071), (0.7071, -0.7071), (-0.7071, -0.7071)
            ]
            best_dir = None
            best_score = -float('inf')

            for dx_d, dy_d in dirs:
                new_x = x[i] + dx_d * 0.01
                new_y = y[i] + dy_d * 0.01
                if new_x < r[i] or new_x > w - r[i] or new_y < r[i] or new_y > h - r[i]:
                    continue

                min_gap = float('inf')
                for j in range(n):
                    if i == j:
                        continue
                    dx = new_x - x[j]
                    dy = new_y - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    gap = d - (r[i] + r[j])
                    if gap < min_gap:
                        min_gap = gap

                wall_dist = min(new_x - r[i], w - new_x - r[i], new_y - r[i], h - new_y - r[i])
                score = min(min_gap, wall_dist)

                if score > best_score:
                    best_score = score
                    best_dir = (dx_d, dy_d)

            if best_dir is None:
                return False

            dx_d, dy_d = best_dir
            for step in [0.06, 0.04, 0.03, 0.02, 0.015, 0.01, 0.005, 0.003, 0.001]:
                new_x = x[i] + dx_d * step
                new_y = y[i] + dy_d * step
                if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                    x[i] = new_x
                    y[i] = new_y
                    return True
            return False

        max_rounds = 40
        for rnd in range(max_rounds):
            if time.time() > time_budget:
                break
            improved = False
            for i in range(n):
                if time.time() > time_budget:
                    break
                if try_expand_circle(i):
                    improved = True
                r_wall = min(x[i], w - x[i], y[i], h - y[i])
                r_max = r_wall
                for j in range(n):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    r_allowed = d - r[j]
                    if r_allowed < r_max:
                        r_max = r_allowed
                r_max = max(0.0, r_max)
                if r_max - r[i] < 0.01:
                    if try_move_circle(i):
                        improved = True
            if not improved:
                break

        for rnd in range(20):
            if time.time() > time_budget:
                break
            improved = False
            for i in range(n):
                if time.time() > time_budget:
                    break
                if try_expand_circle(i):
                    improved = True
            if not improved:
                break

        for _ in range(100):
            if time.time() > time_budget:
                break
            lo, hi = 1.0, 1.05
            for _ in range(15):
                mid = (lo + hi) / 2.0
                if _is_valid(x, y, r * mid, w, h):
                    lo = mid
                else:
                    hi = mid
            if lo > 1.0 + 1e-9:
                r = r * lo
            else:
                break

        if _is_valid(x, y, r, w, h):
            z_new = np.empty(3 * N + 1)
            z_new[0::3][:n] = x
            z_new[1::3][:n] = y
            z_new[2::3][:n] = r
            z_new[-1] = w
            return z_new
        return None

    def _is_valid_move(x, y, r, w, h, i, new_x, new_y):
        if new_x < r[i] or new_x > w - r[i]:
            return False
        if new_y < r[i] or new_y > h - r[i]:
            return False
        for j in range(N):
            if i == j:
                continue
            dx = new_x - x[j]
            dy = new_y - y[j]
            d2 = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            if d2 < r_sum * r_sum - 1e-9:
                return False
        return True

    def _constraint_refine(z, time_budget, tight=False):
        """SLSQP refinement with penalty objective."""
        z = z.copy().astype(float)
        n = N
        penalty = 1000.0 if tight else 500.0
        ftol = 1e-18 if tight else 1e-15
        maxls = 200 if tight else 150

        def obj_refine(zv):
            return _vectorized_objective(zv, penalty)

        bounds = []
        for i in range(n):
            bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
        bounds.append((0.3, 1.7))

        maxiter = int(min(2000 if tight else 1000, time_budget * 200))
        try:
            res = minimize(obj_refine, z, method='SLSQP',
                           bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': ftol, 'maxls': maxls})

            z_opt = res.x
            x = z_opt[0::3][:n]
            y = z_opt[1::3][:n]
            r = z_opt[2::3][:n]
            w = z_opt[-1]
            h = 2.0 - w

            if not _is_valid(x, y, r, w, h):
                for factor in [0.999, 0.998, 0.997, 0.995, 0.99]:
                    r_trial = r * factor
                    if _is_valid(x, y, r_trial, w, h):
                        r = r_trial
                        break

            if _is_valid(x, y, r, w, h):
                z_opt[2::3][:n] = r
                return z_opt
            return None
        except Exception:
            return None

    def _coordinate_descent_polish(z, time_budget):
        """Coordinate-descent polish."""
        z = z.copy().astype(float)
        n = N
        x = z[0::3][:n].copy()
        y = z[1::3][:n].copy()
        r = z[2::3][:n].copy()
        w = z[-1]
        h = 2.0 - w

        def max_radius_at_position(px, py, other_r, skip_i):
            r_wall = min(px, w - px, py, h - py)
            r_max = r_wall
            for j in range(n):
                if j == skip_i:
                    continue
                dx = px - x[j]
                dy = py - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - other_r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            return max(0.0, r_max)

        for rnd in range(30):
            if time.time() > time_budget:
                break
            improved = False
            for i in range(n):
                if time.time() > time_budget:
                    break
                best_r = r[i]
                best_x = x[i]
                best_y = y[i]

                for dx_p in [-0.03, -0.015, -0.005, 0.005, 0.015, 0.03]:
                    for dy_p in [-0.03, -0.015, -0.005, 0.005, 0.015, 0.03]:
                        px = x[i] + dx_p
                        py = y[i] + dy_p
                        if px < 0 or px > w or py < 0 or py > h:
                            continue
                        r_test = max_radius_at_position(px, py, r, i)
                        if r_test > best_r + 1e-9:
                            r_new = r.copy()
                            r_new[i] = r_test
                            x_new = x.copy()
                            y_new = y.copy()
                            x_new[i] = px
                            y_new[i] = py
                            if _is_valid(x_new, y_new, r_new, w, h):
                                best_r = r_test
                                best_x = px
                                best_y = py

                if best_r > r[i] + 1e-9:
                    x[i] = best_x
                    y[i] = best_y
                    r[i] = best_r
                    improved = True
            if not improved:
                break

        for _ in range(50):
            if time.time() > time_budget:
                break
            lo, hi = 1.0, 1.02
            for _ in range(12):
                mid = (lo + hi) / 2.0
                if _is_valid(x, y, r * mid, w, h):
                    lo = mid
                else:
                    hi = mid
            if lo > 1.0 + 1e-9:
                r = r * lo
            else:
                break

        if _is_valid(x, y, r, w, h):
            z_new = np.empty(3 * N + 1)
            z_new[0::3][:n] = x
            z_new[1::3][:n] = y
            z_new[2::3][:n] = r
            z_new[-1] = w
            return z_new
        return None

    # Phase 1: ADMM from multiple initial guesses
    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 40:
            break

        z0_relaxed = _force_relax(z0, max_iter=300)

        # ADMM optimization
        admm_budget = deadline - (time.time() - start_time) - 35
        if admm_budget < 5:
            break

        z_admm = _admm_pack(z0_relaxed, time_budget=start_time + admm_budget)
        if z_admm is not None:
            valid, val = check_validity(z_admm)
            if valid and val > best_value:
                best_value = val
                best_result = z_admm.copy()

        # SLSQP from ADMM result
        if z_admm is not None and time.time() < deadline - 30:
            remaining = deadline - (time.time() - start_time) - 25
            if remaining > 5:
                try:
                    res = minimize(_vectorized_objective, z_admm, method='SLSQP',
                                   bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                                   options={'maxiter': int(min(300, remaining * 30)),
                                            'ftol': 1e-12, 'maxls': 80})
                    z_opt = res.x
                    valid, val = check_validity(z_opt)
                    if valid and val > best_value:
                        best_value = val
                        best_result = z_opt.copy()
                except Exception:
                    pass

    # Fallback if ADMM didn't find anything
    if best_result is None:
        for i, z0 in enumerate(guesses[:5]):
            if time.time() > deadline - 30:
                break
            z0_relaxed = _force_relax(z0, max_iter=500)
            remaining = deadline - (time.time() - start_time) - 25
            if remaining < 5:
                break
            try:
                res = minimize(_vectorized_objective, z0_relaxed, method='SLSQP',
                               bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                               options={'maxiter': int(min(500, remaining * 40)),
                                        'ftol': 1e-12, 'maxls': 80})
                z_opt = res.x
                valid, val = check_validity(z_opt)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_opt.copy()
            except Exception:
                continue

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Phase 2: Polish
    if time.time() < deadline - 20:
        expanded = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 12)
        if expanded is not None:
            valid, val = check_validity(expanded)
            if valid and val > best_value:
                best_value = val
                best_result = expanded

    if best_result is not None and time.time() < deadline - 15:
        refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 8, tight=True)
        if refined is not None:
            valid, val = check_validity(refined)
            if valid and val > best_value:
                best_value = val
                best_result = refined

    if best_result is not None and time.time() < deadline - 10:
        expanded2 = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 5)
        if expanded2 is not None:
            valid, val = check_validity(expanded2)
            if valid and val > best_value:
                best_value = val
                best_result = expanded2

    if best_result is not None and time.time() < deadline - 6:
        cd_polished = _coordinate_descent_polish(best_result, deadline - (time.time() - start_time) - 3)
        if cd_polished is not None:
            valid, val = check_validity(cd_polished)
            if valid and val > best_value:
                best_value = val
                best_result = cd_polished

    if best_result is not None and time.time() < deadline - 3:
        final_refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 1, tight=True)
        if final_refined is not None:
            valid, val = check_validity(final_refined)
            if valid and val > best_value:
                best_value = val
                best_result = final_refined

    # Format output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    for _ in range(20):
        r_trial = r * (1 - 1e-8)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True
