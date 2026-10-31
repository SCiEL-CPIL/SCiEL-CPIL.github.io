"""Circle packing in perimeter-4 rectangle: SLSQP + smooth distance-field polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    best_value = 0.0
    best_result = None

    # --- Vectorized objective for penalty-based SLSQP ---
    penalty_weight = 500.0

    def objective(z):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w < 0.25 or h < 0.25 or np.any(r < 0):
            return 1e6

        obj = -np.sum(r)

        # Vectorized pairwise overlap penalty
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(N, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        obj += penalty_weight * np.sum(overlap * overlap)

        # Vectorized boundary penalty
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)

        obj += penalty_weight * (
            np.sum(left_viol * left_viol) +
            np.sum(right_viol * right_viol) +
            np.sum(bottom_viol * bottom_viol) +
            np.sum(top_viol * top_viol)
        )

        # Keep aspect ratio in reasonable range
        if w < 0.5:
            obj += penalty_weight * (0.5 - w) ** 2
        elif w > 1.5:
            obj += penalty_weight * (w - 1.5) ** 2

        return obj

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

    # --- Generate diverse initial guesses ---
    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8, 1.1, 0.9, 0.7, 1.4, 0.65, 1.6]
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

            # Hexagonal variant
            z_hex = np.zeros(3 * N + 1)
            hex_r = min(w0, h0) / (side * 1.8)
            hex_dx = hex_r * 2.0
            hex_dy = hex_r * np.sqrt(3)
            count = 0
            for row in range(side + 2):
                for col in range(side + 2):
                    if count >= N:
                        break
                    cx = col * hex_dx + (row % 2) * (hex_dx / 2) + hex_r
                    cy = row * hex_dy + hex_r
                    if cx + hex_r <= w0 and cy + hex_r <= h0:
                        z_hex[3 * count] = cx
                        z_hex[3 * count + 1] = cy
                        z_hex[3 * count + 2] = hex_r
                        count += 1
            if count == N:
                z_hex[-1] = w0
                guesses.append(z_hex)

        rng = np.random.default_rng(42)
        for _ in range(8):
            w0 = rng.uniform(0.55, 1.45)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.05, 0.11)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    # --- Phase 1: Penalty-based SLSQP from multiple starts ---
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 10:
            break

        z0_relaxed = _force_relax_unified(z0, max_iter=100)

        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 10
        maxiter = int(min(250, max(60, remaining * 25)))

        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 60})

            z_opt = res.x
            valid, val = check_validity(z_opt)

            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.93
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()

        except Exception:
            continue

    # --- Phase 2: Hard-constraint SLSQP refinement ---
    if best_result is not None and time.time() < deadline - 18:
        refined = _hard_constraint_refine(best_result, deadline - (time.time() - start_time))
        if refined is not None:
            valid, val = check_validity(refined)
            if valid and val > best_value:
                best_value = val
                best_result = refined

    # --- Phase 3: COBYLA refinement ---
    if best_result is not None and time.time() < deadline - 12:
        cobyla_refined = _cobyla_refine(best_result, deadline - (time.time() - start_time))
        if cobyla_refined is not None:
            valid, val = check_validity(cobyla_refined)
            if valid and val > best_value:
                best_value = val
                best_result = cobyla_refined

    # --- Phase 4: Distance-field polish with radius growth ---
    if best_result is not None and time.time() < deadline - 12:
        polished = _distance_field_polish(best_result, deadline - (time.time() - start_time))
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

    # --- Phase 5: Aspect ratio optimization ---
    if best_result is not None and time.time() < deadline - 6:
        aspect_refined = _aspect_ratio_optimize(best_result, deadline - (time.time() - start_time))
        if aspect_refined is not None:
            valid, val = check_validity(aspect_refined)
            if valid and val > best_value:
                best_value = val
                best_result = aspect_refined

    # --- Phase 6: High-precision final polish ---
    if best_result is not None and time.time() < deadline - 2:
        final_polished = _high_precision_polish(best_result, deadline - (time.time() - start_time))
        if final_polished is not None:
            valid, val = check_validity(final_polished)
            if valid and val > best_value:
                best_value = val
                best_result = final_polished

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]

    r = r * (1 - 1e-6)

    return np.column_stack((x, y, r))


def _force_relax_unified(z, max_iter=150):
    """Force-directed relaxation in unified parameter space."""
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.0001

    for _ in range(max_iter):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        fx = np.zeros(n)
        fy = np.zeros(n)

        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 100
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 100
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 100
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 100

        max_overlap = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_sq = dx * dx + dy * dy
                min_d = r[i] + r[j]
                if d_sq < min_d * min_d and d_sq > 1e-12:
                    d = np.sqrt(d_sq)
                    overlap = min_d - d
                    max_overlap = max(max_overlap, overlap)
                    f = overlap * 15
                    fx[i] += (dx / d) * f
                    fy[i] += (dy / d) * f
                    fx[j] -= (dx / d) * f
                    fy[j] -= (dy / d) * f

        if max_overlap < 1e-8:
            break

        force_mag = np.max(np.sqrt(fx * fx + fy * fy))
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


def _hard_constraint_refine(z_init, time_budget):
    """Refine using SLSQP with explicit hard inequality constraints."""
    if time_budget < 3:
        return None

    z = z_init.copy().astype(float)
    n = N

    constraints = []

    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: p[3 * idx] - p[3 * idx + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: p[-1] - p[3 * idx] - p[3 * idx + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: p[3 * idx + 1] - p[3 * idx + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: (2.0 - p[-1]) - p[3 * idx + 1] - p[3 * idx + 2]})

    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (
                    (p[3 * i] - p[3 * j]) ** 2 +
                    (p[3 * i + 1] - p[3 * j + 1]) ** 2 -
                    (p[3 * i + 2] + p[3 * j + 2]) ** 2
                )
            })

    def obj(p):
        r = p[2::3][:n]
        return -np.sum(r)

    bounds = [(0, 2)] * (3 * n) + [(0.3, 1.7)]

    maxiter = int(min(350, max(80, (time_budget - 3) * 30)))

    try:
        res = minimize(obj, z, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 80})

        if res.success or res.fun < obj(z):
            z_new = res.x
            x = z_new[0::3][:n]
            y = z_new[1::3][:n]
            r = z_new[2::3][:n]
            w = z_new[-1]
            h = 2.0 - w
            if _is_valid(x, y, r, w, h):
                return z_new
            r_shrunk = r * 0.995
            if _is_valid(x, y, r_shrunk, w, h):
                z_new[2::3][:n] = r_shrunk
                return z_new
    except Exception:
        pass

    return None


def _cobyla_refine(z_init, time_budget):
    """Refine using COBYLA (derivative-free)."""
    if time_budget < 3:
        return None

    z = z_init.copy().astype(float)
    n = N

    penalty = 500.0

    def obj_cobyla(p):
        x = p[0::3][:n]
        y = p[1::3][:n]
        r = p[2::3][:n]
        w = p[-1]
        h = 2.0 - w

        if w < 0.3 or h < 0.3:
            return 1e6

        val = -np.sum(r)

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
        overlap = np.maximum(r_sum[iu] - d_pairs, 0.0)
        val += penalty * np.sum(overlap * overlap)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        val += penalty * (np.sum(left_viol**2) + np.sum(right_viol**2) +
                          np.sum(bottom_viol**2) + np.sum(top_viol**2))

        return val

    maxiter = int(min(500, max(120, (time_budget - 3) * 40)))

    try:
        res = minimize(obj_cobyla, z, method='COBYLA',
                       options={'maxiter': maxiter, 'rhobeg': 0.005, 'catol': 1e-10})

        z_new = res.x
        x = z_new[0::3][:n]
        y = z_new[1::3][:n]
        r = z_new[2::3][:n]
        w = z_new[-1]
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            return z_new
        r_shrunk = r * 0.99
        if _is_valid(x, y, r_shrunk, w, h):
            z_new[2::3][:n] = r_shrunk
            return z_new
    except Exception:
        pass

    return None


def _distance_field_polish(z, time_budget):
    """Polish using a smooth distance-field formulation with gradient-based radius growth."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # First, do a standard force-directed relaxation to ensure feasibility
    dt = 0.003
    max_iter = int(min(200, time_budget * 30))
    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 40
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 40
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 40
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 40

        max_overlap = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_sq = dx * dx + dy * dy
                min_d = r[i] + r[j]
                if d_sq < min_d * min_d and d_sq > 1e-12:
                    d = np.sqrt(d_sq)
                    overlap = min_d - d
                    max_overlap = max(max_overlap, overlap)
                    fx[i] += (dx / d) * overlap * 20
                    fy[i] += (dy / d) * overlap * 20
                    fx[j] -= (dx / d) * overlap * 20
                    fy[j] -= (dy / d) * overlap * 20

        if max_overlap < 1e-9:
            break

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    if not _is_valid(x, y, r, w, h):
        return None

    # Smooth distance field with vectorized gradient computation
    beta = 30.0

    def compute_free_space_and_grad(x_pos, y_pos, r_pos, w_val, h_val, beta_val):
        """Compute soft-min free space and its gradient w.r.t. positions, vectorized."""
        n_c = len(x_pos)
        free_r = np.zeros(n_c)
        grad_x = np.zeros(n_c)
        grad_y = np.zeros(n_c)

        # Precompute pairwise distances
        dx_mat = x_pos[:, None] - x_pos[None, :]
        dy_mat = y_pos[:, None] - y_pos[None, :]
        dist_mat = np.sqrt(np.maximum(dx_mat * dx_mat + dy_mat * dy_mat, 1e-15))
        np.fill_diagonal(dist_mat, np.inf)

        for i in range(n_c):
            # All distances that constrain circle i
            bounds = np.array([x_pos[i], w_val - x_pos[i], y_pos[i], h_val - y_pos[i]])
            circle_dists = dist_mat[i] - r_pos
            circle_dists[i] = np.inf  # exclude self
            all_dists = np.concatenate([bounds, circle_dists])

            # Soft min
            v_min = np.min(all_dists)
            exp_terms = np.exp(-beta_val * (all_dists - v_min))
            sum_exp = np.sum(exp_terms)
            free_r[i] = v_min - (1.0 / beta_val) * np.log(sum_exp)

            # Gradient: d/dx_i of soft_min
            # soft_min = -1/beta * log(sum_j exp(-beta * d_j))
            # d(soft_min)/d(x_i) = (1/beta) * sum_j [exp(-beta*d_j)/sum_exp * beta * d(d_j)/d(x_i)]
            # = sum_j [w_j * d(d_j)/d(x_i)] where w_j = exp(-beta*d_j)/sum_exp

            weights = exp_terms / sum_exp

            # Boundary gradients
            # d(x_i)/d(x_i) = 1, d(w-x_i)/d(x_i) = -1
            # d(y_i)/d(y_i) = 1, d(h-y_i)/d(y_i) = -1
            gx = weights[0] * 1.0 + weights[1] * (-1.0)
            gy = weights[2] * 1.0 + weights[3] * (-1.0)

            # Circle distance gradients: d(dist_ij - r_j)/d(x_i) = (x_i - x_j)/dist_ij
            for j in range(n_c):
                if j == i:
                    continue
                d_ij = dist_mat[i, j]
                if d_ij > 1e-10:
                    gx += weights[4 + j] * (x_pos[i] - x_pos[j]) / d_ij
                    gy += weights[4 + j] * (y_pos[i] - y_pos[j]) / d_ij

            grad_x[i] = gx
            grad_y[i] = gy

        return free_r, grad_x, grad_y

    # Gradient-based optimization of positions
    lr = 0.003
    max_steps = int(min(300, time_budget * 25))

    for step in range(max_steps):
        if time.time() > time_budget:
            break

        free_r, grad_x, grad_y = compute_free_space_and_grad(x, y, r, w, h, beta)

        # Adaptive learning rate
        grad_mag = np.max(np.sqrt(grad_x ** 2 + grad_y ** 2))
        if grad_mag > 0.5:
            lr = max(0.001, lr * 0.7)
        elif grad_mag < 0.05:
            lr = min(0.005, lr * 1.2)

        x += lr * grad_x
        y += lr * grad_y

        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

        # Grow radii toward free space
        new_free, _, _ = compute_free_space_and_grad(x, y, r, w, h, beta)
        grow = np.clip((new_free - r) / np.maximum(r, 1e-6), 0, 0.015)
        r_new = r * (1 + grow)

        if _is_valid(x, y, r_new, w, h):
            r = r_new
        else:
            r_new = r * (1 + grow * 0.5)
            if _is_valid(x, y, r_new, w, h):
                r = r_new

    # Force-directed cleanup
    for _ in range(30):
        fx = np.zeros(n)
        fy = np.zeros(n)
        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 40
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 40
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 40
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 40
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 20
                    fy[i] += (dy / d) * overlap * 20
                    fx[j] -= (dx / d) * overlap * 20
                    fy[j] -= (dy / d) * overlap * 20
        x += 0.002 * fx
        y += 0.002 * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Aggressive radius growth
    grow_factor = 1.001
    min_grow = 1.0001
    for _ in range(300):
        if time.time() > time_budget:
            break
        if grow_factor < min_grow:
            break

        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
            grow_factor = max(min_grow, grow_factor * 0.95)
            for _it in range(15):
                fx = np.zeros(n)
                fy = np.zeros(n)
                for i in range(n):
                    if x[i] - r[i] < 0:
                        fx[i] += (x[i] - r[i]) * 40
                    if x[i] + r[i] > w:
                        fx[i] -= (x[i] + r[i] - w) * 40
                    if y[i] - r[i] < 0:
                        fy[i] += (y[i] - r[i]) * 40
                    if y[i] + r[i] > h:
                        fy[i] -= (y[i] + r[i] - h) * 40
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        d = np.sqrt(dx * dx + dy * dy)
                        min_d = r[i] + r[j]
                        if d < min_d and d > 1e-12:
                            overlap = min_d - d
                            fx[i] += (dx / d) * overlap * 20
                            fy[i] += (dy / d) * overlap * 20
                            fx[j] -= (dx / d) * overlap * 20
                            fy[j] -= (dy / d) * overlap * 20
                x += 0.002 * fx
                y += 0.002 * fy
                x = np.clip(x, r, w - r)
                y = np.clip(y, r, h - r)
        else:
            sorted_idx = np.argsort(r)
            success = False
            for k in range(3, 0, -1):
                r_trial2 = r.copy()
                r_trial2[sorted_idx[:k]] *= grow_factor
                if _is_valid(x, y, r_trial2, w, h):
                    r = r_trial2
                    success = True
                    break
            if not success:
                grow_factor *= 0.9
                if grow_factor < min_grow:
                    break

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _aspect_ratio_optimize(z_init, time_budget):
    """Try different aspect ratios by scaling positions and re-relaxing."""
    if time_budget < 5:
        return None

    z = z_init.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    best_val = float(np.sum(r))
    best_x, best_y, best_r, best_w = x.copy(), y.copy(), r.copy(), w

    # Try a range of aspect ratios
    aspect_targets = [w * 0.9, w * 0.85, w * 1.1, w * 1.15, w * 0.8, w * 1.2, w * 0.75, w * 1.25]
    # Also try fixed good aspects
    aspect_targets.extend([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 0.7, 1.4])

    for w_target in aspect_targets:
        if time.time() > time_budget:
            break
        if w_target < 0.5 or w_target > 1.5:
            continue

        h_target = 2.0 - w_target
        if h_target < 0.5:
            continue

        # Scale positions to new rectangle
        x_new = x * (w_target / w)
        y_new = y * (h_target / h)
        r_new = r * min(w_target / w, h_target / h) * 0.95  # shrink to fit

        # Force-directed relaxation
        dt = 0.002
        for _it in range(100):
            if time.time() > time_budget:
                break
            fx = np.zeros(n)
            fy = np.zeros(n)
            max_overlap = 0.0

            for i in range(n):
                if x_new[i] - r_new[i] < 0:
                    fx[i] += (x_new[i] - r_new[i]) * 40
                if x_new[i] + r_new[i] > w_target:
                    fx[i] -= (x_new[i] + r_new[i] - w_target) * 40
                if y_new[i] - r_new[i] < 0:
                    fy[i] += (y_new[i] - r_new[i]) * 40
                if y_new[i] + r_new[i] > h_target:
                    fy[i] -= (y_new[i] + r_new[i] - h_target) * 40

            for i in range(n):
                for j in range(i + 1, n):
                    dx = x_new[i] - x_new[j]
                    dy = y_new[i] - y_new[j]
                    d_sq = dx * dx + dy * dy
                    min_d = r_new[i] + r_new[j]
                    if d_sq < min_d * min_d and d_sq > 1e-12:
                        d = np.sqrt(d_sq)
                        overlap = min_d - d
                        max_overlap = max(max_overlap, overlap)
                        fx[i] += (dx / d) * overlap * 20
                        fy[i] += (dy / d) * overlap * 20
                        fx[j] -= (dx / d) * overlap * 20
                        fy[j] -= (dy / d) * overlap * 20

            if max_overlap < 1e-9:
                break

            x_new += dt * fx
            y_new += dt * fy
            x_new = np.clip(x_new, r_new, w_target - r_new)
            y_new = np.clip(y_new, r_new, h_target - r_new)

        if not _is_valid(x_new, y_new, r_new, w_target, h_target):
            continue

        # Try to grow radii
        grow_factor = 1.0005
        for _ in range(50):
            if time.time() > time_budget:
                break
            r_trial = r_new * grow_factor
            if _is_valid(x_new, y_new, r_trial, w_target, h_target):
                r_new = r_trial
                # Quick relax
                for _it2 in range(10):
                    fx = np.zeros(n)
                    fy = np.zeros(n)
                    for i in range(n):
                        if x_new[i] - r_new[i] < 0:
                            fx[i] += (x_new[i] - r_new[i]) * 40
                        if x_new[i] + r_new[i] > w_target:
                            fx[i] -= (x_new[i] + r_new[i] - w_target) * 40
                        if y_new[i] - r_new[i] < 0:
                            fy[i] += (y_new[i] - r_new[i]) * 40
                        if y_new[i] + r_new[i] > h_target:
                            fy[i] -= (y_new[i] + r_new[i] - h_target) * 40
                    for i in range(n):
                        for j in range(i + 1, n):
                            dx = x_new[i] - x_new[j]
                            dy = y_new[i] - y_new[j]
                            d = np.sqrt(dx * dx + dy * dy)
                            min_d = r_new[i] + r_new[j]
                            if d < min_d and d > 1e-12:
                                overlap = min_d - d
                                fx[i] += (dx / d) * overlap * 20
                                fy[i] += (dy / d) * overlap * 20
                                fx[j] -= (dx / d) * overlap * 20
                                fy[j] -= (dy / d) * overlap * 20
                    x_new += 0.001 * fx
                    y_new += 0.001 * fy
                    x_new = np.clip(x_new, r_new, w_target - r_new)
                    y_new = np.clip(y_new, r_new, h_target - r_new)
            else:
                break

        val = float(np.sum(r_new))
        if val > best_val:
            best_val = val
            best_x, best_y, best_r, best_w = x_new.copy(), y_new.copy(), r_new.copy(), w_target

    if best_val > float(np.sum(r)):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = best_x
        z_new[1::3][:n] = best_y
        z_new[2::3][:n] = best_r
        z_new[-1] = best_w
        return z_new
    return None


def _high_precision_polish(z, time_budget):
    """High-precision final polish with very tight tolerances and small steps."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Very fine force-directed relaxation
    dt = 0.0005
    max_iter = int(min(600, time_budget * 60))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)
        max_overlap = 0.0

        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 200
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 200
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 200
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 200

        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_sq = dx * dx + dy * dy
                min_d = r[i] + r[j]
                if d_sq < min_d * min_d and d_sq > 1e-14:
                    d = np.sqrt(d_sq)
                    overlap = min_d - d
                    max_overlap = max(max_overlap, overlap)
                    f = overlap * 30
                    fx[i] += (dx / d) * f
                    fy[i] += (dy / d) * f
                    fx[j] -= (dx / d) * f
                    fy[j] -= (dy / d) * f

        if max_overlap < 1e-10:
            break

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Very fine radius growth
    grow_factor = 1.0001
    for _ in range(300):
        if time.time() > time_budget:
            break

        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h, tol=1e-10):
            r = r_trial
            for _it in range(15):
                fx = np.zeros(n)
                fy = np.zeros(n)
                for i in range(n):
                    if x[i] - r[i] < 0:
                        fx[i] += (x[i] - r[i]) * 200
                    if x[i] + r[i] > w:
                        fx[i] -= (x[i] + r[i] - w) * 200
                    if y[i] - r[i] < 0:
                        fy[i] += (y[i] - r[i]) * 200
                    if y[i] + r[i] > h:
                        fy[i] -= (y[i] + r[i] - h) * 200
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        d = np.sqrt(dx * dx + dy * dy)
                        min_d = r[i] + r[j]
                        if d < min_d and d > 1e-14:
                            overlap = min_d - d
                            fx[i] += (dx / d) * overlap * 30
                            fy[i] += (dy / d) * overlap * 30
                            fx[j] -= (dx / d) * overlap * 30
                            fy[j] -= (dy / d) * overlap * 30
                x += 0.001 * fx
                y += 0.001 * fy
                x = np.clip(x, r, w - r)
                y = np.clip(y, r, h - r)
        else:
            sorted_idx = np.argsort(r)
            for k in range(2, 0, -1):
                r_trial2 = r.copy()
                r_trial2[sorted_idx[:k]] *= grow_factor
                if _is_valid(x, y, r_trial2, w, h, tol=1e-10):
                    r = r_trial2
                    break
            else:
                break

    # Final squeeze: try growing the largest circles selectively
    for _squeeze in range(50):
        if time.time() > time_budget:
            break
        sorted_idx = np.argsort(-r)  # largest first
        improved = False
        for idx in sorted_idx[:5]:
            r_trial = r.copy()
            r_trial[idx] *= 1.0005
            if _is_valid(x, y, r_trial, w, h, tol=1e-10):
                r = r_trial
                improved = True
                # Quick relax
                for _it in range(10):
                    fx = np.zeros(n)
                    fy = np.zeros(n)
                    for i in range(n):
                        if x[i] - r[i] < 0:
                            fx[i] += (x[i] - r[i]) * 200
                        if x[i] + r[i] > w:
                            fx[i] -= (x[i] + r[i] - w) * 200
                        if y[i] - r[i] < 0:
                            fy[i] += (y[i] - r[i]) * 200
                        if y[i] + r[i] > h:
                            fy[i] -= (y[i] + r[i] - h) * 200
                    for i in range(n):
                        for j in range(i + 1, n):
                            dx = x[i] - x[j]
                            dy = y[i] - y[j]
                            d = np.sqrt(dx * dx + dy * dy)
                            min_d = r[i] + r[j]
                            if d < min_d and d > 1e-14:
                                overlap = min_d - d
                                fx[i] += (dx / d) * overlap * 30
                                fy[i] += (dy / d) * overlap * 30
                                fx[j] -= (dx / d) * overlap * 30
                                fy[j] -= (dy / d) * overlap * 30
                    x += 0.0005 * fx
                    y += 0.0005 * fy
                    x = np.clip(x, r, w - r)
                    y = np.clip(y, r, h - r)
                break
        if not improved:
            break

    if _is_valid(x, y, r, w, h, tol=1e-10):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _is_valid(cx, cy, r, W, H, tol=1e-9):
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
