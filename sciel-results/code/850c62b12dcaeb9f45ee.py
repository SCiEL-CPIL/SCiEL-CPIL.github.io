import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (10s buffer before 150s kill)

    # Initial position: simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pos = grid[:N].copy()

    def compute_radius(p):
        if not np.all(np.isfinite(p)):
            return -1.0
        if np.any(p < -0.5) or np.any(p > 0.5):
            return -1.0
        wall_dists = 0.5 - np.abs(p)
        min_wall = np.min(wall_dists)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        r = min(min_wall, min_pair / 2.0)
        return r

    def vec_soft_min_radius(p, beta):
        """Vectorized soft-min of wall distances and half-pair distances."""
        wall_dists = 0.5 - np.abs(p)  # (N, 3)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2)) / 2.0
        np.fill_diagonal(dists, np.inf)
        # Collect all finite values
        all_vals = np.concatenate([wall_dists.ravel(), dists.ravel()])
        all_vals = all_vals[np.isfinite(all_vals)]
        if len(all_vals) == 0:
            return -1e10
        max_val = np.max(all_vals)
        shifted = all_vals - max_val
        log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta + max_val
        return log_sum

    def local_search_lbfgs(initial_pos, time_limit, beta_start=15.0, beta_end=80.0):
        p = initial_pos.copy()
        current_r = compute_radius(p)
        best_r_local = current_r
        best_p_local = p.copy()

        num_stages = 6
        betas = np.linspace(beta_start, beta_end, num_stages)

        for beta in betas:
            if time.time() > time_limit:
                break

            def obj_and_grad(x):
                p2 = x.reshape(N, 3)
                wall_dists = 0.5 - np.abs(p2)  # (N, 3)

                # Vectorized pair distances
                diff = p2[:, np.newaxis, :] - p2[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff ** 2, axis=2)) / 2.0
                np.fill_diagonal(dists, np.inf)

                # Collect all values for soft-min
                all_vals = np.concatenate([wall_dists.ravel(), dists.ravel()])
                max_val = np.max(all_vals)
                shifted = all_vals - max_val
                exp_vals = np.exp(-beta * shifted)
                sum_exp = np.sum(exp_vals)

                soft_min_val = max_val - np.log(sum_exp) / beta

                # Gradient computation
                grad = np.zeros_like(p2)

                # Wall gradient
                wall_exp = exp_vals[:N * 3]
                wall_w = wall_exp / sum_exp
                signs = np.sign(p2)
                signs[signs == 0] = 1.0
                # d(soft_min)/dx_k = beta * sum_i w_i * sign_i (for wall dist = 0.5 - |x|)
                grad += beta * np.reshape(wall_w, (N, 3)) * signs

                # Pair gradient
                pair_exp = exp_vals[N * 3:]
                pair_w = np.reshape(pair_exp, (N, N)) / sum_exp

                # For each pair (i,j), d(dist_ij/2)/dx_i = 0.5 * (p_i - p_j)/|p_i - p_j|
                # d(soft_min)/dx_i = -beta * sum_j w_ij * 0.5 * unit_ij
                # d(soft_min)/dx_j = +beta * sum_i w_ij * 0.5 * unit_ij

                # Vectorized: grad[i] -= beta * sum_j w_ij * 0.5 * (p_i - p_j)/|p_i - p_j|
                # grad[j] += beta * sum_i w_ij * 0.5 * (p_i - p_j)/|p_i - p_j|

                diff_full = p2[:, np.newaxis, :] - p2[np.newaxis, :, :]
                dists_full = np.sqrt(np.sum(diff_full ** 2, axis=2))
                # Avoid division by zero
                dists_full_safe = np.where(dists_full > 1e-12, dists_full, 1.0)
                unit_vecs = diff_full / dists_full_safe[:, :, np.newaxis]
                # Zero out diagonal
                unit_vecs[np.arange(N), np.arange(N)] = 0

                # grad[i] contribution from pairs: -beta * 0.5 * sum_j w_ij * unit_ij
                pair_grad = beta * 0.5 * np.sum(pair_w[:, :, np.newaxis] * unit_vecs, axis=1)
                grad -= pair_grad

                return -soft_min_val, -grad.reshape(-1)

            # Use L-BFGS-B
            res = minimize(
                obj_and_grad,
                p.reshape(-1),
                jac=True,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 300, 'maxfun': 20000, 'gtol': 1e-8}
            )

            new_p = res.x.reshape(N, 3)
            new_r = compute_radius(new_p)

            if new_r > best_r_local:
                best_r_local = new_r
                best_p_local = new_p.copy()
                p = new_p.copy()

            if time.time() > time_limit:
                break

        # Subgradient refinement pass
        p = best_p_local.copy()
        for iteration in range(200):
            if time.time() > time_limit:
                break

            # Find active constraints
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists)
            min_pair_idx = np.unravel_index(np.argmin(dists), (N, N))
            i, j = min_pair_idx[0], min_pair_idx[1]

            wall_dists = 0.5 - np.abs(p)
            min_wall_dist = np.min(wall_dists)
            min_wall_idx = np.unravel_index(np.argmin(wall_dists), (N, 3))
            wi, wk = min_wall_idx

            current_r = min(min_pair_dist / 2.0, min_wall_dist)

            # Adaptive step size
            step = max(0.001, 0.005 * (1.0 - iteration / 200.0))

            # Move spheres in min pair apart
            d_vec = p[i] - p[j]
            dist = np.linalg.norm(d_vec)
            if dist > 1e-8:
                d_vec = d_vec / dist
                p[i] += step * d_vec
                p[j] -= step * d_vec

            # Move sphere in min wall inward
            sign = 1.0 if p[wi, wk] > 0 else -1.0
            p[wi, wk] -= step * sign

            p = np.clip(p, -0.5, 0.5)

            new_r = compute_radius(p)
            if new_r > best_r_local:
                best_r_local = new_r
                best_p_local = p.copy()
            else:
                # Revert if not improving
                p = best_p_local.copy()

        return best_p_local, best_r_local

    # Run local search from grid
    time_left = deadline - time.time()
    best_pos = pos.copy()
    best_r = compute_radius(best_pos)

    if time_left > 5:
        grid_pos, grid_r = local_search_lbfgs(pos, time.time() + time_left * 0.35)
        if grid_r > best_r:
            best_r = grid_r
            best_pos = grid_pos

    # Try random restarts with remaining time
    time_left = deadline - time.time()
    if time_left > 15:
        rng = np.random.RandomState(42)
        rand_pos = rng.uniform(-0.45, 0.45, size=(N, 3))
        rand_pos, rand_r = local_search_lbfgs(rand_pos, time.time() + time_left * 0.5)
        if rand_r > best_r:
            best_r = rand_r
            best_pos = rand_pos

    # Another random restart if time remains
    time_left = deadline - time.time()
    if time_left > 15:
        rng = np.random.RandomState(123)
        rand_pos2 = rng.uniform(-0.45, 0.45, size=(N, 3))
        rand_pos2, rand_r2 = local_search_lbfgs(rand_pos2, time.time() + time_left * 0.6)
        if rand_r2 > best_r:
            best_r = rand_r2
            best_pos = rand_pos2

    # Final subgradient polish on best result
    time_left = deadline - time.time()
    if time_left > 3:
        p = best_pos.copy()
        for iteration in range(300):
            if time.time() > time_left:
                break

            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists)
            min_pair_idx = np.unravel_index(np.argmin(dists), (N, N))
            i, j = min_pair_idx[0], min_pair_idx[1]

            wall_dists = 0.5 - np.abs(p)
            min_wall_dist = np.min(wall_dists)
            min_wall_idx = np.unravel_index(np.argmin(wall_dists), (N, 3))
            wi, wk = min_wall_idx

            current_r = min(min_pair_dist / 2.0, min_wall_dist)

            step = max(0.0005, 0.002 * (1.0 - iteration / 300.0))

            d_vec = p[i] - p[j]
            dist = np.linalg.norm(d_vec)
            if dist > 1e-8:
                d_vec = d_vec / dist
                p[i] += step * d_vec
                p[j] -= step * d_vec

            sign = 1.0 if p[wi, wk] > 0 else -1.0
            p[wi, wk] -= step * sign

            p = np.clip(p, -0.5, 0.5)

            new_r = compute_radius(p)
            if new_r > best_r:
                best_r = new_r
                best_pos = p.copy()

    # Final validity check
    final_r = compute_radius(best_pos)
    if final_r < 0 or not np.all(np.isfinite(best_pos)):
        best_pos = pos.copy()

    return best_pos
