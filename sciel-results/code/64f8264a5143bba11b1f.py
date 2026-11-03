import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin

    # 1. Initialization: Hexagonal Lattice
    s_init = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    s_gen = s_init * 0.95
    y_step = s_gen * np.sqrt(3) / 2
    max_j = int(1.0 / y_step) + 2
    max_i = int(1.0 / s_gen) + 2

    points_list = []
    for j in range(-max_j, max_j + 1):
        y = j * y_step
        offset = 0 if j % 2 == 0 else s_gen / 2
        for i in range(-max_i, max_i + 1):
            x = i * s_gen + offset
            if np.hypot(x, y) < 1.0:
                points_list.append([x, y])

    points = np.array(points_list)

    if len(points) > N:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        points = points[idx]
    elif len(points) < N:
        while len(points) < N:
            p = np.random.uniform(-1, 1, 2)
            if np.linalg.norm(p) < 1.0:
                points = np.vstack([points, p])
        points = points[:N]

    points = points.copy()
    best_points = points.copy()
    best_score = 0.0

    # Precompute indices for pairwise distances
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    num_pairs = len(i_idx)

    # Optimization parameters
    alpha = 0.02
    cooling = 0.9999
    k_soft_start = 5.0
    k_soft_end = 200.0
    max_iter = 10000
    iter_count = 0
    stall_count = 0
    stall_threshold = 100

    k_soft = k_soft_start

    # Phase 1: Main optimization
    while time.time() - start_time < time_limit * 0.7 and iter_count < max_iter:
        # Compute pairwise distances efficiently
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)

        # Compute boundary distances
        rad = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - rad

        # All constraint values
        pair_vals = dists[i_idx, j_idx] / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])

        current_min = np.min(all_vals)

        if current_min > best_score:
            best_score = current_min
            best_points = points.copy()
            stall_count = 0
        else:
            stall_count += 1

        # Compute Soft-Min Weights using log-sum-exp for stability
        shifted_vals = all_vals - current_min
        max_shift = np.max(shifted_vals)
        if max_shift > 50.0:
            mask_close = shifted_vals < 50.0
            weights = np.zeros_like(all_vals)
            if np.any(mask_close):
                close_vals = shifted_vals[mask_close]
                close_weights = np.exp(-k_soft * close_vals)
                weights[mask_close] = close_weights
                sum_weights = np.sum(weights)
                if sum_weights > 0:
                    weights /= sum_weights
                else:
                    weights = np.ones_like(all_vals) / len(all_vals)
            else:
                weights = np.ones_like(all_vals) / len(all_vals)
        else:
            weights = np.exp(-k_soft * shifted_vals)
            sum_weights = np.sum(weights)
            if sum_weights > 0:
                weights /= sum_weights
            else:
                weights = np.ones_like(all_vals) / len(all_vals)

        # Compute Gradients
        grads = np.zeros_like(points)

        # Pair contributions
        pair_weights = weights[:num_pairs]
        d_pair = dists[i_idx, j_idx]
        mask_nonzero = d_pair > 1e-9
        if np.any(mask_nonzero):
            vecs = np.zeros((num_pairs, 2))
            vecs[mask_nonzero] = (diff[i_idx[mask_nonzero], j_idx[mask_nonzero]] / d_pair[mask_nonzero, np.newaxis]) * pair_weights[mask_nonzero, np.newaxis]

            np.add.at(grads, i_idx, vecs)
            np.add.at(grads, j_idx, -vecs)

        # Boundary contributions
        bound_weights = weights[num_pairs:]
        rad_safe = np.where(rad > 1e-9, rad, 1.0)
        bound_grads = - (points / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads

        # Normalize gradient to prevent large steps
        grad_norm = np.linalg.norm(grads)
        if grad_norm > 1.0:
            grads /= grad_norm

        # Update points
        points += alpha * grads

        # Project back inside circle
        rad_new = np.linalg.norm(points, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            points[mask] /= rad_new[mask, np.newaxis]

        # Cool down
        alpha *= cooling

        # Increase k_soft over time
        progress = iter_count / max_iter
        k_soft = k_soft_start + (k_soft_end - k_soft_start) * (progress ** 2)

        # Random perturbation to escape local minima
        if stall_count > stall_threshold:
            noise_scale = alpha * 3.0
            points += np.random.normal(0, noise_scale, size=points.shape)
            rad_new = np.linalg.norm(points, axis=1)
            mask = rad_new > 1.0
            if np.any(mask):
                points[mask] /= rad_new[mask, np.newaxis]
            stall_count = 0
            alpha *= 0.8

        iter_count += 1

    # Phase 2: High-precision polish with adaptive step size
    points = best_points.copy()
    polish_alpha = 0.001
    polish_k = 500.0
    polish_time_limit = time_limit * 0.95
    polish_iter = 0
    max_polish_iter = 5000

    while time.time() - start_time < polish_time_limit and polish_iter < max_polish_iter:
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)

        rad = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - rad

        pair_vals = dists[i_idx, j_idx] / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])

        current_min = np.min(all_vals)

        if current_min > best_score:
            best_score = current_min
            best_points = points.copy()

        # Very sharp soft-min for polish
        shifted_vals = all_vals - current_min
        mask_close = shifted_vals < 20.0
        weights = np.zeros_like(all_vals)
        if np.any(mask_close):
            close_vals = shifted_vals[mask_close]
            close_weights = np.exp(-polish_k * close_vals)
            weights[mask_close] = close_weights
            sum_weights = np.sum(weights)
            if sum_weights > 0:
                weights /= sum_weights
            else:
                weights = np.ones_like(all_vals) / len(all_vals)
        else:
            weights = np.ones_like(all_vals) / len(all_vals)

        grads = np.zeros_like(points)

        pair_weights = weights[:num_pairs]
        d_pair = dists[i_idx, j_idx]
        mask_nonzero = d_pair > 1e-9
        if np.any(mask_nonzero):
            vecs = np.zeros((num_pairs, 2))
            vecs[mask_nonzero] = (diff[i_idx[mask_nonzero], j_idx[mask_nonzero]] / d_pair[mask_nonzero, np.newaxis]) * pair_weights[mask_nonzero, np.newaxis]

            np.add.at(grads, i_idx, vecs)
            np.add.at(grads, j_idx, -vecs)

        bound_weights = weights[num_pairs:]
        rad_safe = np.where(rad > 1e-9, rad, 1.0)
        bound_grads = - (points / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads

        grad_norm = np.linalg.norm(grads)
        if grad_norm > 1.0:
            grads /= grad_norm

        # Adaptive step: try to increase if moving toward feasibility
        points += polish_alpha * grads

        rad_new = np.linalg.norm(points, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            points[mask] /= rad_new[mask, np.newaxis]

        # Check if we improved
        diff2 = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists2 = np.sqrt(np.sum(diff2**2, axis=2))
        np.fill_diagonal(dists2, np.inf)
        rad2 = np.linalg.norm(points, axis=1)
        bound_dists2 = 1.0 - rad2
        pair_vals2 = dists2[i_idx, j_idx] / 2.0
        all_vals2 = np.concatenate([pair_vals2, bound_dists2])
        current_min2 = np.min(all_vals2)

        if current_min2 > current_min:
            polish_alpha *= 1.05  # Can take bigger steps
        else:
            polish_alpha *= 0.8   # Step was too large

        polish_k *= 1.001
        polish_iter += 1

    return best_points
