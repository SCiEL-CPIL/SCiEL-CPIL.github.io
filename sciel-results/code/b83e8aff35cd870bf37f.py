import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _compute_radius(centers):
    """
    Computes the maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0

    # Wall constraints: distance from center to nearest wall
    abs_c = np.abs(centers)
    max_abs = np.max(abs_c, axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)

    if min_wall_dist < -1e-12:
        return 0.0

    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))

    # Set diagonal to infinity to exclude self-distance
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)

    # The radius is half the minimum pairwise distance, limited by wall distance
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _energy_and_gradient(x, R):
    """
    Calculates the repulsion energy and gradient for a target radius R.
    Energy is the sum of squared violations of the distance constraints.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)

    energy = 0.0
    grad = np.zeros_like(centers)

    # 1. Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)

    i_idx, j_idx = np.triu_indices(n, k=1)
    d_ij = dists[i_idx, j_idx]
    mask_pairs = d_ij < 2.0 * R

    if np.any(mask_pairs):
        v_ij = 2.0 * R - d_ij[mask_pairs]
        d_valid = d_ij[mask_pairs]
        i_valid = i_idx[mask_pairs]
        j_valid = j_idx[mask_pairs]

        energy += np.sum(v_ij**2)

        # Gradient of sum (2R - d)^2 w.r.t centers
        factors = 2.0 * v_ij / d_valid
        vecs = centers[i_valid] - centers[j_valid]
        forces = factors[:, np.newaxis] * vecs

        np.add.at(grad, i_valid, -forces)
        np.add.at(grad, j_valid, forces)

    # 2. Wall terms
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs

    mask_wall = wall_dists < R
    if np.any(mask_wall):
        violations_wall = R - wall_dists[mask_wall]
        i_valid_wall = np.where(mask_wall)[0]

        energy += np.sum(violations_wall**2)

        for idx in i_valid_wall:
            v = violations_wall[np.where(i_valid_wall == idx)[0][0]]
            max_k = np.argmax(abs_centers[idx])

            # Gradient of (R - (0.5 - |x_k|))^2
            grad[idx, max_k] += 2.0 * v * np.sign(centers[idx, max_k])

    return energy, grad.flatten()

def _soft_min_obj_and_grad(x, R, beta):
    """
    Vectorized smooth surrogate: soft-min of constraint slacks via log-sum-exp.
    We want to maximize min(slack), so minimize -soft_min(slack).
    """
    centers = x.reshape(-1, 3)
    n = len(centers)

    # Pairwise slacks (upper triangle)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)

    i_idx, j_idx = np.triu_indices(n, k=1)
    d_ij = dists[i_idx, j_idx]
    pair_slack = d_ij - 2.0 * R  # shape (M,)
    M = len(pair_slack)

    # Wall slacks
    abs_centers = np.abs(centers)
    wall_slack = 0.5 - abs_centers - R  # shape (N, 3)
    wall_slack_flat = wall_slack.flatten()

    # All slacks
    all_slack = np.concatenate([pair_slack, wall_slack_flat])
    total = len(all_slack)

    # -soft_min(slack) = (1/beta)*log(sum(exp(-beta*slack)))
    # Numerically stable
    min_slack = np.min(all_slack)
    shifted = -beta * (all_slack - min_slack)
    ex = np.exp(shifted)
    s = np.sum(ex)
    obj = (beta * min_slack + np.log(s)) / beta

    # Gradient weights: p_k = exp(-beta*slack_k) / sum(exp(-beta*slack_k))
    p = ex / s  # shape (total,)

    grad = np.zeros_like(centers)

    # Vectorized pairwise gradient
    p_pair = p[:M]  # shape (M,)
    vecs = centers[i_idx] - centers[j_idx]  # shape (M, 3)
    inv_d = 1.0 / d_ij  # shape (M,)
    
    # grad[i] += -p_pair * vecs / d
    # grad[j] += -p_pair * (-vecs) / d = +p_pair * vecs / d
    forces = p_pair[:, np.newaxis] * vecs * inv_d[:, np.newaxis]  # shape (M, 3)
    np.add.at(grad, i_idx, -forces)
    np.add.at(grad, j_idx, forces)

    # Vectorized wall gradient
    p_wall = p[M:].reshape(n, 3)
    signs = np.sign(centers)
    # d(wall_slack)/d(centers[i,k]) = -sign(centers[i,k])
    # grad[i,k] += -p_wall[i,k] * (-sign(centers[i,k])) = p_wall[i,k] * sign(centers[i,k])
    grad += p_wall * signs

    return obj, grad.flatten()

def _minimize_energy(init_centers, target_R, max_iter=500, ftol=1e-14, gtol=1e-12):
    """
    Minimizes the energy for a given target radius.
    """
    def objective_and_grad(x):
        val, g = _energy_and_gradient(x, target_R)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)

    try:
        res = minimize(
            objective_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
        )
        new_centers = res.x.reshape(N, 3)
        final_energy, _ = _energy_and_gradient(new_centers, target_R)
        return new_centers, final_energy, res.success
    except Exception:
        return init_centers, np.inf, False

def _minimize_soft_min(init_centers, target_R, beta, max_iter=1000, ftol=1e-15, gtol=1e-12):
    """
    High-precision polish using soft-min surrogate.
    """
    def objective_and_grad(x):
        val, g = _soft_min_obj_and_grad(x, target_R, beta)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)

    try:
        res = minimize(
            objective_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
        )
        new_centers = res.x.reshape(N, 3)
        return new_centers
    except Exception:
        return init_centers

def _generate_initial_positions():
    """
    Hybrid initialization: structured lattice with symmetry breaking.
    Uses a 5x4x5 grid with jitter for good spatial coverage.
    """
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)

    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    rng = np.random.default_rng(42)
    pos = grid[:N].copy() + rng.normal(0, 0.008, size=(N, 3))
    return np.clip(pos, -0.49, 0.49)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # 1. Initialization
    best_pos = _generate_initial_positions()
    
    # Initial optimization to get a good baseline
    initial_r = 0.1
    best_pos, energy, _ = _minimize_energy(best_pos, initial_r, max_iter=200)
    best_r = _compute_radius(best_pos)

    if best_r <= 0:
        rng = np.random.default_rng(42)
        best_pos = rng.uniform(-0.45, 0.45, size=(N, 3))
        best_r = _compute_radius(best_pos)

    # 2. Iterative Radius Increase with adaptive step
    current_r = best_r
    step_size = 0.001
    
    consecutive_failures = 0
    max_failures = 3

    rng_search = np.random.default_rng(123)

    while time.time() < deadline - 25.0:
        target_r = current_r + step_size

        if consecutive_failures >= max_failures:
            step_size *= 0.7
            consecutive_failures = 0
            if step_size < 1e-8:
                break

        # Try from current best position
        start_pos = np.clip(best_pos, -0.5, 0.5)
        
        # Use soft-min for the main optimization - it's smoother and faster
        max_iter = 400 if current_r < 0.11 else 600
        new_pos = _minimize_soft_min(start_pos, target_r, beta=100.0, max_iter=max_iter)
        
        actual_r = _compute_radius(new_pos)
        
        if actual_r >= target_r - 1e-7:
            best_pos = new_pos
            best_r = actual_r
            current_r = actual_r
            consecutive_failures = 0
        elif actual_r > current_r:
            best_pos = new_pos
            best_r = actual_r
            current_r = actual_r
            consecutive_failures = 0
        else:
            # Try a perturbed start
            perturbed = best_pos + rng_search.normal(0, 0.003, size=(N, 3))
            perturbed = np.clip(perturbed, -0.5, 0.5)
            new_pos2 = _minimize_soft_min(perturbed, target_r, beta=100.0, max_iter=300)
            actual_r2 = _compute_radius(new_pos2)
            if actual_r2 > current_r:
                best_pos = new_pos2
                best_r = actual_r2
                current_r = actual_r2
                consecutive_failures = 0
            else:
                consecutive_failures += 1

    # 3. Final high-precision polish
    if time.time() < deadline - 5.0:
        # Multiple beta values for refinement
        betas = [80.0, 150.0, 300.0]
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
            polished = _minimize_soft_min(best_pos, current_r, beta=beta, max_iter=800)
            r_polished = _compute_radius(polished)
            if r_polished > best_r:
                best_pos = polished
                best_r = r_polished
                current_r = r_polished

        # Final energy-based polish with tight tolerances
        if time.time() < deadline - 3.0:
            final_pos, _, _ = _minimize_energy(best_pos, current_r, max_iter=1500, ftol=1e-16, gtol=1e-14)
            r_final = _compute_radius(final_pos)
            if r_final > best_r:
                best_pos = final_pos
                best_r = r_final

    # Ensure valid output
    best_pos = np.clip(best_pos, -0.5, 0.5)
    if not np.all(np.isfinite(best_pos)):
        best_pos = _generate_initial_positions()

    return best_pos
