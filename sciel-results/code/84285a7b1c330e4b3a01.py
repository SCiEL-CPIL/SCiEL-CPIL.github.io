import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers in [-0.5, 0.5]^3.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0

    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)

    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dists_sq))

    return min(min_wall_dist, min_pair_dist / 2.0)


def _soft_min_objective(centers_flat, beta, iu):
    """
    Objective function for L-BFGS-B.
    Minimizes the soft-min approximation of the radius.
    r(p) approx -1/beta * ln(sum(exp(-beta * x_k)))
    where x_k are the constraints (distances).
    We want to maximize r, so we minimize -r, which is 1/beta * ln(sum(exp(-beta * x_k))).
    """
    centers = centers_flat.reshape(-1, 3)

    # 1. Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()

    # 2. Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    pair_vals = np.sqrt(dists_sq) / 2.0
    pair_vals_unique = pair_vals[iu]

    all_vals = np.concatenate([wall_vals, pair_vals_unique])

    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta

    return obj


def _build_initial_grid(nx, ny, nz, noise_scale=0.005, seed=42):
    """Build a grid initialization with small noise to break symmetry."""
    margin = 0.02
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    # Add small noise to break perfect symmetry
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, noise_scale, centers.shape)

    # Clip to valid range
    centers = np.clip(centers, -0.499, 0.499)

    return centers


def _optimize_from_initial(centers, deadline, beta_start=50.0, beta_end=5000.0, n_beta_steps=8):
    """
    Run L-BFGS-B optimization with increasing beta from initial centers.
    Returns (best_centers, best_radius).
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)

    # Precompute upper triangle indices
    iu = np.triu_indices(N, k=1)

    # Create beta schedule: geometric progression from beta_start to beta_end
    betas = np.geomspace(beta_start, beta_end, n_beta_steps)

    for beta in betas:
        if time.time() > deadline:
            break

        x0 = best_centers.flatten()

        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )

            # Check if we improved
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)

            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass

        # Check time
        if time.time() > deadline - 5.0:
            break

    # Final polish pass with very high beta and tight tolerances
    if time.time() < deadline - 10.0:
        try:
            x0 = best_centers.flatten()
            res = minimize(
                _soft_min_objective,
                x0,
                args=(20000.0, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )

            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)

            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass

    return best_centers, best_radius


def construct():
    start_time = time.time()
    # Safety margin: 110s is well within 120s.
    deadline = start_time + 110.0

    best_centers = None
    best_radius = -np.inf

    # Multiple grid configurations with different noise levels
    grid_configs = [
        (5, 5, 4),
        (4, 5, 5),
        (5, 4, 5),
        (10, 5, 2),
        (5, 10, 2),
    ]

    noise_scales = [0.003, 0.005, 0.01]

    # Time allocation: focus on top configurations
    # We'll try each grid config with different noise levels
    total_configs = len(grid_configs) * len(noise_scales)
    time_per_config = (deadline - start_time) / total_configs

    config_idx = 0
    for nx, ny, nz in grid_configs:
        for noise_scale in noise_scales:
            if time.time() > deadline - 5.0:
                break

            config_idx += 1
            seed = 42 + config_idx

            centers = _build_initial_grid(nx, ny, nz, noise_scale, seed)
            r_init = _compute_radius(centers)

            if r_init > best_radius:
                best_radius = r_init
                best_centers = centers.copy()

            # Allocate time for this configuration
            slice_deadline = min(time.time() + time_per_config, deadline - 2.0)

            # Optimize from this initial
            opt_centers, opt_radius = _optimize_from_initial(
                centers, slice_deadline,
                beta_start=50.0, beta_end=5000.0, n_beta_steps=7
            )

            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers

    # Final polish on best found
    if best_centers is not None and time.time() < deadline - 2.0:
        c_final, r_final = _optimize_from_initial(best_centers, deadline - 0.5,
                                                   beta_start=5000.0, beta_end=20000.0, n_beta_steps=3)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = c_final.copy()

    if best_centers is None:
        best_centers = _build_initial_grid(5, 5, 4, 0.01)

    return best_centers

# EVOLVE-BLOCK-END
