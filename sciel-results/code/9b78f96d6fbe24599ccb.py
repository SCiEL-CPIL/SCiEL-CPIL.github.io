"""
Structural change: replace the ad-hoc SA + L-BFGS soft-min pipeline with a
deterministic, differentiable repulsion-energy solver.

The core idea is to model the packing as a smooth energy minimization where
the objective is a log-sum-exp (soft-min) of all constraint "violations" for
a fixed target radius. Because the energy landscape is smooth and
differentiable, we can use scipy's L-BFGS-B directly on the full 100-D
coordinate space without needing a discrete search phase like simulated
annealing. The "global" search is achieved by annealing the target radius
from a small value up to the optimum, which effectively cools the system
into the best packing configuration.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
DEADLINE = 147.0  # Safety margin before 150s kill

# Precompute upper-triangle indices for pairwise operations
_IU = np.triu_indices(N, k=1)
_NPAIRS = len(_IU[0])


def _compute_radius(centers):
    """Compute the true maximum radius for a given configuration."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0

    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))

    if min_wall < 0:
        return 0.0

    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair = np.sqrt(np.min(dist_sq)) / 2.0

    return min(min_pair, min_wall)


def _soft_violation_obj(x, r_target, beta):
    """
    Objective: Minimize smooth violation for target radius r_target.
    Uses log-sum-exp for numerical stability.
    """
    centers = x.reshape(-1, 2)

    # Wall violations: we want wall_dist >= r_target
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    wall_viol = r_target - walls

    # Pair violations: we want dist >= 2*r_target
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    pair_viol = 2.0 * r_target - dists

    # Soft sum of violations: sum(log(1+exp(beta*viol))/beta)
    # This is 0 when all violations are <= 0 (feasible) and grows when violated
    z_wall = np.clip(beta * wall_viol, -50, 50)
    z_pair = np.clip(beta * pair_viol, -50, 50)

    soft_wall = np.log1p(np.exp(z_wall)) / beta
    soft_pair = np.log1p(np.exp(z_pair)) / beta

    return float(np.sum(soft_wall) + np.sum(soft_pair))


def _soft_violation_grad(x, r_target, beta):
    """
    Gradient of the soft violation objective. Fully vectorized.
    """
    centers = x.reshape(-1, 2)
    grad = np.zeros_like(x)

    # Wall gradients
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    wall_viol = r_target - walls

    z_wall = np.clip(beta * wall_viol, -50, 50)
    sig_wall = 1.0 / (1.0 + np.exp(-z_wall))  # sigmoid

    # Determine which wall is limiting for each circle
    wx = wall_dists_x < wall_dists_y
    wy = wall_dists_y < wall_dists_x
    wt = ~(wx | wy)

    idx_wx = np.nonzero(wx)[0]
    idx_wy = np.nonzero(wy)[0]
    idx_wt = np.nonzero(wt)[0]

    if len(idx_wx) > 0:
        grad[2 * idx_wx] += sig_wall[wx] * np.sign(centers[idx_wx, 0])
    if len(idx_wy) > 0:
        grad[2 * idx_wy + 1] += sig_wall[wy] * np.sign(centers[idx_wy, 1])
    if len(idx_wt) > 0:
        grad[2 * idx_wt] += 0.5 * sig_wall[wt] * np.sign(centers[idx_wt, 0])
        grad[2 * idx_wt + 1] += 0.5 * sig_wall[wt] * np.sign(centers[idx_wt, 1])

    # Pair gradients
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff_vec = ci - cj
    dists = np.sqrt(np.sum(diff_vec ** 2, axis=1))
    dists_safe = np.maximum(dists, 1e-12)
    pair_viol = 2.0 * r_target - dists

    z_pair = np.clip(beta * pair_viol, -50, 50)
    sig_pair = 1.0 / (1.0 + np.exp(-z_pair))

    # d(soft_pair)/d(c_i) = -sig * (c_i - c_j) / dist
    factor = sig_pair / dists_safe
    contrib = factor[:, None] * diff_vec

    np.add.at(grad, 2 * _IU[0], -contrib[:, 0])
    np.add.at(grad, 2 * _IU[0] + 1, -contrib[:, 1])
    np.add.at(grad, 2 * _IU[1], contrib[:, 0])
    np.add.at(grad, 2 * _IU[1] + 1, contrib[:, 1])

    return grad


def _initial_grid():
    """Create initial grid layout with slight jitter."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N].copy()

    rng = np.random.default_rng(12345)
    grid = grid + rng.normal(0, 0.005, grid.shape)
    grid = np.clip(grid, -0.49, 0.49)

    return grid


def _anneal_radius(x0, beta_schedule, time_budget):
    """
    Anneal the target radius from small to large, optimizing coordinates at each step.
    This provides a smooth path through the configuration space.
    """
    start_time = time.time()
    x = x0.copy()
    best_x = x.copy()
    best_r = _compute_radius(x)

    # Determine a reasonable starting radius
    r_start = 0.01
    r_end = 0.075  # Slightly above known best

    # Create a schedule of radii to try
    n_steps = len(beta_schedule)
    radii = np.linspace(r_start, r_end, n_steps)

    for i, (r_target, beta) in enumerate(zip(radii, beta_schedule)):
        if time.time() - start_time > time_budget:
            break

        def obj_and_grad(x):
            if time.time() - start_time > time_budget:
                return 0.0, np.zeros_like(x)
            f = _soft_violation_obj(x, r_target, beta)
            g = _soft_violation_grad(x, r_target, beta)
            return f, g

        try:
            result = opt.minimize(
                obj_and_grad, x, jac=True, method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8,
                         'maxfun': 500, 'disp': False}
            )
            x = result.x
            x = np.clip(x, -0.5, 0.5)

            # Check if this configuration is feasible for r_target
            r_actual = _compute_radius(x)
            if r_actual > best_r:
                best_r = r_actual
                best_x = x.copy()

        except Exception:
            pass

    return best_x, best_r


def _high_precision_polish(centers, time_budget):
    """
    Final high-precision polish using very tight tolerances and high beta.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)

    # Try to push the radius up slightly with high-precision optimization
    r_target = best_r * 1.001  # Try to achieve slightly more
    beta = 100.0  # High beta for sharpness

    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f = _soft_violation_obj(x, r_target, beta)
        g = _soft_violation_grad(x, r_target, beta)
        return f, g

    try:
        result = opt.minimize(
            obj_and_grad, best.flatten(), jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12,
                     'maxfun': 2000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        r_polished = _compute_radius(polished)
        if r_polished > best_r:
            best_r = r_polished
            best = polished
    except Exception:
        pass

    # Also try optimizing for the current best radius with even higher beta
    r_target = best_r
    beta = 200.0

    def obj_and_grad2(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f = _soft_violation_obj(x, r_target, beta)
        g = _soft_violation_grad(x, r_target, beta)
        return f, g

    try:
        result = opt.minimize(
            obj_and_grad2, best.flatten(), jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12,
                     'maxfun': 2000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        r_polished = _compute_radius(polished)
        if r_polished > best_r:
            best_r = r_polished
            best = polished
    except Exception:
        pass

    return best, best_r


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    # Initial configuration
    best_centers = _initial_grid()
    best_radius = _compute_radius(best_centers)

    # Phase 1: Anneal radius with increasing beta
    # Start with low beta (smooth landscape) and increase for sharpness
    beta_schedule = [10.0, 20.0, 50.0, 100.0]
    sa_budget = min(80.0, (deadline - time.time()) * 0.6)
    if sa_budget > 10:
        best_centers, best_radius = _anneal_radius(
            best_centers.flatten(), beta_schedule, sa_budget
        )

    # Phase 2: High-precision polish
    if time.time() < deadline - 20:
        best_centers, best_radius = _high_precision_polish(
            best_centers, time_budget=min(25, deadline - time.time() - 10)
        )

    # Phase 3: Try a few more polish passes with different starting points
    # Slightly perturb the best solution and re-optimize
    rng = np.random.default_rng(42)
    if time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            perturbed = best_centers + rng.normal(0, 0.001, best_centers.shape)
            perturbed = np.clip(perturbed, -0.5, 0.5)

            r_target = best_radius * 1.0005
            beta = 150.0

            def obj_and_grad(x):
                if time.time() - start_time > deadline:
                    return 0.0, np.zeros_like(x)
                f = _soft_violation_obj(x, r_target, beta)
                g = _soft_violation_grad(x, r_target, beta)
                return f, g

            try:
                result = opt.minimize(
                    obj_and_grad, perturbed.flatten(), jac=True, method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (2 * N),
                    options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12,
                             'maxfun': 1000, 'disp': False}
                )
                polished = result.x.reshape(-1, 2)
                polished = np.clip(polished, -0.5, 0.5)
                r_polished = _compute_radius(polished)
                if r_polished > best_radius:
                    best_radius = r_polished
                    best_centers = polished
            except Exception:
                pass

    # Final validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()

    return best_centers
