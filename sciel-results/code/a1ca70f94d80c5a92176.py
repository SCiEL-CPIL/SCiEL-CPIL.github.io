"""
Packing 200 equal circles in a unit square [-1/2, 1/2]^2.
Strategy:
1. Initialize with multiple diverse patterns (hexagonal variants, grid).
2. Minimize a smooth repulsive energy via L-BFGS-B with analytic gradient.
3. Refine via improved local search with adaptive steps.
4. Final high-precision polish.
"""

N = 200
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0

    dists_x = 0.5 - np.abs(c[:, 0])
    dists_y = 0.5 - np.abs(c[:, 1])
    min_boundary = np.minimum(np.min(dists_x), np.min(dists_y))

    if n < 2:
        return min_boundary

    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists) if len(dists) > 0 else np.inf

    return min(min_pair_dist / 2.0, min_boundary)

def _energy_and_grad(params, power=4.0):
    """
    Compute smooth repulsive energy and its gradient.
    Energy = sum_{i<j} 1/d_ij^power + sum_i sum_wall 1/d_iw^2
    For power=4: E_pair = sum_{i<j} 1/d_ij^4, grad = -4 * (x_i-x_j) / d_ij^6
    """
    c = params.reshape(N, 2)
    x = c[:, 0]
    y = c[:, 1]

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx**2 + dy**2
    np.fill_diagonal(d2, 1.0)

    # For power=4: E_pair = sum_{i<j} 1/d^4 = sum_{i<j} 1/d2^2
    # d/dx_i [1/d2^2] = -2 * d2^-3 * 2*(x_i-x_j) = -4*(x_i-x_j)*d2^-3
    d2_safe = d2.copy()
    np.fill_diagonal(d2_safe, 1.0)
    d2_neg2 = 1.0 / (d2_safe**2)
    d2_neg3 = 1.0 / (d2_safe**3)
    np.fill_diagonal(d2_neg2, 0.0)
    np.fill_diagonal(d2_neg3, 0.0)

    E_pair = np.triu(d2_neg2, 1).sum()

    # Gradient
    grad_x = -4.0 * (dx * d2_neg3).sum(axis=1)
    grad_y = -4.0 * (dy * d2_neg3).sum(axis=1)

    eps = 1e-8
    dx_w = np.clip(0.5 - x, eps, None)
    dx_w_neg = np.clip(0.5 + x, eps, None)
    dy_w = np.clip(0.5 - y, eps, None)
    dy_w_neg = np.clip(0.5 + y, eps, None)

    E_wall = (1.0/dx_w**2 + 1.0/dx_w_neg**2 + 1.0/dy_w**2 + 1.0/dy_w_neg**2).sum()

    grad_x += 2.0 / dx_w**3 - 2.0 / dx_w_neg**3
    grad_y += 2.0 / dy_w**3 - 2.0 / dy_w_neg**3

    total_E = E_pair + E_wall
    total_grad = np.concatenate([grad_x, grad_y])

    return total_E, total_grad

def _initialize_hexagonal(d_spacing=None):
    """Initialize with a hexagonal pattern with given spacing."""
    if d_spacing is None:
        d_spacing = 0.07

    points = []
    row_spacing = d_spacing * np.sqrt(3) / 2
    x_offset = d_spacing / 2

    y = -0.5 + d_spacing / 2
    row = 0
    while y <= 0.5 - d_spacing / 2:
        x = -0.5 + d_spacing / 2
        if row % 2 == 1:
            x += x_offset
        while x <= 0.5 - d_spacing / 2:
            points.append([x, y])
            x += d_spacing
        y += row_spacing
        row += 1

    points = np.array(points)
    if len(points) >= N:
        c = points[:N].copy()
        c = np.clip(c, -0.49, 0.49)
        return c
    return None

def _initialize_grid():
    """Initialize with a grid pattern."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _initialize_random(seed=42):
    """Initialize with random points."""
    rng = np.random.RandomState(seed)
    c = rng.uniform(-0.45, 0.45, size=(N, 2))
    return c

def _local_polish(c, max_iters=5000, time_limit=25.0, base_step=1e-4):
    """
    Fast local search using gradient of the smooth energy.
    Moves circles in the direction that reduces repulsive energy.
    """
    start_time = time.time()
    c = c.copy()
    current_r = _compute_radius(c)
    best_c = c.copy()
    best_r = current_r
    step = base_step
    stall_count = 0
    max_stall = 50

    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break

        # Compute energy gradient (fast, O(N^2) but vectorized)
        _, grad = _energy_and_grad(c.flatten(), power=4.0)
        grad = grad.reshape(N, 2)

        # Normalize gradient to get direction
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norm = np.where(grad_norm > 1e-12, grad_norm, 1.0)
        grad_dir = grad / grad_norm

        # Move all circles in the gradient direction (away from each other and walls)
        new_c = c + step * grad_dir

        # Clip to stay within bounds
        new_c = np.clip(new_c, -0.499, 0.499)

        new_r = _compute_radius(new_c)
        if new_r > current_r + 1e-12:
            c = new_c
            current_r = new_r
            stall_count = 0
            step = min(step * 1.05, 5e-4)
            if current_r > best_r:
                best_r = current_r
                best_c = c.copy()
        else:
            # Try smaller step
            for factor in [0.5, 0.25, 0.1]:
                new_c2 = c + step * factor * grad_dir
                new_c2 = np.clip(new_c2, -0.499, 0.499)
                new_r2 = _compute_radius(new_c2)
                if new_r2 > current_r + 1e-12:
                    c = new_c2
                    current_r = new_r2
                    stall_count = 0
                    step = min(step * 1.02, 5e-4)
                    if current_r > best_r:
                        best_r = current_r
                        best_c = c.copy()
                    break
            else:
                stall_count += 1
                step = max(step * 0.8, 1e-6)
                if stall_count > max_stall:
                    break

    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145

    # Generate multiple initializations
    inits = []

    # Hexagonal with different spacings
    for d in np.linspace(0.065, 0.075, 5):
        c_hex = _initialize_hexagonal(d)
        if c_hex is not None:
            inits.append(c_hex)

    # Grid
    inits.append(_initialize_grid())

    # Random with different seeds
    for seed in [42, 123, 456]:
        inits.append(_initialize_random(seed))

    # Evaluate each initialization and pick the best few
    init_results = []
    for c_init in inits:
        c_init = np.clip(c_init, -0.49, 0.49)
        r_init = _compute_radius(c_init)
        init_results.append((r_init, c_init))

    # Sort by radius and take top 3
    init_results.sort(key=lambda x: x[0], reverse=True)
    top_inits = [c for r, c in init_results[:3]]

    best_c = top_inits[0].copy()
    best_r = _compute_radius(best_c)

    # Run L-BFGS-B on the best initialization with power=4
    if time.time() < deadline - 50:
        try:
            bounds = [(-0.499, 0.499)] * (2 * N)

            res = minimize(
                _energy_and_grad,
                best_c.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 5000, 'maxls': 50}
            )

            c_opt = np.clip(res.x.reshape(N, 2), -0.499, 0.499)
            r_opt = _compute_radius(c_opt)

            if r_opt > best_r:
                best_r = r_opt
                best_c = c_opt.copy()

        except Exception:
            pass

    # If time permits, try L-BFGS-B on the second-best init too
    if time.time() < deadline - 35 and len(top_inits) > 1:
        try:
            c2 = top_inits[1]
            bounds = [(-0.499, 0.499)] * (2 * N)

            res2 = minimize(
                _energy_and_grad,
                c2.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 3000, 'maxls': 50}
            )

            c_opt2 = np.clip(res2.x.reshape(N, 2), -0.499, 0.499)
            r_opt2 = _compute_radius(c_opt2)

            if r_opt2 > best_r:
                best_r = r_opt2
                best_c = c_opt2.copy()

        except Exception:
            pass

    # Local polish using gradient-based method
    if time.time() < deadline - 10:
        time_for_polish = min(30.0, deadline - time.time() - 10)
        c_polished, r_polished = _local_polish(best_c, max_iters=10000, time_limit=time_for_polish, base_step=2e-4)
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished.copy()

    # Final high-precision polish with smaller steps
    if time.time() < deadline - 3:
        time_for_final = min(8.0, deadline - time.time() - 3)
        c_final, r_final = _local_polish(best_c, max_iters=5000, time_limit=time_for_final, base_step=5e-5)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()

    best_c = np.clip(best_c, -0.5, 0.5)

    return best_c
