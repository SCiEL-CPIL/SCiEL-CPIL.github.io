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

def _energy_and_grad(params, power=2.0):
    """
    Compute smooth repulsive energy and its gradient.
    Energy = sum_{i<j} 1/d_ij^power + sum_i sum_wall 1/d_iw^2
    """
    c = params.reshape(N, 2)
    x = c[:, 0]
    y = c[:, 1]

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx**2 + dy**2
    np.fill_diagonal(d2, 1.0)

    # Use d^power in denominator: 1/d^power = 1/(d2)^(power/2)
    # For power=2: 1/d2
    # For power=4: 1/d2^2
    if power == 2.0:
        inv_d = 1.0 / d2
        inv_d_pow2 = inv_d**2  # for gradient: d/dx (1/d2) = -2*dx/d2^2
    elif power == 4.0:
        inv_d = 1.0 / (d2**2)
        inv_d_pow2 = (1.0 / (d2**2))**2  # for gradient
    else:
        # General: 1/d^p = 1/(d2)^(p/2)
        half_p = power / 2.0
        inv_d = 1.0 / (d2**half_p)
        # d/dx_i [1/(d2)^(p/2)] = -(p/2) * (d2)^(p/2-1) * 2*(x_i-x_j) = -p * (x_i-x_j) * (d2)^(p/2-1)
        # = -p * dx * (d2)^((p-2)/2)
        # Let's just compute numerically if power is not 2 or 4
        inv_d_pow2 = None

    np.fill_diagonal(inv_d, 0.0)

    E_pair = np.triu(inv_d, 1).sum()

    # Gradient
    grad_x = np.zeros(N)
    grad_y = np.zeros(N)

    if power == 2.0:
        inv_d4 = inv_d**2
        np.fill_diagonal(inv_d4, 0.0)
        grad_x -= 2.0 * (dx * inv_d4).sum(axis=1)
        grad_y -= 2.0 * (dy * inv_d4).sum(axis=1)
    elif power == 4.0:
        # d/dx_i [1/d2^2] = -4 * dx * d2^-3
        d2_safe = d2.copy()
        np.fill_diagonal(d2_safe, 1.0)
        d2_neg3 = 1.0 / (d2_safe**3)
        np.fill_diagonal(d2_neg3, 0.0)
        grad_x -= 4.0 * (dx * d2_neg3).sum(axis=1)
        grad_y -= 4.0 * (dy * d2_neg3).sum(axis=1)
    else:
        # Fallback: finite difference would be too slow, use power=2
        inv_d4 = inv_d**2
        np.fill_diagonal(inv_d4, 0.0)
        grad_x -= 2.0 * (dx * inv_d4).sum(axis=1)
        grad_y -= 2.0 * (dy * inv_d4).sum(axis=1)

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
    Improved local search to maximize minimum distance.
    Uses adaptive step sizes and tries multiple perturbations.
    """
    start_time = time.time()
    c = c.copy()
    current_r = _compute_radius(c)
    best_c = c.copy()
    best_r = current_r
    step = base_step
    stall_count = 0
    max_stall = 100

    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break

        dist_matrix = np.sqrt(np.sum((c[:, np.newaxis, :] - c[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf)

        min_val = np.min(dist_matrix)
        i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)

        bdists = np.minimum(
            np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0]),
            np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1])
        )
        min_boundary_val = np.min(bdists)
        b_idx = np.argmin(bdists)

        improved = False

        if min_val / 2.0 < min_boundary_val:
            # Move i and j apart
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                unit_vec = vec / norm

                # Try multiple step sizes
                for s in [step, step * 0.5, step * 2.0]:
                    new_ci = c[i] + unit_vec * s
                    new_cj = c[j] - unit_vec * s

                    if np.all(np.abs(new_ci) < 0.5) and np.all(np.abs(new_cj) < 0.5):
                        old_ci, old_cj = c[i].copy(), c[j].copy()
                        c[i] = new_ci
                        c[j] = new_cj
                        new_r = _compute_radius(c)
                        if new_r > current_r:
                            current_r = new_r
                            improved = True
                            break
                        else:
                            c[i] = old_ci
                            c[j] = old_cj

                if not improved:
                    # Try random perturbation of the pair
                    for _ in range(3):
                        rand_dir = np.random.randn(2)
                        rand_dir /= np.linalg.norm(rand_dir)
                        new_ci = c[i] + rand_dir * step * 0.5
                        new_cj = c[j] - rand_dir * step * 0.5
                        if np.all(np.abs(new_ci) < 0.5) and np.all(np.abs(new_cj) < 0.5):
                            old_ci, old_cj = c[i].copy(), c[j].copy()
                            c[i] = new_ci
                            c[j] = new_cj
                            new_r = _compute_radius(c)
                            if new_r > current_r:
                                current_r = new_r
                                improved = True
                                break
                            else:
                                c[i] = old_ci
                                c[j] = old_cj

        else:
            # Move boundary circle inwards
            x, y = c[b_idx]
            d_left = 0.5 - x
            d_right = x + 0.5
            d_down = 0.5 - y
            d_up = y + 0.5

            min_d = min(d_left, d_right, d_down, d_up)

            for s in [step, step * 0.5, step * 2.0]:
                if min_d == d_left:
                    new_pt = c[b_idx] + np.array([s, 0])
                elif min_d == d_right:
                    new_pt = c[b_idx] + np.array([-s, 0])
                elif min_d == d_down:
                    new_pt = c[b_idx] + np.array([0, s])
                else:
                    new_pt = c[b_idx] + np.array([0, -s])

                if np.all(np.abs(new_pt) < 0.5):
                    old_pt = c[b_idx].copy()
                    c[b_idx] = new_pt
                    new_r = _compute_radius(c)
                    if new_r > current_r:
                        current_r = new_r
                        improved = True
                        break
                    else:
                        c[b_idx] = old_pt

        if improved:
            stall_count = 0
            step = min(step * 1.1, 1e-3)  # Increase step slightly
            if current_r > best_r:
                best_r = current_r
                best_c = c.copy()
        else:
            stall_count += 1
            step = max(step * 0.7, 1e-6)  # Decrease step
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

    # Run L-BFGS-B on the best initialization
    if time.time() < deadline - 40:
        try:
            bounds = [(-0.499, 0.499)] * (2 * N)

            res = minimize(
                _energy_and_grad,
                best_c.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 2000}
            )

            c_opt = np.clip(res.x.reshape(N, 2), -0.499, 0.499)
            r_opt = _compute_radius(c_opt)

            if r_opt > best_r:
                best_r = r_opt
                best_c = c_opt.copy()

        except Exception:
            pass

    # If time permits, try L-BFGS-B on the second-best init too
    if time.time() < deadline - 30 and len(top_inits) > 1:
        try:
            c2 = top_inits[1]
            bounds = [(-0.499, 0.499)] * (2 * N)

            res2 = minimize(
                _energy_and_grad,
                c2.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000}
            )

            c_opt2 = np.clip(res2.x.reshape(N, 2), -0.499, 0.499)
            r_opt2 = _compute_radius(c_opt2)

            if r_opt2 > best_r:
                best_r = r_opt2
                best_c = c_opt2.copy()

        except Exception:
            pass

    # Local polish
    if time.time() < deadline - 5:
        time_for_polish = min(25.0, deadline - time.time() - 5)
        c_polished, r_polished = _local_polish(best_c, max_iters=5000, time_limit=time_for_polish)
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished.copy()

    # Final high-precision polish with smaller steps
    if time.time() < deadline - 2:
        time_for_final = min(5.0, deadline - time.time() - 2)
        c_final, r_final = _local_polish(best_c, max_iters=2000, time_limit=time_for_final, base_step=1e-5)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()

    best_c = np.clip(best_c, -0.5, 0.5)

    return best_c
