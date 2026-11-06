"""
Pack 200 equal circles in the unit square [-1/2, 1/2]^2 to maximize common radius.
Uses hexagonal grid initialization followed by L-BFGS-B optimization of a
soft-min surrogate of the minimum distance, with multiple restarts.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 200
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s kill


def _remaining_time():
    return TIME_LIMIT - (time.time() - START_TIME)


def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    if len(centers) == 0:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Wall distances: min_i (0.5 - |c_i|)
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    wall_dists = np.minimum(dx, dy)
    min_wall = np.min(wall_dists)
    
    if min_wall < 0:
        return 0.0
        
    if N < 2:
        return float(min_wall)
        
    d = pdist(centers)
    min_pair = np.min(d)
    pair_term = min_pair / 2.0
    
    return min(float(min_wall), float(pair_term))


def _soft_min_objective_and_grad(x, beta=100.0):
    """
    Soft-min objective for L-BFGS-B.
    We want to maximize r = min(min_pair_dist/2, min_wall_dist).
    Equivalent to minimizing -r.
    
    Uses log-sum-exp trick: -min(d_i) ≈ (1/beta) * log(sum(exp(-beta * d_i)))
    for large beta.
    
    x is a flat array of shape (2*N,).
    Returns (objective_value, gradient).
    """
    centers = x.reshape(N, 2)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    wall_dists = np.minimum(dx, dy)
    
    # Combine all "distance" values: pairwise/2 and wall distances
    # We want min of all these.
    pair_half_dists = dists / 2.0
    # Flatten pair distances (upper triangle only to avoid double counting)
    iu = np.triu_indices(N, k=1)
    pair_vals = pair_half_dists[iu]
    all_vals = np.concatenate([pair_vals, wall_dists])
    
    # Soft min: (1/beta) * log(sum(exp(-beta * vals)))
    # For numerical stability, subtract min
    min_val = np.min(all_vals)
    if min_val < 1e-10:
        # Degenerate case
        return (1.0, np.zeros_like(x))
    
    exp_terms = np.exp(-beta * (all_vals - min_val))
    soft_min_val = min_val + np.log(np.sum(exp_terms)) / beta
    
    # Gradient with respect to centers
    # d(soft_min)/d(val_k) = exp(-beta*(val_k - min_val)) / sum(exp(...))
    weights = exp_terms / np.sum(exp_terms)
    
    # Gradient of pairwise distances
    grad_centers = np.zeros((N, 2))
    
    # For each pair (i,j), d/d(c_i) of dist(i,j)/2 = (c_i - c_j) / (2 * dist(i,j))
    # d/d(c_j) of dist(i,j)/2 = (c_j - c_i) / (2 * dist(i,j))
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        w = weights[idx]
        d = dists[i, j]
        if d < 1e-10:
            continue
        # Gradient contribution from pair (i,j)
        diff_ij = centers[i] - centers[j]
        grad_ij = diff_ij / (2.0 * d)
        grad_centers[i] += w * grad_ij
        grad_centers[j] -= w * grad_ij
    
    # Gradient of wall distances
    for idx in range(N):
        w = weights[len(pair_vals) + idx]
        # d/d(c_i) of min(0.5 - |x_i|, 0.5 - |y_i|)
        # This is piecewise. Use subgradient.
        x_i = centers[idx, 0]
        y_i = centers[idx, 1]
        dx_i = 0.5 - abs(x_i)
        dy_i = 0.5 - abs(y_i)
        
        if dx_i <= dy_i:
            # Wall distance is determined by x
            if x_i > 0:
                grad_centers[idx, 0] -= w  # d(0.5 - x)/dx = -1
            else:
                grad_centers[idx, 0] += w  # d(0.5 + x)/dx = 1
        else:
            # Wall distance is determined by y
            if y_i > 0:
                grad_centers[idx, 1] -= w
            else:
                grad_centers[idx, 1] += w
    
    # Objective is -soft_min, so gradient is -grad_soft_min
    obj = -soft_min_val
    grad = -grad_centers.flatten()
    
    return (obj, grad)


def _hexagonal_grid(n_circles):
    """
    Create a hexagonal packing arrangement for n_circles circles.
    Returns an array of shape (n_circles, 2).
    """
    best_centers = None
    best_r = 0.0

    # Try different numbers of rows
    # For N=200, rows around 10-20 are good.
    for m in range(10, 25):
        base_per_row = N // m
        extra = N % m

        row_counts = []
        for i in range(m):
            if i < extra:
                row_counts.append(base_per_row + 1)
            else:
                row_counts.append(base_per_row)

        max_row_count = max(row_counts)

        # Estimate radius
        # Width constraint: max_row_count * 2r <= 1  => r <= 1/(2*max_row_count)
        # Height constraint: (m-1)*sqrt(3)*r + 2r <= 1 => r <= 1/((m-1)*sqrt(3) + 2)
        r_width = 1.0 / (2.0 * max_row_count)
        r_height = 1.0 / ((m - 1) * np.sqrt(3) + 2)
        r_est = min(r_width, r_height)

        if r_est <= best_r and best_centers is not None:
            continue

        dx = 2.0 * r_est
        dy = np.sqrt(3) * r_est

        centers = []
        y_offset = -0.5 + r_est

        for row_idx, row_count in enumerate(row_counts):
            x_offset = 0.0
            if row_idx % 2 == 1:
                x_offset = dx / 2.0

            if row_count > 0:
                row_width = (row_count - 1) * dx
                start_x = -row_width / 2.0 + x_offset
            else:
                start_x = 0.0

            for col_idx in range(row_count):
                x = start_x + col_idx * dx
                y = y_offset + row_idx * dy
                centers.append([x, y])

        centers = np.array(centers)

        # Center the arrangement in the square
        centers[:, 0] -= np.mean(centers[:, 0])
        centers[:, 1] -= np.mean(centers[:, 1])

        # Clip to be within bounds
        centers = np.clip(centers, -0.5 + 1e-9, 0.5 - 1e-9)

        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()

    return best_centers if best_centers is not None else np.random.uniform(-0.5, 0.5, (N, 2))


def _refine_lbfgs(centers, beta=100.0, max_time=10.0):
    """
    Refine the packing using L-BFGS-B on a soft-min objective.
    """
    x0 = centers.flatten()
    
    # Define objective and gradient wrapper
    def obj_and_grad(x):
        if _remaining_time() < 2.0:
            return (0.0, np.zeros_like(x))
        return _soft_min_objective_and_grad(x, beta=beta)
    
    # Bounds: centers must be in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_r = _compute_radius(centers)
    best_x = x0.copy()
    
    try:
        result = minimize(
            lambda x: obj_and_grad(x)[0],
            x0,
            jac=lambda x: obj_and_grad(x)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 200,
                'maxfun': 2000,
                'ftol': 1e-8,
                'gtol': 1e-6,
                'maxls': 50,
            }
        )
        
        if result.x is not None and np.all(np.isfinite(result.x)):
            new_centers = result.x.reshape(N, 2)
            new_r = _compute_radius(new_centers)
            if new_r > best_r:
                best_r = new_r
                best_x = result.x.copy()
    except Exception:
        pass
    
    return best_x.reshape(N, 2), best_r


def construct():
    """Construct optimal circle packing."""
    # Start with hexagonal grid
    centers = _hexagonal_grid(N)
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Refine with L-BFGS-B using soft-min objective
    if _remaining_time() > 5.0:
        try:
            refined_centers, refined_radius = _refine_lbfgs(centers, beta=100.0)
            if refined_radius > best_radius:
                best_radius = refined_radius
                best_centers = refined_centers.copy()
        except Exception:
            pass
    
    # Try with different beta values for multi-scale optimization
    if _remaining_time() > 10.0:
        for beta_val in [50.0, 200.0, 500.0]:
            if _remaining_time() < 5.0:
                break
            try:
                refined_centers, refined_radius = _refine_lbfgs(best_centers, beta=beta_val)
                if refined_radius > best_radius:
                    best_radius = refined_radius
                    best_centers = refined_centers.copy()
            except Exception:
                pass
    
    # Perturbed restarts with L-BFGS-B
    if _remaining_time() > 15.0:
        n_restarts = 3
        for restart in range(n_restarts):
            if _remaining_time() < 5.0:
                break
            perturbed = best_centers + np.random.randn(N, 2) * 0.02
            perturbed = np.clip(perturbed, -0.5, 0.5)
            try:
                refined_centers, refined_radius = _refine_lbfgs(perturbed, beta=100.0)
                if refined_radius > best_radius:
                    best_radius = refined_radius
                    best_centers = refined_centers.copy()
            except Exception:
                pass
    
    # Final safety clip
    best_centers = np.clip(best_centers, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_centers
