"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _score(points):
    """Compute the packing score: min of pairwise distances/2 and (1 - |c_i|)."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    norms = np.sqrt(np.sum(points * points, axis=-1))
    min_boundary = np.min(1.0 - norms)
    
    return min(min_pair, min_boundary)


def _compute_score(points):
    """Alias for _score."""
    return _score(points)


def _soft_min_radius(points, beta=50.0):
    """Smooth surrogate: soft-min of pairwise distances/2 and boundary distances."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    norms = np.sqrt(np.sum(points * points, axis=-1))
    boundary_vals = 1.0 - norms
    
    # Combine all values into one array
    all_vals = np.concatenate([pair_vals.flatten(), boundary_vals])
    all_vals = all_vals[all_vals > 0]  # remove inf
    
    if len(all_vals) == 0:
        return 0.0
    
    # Log-sum-exp for soft-min
    m = np.min(all_vals)
    log_sum = np.log(np.sum(np.exp(-beta * (all_vals - m))))
    soft_min = m - np.log(log_sum) / beta
    
    return soft_min


def _soft_min_grad(points, beta=50.0):
    """Gradient of the soft-min radius surrogate."""
    N = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    norms = np.sqrt(np.sum(points * points, axis=-1))
    boundary_vals = 1.0 - norms
    
    all_vals = np.concatenate([pair_vals.flatten(), boundary_vals])
    valid_mask = all_vals > 0
    all_vals_valid = all_vals[valid_mask]
    
    if len(all_vals_valid) == 0:
        return np.zeros_like(points)
    
    m = np.min(all_vals_valid)
    exp_vals = np.exp(-beta * (all_vals_valid - m))
    log_sum = np.log(np.sum(exp_vals))
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros((N, 2))
    
    # Pairwise contributions
    pair_flat = pair_vals.flatten()
    boundary_flat = boundary_vals
    all_flat = np.concatenate([pair_flat, boundary_flat])
    all_valid_flat = all_flat[valid_mask]
    
    # Reconstruct which indices are pairs vs boundary
    n_pairs = pair_flat.size
    for idx, w in enumerate(weights):
        if idx < n_pairs:
            # This is a pair contribution
            flat_idx = idx
            i = flat_idx // N
            j = flat_idx % N
            if i == j:
                continue
            d = dists[i, j]
            if d < 1e-10:
                continue
            # d/2 derivative w.r.t. points[i] and points[j]
            direction = (points[i] - points[j]) / d
            grad[i] += w * direction / 2.0
            grad[j] -= w * direction / 2.0
        else:
            # Boundary contribution
            b_idx = idx - n_pairs
            norm = norms[b_idx]
            if norm < 1e-10:
                continue
            # (1 - |c_i|) derivative w.r.t. points[i]
            grad[b_idx] += w * (-points[b_idx] / norm)
    
    return grad


def _hex_lattice_init(n=50):
    """Generate a hexagonal lattice initial configuration with better spacing."""
    best_points = None
    best_score = -np.inf
    
    for spacing in [0.22, 0.23, 0.24, 0.25, 0.26, 0.27]:
        points = []
        for j in range(-4, 5):
            for i in range(-4, 5):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 0.9**2:
                    points.append([x, y])
        
        if len(points) < n:
            continue
            
        points = np.array(points)
        radii = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(radii)[:n]
        points = points[idx]
        
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    if best_points is None:
        best_points = np.random.uniform(-0.8, 0.8, (n, 2))
    
    return best_points


def _lbfgs_optimize(points, deadline=None, beta=50.0):
    """Use L-BFGS-B to maximize the soft-min radius surrogate."""
    N = points.shape[0]
    x0 = points.flatten().copy()
    
    def neg_soft_min(x):
        pts = x.reshape(N, 2)
        return -_soft_min_radius(pts, beta=beta)
    
    def neg_soft_min_grad(x):
        pts = x.reshape(N, 2)
        g = _soft_min_grad(pts, beta=beta)
        return -g.flatten()
    
    bounds = [(-0.99, 0.99)] * (N * 2)
    
    result = minimize(
        neg_soft_min, x0, jac=neg_soft_min_grad, method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 50}
    )
    
    if result.fun < neg_soft_min(x0):
        return result.x.reshape(N, 2)
    return points


def _vectorized_polish(points, n_iter=2000, deadline=None, step_scale=1.0):
    """Fully vectorized local optimization with adaptive step size."""
    N = points.shape[0]
    points = points.copy()
    best_score = _score(points)
    step_size = 0.005 * step_scale
    patience = 0
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline - 2.0:
            break
        
        # Compute forces - fully vectorized
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, 1.0)
        
        # Only consider close pairs: force = (1/d) * direction for d < threshold
        threshold = 0.35
        close_mask = (dists < threshold) & (dists > 1e-6)
        
        # Force magnitude: 0.005 / d for close pairs
        force_mag = np.zeros_like(dists)
        force_mag[close_mask] = 0.005 / dists[close_mask]
        
        # Direction: (points[i] - points[j]) / d
        # For each pair (i,j), add force_mag[i,j] * direction[i,j] to forces[i]
        # and subtract from forces[j]
        direction = diff / dists[:, :, np.newaxis]
        direction[~close_mask] = 0
        
        forces = np.zeros_like(points)
        for i in range(N):
            forces[i] += np.sum(force_mag[i, :, np.newaxis] * direction[i], axis=0)
        
        # Boundary force: push inward if norm > 0.85
        norms = np.hypot(points[:, 0], points[:, 1])
        boundary_mask = norms > 0.85
        if np.any(boundary_mask):
            boundary_force = np.zeros_like(points)
            for i in np.where(boundary_mask)[0]:
                boundary_force[i] = -0.005 * (norms[i] - 0.85) * points[i] / norms[i]
            forces += boundary_force
        
        # Update positions
        new_points = points + forces * step_size
        
        # Project into unit disk
        new_norms = np.hypot(new_points[:, 0], new_points[:, 1])
        mask = new_norms > 0.99
        if np.any(mask):
            scale = 0.99 / new_norms[mask]
            new_points[mask] *= scale[:, np.newaxis]
        
        new_score = _score(new_points)
        if new_score > best_score:
            points = new_points
            best_score = new_score
            patience = 0
            step_size = min(step_size * 1.05, 0.03)
        else:
            patience += 1
            step_size *= 0.95
            if patience > 100:
                break
    
    return points


def _simulated_annealing(points, deadline=None, n_restarts=5):
    """Simulated annealing to escape local minima."""
    N = points.shape[0]
    best_points = points.copy()
    best_score = _score(best_points)
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline - 10.0:
            break
        
        # Perturb current best
        perturbed = best_points + np.random.normal(0, 0.02, (N, 2))
        norms = np.hypot(perturbed[:, 0], perturbed[:, 1])
        mask = norms > 0.98
        if np.any(mask):
            scale = 0.98 / norms[mask]
            perturbed[mask] *= scale[:, np.newaxis]
        
        # Optimize the perturbed configuration
        polished = _vectorized_polish(perturbed, n_iter=500, deadline=deadline, step_scale=1.0)
        polished_score = _score(polished)
        
        if polished_score > best_score:
            best_points = polished
            best_score = polished_score
    
    return best_points


def _final_high_precision_polish(points, deadline=None):
    """High-precision final polish with very small steps and higher beta."""
    N = points.shape[0]
    points = points.copy()
    best_score = _score(points)
    step_size = 0.001
    
    for it in range(3000):
        if deadline is not None and time.time() > deadline - 1.0:
            break
        
        # Compute forces - vectorized
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, 1.0)
        
        threshold = 0.28
        close_mask = (dists < threshold) & (dists > 1e-6)
        
        force_mag = np.zeros_like(dists)
        force_mag[close_mask] = 0.001 / dists[close_mask]
        
        direction = diff / dists[:, :, np.newaxis]
        direction[~close_mask] = 0
        
        forces = np.zeros_like(points)
        for i in range(N):
            forces[i] += np.sum(force_mag[i, :, np.newaxis] * direction[i], axis=0)
        
        norms = np.hypot(points[:, 0], points[:, 1])
        boundary_mask = norms > 0.95
        if np.any(boundary_mask):
            for i in np.where(boundary_mask)[0]:
                forces[i] += -0.001 * (norms[i] - 0.95) * points[i] / norms[i]
        
        new_points = points + forces * step_size
        
        new_norms = np.hypot(new_points[:, 0], new_points[:, 1])
        mask = new_norms > 0.999
        if np.any(mask):
            scale = 0.999 / new_norms[mask]
            new_points[mask] *= scale[:, np.newaxis]
        
        new_score = _score(new_points)
        if new_score > best_score:
            points = new_points
            best_score = new_score
        else:
            step_size *= 0.97
    
    return points


def construct():
    """Construct optimal packing of 50 circles in unit circle."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initial configuration from hexagonal lattice
    init_points = _hex_lattice_init(N)
    best_points = init_points.copy()
    best_score = _score(best_points)
    
    # L-BFGS-B optimization with smooth surrogate
    if time.time() < deadline - 80.0:
        lbfgs_points = _lbfgs_optimize(init_points, deadline=deadline, beta=50.0)
        lbfgs_score = _score(lbfgs_points)
        if lbfgs_score > best_score:
            best_points = lbfgs_points
            best_score = lbfgs_score
    
    # Vectorized local polish
    if time.time() < deadline - 40.0:
        polished_points = _vectorized_polish(best_points, n_iter=1500, deadline=deadline)
        polished_score = _score(polished_points)
        if polished_score > best_score:
            best_points = polished_points
            best_score = polished_score
    
    # Simulated annealing to escape local minima
    if time.time() < deadline - 30.0:
        sa_points = _simulated_annealing(best_points, deadline=deadline, n_restarts=3)
        sa_score = _score(sa_points)
        if sa_score > best_score:
            best_points = sa_points
            best_score = sa_score
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        final_points = _final_high_precision_polish(best_points, deadline=deadline)
        final_score = _score(final_points)
        if final_score > best_score:
            best_points = final_points
            best_score = final_score
    
    # Final check: ensure all points are strictly inside
    norms = np.hypot(best_points[:, 0], best_points[:, 1])
    mask = norms >= 0.999
    if np.any(mask):
        scale = 0.999 / norms[mask]
        best_points[mask] *= scale[:, np.newaxis]
    
    return best_points
# EVOLVE-BLOCK-END
