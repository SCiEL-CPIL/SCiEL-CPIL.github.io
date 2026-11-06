"""Improved packing algorithm for 50 circles in a unit circle."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _hexagonal_seed(n, jitter=0.0):
    """Generate a hexagonal lattice seed for n circles."""
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= n:
            idx = np.argsort(radius[mask], kind="stable")[:n]
            result = points[idx].copy()
            if jitter > 0:
                result += np.random.normal(0, jitter, size=result.shape)
            return result
        spacing *= 0.95

def _ring_seed(n):
    """Generate a ring-based seed for n circles."""
    # Place circles in concentric rings
    centers = []
    # Center circle
    centers.append([0.0, 0.0])
    remaining = n - 1
    
    ring = 1
    while remaining > 0:
        # Number of circles on this ring
        ring_count = min(remaining, int(2 * np.pi * ring * 0.15))
        if ring_count < 1:
            ring_count = 1
        radius = ring * 0.18
        for i in range(ring_count):
            angle = 2 * np.pi * i / ring_count
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
        remaining -= ring_count
        ring += 1
    
    return np.array(centers[:n])

def _random_seed(n):
    """Generate a random seed for n circles."""
    angles = np.random.uniform(0, 2 * np.pi, n)
    radii = np.sqrt(np.random.uniform(0, 1, n)) * 0.95
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _compute_score(points):
    """Compute the packing score: min of pairwise dist/2 and boundary dist."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    radii = np.sqrt(np.sum(points ** 2, axis=1))
    min_boundary = np.min(1.0 - radii)
    return min(min_pair, min_boundary)

def _soft_min_objective(x, beta):
    """
    Objective function: minimize -soft_min(y) where y are the distances.
    soft_min(y) approximates min(y)
    We want to maximize min(y), so minimize -min(y) ≈ -soft_min(y)
    """
    pts = x.reshape(-1, 2)
    
    # Pairwise distances / 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    
    iu = np.triu_indices(N, k=1)
    pair_dists = np.sqrt(dists_sq[iu]) / 2.0
    
    # Boundary distances
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    bound_dists = 1.0 - radii
    
    y = np.concatenate([pair_dists, bound_dists])
    
    # Soft min: soft_min(y) = -1/beta * log(sum(exp(-beta * y_k)))
    # For numerical stability, shift by min
    y_min = np.min(y)
    diffs = y - y_min
    exp_terms = np.exp(-beta * diffs)
    log_sum = np.log(np.sum(exp_terms))
    
    # soft_min(y) = y_min - 1/beta * log_sum
    # We want to minimize -soft_min(y) = -y_min + 1/beta * log_sum
    obj = -y_min + (1.0 / beta) * log_sum
    return obj

def _gradient(x, beta):
    """
    Gradient of the soft-min objective.
    f(x) = -soft_min(y(x)) = y_min(x) - 1/beta * log(sum(exp(-beta*(y_k(x) - y_min(x)))))
    
    Using the log-sum-exp form directly:
    g(x) = 1/beta * log(sum_k exp(-beta * y_k(x)))
    dg/dx = -sum_k (w_k * dy_k/dx) where w_k = exp(-beta*y_k) / sum(exp(-beta*y_j))
    
    But our objective is -soft_min = -y_min + g(x), so we need to account for y_min derivative.
    Actually, let's use the direct form:
    f(x) = -1/beta * log(sum_k exp(-beta * y_k(x)))
    df/dx = sum_k (w_k * dy_k/dx) where w_k = exp(-beta*y_k) / sum(exp(-beta*y_j))
    
    Wait, let me reconsider. We want to minimize f(x) = -soft_min(y(x)).
    soft_min(y) = -1/beta * log(sum_k exp(-beta * y_k))
    So f(x) = 1/beta * log(sum_k exp(-beta * y_k))
    df/dx = 1/beta * (1/sum(exp)) * sum(exp(-beta*y_k) * (-beta * dy_k/dx))
    df/dx = -sum_k (w_k * dy_k/dx)
    where w_k = exp(-beta*y_k) / sum(exp(-beta*y_j))
    """
    pts = x.reshape(-1, 2)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    
    iu = np.triu_indices(N, k=1)
    pair_dists = np.sqrt(dists_sq[iu]) / 2.0
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    bound_dists = 1.0 - radii
    
    y_all = np.concatenate([pair_dists, bound_dists])
    y_min = np.min(y_all)
    
    # Weights w_k = exp(-beta * (y_k - y_min)) / sum(exp(-beta * (y_j - y_min)))
    exp_pair = np.exp(-beta * (pair_dists - y_min))
    exp_bound = np.exp(-beta * (bound_dists - y_min))
    
    S = np.sum(exp_pair) + np.sum(exp_bound)
    
    w_pair = exp_pair / S
    w_bound = exp_bound / S
    
    grad = np.zeros_like(x)
    
    # Pairwise contributions
    i_idx, j_idx = iu
    d_ij = 2.0 * pair_dists
    d_ij_safe = np.where(d_ij > 1e-12, d_ij, 1.0)
    
    diff_ij = pts[i_idx] - pts[j_idx]
    unit_vecs = diff_ij / d_ij_safe[:, np.newaxis]
    
    # dy_k/dp_i = 0.5 * (p_i - p_j) / d_ij
    # dy_k/dp_j = 0.5 * (p_j - p_i) / d_ij
    # df/dx = -sum(w_k * dy_k/dx)
    # So grad[p_i] += -w_k * 0.5 * unit_vec
    # grad[p_j] += -w_k * 0.5 * (-unit_vec) = w_k * 0.5 * unit_vec
    
    grad_i = -w_pair[:, np.newaxis] * 0.5 * unit_vecs
    grad_j = w_pair[:, np.newaxis] * 0.5 * unit_vecs
    
    np.add.at(grad, i_idx, grad_i)
    np.add.at(grad, j_idx, grad_j)
    
    # Boundary contributions
    # dy_k/dp_i = -p_i / ||p_i||
    # df/dx += -w_k * (-p_i / ||p_i||) = w_k * p_i / ||p_i||
    radii_safe = np.where(radii > 1e-12, radii, 1.0)
    unit_radii = pts / radii_safe[:, np.newaxis]
    
    grad_bound = w_bound[:, np.newaxis] * unit_radii
    
    grad += grad_bound
    
    return grad

def _exact_hill_climb(centers, current_r, deadline):
    """Exact hill climbing with distance matrix updates."""
    dist_mat = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=-1)
    np.fill_diagonal(dist_mat, np.inf)
    
    boundary_dists = 1.0 - np.linalg.norm(centers, axis=1)
    
    step_size = 0.005
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline and step_size > 0.0001:
        i = np.random.randint(N)
        
        for _ in range(20):
            delta = np.random.normal(0, step_size, size=2)
            new_c = centers[i] + delta
            r_new_c = np.linalg.norm(new_c)
            if r_new_c > 1.0:
                continue
            
            old_row = dist_mat[i].copy()
            new_dists = np.linalg.norm(centers - new_c, axis=1)
            new_dists[i] = np.inf
            dist_mat[i] = new_dists
            dist_mat[:, i] = new_dists
            
            min_pair = np.min(dist_mat) / 2.0
            boundary_dists[i] = 1.0 - r_new_c
            min_bound = np.min(boundary_dists)
            
            candidate_r = min(min_pair, min_bound)
            
            if candidate_r > current_r + 1e-12:
                centers[i] = new_c
                current_r = candidate_r
                no_improve_count = 0
                step_size *= 1.01
            else:
                dist_mat[i] = old_row
                dist_mat[:, i] = old_row
                boundary_dists[i] = 1.0 - np.linalg.norm(centers[i])
                no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            step_size *= 0.85
            no_improve_count = 0
    
    return centers, current_r

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s kill
    deadline = start_time + timeout
    
    best_centers = None
    best_r = -np.inf
    
    # Generate multiple diverse seeds
    seeds = []
    seeds.append(_hexagonal_seed(N, jitter=0.0))
    seeds.append(_hexagonal_seed(N, jitter=0.02))
    seeds.append(_ring_seed(N))
    seeds.append(_random_seed(N))
    
    # Add a few more random seeds
    for _ in range(3):
        seeds.append(_random_seed(N))
    
    # For each seed, run optimization with increasing beta
    betas = [20.0, 50.0, 100.0, 200.0, 400.0]
    
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    for seed_idx, seed in enumerate(seeds):
        if time.time() > deadline - 10:
            break
        
        x0 = seed.flatten()
        current_r = _compute_score(seed)
        
        if current_r > best_r:
            best_r = current_r
            best_centers = seed.copy()
        
        for beta in betas:
            if time.time() > deadline - 15:
                break
            
            try:
                res = minimize(
                    _soft_min_objective,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=_gradient,
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 800}
                )
                
                if hasattr(res, 'x') and res.x is not None:
                    new_centers = res.x.reshape(-1, 2)
                    # Clip to valid range
                    new_centers = np.clip(new_centers, -1.0, 1.0)
                    new_r = _compute_score(new_centers)
                    
                    if new_r > best_r:
                        best_r = new_r
                        best_centers = new_centers.copy()
                        x0 = new_centers.flatten()
                    else:
                        # Keep the better of current and new for next iteration
                        if new_r > current_r:
                            x0 = new_centers.flatten()
                            current_r = new_r
            except Exception:
                pass
    
    # Final polish with exact hill climbing
    if best_centers is not None:
        if time.time() < deadline - 5:
            best_centers, best_r = _exact_hill_climb(best_centers, best_r, deadline)
        
        # Final validation
        radii = np.linalg.norm(best_centers, axis=1)
        if np.any(radii >= 1.0):
            # Scale down points that are outside
            for i in range(N):
                if radii[i] >= 1.0:
                    best_centers[i] = best_centers[i] * (1.0 - 1e-8) / radii[i]
            best_r = _compute_score(best_centers)
    
    # If no valid result, return a simple fallback
    if best_centers is None or not np.all(np.isfinite(best_centers)):
        # Fallback: hexagonal seed
        best_centers = _hexagonal_seed(N)
        radii = np.linalg.norm(best_centers, axis=1)
        if np.any(radii >= 1.0):
            best_centers = best_centers * (1.0 - 1e-8) / radii
    
    return best_centers
