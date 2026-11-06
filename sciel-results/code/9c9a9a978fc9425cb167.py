"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hexagonal_seed(n):
    """Generate a hexagonal lattice seed for n circles."""
    for trial in range(100):
        spacing = 2.0 / (np.sqrt(n) + 1.0) * (1.0 + trial * 0.005)
        k = int(1.0 / spacing) + 3
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0
        points = points[mask]
        if len(points) >= n:
            idx = np.argsort(radius[mask])[:n]
            return points[idx]
    # Fallback: random points in disk
    angles = np.random.uniform(0, 2 * np.pi, n)
    radii = np.sqrt(np.random.uniform(0, 0.8, n))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _compute_radius(points):
    """Compute the actual packing radius: min of pairwise dist/2 and boundary dist."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    radii = np.sqrt(np.sum(points ** 2, axis=1))
    min_boundary = np.min(1.0 - radii)
    return min(min_pair, min_boundary)


def _smooth_objective(x, n, beta=50.0):
    """
    Smooth approximation of maximizing the minimum radius.
    We minimize: sum over pairs of exp(-beta * d_ij) + sum over points of exp(-beta * (1-|c_i|))
    This is smooth and its minimum approximates the point where all critical distances are equalized.
    """
    pts = x.reshape(-1, 2)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Boundary distances
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    boundary_dists = 1.0 - radii
    
    # Smooth "min" via negative log-sum-exp approximation
    # For pairs: we want to maximize the min distance, so minimize sum of exp(-beta*d)
    pair_term = np.sum(np.exp(-beta * dists))
    # For boundary: maximize min(1-|c_i|), minimize sum of exp(-beta*(1-|c_i|))
    boundary_term = np.sum(np.exp(-beta * boundary_dists))
    
    # Penalty for going outside disk (shouldn't happen with bounds, but safety)
    outside_penalty = np.sum(np.maximum(radii - 1.0, 0.0) ** 2)
    
    return pair_term + boundary_term + 1000.0 * outside_penalty


def _smooth_gradient(x, n, beta=50.0):
    """Gradient of the smooth objective."""
    pts = x.reshape(-1, 2)
    
    # Pairwise
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # grad of sum exp(-beta * d_ij) w.r.t. pts[k]
    # d/dx_k sum_{i<j} exp(-beta*d_ij) where k is in pair
    # = sum_{j!=k} exp(-beta*d_kj) * (-beta) * (pts[k] - pts[j]) / d_kj
    grad = np.zeros_like(pts)
    for k in range(n):
        d_k = dists[k].copy()
        d_k[k] = np.inf
        weights = beta * np.exp(-beta * d_k) / d_k
        grad[k] = -np.sum(weights[:, np.newaxis] * (pts[k] - pts), axis=0)
    
    # Boundary
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    # d/dx_k sum exp(-beta*(1-|c_k|)) = sum exp(-beta*(1-|c_k|)) * (-beta) * (-x_k/|x_k|)
    # = sum exp(-beta*(1-|c_k|)) * beta * x_k / |x_k|
    boundary_weights = beta * np.exp(-beta * (1.0 - radii)) / np.maximum(radii, 1e-10)
    grad += boundary_weights[:, np.newaxis] * pts
    
    # Outside penalty gradient
    outside_mask = radii > 1.0
    if np.any(outside_mask):
        grad[outside_mask] += 2000.0 * (radii[outside_mask] - 1.0)[:, np.newaxis] * pts[outside_mask] / radii[outside_mask, np.newaxis]
    
    return grad.flatten()


def _local_refine(centers, time_limit=5.0):
    """Local refinement using small perturbations to escape local minima."""
    start_time = time.time()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    step_size = 0.02
    n = len(centers)
    
    while time.time() - start_time < time_limit:
        improved = False
        for _ in range(20):
            i = np.random.randint(n)
            delta = np.random.normal(0, step_size, size=2)
            new_c = centers[i].copy()
            new_c += delta
            
            # Check if new position is valid
            if np.linalg.norm(new_c) > 1.0:
                continue
            
            # Temporarily update and check
            old_c = centers[i].copy()
            centers[i] = new_c
            new_r = _compute_radius(centers)
            
            if new_r > best_r + 1e-12:
                best_r = new_r
                best_centers = centers.copy()
                improved = True
                step_size *= 1.05
            else:
                centers[i] = old_c
        
        if not improved:
            step_size *= 0.9
            if step_size < 0.001:
                break
    
    return best_centers, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Start with hexagonal lattice
    centers = _hexagonal_seed(N)
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Optimization using L-BFGS-B with smooth objective
    bounds = [(-0.99, 0.99)] * (2 * N)
    
    # Try a few different beta values
    for beta in [30.0, 50.0, 80.0]:
        if time.time() > deadline - 20.0:
            break
        
        try:
            result = minimize(
                _smooth_objective,
                centers.flatten(),
                args=(N, beta),
                method='L-BFGS-B',
                jac=lambda x, n, b: _smooth_gradient(x, n, b),
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 500}
            )
            
            new_centers = result.x.reshape(-1, 2)
            # Ensure points are within disk
            radii = np.linalg.norm(new_centers, axis=1)
            mask = radii <= 1.0
            if np.all(mask):
                new_r = _compute_radius(new_centers)
                if new_r > best_r:
                    best_r = new_r
                    best_centers = new_centers.copy()
                    centers = new_centers.copy()
        except Exception:
            pass
    
    # Local refinement to polish the result
    if time.time() < deadline - 10.0:
        best_centers, best_r = _local_refine(best_centers, time_limit=min(10.0, deadline - time.time() - 5.0))
    
    # Final check: ensure validity
    final_r = _compute_radius(best_centers)
    if final_r > 0:
        return best_centers
    else:
        # Fallback to hexagonal lattice
        return _hexagonal_seed(N)

# EVOLVE-BLOCK-END
