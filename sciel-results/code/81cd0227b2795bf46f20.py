"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hexagonal_initial(n, seed=0):
    """Generate a hexagonal lattice initialization and select n points near center."""
    rng = np.random.RandomState(seed)
    # Generate a grid of hexagonal lattice points
    # We want enough points to choose from
    max_radius = 1.0
    # Estimate spacing: for n points in unit disk, area per point ~ pi/n
    # Hexagonal packing: area per point = s^2 * sqrt(3)/2
    # s^2 = 2*pi/(n*sqrt(3))
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    
    # Generate lattice points
    points = []
    k = int(max_radius / s) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= max_radius - s / 2.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < n:
        # Fallback: random points
        angles = rng.uniform(0, 2 * np.pi, n)
        radii = rng.uniform(0, 0.9, n)
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return points
    
    # Select n points closest to center
    radii = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(radii, kind='stable')[:n]
    return points[idx]

def _compute_score(centers):
    """Compute the packing radius for given centers."""
    n = len(centers)
    if n < 2:
        return min(1.0 - np.max(np.hypot(centers[:, 0], centers[:, 1])), 1.0)
    
    # Compute all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Minimum pairwise distance / 2
    min_pair_dist = np.min(dists[np.triu_indices(n, k=1)]) / 2.0
    
    # Minimum distance from boundary
    radii = np.hypot(centers[:, 0], centers[:, 1])
    min_boundary = 1.0 - np.max(radii)
    
    return min(min_pair_dist, min_boundary)

def _soft_min_obj(centers, beta=50.0):
    """
    Objective function to minimize: -soft_min of all constraints.
    We want to maximize r = min(min_pair_dist/2, min_boundary).
    So we minimize -r using a smooth approximation.
    """
    n = len(centers)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    
    # Upper triangular indices for pairwise
    iu = np.triu_indices(n, k=1)
    pair_dists_sq = dists_sq[iu]
    
    # Boundary distances
    radii_sq = np.sum(centers ** 2, axis=1)
    boundary_dists = 1.0 - np.sqrt(radii_sq)
    
    # All constraint values (we want these to be large)
    # Pair constraints: dist/2 = sqrt(dist_sq)/2
    # Boundary constraints: 1 - r_i
    pair_vals = np.sqrt(pair_dists_sq) / 2.0
    all_vals = np.concatenate([pair_vals, boundary_dists])
    
    # Soft minimum using log-sum-exp
    # soft_min(x) = -log(sum(exp(-beta * x))) / beta
    # This approximates min(x) as beta -> infinity
    scaled = -beta * all_vals
    # For numerical stability
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -log_sum / beta
    
    return -soft_min_val

def _soft_min_obj_grad(centers, beta=50.0):
    """Gradient of the soft min objective."""
    n = len(centers)
    
    # Pairwise
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    
    iu = np.triu_indices(n, k=1)
    pair_dists_sq = dists_sq[iu]
    pair_dists = np.sqrt(pair_dists_sq + 1e-12)
    
    # Boundary
    radii_sq = np.sum(centers ** 2, axis=1)
    radii = np.sqrt(radii_sq + 1e-12)
    boundary_dists = 1.0 - radii
    
    pair_vals = pair_dists / 2.0
    all_vals = np.concatenate([pair_vals, boundary_dists])
    
    # Soft min weights
    scaled = -beta * all_vals
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp
    
    # Gradient of -soft_min = sum(weights_i * grad(-val_i))... wait
    # soft_min = -log(sum(exp(-beta * x))) / beta
    # d(soft_min)/dx_i = (beta * sum(exp(-beta*x_j) * (-x_j') )) / (beta * sum(exp(-beta*x_j)))
    #                  = -sum(weights_j * x_j')
    # So d(-soft_min)/dx_i = sum(weights_j * x_j')
    
    # We need gradients of each val_j with respect to centers
    grad = np.zeros_like(centers)
    
    # Pair contributions
    n_pairs = len(pair_vals)
    for idx in range(n_pairs):
        i, j = iu[0][idx], iu[1][idx]
        w = weights[idx]
        # val = dist(i,j) / 2
        # d(dist)/d(c_i) = (c_i - c_j) / dist
        # d(val)/d(c_i) = (c_i - c_j) / (2 * dist)
        d = centers[i] - centers[j]
        dist = pair_dists[idx]
        if dist > 1e-10:
            grad[i] += w * d / (2.0 * dist)
            grad[j] -= w * d / (2.0 * dist)
    
    # Boundary contributions
    for i in range(n):
        w = weights[n_pairs + i]
        # val = 1 - |c_i|
        # d(val)/d(c_i) = -c_i / |c_i|
        r = radii[i]
        if r > 1e-10:
            grad[i] -= w * centers[i] / r
    
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial guess from hexagonal lattice
    centers = _hexagonal_initial(N, seed=0)
    best_centers = centers.copy()
    best_score = _compute_score(centers)
    
    # Multi-restart L-BFGS-B optimization
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # For first restart, use the hexagonal init; for others, perturb
        if restart == 0:
            x0 = centers.copy()
        else:
            # Perturb the best solution found so far
            rng = np.random.RandomState(restart)
            noise_scale = 0.05 * (1.0 - restart / float(n_restarts))
            x0 = best_centers + rng.normal(0, noise_scale, size=best_centers.shape)
            # Also try different initializations
            if restart % 5 == 0:
                x0 = _hexagonal_initial(N, seed=restart)
        
        # Ensure valid starting point (inside unit circle)
        radii = np.hypot(x0[:, 0], x0[:, 1])
        mask = radii < 1.0
        if not np.all(mask):
            x0[~mask] *= 0.9 * (1.0 / radii[~mask])[:, np.newaxis]
        
        # Run L-BFGS-B
        try:
            result = minimize(
                _soft_min_obj,
                x0,
                jac=_soft_min_obj_grad,
                method='L-BFGS-B',
                args=(50.0,),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < _soft_min_obj(x0, 50.0):
                new_centers = result.x
                # Check validity
                radii = np.hypot(new_centers[:, 0], new_centers[:, 1])
                if np.all(radii < 1.0) and np.all(np.isfinite(new_centers)):
                    new_score = _compute_score(new_centers)
                    if new_score > best_score:
                        best_score = new_score
                        best_centers = new_centers.copy()
        except Exception:
            pass
    
    # Final polish: use a higher beta for more accurate soft min
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                _soft_min_obj,
                best_centers,
                jac=_soft_min_obj_grad,
                method='L-BFGS-B',
                args=(100.0,),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.fun < _soft_min_obj(best_centers, 100.0):
                new_centers = result.x
                radii = np.hypot(new_centers[:, 0], new_centers[:, 1])
                if np.all(radii < 1.0) and np.all(np.isfinite(new_centers)):
                    new_score = _compute_score(new_centers)
                    if new_score > best_score:
                        best_score = new_score
                        best_centers = new_centers.copy()
        except Exception:
            pass
    
    # Ensure the result is valid
    radii = np.hypot(best_centers[:, 0], best_centers[:, 1])
    if not np.all(radii < 1.0):
        # Fallback to hexagonal
        best_centers = _hexagonal_initial(N, seed=0)
    
    return best_centers
# EVOLVE-BLOCK-END
