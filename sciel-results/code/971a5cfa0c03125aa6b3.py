"""Packing 300 equal circles in a unit circle using gradient-based optimization."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_score(points):
    """Compute the packing score: min(min pairwise dist / 2, min (1 - |c_i|))."""
    n = len(points)
    # Pairwise distances
    # Use broadcasting for all pairs
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to boundary
    radii = np.sqrt(np.sum(points * points, axis=1))
    min_boundary = np.min(1.0 - radii)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _negative_score(points_flat):
    """Negative score for minimization."""
    points = points_flat.reshape(-1, 2)
    n = len(points)
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to boundary
    radii = np.sqrt(np.sum(points * points, axis=1))
    min_boundary = np.min(1.0 - radii)
    
    score = min(min_pair_dist / 2.0, min_boundary)
    return -score


def _negative_score_grad(points_flat):
    """Gradient of negative score. Approximate using finite differences for simplicity
    but we'll use a more efficient approach."""
    points = points_flat.reshape(-1, 2)
    n = len(points)
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Distance to boundary
    radii = np.sqrt(np.sum(points * points, axis=1))
    boundary_dists = 1.0 - radii
    
    min_pair_dist = np.min(dists)
    min_boundary = np.min(boundary_dists)
    
    score = min(min_pair_dist / 2.0, min_boundary)
    
    # For gradient, we need to know which constraint is active
    # This is complex; let's use a smooth approximation approach instead
    # We'll use L-BFGS-B with numerical gradients via scipy
    
    return None  # Let scipy handle numerical gradients


def _smooth_negative_score(points_flat, beta=100.0):
    """Smooth approximation of negative score using log-sum-exp."""
    points = points_flat.reshape(-1, 2)
    n = len(points)
    
    # Pairwise squared distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # We want to maximize the minimum distance, so minimize the max of (1/dist^2)
    # Or use a different smooth approximation
    # Let's use: -log(sum(exp(-beta * dist))) which approximates -min(dist)
    # Actually, let's think differently.
    
    # Score = min(min_pair_dist/2, min_boundary)
    # -score = max(-min_pair_dist/2, -min_boundary)
    # max(a, b) ≈ (1/beta) * log(exp(beta*a) + exp(beta*b))
    
    # But we need to compute min_pair_dist and min_boundary first
    dists = np.sqrt(dists_sq)
    min_pair_dist = np.min(dists)
    
    radii = np.sqrt(np.sum(points * points, axis=1))
    boundary_dists = 1.0 - radii
    min_boundary = np.min(boundary_dists)
    
    a = -min_pair_dist / 2.0
    b = -min_boundary
    
    # Smooth max
    beta_safe = min(beta, 1000.0)
    m = max(a, b)
    smooth_max = m + (1.0 / beta_safe) * np.log(np.exp(beta_safe * (a - m)) + np.exp(beta_safe * (b - m)))
    
    return smooth_max


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Initial configuration: hexagonal lattice
    def _init_hexagonal(n_points):
        spacing = 2.0 / (np.sqrt(n_points) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points = points[radius <= 1.0 - spacing / 2.0]
            if len(points) >= n_points:
                # Select the n_points closest to center
                r = np.hypot(points[:, 0], points[:, 1])
                idx = np.argsort(r, kind='stable')[:n_points]
                return points[idx]
            spacing *= 0.98
    
    best_points = _init_hexagonal(N)
    best_score = _compute_score(best_points)
    
    # Try multiple restarts with perturbed initial conditions
    rng = np.random.default_rng(42)
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            # Start from hexagonal
            x0 = best_points.flatten()
        else:
            # Perturbed version
            x0 = best_points.flatten().copy()
            # Add small random perturbation
            perturbation_scale = 0.01 * (restart + 1)
            x0 += rng.standard_normal(2 * N) * perturbation_scale
            # Ensure points are inside unit circle with margin
            for i in range(N):
                r = np.sqrt(x0[2*i]**2 + x0[2*i+1]**2)
                if r > 0.95:
                    scale = 0.95 / r
                    x0[2*i] *= scale
                    x0[2*i+1] *= scale
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                _smooth_negative_score,
                x0,
                args=(200.0,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
            )
            
            if result.success or result.fun < _smooth_negative_score(best_points.flatten(), 200.0):
                new_points = result.x.reshape(-1, 2)
                new_score = _compute_score(new_points)
                if new_score > best_score:
                    best_points = new_points
                    best_score = new_score
        except Exception as e:
            pass
    
    # Final refinement with higher beta
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                _smooth_negative_score,
                best_points.flatten(),
                args=(1000.0,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 20}
            )
            new_points = result.x.reshape(-1, 2)
            new_score = _compute_score(new_points)
            if new_score > best_score:
                best_points = new_points
                best_score = new_score
        except Exception:
            pass
    
    # Ensure validity: all points inside unit circle
    radii = np.sqrt(np.sum(best_points * best_points, axis=1))
    if np.any(radii >= 1.0):
        # Scale down slightly
        max_r = np.max(radii)
        if max_r > 0:
            best_points *= (1.0 - 1e-10) / max_r
    
    return best_points
# EVOLVE-BLOCK-END
