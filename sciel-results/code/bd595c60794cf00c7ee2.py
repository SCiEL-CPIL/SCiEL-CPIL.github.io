"""
Improved packing of 50 equal spheres in a unit cube.

Strategy:
1. Start with BCC or grid initial configurations.
2. Use smooth surrogate objective (soft-min) with gradient-based optimization.
3. Final polish pass with exact objective using scipy L-BFGS-B.
4. Multiple restarts with perturbation to escape local minima.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE_SECONDS = 115  # Safety margin before 120s limit

def _score(points):
    """Compute the maximum valid radius for the given sphere centers."""
    if points.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    if np.any(points > 0.5) or np.any(points < -0.5):
        return 0.0
        
    # Distance between all pairs
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Distance to walls: cube is [-0.5, 0.5]^3
    wall_dists = 0.5 - np.abs(points)
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)

def _soft_min_energy(points, beta=100.0):
    """
    Smooth surrogate for negative min distance using log-sum-exp.
    Returns negative of soft-min of all pair distances and wall distances.
    """
    # Pair distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, 1e10)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(points)
    wall_dists_sq = wall_dists ** 2
    
    # Combine all distances
    all_dists_sq = np.concatenate([dists_sq.flatten(), wall_dists_sq.flatten()])
    
    # Soft min using log-sum-exp: -log(sum(exp(-beta * x))) / beta
    # For numerical stability, subtract min first
    min_sq = np.min(all_dists_sq)
    exp_vals = np.exp(-beta * (all_dists_sq - min_sq))
    soft_min_sq = min_sq + np.log(np.sum(exp_vals)) / beta
    
    return soft_min_sq  # We want to maximize this, so minimize negative

def _soft_min_grad(points, beta=100.0):
    """Gradient of soft_min_energy with respect to points."""
    # Pair distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, 1e10)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(points)
    wall_dists_sq = wall_dists ** 2
    
    # Combine all distances
    all_dists_sq = np.concatenate([dists_sq.flatten(), wall_dists_sq.flatten()])
    
    # Soft min weights
    min_sq = np.min(all_dists_sq)
    exp_vals = np.exp(-beta * (all_dists_sq - min_sq))
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(points)
    
    # Pair distance gradients: d/dx_i |x_i - x_j|^2 = 2(x_i - x_j)
    for i in range(N):
        for j in range(i + 1, N):
            idx = i * N + j
            w = weights[idx]
            d = points[i] - points[j]
            grad[i] += 2 * w * d
            grad[j] -= 2 * w * d
    
    # Wall distance gradients: d/dx_i (0.5 - |x_i|)^2 = -2(0.5 - |x_i|)*sign(x_i)
    for i in range(N):
        for k in range(3):
            idx = N * N + i * 3 + k
            w = weights[idx]
            x = points[i, k]
            if abs(x) < 1e-10:
                continue
            grad[i, k] += -2 * w * (0.5 - abs(x)) * np.sign(x)
    
    return grad

def _initial_grid(n):
    """Create an initial grid-like configuration."""
    m = int(np.ceil(n ** (1.0 / 3.0)))
    # Adjust to fit n spheres
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Shuffle and take first n
    idx = np.random.permutation(len(grid))
    return grid[idx[:n]].copy()

def _initial_bcc(n):
    """Create an initial BCC (body-centered cubic) configuration."""
    # BCC has 2 points per unit cell: corners and center
    m = int(np.ceil((n / 2.0) ** (1.0 / 3.0)))
    
    points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                # Corner
                points.append([
                    (i + 0.5) / m - 0.5,
                    (j + 0.5) / m - 0.5,
                    (k + 0.5) / m - 0.5
                ])
                # Body center
                if len(points) < n:
                    points.append([
                        (i + 0.5) / m - 0.5,
                        (j + 0.5) / m - 0.5,
                        (k + 0.5) / m - 0.5
                    ])
    
    # Scale to fit in [-0.5, 0.5]
    points = np.array(points[:n])
    max_abs = np.max(np.abs(points))
    if max_abs > 0:
        points = points * (0.45 / max_abs)
    
    return points

def _optimize(points, max_time):
    """Optimize the sphere configuration using smooth surrogate then exact polish."""
    start_time = time.time()
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Phase 1: Smooth optimization with L-BFGS-B
    beta = 50.0
    
    def neg_soft_min(x):
        pts = x.reshape(N, 3)
        return -_soft_min_energy(pts, beta)
    
    def neg_soft_min_grad(x):
        pts = x.reshape(N, 3)
        return -_soft_min_grad(pts, beta)
    
    x0 = points.flatten()
    
    try:
        result = minimize(
            neg_soft_min, x0,
            jac=neg_soft_min_grad,
            method='L-BFGS-B',
            bounds=[(-0.499, 0.499)] * (N * 3),
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < neg_soft_min(x0):
            new_points = result.x.reshape(N, 3)
            new_points = np.clip(new_points, -0.5, 0.5)
            new_score = _score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
    except Exception:
        pass
    
    # Phase 2: Exact polish with L-BFGS-B on the true objective
    if time.time() - start_time < max_time - 5:
        def neg_exact_score(x):
            pts = x.reshape(N, 3)
            return -_score(pts)
        
        try:
            result2 = minimize(
                neg_exact_score, best_points.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.499, 0.499)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result2.success:
                new_points = result2.x.reshape(N, 3)
                new_points = np.clip(new_points, -0.5, 0.5)
                new_score = _score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
        except Exception:
            pass
    
    return best_points, best_score

def construct():
    """Construct the sphere centers."""
    start_time = time.time()
    
    # Multiple restarts with different initializations
    best_points = None
    best_score = 0.0
    
    n_restarts = 0
    while time.time() - start_time < DEADLINE_SECONDS:
        n_restarts += 1
        
        # Alternate between grid and BCC initializations
        if n_restarts % 2 == 1:
            points = _initial_grid(N)
        else:
            points = _initial_bcc(N)
        
        # Add random perturbation to break symmetry
        perturbation_scale = 0.02 if n_restarts <= 5 else 0.05
        points += np.random.randn(N, 3) * perturbation_scale
        points = np.clip(points, -0.48, 0.48)
        
        # Calculate remaining time for this restart
        remaining = DEADLINE_SECONDS - (time.time() - start_time)
        if remaining < 10:
            break
        
        # Optimize
        opt_points, opt_score = _optimize(points, remaining)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points.copy()
        
        # If we're close to the known best, do a final polish
        if best_score > 0.135 and time.time() - start_time < DEADLINE_SECONDS - 20:
            # Try to improve with smaller perturbations
            for _ in range(3):
                if time.time() - start_time > DEADLINE_SECONDS - 10:
                    break
                perturbed = best_points + np.random.randn(N, 3) * 0.005
                perturbed = np.clip(perturbed, -0.48, 0.48)
                opt_points, opt_score = _optimize(perturbed, min(20, DEADLINE_SECONDS - (time.time() - start_time)))
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
    
    if best_points is None:
        # Fallback
        best_points = _initial_grid(N)
    
    return best_points

# For quick testing
if __name__ == "__main__":
    centers = construct()
    print(f"Score: {_score(centers)}")
    print(f"Shape: {centers.shape}")
