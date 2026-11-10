"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import sys


def _compute_score(centers):
    """Compute the minimum radius for the given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Vectorized pairwise distance computation
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)


def _soft_min_radius(centers, beta=500.0):
    """
    Compute a soft-min approximation of the minimum radius.
    Uses log-sum-exp for the pairwise distances and wall distances.
    """
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Wall distances squared
    wall_dists_sq = (0.5 - np.abs(centers)) ** 2
    wall_dists_sq_flat = wall_dists_sq.flatten()
    
    # Combine all constraints: pairwise distances (need to halve) and wall distances
    # We want min of: dists/2 and wall_dists
    # Equivalently min of: dists_sq/4 and wall_dists_sq
    # Let's work with squared values for numerical stability
    
    # For pairwise: constraint is dist >= 2r, so dist_sq >= 4r^2
    # For wall: constraint is wall_dist >= r, so wall_dist_sq >= r^2
    
    # We want to maximize r such that all constraints hold.
    # Using soft-min on the "slack" values.
    
    # Pairwise slacks: dist_sq/4 - r^2 should be >= 0
    # Wall slacks: wall_dist_sq - r^2 should be >= 0
    
    # Instead, let's use a different approach:
    # Define g_i = min(dists_sq[i]/4, wall_dists_sq[j])
    # We want to maximize r such that r^2 <= g_i for all i,j
    
    # Soft min of all the "limiting" values
    # Let's compute soft-min of (dists_sq/4) and (wall_dists_sq)
    
    # Flatten pairwise (upper triangle only to avoid duplicates, but for soft-min it's ok to include all)
    # Actually for soft-min, we can include all non-diagonal entries
    pair_vals = dists_sq / 4.0
    np.fill_diagonal(pair_vals, np.inf)
    pair_vals_flat = pair_vals.flatten()
    
    all_vals = np.concatenate([pair_vals_flat, wall_dists_sq_flat])
    
    # Soft min: -log(sum(exp(-beta * x))) / beta
    # For numerical stability
    min_val = np.min(all_vals)
    exp_vals = np.exp(-beta * (all_vals - min_val))
    soft_min = min_val - np.log(np.sum(exp_vals)) / beta
    
    # The radius is sqrt(soft_min)
    return np.sqrt(soft_min)


def _negative_soft_min(centers, beta=500.0):
    """Negative of soft-min radius for minimization."""
    return -_soft_min_radius(centers, beta)


def _soft_min_gradient(centers, beta=500.0):
    """
    Compute gradient of the soft-min radius with respect to centers.
    This is expensive but allows gradient-based optimization.
    """
    n = len(centers)
    grad = np.zeros_like(centers)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Wall distances squared
    wall_dists_sq = (0.5 - np.abs(centers)) ** 2
    wall_dists_sq_flat = wall_dists_sq.flatten()
    
    # Combine
    pair_vals = dists_sq / 4.0
    np.fill_diagonal(pair_vals, np.inf)
    pair_vals_flat = pair_vals.flatten()
    
    all_vals = np.concatenate([pair_vals_flat, wall_dists_sq_flat])
    
    # Soft min weights
    min_val = np.min(all_vals)
    exp_vals = np.exp(-beta * (all_vals - min_val))
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient of soft_min = -sum(weights * grad(x_i)) / beta ... 
    # Actually d(soft_min)/dx = -sum(weights_i * dx_i/dx) / beta * ... 
    # Let me re-derive:
    # soft_min = min_val - log(sum(exp(-beta*(x_i - min_val))))/beta
    # d(soft_min)/dx_j = -1/beta * d/dx_j [log(sum(exp(-beta*(x_i - min_val))))]
    # = -1/beta * [sum_i exp(-beta*(x_i-min_val)) * (-beta) * dx_i/dx_j] / sum_i exp(-beta*(x_i-min_val))
    # = sum_i [exp(-beta*(x_i-min_val)) / sum_k exp(-beta*(x_k-min_val))] * dx_i/dx_j
    # = sum_i weights_i * dx_i/dx_j
    
    # So gradient of soft_min w.r.t. x_j is sum_i weights_i * (dx_i/dx_j)
    # And we want gradient of -soft_min, so negate it.
    
    # For pairwise term i (corresponding to pair (a,b)):
    # x_i = dists_sq[a,b]/4 = |c_a - c_b|^2 / 4
    # dx_i/dc_a = (c_a - c_b)/2
    # dx_i/dc_b = (c_b - c_a)/2
    
    # For wall term i (corresponding to sphere a, dimension k):
    # x_i = (0.5 - |c_ak|)^2
    # dx_i/dc_ak = -2*(0.5 - |c_ak|)*sign(c_ak)
    
    # This is complex to vectorize. Let's use a simpler approach:
    # Use finite differences or just use scipy's numerical gradient.
    
    # For efficiency, let's just return None and let scipy use numerical gradient
    return None


def _optimize_smooth(centers, time_budget, beta=200.0):
    """
    Optimize using smooth surrogate objective with scipy L-BFGS-B.
    """
    start_time = time.time()
    
    # Flatten centers
    x0 = centers.flatten()
    
    def objective(x):
        if time.time() - start_time > time_budget:
            return 1e10  # Penalty for timeout
        c = x.reshape(-1, 3)
        return -_soft_min_radius(c, beta)
    
    # Bounds
    bounds = [(-0.5 + 1e-9, 0.5 - 1e-9)] * (N * 3)
    
    # Optimize
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
    
    best_centers = result.x.reshape(-1, 3)
    best_score = _compute_score(best_centers)
    
    return best_centers, best_score


def _local_search(centers, time_budget):
    """
    Local search using coordinate descent and small perturbations.
    """
    start_time = time.time()
    current = centers.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    # Coordinate descent with small steps
    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005]
    
    while time.time() - start_time < time_budget:
        improved = False
        
        for idx in range(N):
            if time.time() - start_time > time_budget:
                break
            
            for dim in range(3):
                for step in step_sizes:
                    for direction in [-1, 1]:
                        new_centers = current.copy()
                        new_centers[idx, dim] += direction * step
                        new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
                        
                        new_score = _compute_score(new_centers)
                        if new_score > current_score + 1e-12:
                            current = new_centers
                            current_score = new_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best = current.copy()
                            break
        
        if not improved:
            # Try random perturbations
            for _ in range(100):
                if time.time() - start_time > time_budget:
                    break
                
                idx = np.random.randint(N)
                delta = np.random.randn(3) * 0.001
                new_centers = current.copy()
                new_centers[idx] += delta
                new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
                
                new_score = _compute_score(new_centers)
                if new_score > current_score + 1e-12:
                    current = new_centers
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best = current.copy()
        
        if not improved:
            break
    
    return best, best_score


def _bcc_initial():
    """Generate a BCC-like initial configuration for 50 spheres."""
    # Use a 4x4x4 grid = 64 positions, take 50
    m = 4
    spacing = 1.0 / m
    centers = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = -0.5 + (i + 0.5) * spacing
                y = -0.5 + (j + 0.5) * spacing
                z = -0.5 + (k + 0.5) * spacing
                centers.append([x, y, z])
    
    centers = np.array(centers[:N])
    return centers


def _hcp_initial():
    """
    Generate an HCP-like initial configuration.
    Hexagonal close-packed structure projected into the cube.
    """
    # HCP has layers of hexagonally packed spheres
    # Let's create a 3D arrangement based on HCP
    
    centers = []
    
    # Layer 1: hexagonal packing
    # Hexagonal lattice in 2D: points at (i*a + j*a/2, j*a*sqrt(3)/2)
    a = 0.25  # spacing
    
    # Create multiple layers
    n_layers = 5
    layer_spacing = a * np.sqrt(2/3)  # HCP layer spacing
    
    for layer in range(n_layers):
        z = -0.5 + 0.15 + layer * layer_spacing
        if z > 0.45:
            break
        
        # Hexagonal grid in this layer
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * a + j * a / 2
                y = j * a * np.sqrt(3) / 2
                
                # Offset every other layer
                if layer % 2 == 1:
                    x += a / 2
                    y += a * np.sqrt(3) / 6
                
                if -0.45 <= x <= 0.45 and -0.45 <= y <= 0.45:
                    centers.append([x, y, z])
    
    # If we have too many, take the first 50
    centers = np.array(centers)
    if len(centers) > N:
        centers = centers[:N]
    elif len(centers) < N:
        # Fill with random points
        while len(centers) < N:
            centers = np.vstack([centers, np.random.uniform(-0.4, 0.4, 3)])
    
    return centers


def construct():
    """Construct optimal sphere packing."""
    start_time = time.time()
    deadline = 115.0  # safety margin before 150s kill
    
    np.random.seed(42)
    
    # Generate multiple initial configurations
    initial_configs = [
        _bcc_initial(),
        _hcp_initial(),
        _bcc_initial() + np.random.normal(0, 0.02, size=(N, 3)),
    ]
    
    # Clip all to valid range
    for i in range(len(initial_configs)):
        initial_configs[i] = np.clip(initial_configs[i], -0.49, 0.49)
    
    best_centers = initial_configs[0].copy()
    best_score = _compute_score(best_centers)
    
    # Try smooth optimization from each initial config
    for i, initial in enumerate(initial_configs):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining < 20.0:
            break
        
        # Allocate time for smooth optimization
        time_for_smooth = min(remaining / (len(initial_configs) - i), 25.0)
        
        optimized, score = _optimize_smooth(initial, time_budget=time_for_smooth)
        
        if score > best_score:
            best_score = score
            best_centers = optimized.copy()
    
    # Local search polish
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining > 10.0:
        polished, polished_score = _local_search(best_centers, time_budget=min(remaining - 3.0, 30.0))
        if polished_score > best_score:
            best_score = polished_score
            best_centers = polished.copy()
    
    # Final high-precision polish with smaller steps
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining > 5.0:
        # Another round of local search with even smaller steps
        polished2, polished_score2 = _local_search(best_centers, time_budget=min(remaining - 2.0, 15.0))
        if polished_score2 > best_score:
            best_score = polished_score2
            best_centers = polished2.copy()
    
    # Ensure valid output
    best_centers = np.clip(best_centers, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_centers
# EVOLVE-BLOCK-END
