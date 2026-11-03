"""Improved packing: Coulomb-repulsion optimization with multi-restart L-BFGS-B."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
CUBE_HALF = 0.5
DEADLINE = 115.0  # seconds, safety margin before 120s kill

def _coulomb_energy(pos):
    """
    pos: (N, 3) array of positions in [-0.5, 0.5]^3.
    Returns total repulsive energy sum_{i<j} 1/|r_i - r_j|.
    We minimize this to maximize minimum distance.
    """
    # Compute pairwise differences
    # pos[i] - pos[j] for all i < j
    n = pos.shape[0]
    # Use broadcasting: diff[i,j] = pos[i] - pos[j], shape (N, N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    # Distance matrix
    dist = np.linalg.norm(diff, axis=2)
    # Mask diagonal (self-distances = 0)
    np.fill_diagonal(dist, np.inf)
    # Sum of 1/dist for all pairs (each pair counted twice, divide by 2)
    energy = np.sum(1.0 / dist) / 2.0
    return energy

def _coulomb_energy_grad(pos):
    """
    Gradient of Coulomb energy with respect to positions.
    dE/dpos[i] = sum_{j != i} (pos[i] - pos[j]) / |pos[i] - pos[j]|^3
    """
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    # Factor: 1/dist^3
    factor = 1.0 / (dist ** 3)  # (N, N)
    # Set diagonal factor to 0
    np.fill_diagonal(factor, 0.0)
    # grad[i] = sum_j factor[i,j] * diff[i,j]
    # factor has shape (N, N), diff has shape (N, N, 3)
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    return grad

def _make_initial_grid(n, seed=0):
    """Generate an initial grid-like configuration with some randomization."""
    rng = np.random.RandomState(seed)
    m = int(np.ceil(n ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    # Generate all grid points
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    if len(grid) >= n:
        # Shuffle and take first n
        idx = rng.permutation(len(grid))[:n]
        pos = grid[idx].copy()
    else:
        # Fallback: random
        pos = rng.uniform(-0.45, 0.45, size=(n, 3))
    # Add small random perturbation to break symmetry
    pos += rng.uniform(-0.01, 0.01, size=pos.shape)
    # Clip to valid range
    pos = np.clip(pos, -0.49, 0.49)
    return pos

def _make_random_config(n, seed=0):
    """Generate a random configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(-0.45, 0.45, size=(n, 3))
    return pos

def _compute_radius(pos):
    """Compute the valid radius for a configuration."""
    n = pos.shape[0]
    if n < 2:
        return 0.0
    # Min pairwise distance
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    # Min distance to boundary
    boundary_dist = np.min(CUBE_HALF - np.abs(pos))
    r = min(min_pair / 2.0, boundary_dist)
    return r

def construct():
    start_time = time.time()
    
    best_pos = None
    best_radius = 0.0
    
    # Try multiple initial configurations with different seeds
    # Strategy: grid-based with perturbations, and random
    seeds_to_try = list(range(20))
    
    # Also try a few structured configurations
    # Seed 0: near-cubic grid
    # Seeds 1-19: grid with varying perturbations
    
    for seed in seeds_to_try:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15.0:
            break
        
        # Choose initial config
        if seed == 0:
            pos0 = _make_initial_grid(N, seed=0)
        elif seed == 1:
            pos0 = _make_random_config(N, seed=42)
        elif seed == 2:
            # Body-centered cubic-like
            rng = np.random.RandomState(seed)
            m = int(np.ceil(N ** (1.0/3.0)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            if len(grid) >= N:
                idx = rng.permutation(len(grid))[:N]
                pos0 = grid[idx].copy()
            else:
                pos0 = rng.uniform(-0.45, 0.45, size=(N, 3))
            pos0 += rng.uniform(-0.02, 0.02, size=pos0.shape)
            pos0 = np.clip(pos0, -0.49, 0.49)
        else:
            pos0 = _make_initial_grid(N, seed=seed)
        
        # L-BFGS-B optimization
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(
                _coulomb_energy,
                pos0.ravel(),
                jac=_coulomb_energy_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 500,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'maxls': 50,
                    'maxfun': 1000,
                }
            )
        except Exception:
            continue
        
        pos_opt = res.x.reshape(N, 3)
        # Ensure within bounds
        pos_opt = np.clip(pos_opt, -0.5, 0.5)
        
        r = _compute_radius(pos_opt)
        
        if r > best_radius:
            best_radius = r
            best_pos = pos_opt.copy()
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10.0:
            break
    
    # If we haven't found anything good, use the grid
    if best_pos is None:
        best_pos = _make_initial_grid(N, seed=0)
        best_radius = _compute_radius(best_pos)
    
    # Final refinement: try a few more restarts from the best solution
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 20.0:
        for extra_seed in range(5):
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 15.0:
                break
            
            # Perturb the best solution
            rng = np.random.RandomState(100 + extra_seed)
            pos_pert = best_pos + rng.uniform(-0.01, 0.01, size=best_pos.shape)
            pos_pert = np.clip(pos_pert, -0.49, 0.49)
            
            bounds = [(-0.5, 0.5)] * (N * 3)
            try:
                res = minimize(
                    _coulomb_energy,
                    pos_pert.ravel(),
                    jac=_coulomb_energy_grad,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 300,
                        'ftol': 1e-12,
                        'gtol': 1e-10,
                        'maxls': 50,
                    }
                )
            except Exception:
                continue
            
            pos_opt = res.x.reshape(N, 3)
            pos_opt = np.clip(pos_opt, -0.5, 0.5)
            r = _compute_radius(pos_opt)
            
            if r > best_radius:
                best_radius = r
                best_pos = pos_opt.copy()
    
    # Final check: ensure validity
    # Clip to ensure inside cube
    best_pos = np.clip(best_pos, -0.5, 0.5)
    
    return best_pos
