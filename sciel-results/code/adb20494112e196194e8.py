"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum common radius for given centers in [-0.5, 0.5]^3."""
    # Minimum distance to boundary
    boundary_dists = np.array([
        0.5 - np.min(centers, axis=0),
        0.5 + np.min(-centers, axis=0)
    ])
    # Actually, we need min over each coordinate for each point
    # For each point i, distance to boundary is min(0.5 - x_i, 0.5 + x_i, 0.5 - y_i, 0.5 + y_i, 0.5 - z_i, 0.5 + z_i)
    # Which is min(0.5 - |c_ik|) for k in {0,1,2}
    coord_abs = np.abs(centers)  # (N, 3)
    boundary_dist_per_point = 0.5 - coord_abs.max(axis=1)
    min_boundary = boundary_dist_per_point.min()
    
    # Minimum pairwise distance
    # Compute all pairwise distances
    # Use broadcasting: (N, 1, 3) - (1, N, 3)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = dists.min()
    
    r = min(min_boundary, min_pair_dist / 2.0)
    return r

def _objective(x, N):
    """Objective to minimize: negative of the radius (we want to maximize radius)."""
    centers = x.reshape(N, 3)
    # Add large penalty if any point is outside [-0.5, 0.5]^3
    if np.any(np.abs(centers) > 0.5 + 1e-9):
        return 1.0  # Large positive value (we're minimizing, so this is bad)
    
    r = _compute_radius(centers)
    # We want to maximize r, so minimize -r
    # But also need to ensure points are well-separated to avoid degenerate solutions
    # Add a small penalty for very close points to help optimization
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = dists.min()
    
    # Objective: -r + small penalty if min_pair_dist is very small
    # Actually, let's just use -r and let the optimizer handle it
    # But to help convergence, add a soft penalty for overlap
    penalty = 0.0
    if min_pair_dist < 0.01:
        penalty = (0.01 - min_pair_dist) ** 2 * 1000
    
    return -r + penalty

def _objective_and_grad(x, N):
    """Objective and gradient for L-BFGS-B."""
    centers = x.reshape(N, 3)
    
    # Check bounds
    if np.any(np.abs(centers) > 0.5 + 1e-9):
        return 1.0, np.zeros_like(x)
    
    # Compute radius
    coord_abs = np.abs(centers)
    boundary_dist_per_point = 0.5 - coord_abs.max(axis=1)
    min_boundary = boundary_dist_per_point.min()
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = dists.min()
    
    r = min(min_boundary, min_pair_dist / 2.0)
    
    # For gradient, we need to know which constraint is active
    # This is complex, so let's use finite differences or just return the objective
    # and let L-BFGS-B use its own gradient estimation
    
    penalty = 0.0
    if min_pair_dist < 0.01:
        penalty = (0.01 - min_pair_dist) ** 2 * 1000
    
    return -r + penalty, None

def _initial_grid(N):
    """Create initial grid of N points in [-0.5, 0.5]^3."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N]

def _optimize_from_initial(centers, N, deadline):
    """Optimize starting from given initial centers."""
    x0 = centers.flatten()
    
    # Define bounds: each coordinate in [-0.49, 0.49] to stay safely inside
    bounds = [(-0.49, 0.49)] * (N * 3)
    
    best_x = x0.copy()
    best_r = _compute_radius(centers)
    
    # Use L-BFGS-B
    start_time = time.time()
    try:
        result = minimize(
            _objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        if result.fun < 0:  # Valid improvement
            new_centers = result.x.reshape(N, 3)
            new_r = _compute_radius(new_centers)
            if new_r > best_r:
                best_r = new_r
                best_x = result.x.copy()
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline:
        return best_x, best_r
    
    # Try a few more optimizations with different initial perturbations
    for restart in range(3):
        if time.time() > deadline:
            break
        
        # Add small random perturbation to best solution
        noise = np.random.randn(N * 3) * 0.01
        x_perturbed = best_x + noise
        # Clip to bounds
        x_perturbed = np.clip(x_perturbed, -0.49, 0.49)
        
        try:
            result = minimize(
                _objective,
                x_perturbed,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            if result.fun < 0:
                new_centers = result.x.reshape(N, 3)
                new_r = _compute_radius(new_centers)
                if new_r > best_r:
                    best_r = new_r
                    best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x, best_r

def construct():
    """Construct centers of N equal spheres inside the cube [-1/2, 1/2]^3."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with grid
    centers = _initial_grid(N)
    best_x, best_r = centers.flatten(), _compute_radius(centers)
    
    # Optimize from grid
    best_x, best_r = _optimize_from_initial(centers, N, deadline)
    
    if time.time() > deadline:
        return best_x.reshape(N, 3)
    
    # Try random initializations
    np.random.seed(42)
    for restart in range(5):
        if time.time() > deadline:
            break
        
        # Random initial positions
        centers_rand = np.random.uniform(-0.45, 0.45, size=(N, 3))
        best_x_new, best_r_new = _optimize_from_initial(centers_rand, N, deadline)
        
        if best_r_new > best_r:
            best_r = best_r_new
            best_x = best_x_new
    
    return best_x.reshape(N, 3)
# EVOLVE-BLOCK-END
