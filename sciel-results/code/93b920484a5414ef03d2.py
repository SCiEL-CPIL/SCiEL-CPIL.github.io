"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    if len(centers) < 2:
        min_dist_to_wall = np.min(0.5 - np.abs(centers))
        return min_dist_to_wall
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    min_wall_dist = np.min(0.5 - np.abs(centers))
    return min(min_pair_dist / 2.0, min_wall_dist)

def _energy_and_grad(centers, target_r):
    """Compute energy and gradient for repulsion model."""
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # Pairwise repulsion
    # Use a loop for clarity and to avoid creating huge temporary arrays if needed, 
    # but for N=100, vectorization is fine.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Mask for pairs closer than 2*target_r
    mask = dists < 2.0 * target_r
    # We need to iterate over pairs or use vectorized ops
    # Vectorized energy: sum of (2r - d)^2 for d < 2r
    violations = np.where(mask, 2.0 * target_r - dists, 0.0)
    energy += np.sum(violations**2)
    
    # Vectorized gradient contribution from pairs
    # grad_i += sum_j -2(2r - d_ij) * (c_i - c_j) / d_ij
    # Note: (c_i - c_j) / d_ij is the unit vector from j to i.
    # Let's compute unit vectors
    # dists is inf on diagonal, so division is safe (0/inf = 0)
    unit_vecs = diff / dists[:, :, np.newaxis]
    
    # coeff[i, j] = -2 * (2r - d_ij) if mask[i,j] else 0
    coeff = np.where(mask, -2.0 * (2.0 * target_r - dists), 0.0)
    
    # grad[i] = sum_j coeff[i, j] * unit_vecs[i, j]
    pair_grad = np.sum(coeff[:, :, np.newaxis] * unit_vecs, axis=1)
    grad += pair_grad
    
    # Wall repulsion
    # Distances to 6 walls
    # x: 0.5 - c_x, 0.5 + c_x
    # y: 0.5 - c_y, 0.5 + c_y
    # z: 0.5 - c_z, 0.5 + c_z
    
    # Calculate min distance to wall for each sphere
    # We need to know which wall is the closest to compute the gradient direction
    # dists_to_walls shape: (N, 6)
    dw = np.zeros((n, 6))
    dw[:, 0] = 0.5 - centers[:, 0] # left x
    dw[:, 1] = 0.5 + centers[:, 0] # right x
    dw[:, 2] = 0.5 - centers[:, 1] # left y
    dw[:, 3] = 0.5 + centers[:, 1] # right y
    dw[:, 4] = 0.5 - centers[:, 2] # left z
    dw[:, 5] = 0.5 + centers[:, 2] # right z
    
    min_wall_dist = np.min(dw, axis=1)
    wall_violations = np.maximum(0, target_r - min_wall_dist)
    energy += np.sum(wall_violations**2)
    
    # Gradient for walls
    # If min_wall_dist < target_r, we push away from the closest wall.
    # Find index of min wall
    min_idx = np.argmin(dw, axis=1)
    
    # Create gradient contributions
    # For each sphere i, if violated:
    # grad[i] += -2 * (r - w_i) * grad(w_i)
    # grad(w_i) is a vector pointing towards the closest wall? No, w_i is distance TO wall.
    # If wall is left x (dw[:,0]), w_i = 0.5 - c_x. d(w_i)/d(c_x) = -1.
    # So grad(w_i) wrt c_i is (-1, 0, 0).
    # Energy term: (r - w_i)^2. dE/dc_i = 2(r-w_i) * (-1) * d(w_i)/dc_i = -2(r-w_i) * d(w_i)/dc_i.
    # Wait, E = (r - w)^2. dE/dw = -2(r-w). dw/dc = normal vector pointing OUT of cube?
    # Let's check: Left wall x=-0.5. w = 0.5 - c_x. If c_x increases, w decreases.
    # We want to increase w. So force should be in +x direction.
    # dE/dc_x = 2(r-w) * (-1) * (-1) = 2(r-w). Positive. Correct.
    
    wall_grad = np.zeros_like(centers)
    violated = wall_violations > 0
    
    for i in range(n):
        if violated[i]:
            idx = min_idx[i]
            factor = -2.0 * (target_r - min_wall_dist[i])
            # Gradient of w_i wrt c_i
            # 0: -1, 1: +1, 2: -1, 3: +1, 4: -1, 5: +1
            # These are the derivatives of the distance functions.
            # Let's just construct the vector directly.
            # If idx 0 (left x), d(w)/dc = (-1, 0, 0). Force = -2(r-w) * (-1,0,0) = (2(r-w), 0, 0)
            # If idx 1 (right x), d(w)/dc = (1, 0, 0). Force = -2(r-w) * (1,0,0) = (-2(r-w), 0, 0)
            # So force is always away from wall.
            
            if idx == 0: wall_grad[i, 0] += -factor * (-1)
            elif idx == 1: wall_grad[i, 0] += -factor * (1)
            elif idx == 2: wall_grad[i, 1] += -factor * (-1)
            elif idx == 3: wall_grad[i, 1] += -factor * (1)
            elif idx == 4: wall_grad[i, 2] += -factor * (-1)
            elif idx == 5: wall_grad[i, 2] += -factor * (1)
            
    grad += wall_grad
    
    return energy, grad

def _optimize_from_init(centers_init, target_r, max_iter=500, tol=1e-12):
    """Optimize from a given initial configuration using L-BFGS-B."""
    def objective(x):
        c = x.reshape(-1, 3)
        e, g = _energy_and_grad(c, target_r)
        return e, g
    
    # Flatten centers
    x0 = centers_init.flatten()
    
    # Bounds: centers must be in [-0.5, 0.5]
    # Actually, to ensure radius r, they should be in [-0.5+r, 0.5-r]
    # But the penalty handles the wall constraint.
    # Using bounds helps L-BFGS-B stay in the box.
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8, 'disp': False}
    )
    
    return result.x.reshape(-1, 3), result.success, result.fun

def _grid_init(n):
    """Generate grid initial configuration."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def _random_init(n, seed=None):
    """Generate random initial configuration inside the cube."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Strategy 1: Grid initialization
    init_grid = _grid_init(N)
    r_grid = _compute_radius(init_grid)
    best_radius = r_grid
    best_centers = init_grid.copy()
    
    # Strategy 2: Random initializations
    # We will try to optimize from a few random starts if time permits.
    # But grid is usually a strong start for 100 spheres.
    # Let's try to improve the grid solution first.
    
    # Optimization 1: From Grid
    # We need to estimate a good target_r.
    # Start with current best radius.
    target_r = best_radius * 1.01
    
    try:
        optimized, success, energy = _optimize_from_init(
            best_centers, target_r, max_iter=200, tol=1e-10
        )
        r_opt = _compute_radius(optimized)
        if r_opt > best_radius:
            best_radius = r_opt
            best_centers = optimized.copy()
    except Exception:
        pass
    
    # Try to push the radius higher
    # If the energy is 0, we can try increasing target_r and re-optimizing.
    # This is a common strategy: binary search on radius.
    
    # Let's try a few random restarts if time allows
    for seed in range(5):
        if time.time() > deadline - 20.0:
            break
            
        init_rand = _random_init(N, seed=seed)
        r_rand = _compute_radius(init_rand)
        
        # Only optimize if initial radius is decent or if we have lots of time
        # Random starts are usually worse than grid for N=100
        # But maybe a specific random start falls into a better basin?
        # Let's just try one random optimization from the best random start.
        
        # Actually, let's just try to improve the current best solution further.
        # Increase target_r slightly and optimize again.
        
        if best_centers is not None:
            # Perturb the best solution
            perturbed = best_centers + np.random.randn(N, 3) * 0.02
            perturbed = np.clip(perturbed, -0.49, 0.49)
            
            # Use current best radius as target
            target_r_new = best_radius
            try:
                optimized, success, energy = _optimize_from_init(
                    perturbed, target_r_new, max_iter=100, tol=1e-10
                )
                r_opt = _compute_radius(optimized)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = optimized.copy()
            except Exception:
                continue
    
    # Final check and clip
    if best_centers is None:
        best_centers = _grid_init(N)
        
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
# EVOLVE-BLOCK-END
