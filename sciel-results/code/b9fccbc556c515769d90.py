"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N_CIRCLES = 200
HALF_SIDE = 0.5
KNOWN_BEST_R = 0.036612798904

def compute_radius(centers):
    """Compute the maximum valid radius for the given centers."""
    if len(centers) != N_CIRCLES:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > HALF_SIDE + 1e-12):
        return 0.0
    
    min_dist_to_wall = np.min(HALF_SIDE - np.abs(centers), axis=None)
    if min_dist_to_wall <= 0:
        return 0.0
    
    tree = cKDTree(centers)
    # Safe radius for N=200 in unit square
    pairs = tree.query_pairs(r=0.1) 
    if len(pairs) == 0:
        min_pair_dist = 1e9
    else:
        min_pair_dist = min(np.linalg.norm(centers[i] - centers[j]) for i, j in pairs)
    
    r = min(min_dist_to_wall, min_pair_dist / 2)
    return r

def _simple_grid():
    """Fallback simple grid."""
    m = int(np.ceil(np.sqrt(N_CIRCLES)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N_CIRCLES]

def random_grid_initialization():
    """Generate a random grid-like initialization."""
    best_config = None
    best_r = 0.0
    
    for m in range(14, 16):
        for n in range(14, 16):
            if m * n < N_CIRCLES:
                continue
            
            x_ticks = (np.arange(m) + 0.5) / m - 0.5
            y_ticks = (np.arange(n) + 0.5) / n - 0.5
            grid = np.array([(x, y) for y in y_ticks for x in ticks])
            
            idx = np.random.permutation(len(grid))
            centers = grid[idx[:N_CIRCLES]]
            
            r = compute_radius(centers)
            if r > best_r:
                best_r = r
                best_config = centers.copy()
    
    return best_config if best_config is not None else _simple_grid()

def energy_and_gradient(flat_centers, sigma, k_wall):
    """
    Compute energy and gradient for L-BFGS-B.
    Energy is a sum of soft repulsion terms (Lennard-Jones-like) and wall penalties.
    We minimize this energy to maximize the minimum distance.
    
    flat_centers: shape (2*N,)
    sigma: 'soft' minimum distance (we want to push distances above this)
    k_wall: stiffness of wall penalty
    """
    centers = flat_centers.reshape(N_CIRCLES, 2)
    
    # 1. Wall Penalty
    # We want centers to stay within [-0.5, 0.5].
    # Actually, for packing, we want them to stay away from the boundary by at least r.
    # But we don't know r. 
    # A common trick: minimize the energy where the 'potential' goes to infinity at the boundary.
    # Or, simply penalize proximity to the boundary.
    # Let's use a penalty that grows quadratically as points get close to the boundary.
    # dist_to_wall = 0.5 - abs(coord)
    # If dist_to_wall < 0, it's invalid.
    # Let's penalize max(0, (0.5 - abs(coord)) - sigma/2)^2 ? No, that's complex.
    
    # Simpler: Just keep them inside [-0.5, 0.5] using bounds in L-BFGS-B.
    # And use a repulsive force for pairs.
    
    # Pairwise Repulsion
    # We use a potential V(d) = (sigma/d)^12 - 2*(sigma/d)^6 ? No, that has a minimum at d=sigma.
    # We want to *increase* the minimum distance.
    # So we want a potential that is high when d < target_d and low when d >= target_d.
    # V(d) = max(0, target_d - d)^2 is non-differentiable at target_d.
    # V(d) = exp(-(d - target_d)/eps) for d < target_d is smooth.
    # Or simply: V(d) = (1/d)^2 is always repulsive. Minimizing sum(1/d^2) pushes all pairs apart.
    # This is equivalent to maximizing the minimum distance in a loose sense, but it tends to
    # spread points out evenly.
    
    # Let's use V(d) = 1/d^2.
    # Energy = sum_{i<j} 1/d_ij^2
    # This is a classic 'electrostatic' repulsion.
    
    # However, computing the full sum O(N^2) and its gradient in Python might be slow.
    # N=200 -> 20k pairs. 20k * 1000 iters = 20M ops. NumPy can handle this.
    
    # Let's vectorize the energy calculation.
    # d_ij for all pairs.
    # Use cKDTree for efficiency? L-BFGS needs gradient.
    # With N=200, full pairwise distance matrix is 200x200 = 40k floats. This is tiny.
    
    # Distance matrix
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    
    # Avoid division by zero
    dist_sq = np.maximum(dist_sq, 1e-12)
    dist = np.sqrt(dist_sq)
    
    # Mask diagonal
    mask = ~np.eye(N_CIRCLES, dtype=bool)
    
    # Energy: Sum of 1/d^2 for i != j
    # E = 0.5 * sum_{i != j} 1/d_ij^2  (factor 0.5 to avoid double counting if we sum over all i,j)
    # Actually, let's just sum over i < j.
    # But vectorized full sum is easier.
    # E = 0.5 * sum_{i,j} (1/d_ij^2 - delta_ij)
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    energy = 0.5 * np.sum(inv_dist_sq)
    
    # Gradient
    # dE/dx_i = sum_{j != i} dE/dx_ij
    # E = 0.5 * sum_{i,j} (x_i - x_j)^-2
    # Let d^2 = (x_i - x_j)^2. E_ij = 0.5 * (d^2)^-1
    # dE_ij/dx_i = 0.5 * (-1) * (d^2)^-2 * d(d^2)/dx_i
    # d(d^2)/dx_i = 2 * (x_i - x_j)
    # dE_ij/dx_i = -(d^2)^-2 * (x_i - x_j) = -(x_i - x_j) / d^4
    
    # Sum over j
    # grad_i = sum_{j != i} -(x_i - x_j) / d_ij^4
    
    # Vectorized:
    # diff is (N, N, 2). diff[i, j] = x_i - x_j
    # inv_dist_sq_2 is (N, N) = 1/d^4
    inv_dist_4 = inv_dist_sq**2
    np.fill_diagonal(inv_dist_4, 0.0)
    
    # grad = - sum_j diff[i, j] * inv_dist_4[i, j]
    # diff: (N, N, 2)
    # inv_dist_4: (N, N, 1)
    grad_flat = -np.sum(diff * inv_dist_4[:, :, np.newaxis], axis=1)
    
    # Add wall forces?
    # If we use bounds in L-BFGS-B, it handles the walls.
    # But we might want to push them away from walls to allow for larger r.
    # The repulsion 1/d^2 will naturally push them away from each other.
    # If they are in a grid, they will be pushed to corners? No, 1/d^2 is isotropic.
    # In a square, the optimal packing for large N is usually a hexagonal-like lattice.
    # The 1/d^2 potential might favor a specific arrangement.
    
    # Let's add a small "centering" force to keep them in the middle?
    # No, the boundary is the constraint.
    
    return energy, grad_flat.flatten()

def optimize_packing(init_centers, max_time=10.0):
    """
    Run L-BFGS-B to minimize the repulsive energy.
    """
    start_time = time.time()
    x0 = init_centers.flatten()
    
    # Bounds: centers must be in [-0.5, 0.5]
    # But if we want radius r, they should be in [-0.5+r, 0.5-r].
    # We don't know r. 
    # Let's just constrain to [-0.5, 0.5]. The repulsion will push them away from the boundary
    # if they are crowded.
    
    bounds = [(-HALF_SIDE, HALF_SIDE)] * (2 * N_CIRCLES)
    
    result = minimize(
        fun=energy_and_gradient,
        x0=x0,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-8, 'disp': False}
    )
    
    # Check time
    if time.time() - start_time > max_time:
        # L-BFGS doesn't have a perfect time callback, but we can check after.
        # If it took too long, we might want to interrupt, but it's usually fast for N=200.
        pass
        
    best_centers = result.x.reshape(N_CIRCLES, 2)
    
    # Ensure validity
    best_centers = np.clip(best_centers, -HALF_SIDE, HALF_SIDE)
    
    return best_centers

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_config = _simple_grid()
    best_r = compute_radius(best_config)
    
    # Initializations
    inits = [
        random_grid_initialization(),
        _simple_grid(),
    ]
    
    for init in inits:
        r = compute_radius(init)
        if r > best_r:
            best_r = r
            best_config = init.copy()
    
    np.random.seed(42)
    
    # Strategy: 
    # 1. Run L-BFGS-B on several random perturbations of the best grid.
    # 2. L-BFGS-B is a quasi-Newton method, so it should converge faster and find a better local minimum
    #    than simple gradient descent or pairwise pushes.
    # 3. The 1/d^2 potential is a proxy for maximizing min distance.
    
    for trial in range(20):
        if time.time() > deadline - 15.0:
            break
        
        # Perturb best config
        noise_level = 0.01 * (1 - trial / 20) + 0.002
        start_config = best_config + np.random.uniform(-noise_level, noise_level, size=(N_CIRCLES, 2))
        start_config = np.clip(start_config, -HALF_SIDE, HALF_SIDE)
        
        # Optimize
        optimized = optimize_packing(start_config, max_time=5.0)
        
        r = compute_radius(optimized)
        if r > best_r:
            best_r = r
            best_config = optimized.copy()
            
    # Final polish
    if time.time() < deadline - 5.0:
        final = optimize_packing(best_config, max_time=10.0)
        r = compute_radius(final)
        if r > best_r:
            best_r = r
            best_config = final.copy()
    
    return best_config
# EVOLVE-BLOCK-END
