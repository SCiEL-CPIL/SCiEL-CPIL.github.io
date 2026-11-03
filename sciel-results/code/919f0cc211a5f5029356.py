"""Program for packing 100 equal spheres in a unit cube."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Construct centres of 100 equal spheres inside the cube [-1/2, 1/2]^3.
    
    Uses a multi-start local optimization approach:
    1. Generate diverse initial configurations (grid perturbations, random, quasi-random)
    2. For each, run local optimization to maximize the minimum distance
       (which directly maximizes the radius)
    3. Return the best configuration found
    
    The objective function is:
      - min over all pairs (i,j) of |c_i - c_j|  (pairwise separation)
      - min over all points i and coordinates k of (1/2 - |c_ik|)  (distance to boundary)
    
    We maximize the minimum of these, which equals 2*r.
    
    To make this differentiable and amenable to optimization, we use a smooth
    approximation or directly optimize the minimum via a log-sum-exp style
    barrier, but more practically, we can use a penalty-based approach.
    
    Actually, a cleaner approach: we want to maximize r such that:
      |c_i - c_j| >= 2r for all i < j
      1/2 - |c_ik| >= r for all i, k
    
    Equivalently, minimize -r subject to those constraints. But scipy's minimize
    doesn't handle this directly. Instead, we can formulate it as:
    
    Minimize: -min( min_{i<j} |c_i-c_j|/2, min_{i,k} (1/2 - |c_ik|) )
    
    This is non-smooth. A common trick is to use a smooth approximation of the
    minimum function, e.g., the log-sum-exp of negatives:
    
    f(x) = -1/t * log( sum exp(-t * d_k) )  approximates min(d_k) as t -> inf
    
    where d_k are the constraint distances (pairwise distances/2 and boundary distances).
    
    Let's define:
      d_pairs = [|c_i - c_j| / 2 for i < j]
      d_bounds = [1/2 - |c_ik| for i, k]
      all_d = d_pairs + d_bounds
      
      objective = -1/t * log(sum(exp(-t * d)))
    
    As t increases, this approaches -min(d), so minimizing it maximizes min(d).
    
    We'll use a moderate t and multiple restarts.
    """
    
    def compute_min_dist(centres):
        """Compute the minimum of pairwise distances/2 and boundary distances."""
        n = centres.shape[0]
        # Pairwise distances
        diffs = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        # Boundary distances
        abs_centres = np.abs(centres)
        min_bound = np.min(0.5 - abs_centres)
        
        return min(min_pair, min_bound)
    
    def smooth_min_objective(centres, t=50.0):
        """
        Smooth approximation of -min(d) using log-sum-exp.
        centres: (N, 3) array
        Returns a scalar to be minimized.
        """
        n = centres.shape[0]
        
        # Pairwise distances / 2
        diffs = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        d_pairs = dists / 2.0
        # Flatten unique pairs
        i_idx, j_idx = np.triu_indices(n, k=1)
        d_pairs_flat = d_pairs[i_idx, j_idx]
        
        # Boundary distances
        abs_centres = np.abs(centres)
        d_bounds = 0.5 - abs_centres
        d_bounds_flat = d_bounds.flatten()
        
        all_d = np.concatenate([d_pairs_flat, d_bounds_flat])
        
        # Clip to avoid numerical issues with very negative values
        all_d = np.clip(all_d, -1e-6, 1.0)
        
        # Log-sum-exp trick for numerical stability
        max_d = np.max(all_d)
        log_sum_exp = max_d + np.log(np.sum(np.exp(-t * (all_d - max_d))))
        smooth_min = log_sum_exp / t
        
        return -smooth_min
    
    def gradient_smooth_min(centres, t=50.0):
        """
        Compute gradient of the smooth min objective.
        This is complex; let's use finite differences or just let scipy use
        numerical gradients via BFGS/Nelder-Mead.
        """
        # For simplicity, we'll let scipy handle gradients numerically
        # by not providing a jac function
        pass
    
    def optimize_from_initial(initial_centres, t=50.0, maxiter=2000):
        """
        Run local optimization from a given initial configuration.
        Returns the optimized centres and the resulting min distance.
        """
        # Flatten centres for scipy
        x0 = initial_centres.flatten()
        
        def obj(x):
            c = x.reshape(-1, 3)
            # Keep centres within bounds to help optimization
            c = np.clip(c, -0.5, 0.5)
            return smooth_min_objective(c, t=t)
        
        # Use BFGS with numerical gradients
        result = minimize(obj, x0, method='BFGS', options={'maxiter': maxiter, 'gtol': 1e-8})
        
        best_centres = result.x.reshape(-1, 3)
        best_centres = np.clip(best_centres, -0.5, 0.5)
        
        # Evaluate the actual (non-smooth) min distance
        actual_min = compute_min_dist(best_centres)
        
        return best_centres, actual_min
    
    # Generate multiple initial configurations
    def grid_initial(perturb=0.01):
        """Grid-based initial configuration with small perturbations."""
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        grid = grid[:N]
        # Add small random perturbations
        rng = np.random.default_rng(42)
        grid = grid + rng.uniform(-perturb, perturb, size=grid.shape)
        return np.clip(grid, -0.5, 0.5)
    
    def random_initial(seed):
        """Random initial configuration."""
        rng = np.random.default_rng(seed)
        return rng.uniform(-0.45, 0.45, size=(N, 3))
    
    def halton_initial(base_seeds=(2, 3, 5)):
        """Halton sequence for quasi-random distribution."""
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        points = []
        for i in range(1, N + 1):
            point = [halton(i, b) for b in base_seeds]
            # Map to [-0.5, 0.5]
            point = [2 * p - 0.5 for p in point]
            points.append(point)
        return np.array(points)
    
    # Try different initializations and optimization parameters
    best_centres = None
    best_min_dist = -np.inf
    
    # Start with grid (known decent starting point)
    init_configs = []
    
    # Grid with different perturbations
    for perturb in [0.005, 0.01, 0.02]:
        init_configs.append(grid_initial(perturb))
    
    # Halton sequence
    init_configs.append(halton_initial())
    
    # Random with different seeds
    for seed in [0, 1, 2, 3, 4]:
        init_configs.append(random_initial(seed))
    
    # For each initial config, try optimization
    for idx, init in enumerate(init_configs):
        # Try with different t values
        for t in [30.0, 50.0, 80.0]:
            try:
                centres, min_dist = optimize_from_initial(init, t=t, maxiter=1500)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_centres = centres.copy()
            except Exception:
                continue
    
    # If no optimization succeeded, fall back to grid
    if best_centres is None:
        best_centres = grid_initial(0.005)
    
    # Final polish: one more round of optimization from the best found
    try:
        centres, min_dist = optimize_from_initial(best_centres, t=100.0, maxiter=1000)
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_centres = centres
    except Exception:
        pass
    
    # Ensure valid output
    best_centres = np.clip(best_centres, -0.5, 0.5)
    
    return best_centres
# EVOLVE-BLOCK-END
