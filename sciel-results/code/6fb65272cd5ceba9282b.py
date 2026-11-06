"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Best known radius is ~0.11138. Let's try to find a good packing.
    # Strategy: 
    # 1. Start with a reasonable initial configuration (e.g., grid or random with repulsion)
    # 2. Use a local optimization method to maximize the minimum distance
    #    We can maximize the minimum of (distances between centers) and (distance to walls)
    #    This is equivalent to maximizing r such that all constraints are satisfied.
    #    
    # Alternative: Minimize the sum of squared violations, or use a soft-min approach.
    # Let's try: maximize the minimum of all pairwise distances and wall distances.
    # Since scipy minimize minimizes, we can minimize the negative of the min distance.
    # But min is non-smooth. We can use a soft-min (log-sum-exp based) for smoothness.
    # 
    # Actually, a simpler approach: 
    #   Use a random restart + local optimization where we push spheres apart.
    #   Or use the fact that for fixed N in a cube, good packings often have structure.
    #
    # Let's try a simple approach first: 
    #   1. Generate several random initial configurations with a minimum separation.
    #   2. For each, run a local optimization to maximize the minimum distance.
    #   3. Keep the best.
    #
    # For local optimization, we can use L-BFGS-B or Nelder-Mead.
    # The objective: minimize -min(min_{i<j} ||c_i - c_j||, min_i min_k (0.5 - |c_ik|))
    # This is non-smooth. We can use a smooth approximation.
    #
    # Let's use a different approach: 
    #   We want to maximize r such that:
    #     ||c_i - c_j|| >= 2r for all i < j
    #     |c_ik| <= 0.5 - r for all i, k
    #   This is a feasibility problem for given r. We can binary search on r.
    #   For each r, we can try to find a feasible configuration using optimization.
    #
    # But that's complex. Let's try a simpler iterative method:
    #   Start with a configuration, compute the current min distance (radius),
    #   then move spheres to increase the minimum distance.
    #
    # Let's use a gradient-based approach with a smooth surrogate.
    # Define f(x) = -softmin(distances), where distances are all pairwise and wall distances.
    # softmin(a) = -log(sum(exp(-a_i / t)) * t) for small t, but this is for min.
    # Actually, softmin(a) ≈ min(a) as t→0. We can use t=0.01 or so.
    #
    # Let's implement this.

    def compute_distances(centers):
        """Compute all pairwise distances and wall distances."""
        n = len(centers)
        # Pairwise distances
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        # Wall distances: for each center, distance to nearest wall
        # Center is in [-0.5, 0.5]^3, so distance to wall is 0.5 - max(|x|, |y|, |z|)
        # Actually, for a sphere to be inside, we need 0.5 - |c_ik| >= r for all k.
        # So the "wall distance" for center i is min_k (0.5 - |c_ik|)
        abs_centers = np.abs(centers)
        wall_dists = 0.5 - np.max(abs_centers, axis=1)
        # All distances: pairwise and wall
        all_dists = np.concatenate([dists[dists < np.inf], wall_dists])
        return all_dists

    def objective(x):
        """Objective to minimize: negative of soft-min of all distances."""
        centers = x.reshape(-1, 3)
        # Clip to stay in bounds (for safety during optimization)
        centers = np.clip(centers, -0.4999, 0.4999)
        all_dists = compute_distances(centers)
        # Soft-min: -log(sum(exp(-d/t))) * t
        t = 0.01
        # To avoid overflow, subtract max
        max_d = np.max(all_dists)
        exp_terms = np.exp(-(all_dists - max_d) / t)
        softmin = -(t * np.log(np.sum(exp_terms)) + max_d)
        return -softmin

    def gradient(x):
        """Numerical gradient."""
        eps = 1e-7
        n = len(x)
        grad = np.zeros_like(x)
        f0 = objective(x)
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
        return grad

    # Initial configuration: try a grid-like start with some randomization
    def initial_config():
        # 5x5x4 grid = 100 spheres
        # But let's try a few different starts
        starts = []
        
        # Start 1: Regular grid
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks[:4] for y in ticks for x in ticks])
        starts.append(grid[:100])
        
        # Start 2: Random with repulsion
        rng = np.random.RandomState(42)
        for trial in range(5):
            centers = rng.uniform(-0.4, 0.4, (100, 3))
            # Simple repulsion to avoid overlaps
            for _ in range(100):
                moved = False
                for i in range(100):
                    for j in range(i+1, 100):
                        d = np.linalg.norm(centers[i] - centers[j])
                        if d < 0.1:
                            # Push apart
                            if d > 1e-10:
                                dir_vec = (centers[i] - centers[j]) / d
                                centers[i] += 0.005 * dir_vec
                                centers[j] -= 0.005 * dir_vec
                                moved = True
                if not moved:
                    break
            centers = np.clip(centers, -0.49, 0.49)
            starts.append(centers)
        
        return starts

    best_score = -1
    best_centers = None

    starts = initial_config()
    
    for start_idx, centers in enumerate(starts):
        if time.time() > deadline - 20:
            break
            
        x0 = centers.flatten()
        
        # Try a few optimization methods
        for method in ['L-BFGS-B', 'Nelder-Mead']:
            if time.time() > deadline - 10:
                break
                
            try:
                result = minimize(
                    objective, 
                    x0, 
                    method=method,
                    options={'maxiter': 5000, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < best_score:
                    centers_opt = result.x.reshape(-1, 3)
                    # Clip to valid range
                    centers_opt = np.clip(centers_opt, -0.4999, 0.4999)
                    
                    # Compute actual radius
                    all_dists = compute_distances(centers_opt)
                    min_dist = np.min(all_dists)
                    radius = min_dist / 2.0
                    
                    if radius > best_score:
                        best_score = radius
                        best_centers = centers_opt.copy()
                        
            except Exception as e:
                pass

    # If we didn't find anything better, return the grid
    if best_centers is None:
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks[:4] for y in ticks for x in ticks])
        best_centers = grid[:100]
    
    # Final check: ensure all centers are in the cube
    best_centers = np.clip(best_centers, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return best_centers

# EVOLVE-BLOCK-END
