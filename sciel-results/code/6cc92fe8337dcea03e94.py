import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0

    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)

    # Pairwise distances
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)

    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(centers_flat, beta):
    """
    Objective function for L-BFGS-B.
    Minimizes the soft-min approximation of the radius.
    """
    centers = centers_flat.reshape(-1, 3)
    
    # 1. Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    # 2. Pairwise distances / 2
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    # Get upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    # To avoid overflow/underflow, use log-sum-exp trick
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _build_grid(nx, ny, nz, noise_scale=0.005, seed=42):
    """Build a grid of points with noise."""
    margin = 0.02
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    if len(centers) > N:
        # If grid is larger than N, take a random subset or the first N?
        # Taking first N might be biased. Let's just generate exact N if possible or trim.
        # For 5x5x4=100, it's exact.
        # If not exact, we should probably adjust grid or use random placement.
        # Here we assume the grids chosen are exactly N or we trim.
        # Trimming breaks symmetry less than random sampling? Actually random sampling is better for diversity.
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(centers), N, replace=False)
        centers = centers[indices]
        
    # Add small noise to break perfect symmetry
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, noise_scale, centers.shape)
    
    # Clip to valid range
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _random_cluster_init(seed=42):
    """
    Generate a random initialization by placing N points in clusters.
    This helps escape the lattice lock-in.
    """
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-0.45, 0.45, (N, 3))
    # Add some structure by clustering
    n_clusters = 10
    cluster_centers = rng.uniform(-0.4, 0.4, (n_clusters, 3))
    assignments = rng.integers(0, n_clusters, N)
    for i in range(N):
        centers[i] += cluster_centers[assignments[i]]
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _optimize_from_initial(centers, deadline, beta_start=20.0, beta_end=2000.0, n_beta_steps=10):
    """
    Run L-BFGS-B optimization with increasing beta from initial centers.
    Returns (best_centers, best_radius).
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Create beta schedule: geometric progression
    betas = np.geomspace(beta_start, beta_end, n_beta_steps)
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        x0 = best_centers.flatten()
        
        try:
            # L-BFGS-B is good for box-constrained problems
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 50}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 5.0:
            break
    
    return best_centers, best_radius

def _local_refinement_nm(centers, max_iters=500):
    """
    Perform local refinement using Nelder-Mead on the raw radius (negative).
    This handles the non-differentiability of the min function better than gradients
    for the final fine-tuning.
    """
    def neg_radius(c_flat):
        c = c_flat.reshape(-1, 3)
        return -_compute_radius(c)
    
    try:
        res = minimize(
            neg_radius,
            centers.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iters, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        new_centers = res.x.reshape(-1, 3)
        new_centers = np.clip(new_centers, -0.5, 0.5)
        return new_centers
    except Exception:
        return centers

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Define diverse initializations
    # 1. Grids
    grid_configs = [
        (5, 5, 4),
        (4, 5, 5),
        (5, 4, 5),
        (10, 5, 2), # 100 points
        (5, 10, 2),
        (2, 5, 10),
    ]
    
    # 2. Random Clusters
    # We will try a few of these if time permits, but grids are usually better starts.
    
    initializations = []
    
    # Add grid initializations with different noise scales
    for nx, ny, nz in grid_configs:
        for noise in [0.001, 0.005, 0.01]:
            init = _build_grid(nx, ny, nz, noise_scale=noise, seed=np.random.randint(1000))
            initializations.append(init)
            
    # Add a couple of random cluster initializations
    for seed in [1, 2]:
        init = _random_cluster_init(seed=seed)
        initializations.append(init)
        
    # Shuffle initializations to ensure we don't just do grids first if they fail
    np.random.shuffle(initializations)
    
    # Optimize from each initial state
    for init_centers in initializations:
        if time.time() > deadline - 20.0:
            break
            
        # Quick check: if this initial is already worse than best, we might still optimize it 
        # because optimization can improve it significantly. But if it's very bad, skip?
        # Let's just run the optimization.
        
        opt_centers, opt_radius = _optimize_from_initial(
            init_centers, deadline, 
            beta_start=20.0, beta_end=2000.0, n_beta_steps=8
        )
        
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers
    
    # If no valid result, fallback to simple grid
    if best_centers is None:
        best_centers = _build_grid(5, 5, 4, noise_scale=0.005)
        best_radius = _compute_radius(best_centers)
        
    # Final local refinement with Nelder-Mead if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        try:
            # NM is slow, so limit iterations based on time
            max_iters = int(remaining_time * 10) # Rough estimate
            max_iters = max(100, min(max_iters, 1000))
            refined = _local_refinement_nm(best_centers, max_iters=max_iters)
            r_ref = _compute_radius(refined)
            if r_ref > best_radius:
                best_radius = r_ref
                best_centers = refined
        except Exception:
            pass
            
    # One more pass with soft-min if time permits? 
    # If NM improved it, a final soft-min with high beta might polish it.
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        try:
            final_opt, final_rad = _optimize_from_initial(
                best_centers, deadline,
                beta_start=500.0, beta_end=5000.0, n_beta_steps=3
            )
            if final_rad > best_radius:
                best_radius = final_rad
                best_centers = final_opt
        except Exception:
            pass

    return best_centers

# EVOLVE-BLOCK-END
