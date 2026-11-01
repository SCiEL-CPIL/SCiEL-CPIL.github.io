"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

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

def _soft_min_objective(centers_flat, beta, iu):
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
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _build_initial_grid(nx, ny, nz, noise_scale=0.005, seed=42):
    """Build a grid initialization with small noise to break symmetry."""
    margin = 0.02
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Add small noise to break perfect symmetry
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, noise_scale, centers.shape)
    
    # Clip to valid range
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_halton_init(n=100, seed=42):
    """
    Build a Halton sequence based initialization for good space-filling properties.
    """
    def halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    
    rng = np.random.default_rng(seed)
    centers = np.zeros((n, 3))
    for i in range(1, n + 1):
        centers[i - 1, 0] = -0.5 + halton(i, 2)
        centers[i - 1, 1] = -0.5 + halton(i, 3)
        centers[i - 1, 2] = -0.5 + halton(i, 5)
    
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _polish_kdtree(centers, deadline, max_iters=2000):
    """
    High-precision polish using cKDTree for fast nearest-neighbor queries.
    This is much faster than cdist for the local optimization phase.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    rng = np.random.default_rng(12345)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Pick a random point to move
        idx = rng.integers(N)
        
        # Build KDTree for fast nearest neighbor
        tree = cKDTree(best_centers)
        
        # Find the nearest neighbor of this point
        dists, _ = tree.query(best_centers[idx], k=2)
        nearest_dist = dists[1] if len(dists) > 1 else 1.0
        
        # Also compute wall distance
        wall_dist = np.min(0.5 - np.abs(best_centers[idx]))
        
        # The limiting constraint for this point
        limit = min(nearest_dist / 2.0, wall_dist)
        
        # If this point is not on the tightest constraint, skip
        # Actually, let's just do a small random nudge and check
        step_size = 0.001 * (1.0 + 0.5 * np.random.rand())
        delta = rng.normal(0, step_size, 3)
        
        new_pos = best_centers[idx] + delta
        new_pos = np.clip(new_pos, -0.499, 0.499)
        
        # Quick check: will this improve?
        # Compute new wall distance
        new_wall = np.min(0.5 - np.abs(new_pos))
        
        # Compute new nearest neighbor distance
        other_centers = np.delete(best_centers, idx, axis=0)
        if len(other_centers) > 0:
            d = np.linalg.norm(new_pos - other_centers, axis=1)
            new_nn = np.min(d)
        else:
            new_nn = 1.0
        
        new_limit = min(new_nn / 2.0, new_wall)
        
        # Accept if it doesn't make things worse locally
        if new_limit >= limit - 1e-10:
            best_centers[idx] = new_pos
        
        # Every 50 iterations, check global radius
        if (it + 1) % 50 == 0:
            r = _compute_radius(best_centers)
            if r > best_radius:
                best_radius = r
    
    # Final check
    r = _compute_radius(best_centers)
    if r > best_radius:
        best_radius = r
    
    return best_centers, best_radius

def _optimize_from_initial(centers, deadline, beta_start=50.0, beta_end=5000.0, n_beta_steps=8):
    """
    Run L-BFGS-B optimization with increasing beta from initial centers.
    Returns (best_centers, best_radius).
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Precompute upper triangle indices
    iu = np.triu_indices(N, k=1)
    
    # Create beta schedule: geometric progression from beta_start to beta_end
    betas = np.geomspace(beta_start, beta_end, n_beta_steps)
    
    for beta in betas:
        if time.time() > deadline - 30.0:  # Leave time for polish
            break
        
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            # Check if we improved
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 35.0:
            break
    
    return best_centers, best_radius

def construct():
    start_time = time.time()
    # Use 140s deadline to leave safety margin for the 150s limit
    deadline = start_time + 140.0
    
    # Try multiple different grid configurations and noise levels
    grid_configs = [
        (5, 5, 4),
        (4, 5, 5),
        (5, 4, 5),
        (10, 5, 2),
        (5, 10, 2),
        (2, 5, 10),
    ]
    
    noise_scales = [0.003, 0.005, 0.01]
    
    best_centers = None
    best_radius = 0.0
    
    # Phase 1: Grid initializations
    for nx, ny, nz in grid_configs:
        if time.time() > deadline - 40.0:
            break
        
        for noise_scale in noise_scales:
            if time.time() > deadline - 35.0:
                break
            
            centers = _build_initial_grid(nx, ny, nz, noise_scale)
            r_init = _compute_radius(centers)
            
            if r_init > best_radius:
                best_radius = r_init
                best_centers = centers.copy()
            
            # Optimize from this initial
            opt_centers, opt_radius = _optimize_from_initial(
                centers, deadline,
                beta_start=50.0, beta_end=5000.0, n_beta_steps=6
            )
            
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers
    
    # Phase 2: Halton initialization
    if time.time() < deadline - 40.0:
        halton_centers = _build_halton_init()
        r_halton = _compute_radius(halton_centers)
        
        if r_halton > best_radius:
            best_radius = r_halton
            best_centers = halton_centers.copy()
        
        opt_centers, opt_radius = _optimize_from_initial(
            halton_centers, deadline,
            beta_start=50.0, beta_end=5000.0, n_beta_steps=6
        )
        
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers
    
    # Phase 3: Final polish with cKDTree
    if best_centers is not None and time.time() < deadline - 5.0:
        polish_deadline = deadline - 2.0
        best_centers, best_radius = _polish_kdtree(best_centers, polish_deadline)
    
    # If no valid result, fallback to simple grid
    if best_centers is None:
        best_centers = _build_initial_grid(5, 5, 4, 0.01)
    
    return best_centers
# EVOLVE-BLOCK-END
