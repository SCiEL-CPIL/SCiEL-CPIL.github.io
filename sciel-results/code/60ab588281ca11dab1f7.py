import numpy as np
import time
from scipy.optimize import minimize

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
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dists_sq))

    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(centers_flat, beta):
    """
    Objective function for L-BFGS-B.
    Minimizes the soft-min approximation of the radius.
    r(p) approx -1/beta * ln(sum(exp(-beta * x_k)))
    where x_k are the constraints (distances).
    We want to maximize r, so we minimize -r, which is 1/beta * ln(sum(exp(-beta * x_k))).
    """
    centers = centers_flat.reshape(-1, 3)
    
    # 1. Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    # 2. Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
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
    # Objective is 1/beta * ln(sum(exp(-beta * x)))
    # = 1/beta * (max_val + ln(sum_exp))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _build_grid(nx, ny, nz):
    """Build a grid of points."""
    xs = np.linspace(-0.45, 0.45, nx)
    ys = np.linspace(-0.45, 0.45, ny)
    zs = np.linspace(-0.45, 0.45, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    return centers[:N]

def _local_refinement(centers, max_iters=100, step=0.01):
    """
    Perform local refinement using Nelder-Mead on the raw radius (negative).
    This helps escape shallow local minima that the soft-min might miss or settle into.
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
        # Ensure validity
        new_centers = np.clip(new_centers, -0.5, 0.5)
        return new_centers
    except Exception:
        return centers

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Initial guess: Grid
    # Try a few grid configurations to find a good starting point
    best_centers = None
    best_radius = 0.0
    
    grids = [
        (5, 5, 4),
        (5, 4, 5),
        (4, 5, 5),
        (4, 4, 7), # 112 points, take first 100
        (7, 4, 4),
        (4, 7, 4),
    ]
    
    for nx, ny, nz in grids:
        if time.time() > deadline:
            break
        centers = _build_grid(nx, ny, nz)
        # Add small noise to break symmetry
        centers += np.random.normal(0, 0.005, centers.shape)
        centers = np.clip(centers, -0.49, 0.49)
        
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
            
    if best_centers is None:
        # Fallback to simple grid
        best_centers = _build_grid(5, 5, 4)
        best_radius = _compute_radius(best_centers)
        
    # Optimization loop with soft-min
    beta = 10.0
    max_beta = 5000.0
    
    while time.time() < deadline and beta < max_beta:
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
            
        # Increase beta
        beta *= 1.3
        
        # Check time
        if time.time() > deadline:
            break
            
    # Final local refinement with Nelder-Mead if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        try:
            refined = _local_refinement(best_centers, max_iters=200)
            r_ref = _compute_radius(refined)
            if r_ref > best_radius:
                best_radius = r_ref
                best_centers = refined
        except Exception:
            pass
            
    # One more pass with soft-min if time permits and we improved with NM?
    # Or just ensure we return the best we have.
    
    return best_centers

# EVOLVE-BLOCK-END
