import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import qmc
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances using cdist for efficiency
    dists = cdist(centers, centers, metric='euclidean')
    
    # Zero out diagonal
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
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _build_bcc_init():
    """
    Build a BCC (body-centered cubic) lattice initialization.
    """
    nx, ny, nz = 5, 5, 2
    
    xs = np.linspace(-0.45, 0.45, nx)
    ys = np.linspace(-0.45, 0.45, ny)
    zs = np.linspace(-0.45, 0.45, nz)
    
    # Corner positions
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    corners = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Body center positions
    dx = xs[1] - xs[0] if nx > 1 else 1.0
    dy = ys[1] - ys[0] if ny > 1 else 1.0
    dz = zs[1] - zs[0] if nz > 1 else 1.0
    
    body_xs = xs + dx / 2.0
    body_ys = ys + dy / 2.0
    body_zs = zs + dz / 2.0
    
    BX, BY, BZ = np.meshgrid(body_xs, body_ys, body_zs, indexing='ij')
    bodies = np.column_stack((BX.ravel(), BY.ravel(), BZ.ravel()))
    
    all_centers = np.vstack([corners, bodies])
    all_centers = np.clip(all_centers, -0.49, 0.49)
    all_centers += np.random.normal(0, 0.005, all_centers.shape)
    all_centers = np.clip(all_centers, -0.49, 0.49)
    
    return all_centers

def _build_fcc_init():
    """
    Build an FCC (face-centered cubic) lattice initialization.
    """
    # FCC: corners + face centers
    # Use a 4x4x4 grid of unit cells, each with 4 atoms
    nx, ny, nz = 4, 4, 4
    
    # Corner positions
    xs = np.linspace(-0.45, 0.45, nx + 1)
    ys = np.linspace(-0.45, 0.45, ny + 1)
    zs = np.linspace(-0.45, 0.45, nz + 1)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    corners = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Face centers
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    dz = zs[1] - zs[0]
    
    # XY faces
    fxy_x = xs[:-1] + dx / 2
    fxy_y = ys[:-1] + dy / 2
    fxy_z = zs
    FX, FY, FZ = np.meshgrid(fxy_x, fxy_y, fxy_z, indexing='ij')
    fxy = np.column_stack((FX.ravel(), FY.ravel(), FZ.ravel()))
    
    # XZ faces
    fxz_x = xs[:-1] + dx / 2
    fxz_y = ys
    fxz_z = zs[:-1] + dz / 2
    FX, FY, FZ = np.meshgrid(fxz_x, fxz_y, fxz_z, indexing='ij')
    fxz = np.column_stack((FX.ravel(), FY.ravel(), FZ.ravel()))
    
    # YZ faces
    fyz_x = xs
    fyz_y = ys[:-1] + dy / 2
    fyz_z = zs[:-1] + dz / 2
    FX, FY, FZ = np.meshgrid(fyz_x, fyz_y, fyz_z, indexing='ij')
    fyz = np.column_stack((FX.ravel(), FY.ravel(), FZ.ravel()))
    
    all_centers = np.vstack([corners, fxy, fxz, fyz])
    all_centers = np.clip(all_centers, -0.49, 0.49)
    
    # If we have more than N, randomly sample N
    if len(all_centers) > N:
        idx = np.random.choice(len(all_centers), N, replace=False)
        all_centers = all_centers[idx]
    
    all_centers += np.random.normal(0, 0.003, all_centers.shape)
    all_centers = np.clip(all_centers, -0.49, 0.49)
    
    return all_centers

def _build_grid_init():
    """
    Build a simple cubic grid initialization.
    """
    nx, ny, nz = 5, 5, 4
    
    xs = np.linspace(-0.45, 0.45, nx)
    ys = np.linspace(-0.45, 0.45, ny)
    zs = np.linspace(-0.45, 0.45, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    centers += np.random.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    
    return centers

def _build_halton_init():
    """
    Build a Halton sequence initialization for better space-filling.
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = np.zeros((N, 3))
    for i in range(N):
        points[i, 0] = halton(i + 1, 2) - 0.5
        points[i, 1] = halton(i + 1, 3) - 0.5
        points[i, 2] = halton(i + 1, 5) - 0.5
    
    points += np.random.normal(0, 0.02, points.shape)
    points = np.clip(points, -0.49, 0.49)
    
    return points

def _build_sobol_init():
    """
    Build a Sobol sequence initialization for better space-filling.
    """
    sampler = qmc.Sobol(d=3, scramble=True, seed=42)
    points = sampler.random(N)
    points = points - 0.5
    points += np.random.normal(0, 0.01, points.shape)
    points = np.clip(points, -0.49, 0.49)
    return points

def _optimize_from_initial(centers, deadline, beta_start=10.0, beta_end=10000.0):
    """
    Optimize from a given initial configuration using soft-min with increasing beta.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    beta = beta_start
    while time.time() < deadline - 15.0 and beta < beta_end:
        x0 = best_centers.flatten()
        
        # Increase maxiter for higher betas where we need more precision
        maxiter = 300 if beta < 100 else 500
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': maxiter, 'ftol': 1e-13, 'gtol': 1e-10, 'maxls': 100}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
        
        # More aggressive beta progression
        if beta < 20:
            beta *= 1.4
        elif beta < 100:
            beta *= 1.5
        elif beta < 500:
            beta *= 1.6
        else:
            beta *= 1.8
        
        if time.time() > deadline - 15.0:
            break
    
    return best_centers, best_radius

def _local_search_refine(centers, deadline):
    """
    Local search refinement: try small perturbations to improve the minimum distance.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Use gradient-based local search with very high beta
    for beta in [2000.0, 5000.0, 10000.0, 20000.0]:
        if time.time() > deadline - 8.0:
            break
            
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-13, 'maxls': 200}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
    
    # Direct local search: move each point slightly to maximize min distance
    for iteration in range(30):
        if time.time() > deadline - 3.0:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline - 3.0:
                break
            
            center = best_centers[i].copy()
            best_pos = center.copy()
            best_r = best_radius
            
            # Try small moves in each direction with adaptive step sizes
            for axis in range(3):
                for direction in [-1, 1]:
                    for step in [0.0005, 0.001, 0.002, 0.005, 0.01]:
                        trial = center.copy()
                        trial[axis] += direction * step
                        trial = np.clip(trial, -0.499, 0.499)
                        
                        # Quick check: compute min distance from this point to others
                        dists_to_others = np.linalg.norm(best_centers - trial, axis=1)
                        dists_to_others[i] = np.inf
                        min_pair = np.min(dists_to_others) / 2.0
                        
                        # Distance to walls
                        wall_dists = 0.5 - np.abs(trial)
                        min_wall = np.min(wall_dists)
                        
                        trial_r = min(min_pair, min_wall)
                        
                        if trial_r > best_r + 1e-12:
                            best_r = trial_r
                            best_pos = trial.copy()
            
            if np.any(best_pos != center):
                best_centers[i] = best_pos
                improved = True
        
        if not improved:
            break
    
    best_radius = _compute_radius(best_centers)
    return best_centers, best_radius

def _polish(centers, deadline):
    """
    Final polish pass with tighter tolerances and higher beta.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Try a few high-beta optimizations
    for beta in [500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0]:
        if time.time() > deadline - 8.0:
            break
            
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-13, 'maxls': 200}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
    
    # Local search refinement
    best_centers, best_radius = _local_search_refine(best_centers, deadline)
    
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Try multiple initializations
    initializations = [
        ("grid", _build_grid_init),
        ("bcc", _build_bcc_init),
        ("halton", _build_halton_init),
        ("sobol", _build_sobol_init),
        ("fcc", _build_fcc_init),
    ]
    
    # First, find the best initialization
    best_centers = None
    best_radius = 0.0
    
    for name, init_func in initializations:
        if time.time() > deadline - 100.0:
            break
        
        centers = init_func()
        r = _compute_radius(centers)
        
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
    
    # Run optimization on the best initialization
    best_centers, best_radius = _optimize_from_initial(best_centers, deadline, beta_start=10.0, beta_end=10000.0)
    
    # Final polish
    best_centers, best_radius = _polish(best_centers, deadline)
    
    return best_centers

# EVOLVE-BLOCK-END
