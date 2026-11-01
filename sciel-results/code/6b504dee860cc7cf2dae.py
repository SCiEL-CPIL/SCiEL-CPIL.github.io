"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    if not np.all(np.isfinite(centers)):
        return 0.0
    if centers.shape != (N, 3):
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
    The radius is min(wall_dists, pair_dists/2).
    We minimize -radius, so we maximize the soft-min of the constraints.
    Actually, we minimize the soft-min of the *constraints* directly?
    No, we want to maximize r. 
    r = min( constraints ).
    We want to maximize min(constraints).
    Equivalent to minimizing -min(constraints).
    Using soft-min: S_beta(x) = 1/beta * log(sum(exp(-beta * x))).
    As beta -> inf, S_beta(x) -> min(x).
    So we minimize S_beta(constraints).
    """
    centers = centers_flat.reshape(-1, 3)
    
    # 1. Wall distances: 0.5 - |c_i|
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    # 2. Pairwise distances / 2
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    # Only take upper triangle to avoid duplicates and diagonal
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    # To avoid overflow, subtract max
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    
    # If sum_exp is 0 (unlikely with finite values), return large penalty
    if sum_exp == 0:
        return 1e10
        
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _build_bcc_init(n=100, noise_scale=0.005):
    """
    Build a BCC (Body-Centered Cubic) inspired initialization.
    """
    # Approximate grid size for 100 points
    # 10x10x1 BCC layer gives 200 points, we pick 100.
    # Or 5x5x4 BCC gives 2*5*5*4 = 200.
    nx, ny, nz = 5, 5, 4
    margin = 0.02
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    
    all_centers = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Corner points
                all_centers.append([xs[i], ys[j], zs[k]])
                # Body center
                if i < nx-1 and j < ny-1 and k < nz-1:
                    all_centers.append([(xs[i]+xs[i+1])/2, (ys[j]+ys[j+1])/2, (zs[k]+zs[k+1])/2])
    
    all_centers = np.array(all_centers)
    rng = np.random.default_rng(42)
    
    if len(all_centers) >= N:
        idx = rng.choice(len(all_centers), N, replace=False)
        centers = all_centers[idx]
    else:
        centers = all_centers
        # Pad if needed (should not happen with 200 pts)
        while len(centers) < N:
            centers = np.vstack([centers, centers[0]])
    
    centers += rng.normal(0, noise_scale, centers.shape)
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
    for i in range(1, n+1):
        centers[i-1, 0] = -0.5 + halton(i, 2)
        centers[i-1, 1] = -0.5 + halton(i, 3)
        centers[i-1, 2] = -0.5 + halton(i, 5)
    
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _optimize_from_initial(centers, deadline, beta_start=50.0, beta_end=5000.0, n_beta_steps=6):
    """
    Run L-BFGS-B optimization with increasing beta from initial centers.
    Returns (best_centers, best_radius).
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    iu = np.triu_indices(N, k=1)
    betas = np.geomspace(beta_start, beta_end, n_beta_steps)
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 5.0:
            break
    
    return best_centers, best_radius

def _polish(centers, deadline):
    """
    Final high-precision polish pass using very high beta.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    iu = np.triu_indices(N, k=1)
    
    # Use very high beta for sharp minimum
    beta = 20000.0
    
    try:
        res = minimize(
            _soft_min_objective,
            best_centers.flatten(),
            args=(beta, iu),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
        )
        
        new_centers = res.x.reshape(-1, 3)
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        r = _compute_radius(new_centers)
        if r > best_radius:
            best_radius = r
            best_centers = new_centers.copy()
    except Exception:
        pass
        
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s
    
    # Initialize with BCC
    init_centers = _build_bcc_init()
    
    best_centers = init_centers.copy()
    best_r = _compute_radius(best_centers)
    
    # Run optimization from BCC
    c_opt, r_opt = _optimize_from_initial(init_centers, deadline)
    if r_opt > best_r:
        best_r = r_opt
        best_centers = c_opt.copy()
    
    # If time permits, try Halton initialization
    if time.time() < deadline - 30.0:
        init_halton = _build_halton_init()
        c_opt_h, r_opt_h = _optimize_from_initial(init_halton, deadline)
        if r_opt_h > best_r:
            best_r = r_opt_h
            best_centers = c_opt_h.copy()
    
    # Final polish
    if time.time() < deadline - 10.0:
        c_final, r_final = _polish(best_centers, deadline)
        if r_final > best_r:
            best_r = r_final
            best_centers = c_final.copy()
            
    # Final validation
    final_r = _compute_radius(best_centers)
    if final_r > best_r:
        best_r = final_r
        
    return best_centers

# EVOLVE-BLOCK-END
