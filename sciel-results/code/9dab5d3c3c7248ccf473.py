import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
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
    pair_vals = dists / 2.0
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _build_grid_init(nx, ny, nz, noise_scale=0.0, seed=42):
    """Build a grid initialization."""
    if nx == 1: xs = np.array([0.0])
    else: xs = np.linspace(-0.5, 0.5, nx)
    
    if ny == 1: ys = np.array([0.0])
    else: ys = np.linspace(-0.5, 0.5, ny)
    
    if nz == 1: zs = np.array([0.0])
    else: zs = np.linspace(-0.5, 0.5, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    if len(centers) != N:
        # This function is expected to be called with correct dimensions
        # But just in case, we truncate or pad? No, better to fail fast or handle in caller.
        pass

    if noise_scale > 0:
        rng = np.random.default_rng(seed)
        centers += rng.normal(0, noise_scale, centers.shape)
    
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _build_halton_init(seed=42):
    """
    Generate N points using a Halton sequence (low discrepancy).
    This provides a more uniform distribution than a grid.
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    # We need 3 bases, e.g., 2, 3, 5
    bases = [2, 3, 5]
    points = []
    for i in range(1, N + 1):
        pt = [halton(i, b) for b in bases]
        # Map [0,1] to [-0.5, 0.5]
        pt_mapped = [p - 0.5 for p in pt]
        points.append(pt_mapped)
        
    centers = np.array(points)
    
    # Add slight noise to break symmetry
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _polish_repulsion(centers, deadline, iu):
    """
    Local search using repulsion forces.
    """
    centers = centers.copy()
    r_current = _compute_radius(centers)
    best_r = r_current
    best_centers = centers.copy()
    
    step_size = 0.005
    decay = 0.98
    min_step = 1e-5
    margin = 1.02 # Consider pairs within 2% of current radius limit
    
    # We can also try to increase the margin slightly at the start to smooth out
    # but usually starting tight is better for polishing.
    
    while time.time() < deadline - 0.5 and step_size > min_step:
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
        
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Wall forces
        for k in range(3):
            dists = 0.5 - np.abs(centers[:, k])
            limit = r_current * margin
            mask = dists < limit
            if np.any(mask):
                # Direction is towards 0
                dirs = -np.sign(centers[mask, k])
                # If center is 0, sign is 0. But if dist < limit, center is close to wall?
                # dist = 0.5 - |c|. If dist is small, |c| is close to 0.5.
                # So c is close to 0.5 or -0.5. Sign is non-zero.
                # If c is exactly 0, dist is 0.5. If limit < 0.5, mask is false.
                # So dirs is safe.
                strengths = (limit - dists[mask]) / limit
                forces[mask, k] += dirs * strengths
                
        # Pairwise forces
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        limit = 2 * r_current * margin
        mask = (dists < limit) & (dists > 1e-9)
        
        strengths = np.zeros_like(dists)
        strengths[mask] = (limit - dists[mask]) / limit
        
        pair_forces = np.sum(diff * strengths[:, :, np.newaxis], axis=1)
        
        total_forces = forces + pair_forces
        
        # Normalize forces to prevent large jumps
        norms = np.linalg.norm(total_forces, axis=1, keepdims=True)
        norms[norms == 0] = 1
        total_forces = total_forces / norms
        
        centers += step_size * total_forces
        centers = np.clip(centers, -0.5, 0.5)
        
        step_size *= decay
        
    return best_centers, best_r

def _optimize_from_initial(centers, deadline, iu):
    """
    Run optimization from initial centers.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Phase 1: Soft-min optimization
    beta_start = 50.0
    beta_end = 2000.0
    n_beta_steps = 8
    
    if time.time() > deadline - 10.0:
        # Not enough time for full optimization, go straight to polish?
        # Or just return initial?
        # Let's try a quick polish
        return _polish_repulsion(best_centers, deadline, iu)

    for step in range(n_beta_steps):
        if time.time() > deadline - 5.0:
            break
            
        if n_beta_steps == 1:
            beta = beta_start
        else:
            beta = beta_start * (beta_end / beta_start) ** (step / (n_beta_steps - 1))
        
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 20}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
            else:
                # If no improvement, stick with best
                pass
                
        except Exception:
            pass
            
    # Phase 2: Polish with repulsion
    if time.time() < deadline - 1.0:
        polish_centers, polish_radius = _polish_repulsion(best_centers, deadline, iu)
        if polish_radius > best_radius:
            best_radius = polish_radius
            best_centers = polish_centers
            
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    iu = np.triu_indices(N, k=1)
    
    best_centers = None
    best_radius = 0.0
    
    # Initializations
    # 1. Grids
    grid_configs = [
        (4, 5, 5),
        (5, 4, 5),
        (5, 5, 4),
        (10, 5, 2),
        (5, 10, 2),
        (5, 2, 10),
    ]
    
    # 2. Halton
    # 3. BCC (optional if time allows)
    
    for nx, ny, nz in grid_configs:
        if time.time() > deadline - 15.0:
            break
        
        for noise in [0.0, 0.01, 0.02]:
            if time.time() > deadline - 5.0:
                break
            
            centers = _build_grid_init(nx, ny, nz, noise, seed=nx*100+ny*10+nz)
            r_init = _compute_radius(centers)
            
            if r_init > best_radius:
                best_radius = r_init
                best_centers = centers.copy()
            
            opt_centers, opt_radius = _optimize_from_initial(centers, deadline, iu)
            
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers

    # Halton
    if time.time() < deadline - 15.0:
        centers = _build_halton_init()
        r_init = _compute_radius(centers)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = centers.copy()
            
        opt_centers, opt_radius = _optimize_from_initial(centers, deadline, iu)
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers
            
    # Random
    if time.time() < deadline - 15.0:
        rng = np.random.default_rng(123)
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            centers = rng.uniform(-0.45, 0.45, (N, 3))
            r_init = _compute_radius(centers)
            if r_init > best_radius:
                best_radius = r_init
                best_centers = centers.copy()
                
            opt_centers, opt_radius = _optimize_from_initial(centers, deadline, iu)
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers

    if best_centers is None:
        best_centers = _build_grid_init(5, 5, 4, 0.0)
        
    return best_centers
