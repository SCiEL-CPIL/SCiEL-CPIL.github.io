import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

N = 100

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5 + 1e-12):
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
    pair_vals = dists / 2.0
    # Exclude diagonal
    pair_vals_unique = pair_vals[np.triu_indices(N, k=1)]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    # To avoid overflow, subtract max
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    
    # If sum_exp is 0, it means all exp terms underflowed, but max_val is finite.
    # Actually, if beta is large, exp(-beta * x) for x > max_x/beta might be 0.
    # The formula is (max_val + log(sum_exp)) / beta
    # If sum_exp is 0, log(0) is -inf. But max_val is the largest term, so sum_exp >= 1.
    # Wait, exp(neg_beta_x - max_val) <= 1. The max term is exp(0)=1. So sum_exp >= 1.
    
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
        pass

    if noise_scale > 0:
        rng = np.random.default_rng(seed)
        centers += rng.normal(0, noise_scale, centers.shape)
    
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _build_halton_init(seed=42):
    """
    Generate N points using a Halton sequence (low discrepancy).
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    bases = [2, 3, 5]
    points = []
    for i in range(1, N + 1):
        pt = [halton(i, b) for b in bases]
        pt_mapped = [p - 0.5 for p in pt]
        points.append(pt_mapped)
        
    centers = np.array(points)
    
    # Add slight noise to break symmetry
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _polish_repulsion(centers, deadline, r_current):
    """
    Local search using repulsion forces.
    Uses KDTree for efficient neighbor finding.
    """
    centers = centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    step_size = 0.005
    decay = 0.95
    min_step = 1e-6
    
    while time.time() < deadline - 0.5 and step_size > min_step:
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Wall forces
        for k in range(3):
            dists = 0.5 - np.abs(centers[:, k])
            # If a sphere is too close to the wall, push it in
            # Limit is effectively r_current, but we want to maximize min dist.
            # If dist < r_current, it's invalid. We want dist >= r_current.
            # Force should be proportional to how much it violates or is close.
            # Let's use a soft barrier.
            limit = r_current
            mask = dists < limit
            if np.any(mask):
                dirs = -np.sign(centers[mask, k])
                # If center is 0, sign is 0. But dist=0.5 > limit (unless limit > 0.5).
                # So dirs is safe.
                strengths = (limit - dists[mask]) / limit
                forces[mask, k] += dirs * strengths
        
        # Pairwise forces
        # Build KDTree for current positions
        tree = cKDTree(centers)
        # Query pairs within 2 * r_current
        pairs = tree.query_pairs(2 * r_current)
        
        for i, j in pairs:
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-9:
                # Random small push
                forces[i] += np.random.normal(0, 1e-3, 3)
                forces[j] -= np.random.normal(0, 1e-3, 3)
                continue
            
            # We want dist >= 2 * r_current
            # If dist < 2 * r_current, push apart
            limit = 2 * r_current
            if dist < limit:
                strength = (limit - dist) / limit
                unit_vec = diff / dist
                forces[i] += strength * unit_vec
                forces[j] -= strength * unit_vec
        
        # Normalize forces to prevent large jumps
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms[norms == 0] = 1
        forces = forces / norms
        
        centers += step_size * forces
        centers = np.clip(centers, -0.5, 0.5)
        
        # Update best
        r_new = _compute_radius(centers)
        if r_new > best_r:
            best_r = r_new
            best_centers = centers.copy()
        
        step_size *= decay
        
    return best_centers, best_r

def _optimize_from_initial(centers, deadline):
    """
    Run optimization from initial centers.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Phase 1: Soft-min optimization
    # We use a schedule of increasing beta
    beta_schedule = [50, 100, 200, 500, 1000, 2000]
    
    for beta in beta_schedule:
        if time.time() > deadline - 10.0:
            break
            
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 20}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
                
        except Exception:
            pass
            
    # Phase 2: Polish with repulsion
    if time.time() < deadline - 1.0:
        polish_centers, polish_radius = _polish_repulsion(best_centers, deadline, best_radius)
        if polish_radius > best_radius:
            best_radius = polish_radius
            best_centers = polish_centers
            
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
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
    
    for nx, ny, nz in grid_configs:
        if time.time() > deadline - 15.0:
            break
        
        # Try with and without noise
        for noise in [0.0, 0.01, 0.02]:
            if time.time() > deadline - 5.0:
                break
            
            centers = _build_grid_init(nx, ny, nz, noise, seed=nx*100+ny*10+nz)
            r_init = _compute_radius(centers)
            
            if r_init > best_radius:
                best_radius = r_init
                best_centers = centers.copy()
            
            opt_centers, opt_radius = _optimize_from_initial(centers, deadline)
            
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers

    # 2. Halton
    if time.time() < deadline - 15.0:
        centers = _build_halton_init()
        r_init = _compute_radius(centers)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = centers.copy()
            
        opt_centers, opt_radius = _optimize_from_initial(centers, deadline)
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers
            
    # 3. Random
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
                
            opt_centers, opt_radius = _optimize_from_initial(centers, deadline)
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers

    if best_centers is None:
        best_centers = _build_grid_init(5, 5, 4, 0.0)
        
    return best_centers
