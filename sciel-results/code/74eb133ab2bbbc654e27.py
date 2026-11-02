import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

N = 100
_DEADLINE = None

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 145.0
    return _DEADLINE

def _time_left():
    return _get_deadline() - time.time()

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

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
        if len(centers) < N:
            rng = np.random.default_rng(seed)
            extra = rng.uniform(-0.45, 0.45, (N - len(centers), 3))
            centers = np.vstack([centers, extra])
        else:
            centers = centers[:N]

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
    
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_bcc_init():
    """
    Build a BCC (body-centered cubic) lattice initialization.
    """
    # Use a 5x5x2 grid of BCC cells
    nx, ny, nz = 5, 5, 2
    n_cells = nx * ny * nz  # 50
    
    points = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cx = -0.5 + (i + 0.5) * (1.0 / nx)
                cy = -0.5 + (j + 0.5) * (1.0 / ny)
                cz = -0.5 + (k + 0.5) * (1.0 / nz)
                points.append([cx, cy, cz])
                # Body center offset
                points.append([cx + 0.5/nx, cy + 0.5/ny, cz + 0.5/nz])
    
    centers = np.array(points[:N])
    
    # Add slight noise
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _soft_min_objective(centers_flat, beta, iu):
    """
    Objective function for L-BFGS-B.
    Minimizes the soft-min approximation of the negative radius.
    r = min(min_wall, min_pair/2)
    We want to maximize r, so we minimize -r.
    soft_max(-r) = soft_min(r_terms) = -(1/beta) log(sum(exp(-beta * term)))
    Let f(x) = -soft_min(r_terms) = (1/beta) log(sum(exp(-beta * term)))
    """
    centers = centers_flat.reshape(-1, 3)
    
    # 1. Wall distances: r_wall_i = 0.5 - |c_ik|
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    # 2. Pairwise distances / 2: r_pair_ij = |c_i - c_j| / 2
    dists = cdist(centers, centers, metric='euclidean')
    pair_vals = dists / 2.0
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Calculate f(x) = (1/beta) * log(sum(exp(-beta * x)))
    # Note: exp(-beta * x) where x are the radii terms.
    # To maximize min(x), we minimize -min(x) approx -soft_min(x).
    # soft_min(x) = -1/beta log(sum(exp(-beta x)))
    # So objective = -soft_min(x) = 1/beta log(sum(exp(-beta x)))
    
    neg_beta_x = -beta * all_vals
    
    # Numerical stability
    max_val = np.max(neg_beta_x)
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    
    if sum_exp == 0:
        return 0.0
        
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _optimize_lbfgs(centers, deadline, iu, time_budget=10.0):
    """
    Run L-BFGS-B optimization from initial centers.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    start_t = time.time()
    remaining = time_budget
    
    # Adaptive beta schedule
    beta_start = 50.0
    beta_end = 2000.0
    n_beta_steps = 5
    
    if remaining > 5.0:
        for step in range(n_beta_steps):
            if _time_left() < 2.0 or (time.time() - start_t) > remaining - 2.0:
                break
                
            beta = beta_start * (beta_end / beta_start) ** (step / max(n_beta_steps - 1, 1))
            
            x0 = best_centers.flatten()
            
            try:
                max_iter = 200 if step < 2 else 100
                res = minimize(
                    _soft_min_objective,
                    x0,
                    args=(beta, iu),
                    method='L-BFGS-B',
                    bounds=[(-0.4999, 0.4999)] * (N * 3),
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 30}
                )
                
                new_centers = res.x.reshape(-1, 3)
                new_centers = np.clip(new_centers, -0.4999, 0.4999)
                
                # Verify improvement
                new_radius = _compute_radius(new_centers)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = new_centers
                    
            except Exception:
                continue
                
    return best_centers, best_radius

def _polish_repulsion(centers, deadline, iu):
    """
    Local search using repulsion forces with KD-tree acceleration.
    """
    centers = centers.copy()
    r_current = _compute_radius(centers)
    best_r = r_current
    best_centers = centers.copy()
    
    step_size = 0.005
    decay = 0.98
    min_step = 1e-7
    margin = 1.05
    
    no_improve_count = 0
    max_no_improve = 200
    
    while _time_left() > 1.0 and step_size > min_step and no_improve_count < max_no_improve:
        r = _compute_radius(centers)
        if r > best_r + 1e-12:
            best_r = r
            best_centers = centers.copy()
            no_improve_count = 0
            r_current = r # Update current radius for force calculation
        else:
            no_improve_count += 1
            
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Wall forces
        for k in range(3):
            dists = 0.5 - np.abs(centers[:, k])
            limit = max(r_current * margin, 0.01)
            mask = dists < limit
            if np.any(mask):
                dirs = -np.sign(centers[mask, k])
                dirs[dirs == 0] = 1.0
                strengths = (limit - dists[mask]) / limit
                forces[mask, k] += dirs * strengths
                
        # Pairwise forces using KD-tree for efficiency
        tree = cKDTree(centers)
        search_radius = 2 * max(r_current * margin, 0.01)
        
        # Query neighbors
        neighbor_pairs = tree.query_pairs(r=search_radius)
        
        for i, j in neighbor_pairs:
            if i == j:
                continue
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-9 and dist < search_radius:
                strength = (search_radius - dist) / search_radius
                unit_diff = diff / dist
                forces[i] += strength * unit_diff
                forces[j] -= strength * unit_diff
        
        # Normalize forces
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms[norms == 0] = 1
        forces = forces / norms
        
        centers += step_size * forces
        centers = np.clip(centers, -0.4999, 0.4999)
        
        step_size *= decay
        
    return best_centers, best_r

def construct():
    start_time = time.time()
    deadline = _get_deadline()
    
    best_centers = None
    best_radius = 0.0
    
    # Precompute unique indices for upper triangle (excluding diagonal)
    iu = np.triu_indices(N, k=1)
    
    # Initializations
    initializations = []
    
    # Grids
    grid_configs = [
        (4, 5, 5),
        (5, 4, 5),
        (5, 5, 4),
        (10, 5, 2),
        (5, 10, 2),
        (5, 2, 10),
    ]
    
    for nx, ny, nz in grid_configs:
        for noise in [0.0, 0.01, 0.02]:
            centers = _build_grid_init(nx, ny, nz, noise, seed=nx*100+ny*10+nz)
            initializations.append(centers)
            
    # Halton
    initializations.append(_build_halton_init())
    
    # BCC-inspired
    initializations.append(_build_bcc_init())
    
    # Random
    rng = np.random.default_rng(123)
    for _ in range(5):
        centers = rng.uniform(-0.45, 0.45, (N, 3))
        initializations.append(centers)
        
    # Optimize each initialization
    n_inits = len(initializations)
    total_time = _get_deadline() - time.time() - 5.0
    if total_time < 1.0:
        return _build_grid_init(5, 5, 4, 0.0)
        
    time_per_init = total_time / n_inits
    
    for centers in initializations:
        if time.time() > deadline - 5.0:
            break
            
        # Phase 1: L-BFGS-B optimization
        lbfgs_deadline = min(time.time() + time_per_init * 0.6, deadline - 5.0)
        lbfgs_time_budget = max(0.1, lbfgs_deadline - time.time())
        lbfgs_centers, lbfgs_radius = _optimize_lbfgs(centers, lbfgs_deadline, iu, lbfgs_time_budget)
        
        if lbfgs_radius > best_radius:
            best_radius = lbfgs_radius
            best_centers = lbfgs_centers
            
        # Phase 2: Polish
        if time.time() < deadline - 5.0:
            polish_deadline = min(time.time() + time_per_init * 0.4, deadline - 5.0)
            polish_centers, polish_radius = _polish_repulsion(lbfgs_centers, polish_deadline, iu)
            
            if polish_radius > best_radius:
                best_radius = polish_radius
                best_centers = polish_centers

    # Final polish on best result
    if best_centers is not None and time.time() < deadline - 1.0:
        final_centers, final_radius = _polish_repulsion(best_centers, deadline - 0.5, iu)
        if final_radius > best_radius:
            best_radius = final_radius
            best_centers = final_centers

    if best_centers is None:
        best_centers = _build_grid_init(5, 5, 4, 0.0)
        
    return best_centers
