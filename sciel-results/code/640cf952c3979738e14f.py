import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

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
    
    # Pairwise distances using cdist for efficiency
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

def _build_bcc_init():
    """
    Build a BCC (body-centered cubic) lattice initialization.
    """
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
                points.append([cx + 0.5/nx, cy + 0.5/ny, cz + 0.5/nz])
    
    centers = np.array(points[:N])
    
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.005, centers.shape)
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

def _build_sobol_init(seed=42):
    """
    Generate N points using a Sobol sequence (low discrepancy).
    """
    from scipy.stats.qmc import Sobol
    sampler = Sobol(d=3, seed=seed)
    points = sampler.random(N)
    centers = points - 0.5
    
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _polish_annealed(centers, deadline, iu):
    """
    Force-directed polish with simulated annealing characteristics.
    Uses repulsion forces from walls and other spheres.
    Includes reheating to escape local minima.
    """
    centers = centers.copy()
    r_current = _compute_radius(centers)
    best_r = r_current
    best_centers = centers.copy()
    
    # Annealing parameters
    step_size = 0.005
    min_step = 1e-7
    decay = 0.98
    reheat_temp = 0.003
    reheat_prob = 0.05 # Probability of reheating if stuck
    
    no_improve_count = 0
    max_no_improve = 100
    
    rng = np.random.default_rng(12345)
    
    while _time_left() > 1.0 and step_size > min_step:
        r = _compute_radius(centers)
        if r > best_r + 1e-12:
            best_r = r
            best_centers = centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Reheat if stuck
        if no_improve_count > max_no_improve and rng.random() < reheat_prob:
            step_size = max(step_size, reheat_temp)
            no_improve_count = 0
            # Small random perturbation to break symmetry
            centers += rng.normal(0, 0.001, centers.shape)
            centers = np.clip(centers, -0.4999, 0.4999)

        # Compute forces
        forces = np.zeros_like(centers)
        
        # Wall forces
        # We want to push spheres away from walls if they are too close
        # Distance to wall is 0.5 - |c_k|
        # If dist < r_current, it's a violation.
        # We apply force proportional to violation.
        for k in range(3):
            dists = 0.5 - np.abs(centers[:, k])
            # Use a margin to keep them slightly away
            margin = 1.01
            limit = r_current * margin
            mask = dists < limit
            if np.any(mask):
                # Direction: away from wall. 
                # If c_k > 0, wall is at 0.5, so push negative? No, push towards center.
                # Wall at 0.5: if c_k is close to 0.5, push towards 0 (negative direction).
                # Wall at -0.5: if c_k is close to -0.5, push towards 0 (positive direction).
                # So direction is -sign(c_k).
                dirs = -np.sign(centers[mask, k])
                dirs[dirs == 0] = 1.0
                # Strength: linear with penetration
                strengths = (limit - dists[mask]) / limit
                forces[mask, k] += dirs * strengths
                
        # Pairwise forces
        # Use cdist for distances
        dists = cdist(centers, centers, metric='euclidean')
        # Mask out diagonal
        np.fill_diagonal(dists, np.inf)
        
        limit = 2 * r_current * 1.01
        # Find pairs within limit
        # For N=100, 100x100 matrix is fine.
        mask = dists < limit
        
        if np.any(mask):
            # Compute forces vectorized
            # diff[i,j] = centers[i] - centers[j]
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            
            # Strengths: (limit - dist) / limit
            strengths = np.zeros_like(dists)
            strengths[mask] = (limit - dists[mask]) / limit
            # Zero out diagonal
            np.fill_diagonal(strengths, 0)
            
            # Force on i from j is proportional to (c_i - c_j) * strength
            # Sum over j
            pair_forces = np.sum(diff * strengths[:, :, np.newaxis], axis=1)
            forces += pair_forces
        
        # Normalize forces to keep step size meaningful
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms[norms == 0] = 1
        forces = forces / norms
        
        # Update positions
        centers += step_size * forces
        centers = np.clip(centers, -0.4999, 0.4999)
        
        # Decay step size
        step_size *= decay
        
    return best_centers, best_r

def _optimize_from_initial(centers, deadline, iu, time_budget=20.0):
    """
    Run optimization from initial centers.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    start_t = time.time()
    remaining = time_budget
    
    # Phase 1: Soft-min optimization (coarse) to fix major overlaps
    # This is fast and helps get into a good basin for the force-directed method
    if remaining > 5.0:
        beta_start = 50.0
        beta_end = 500.0
        n_beta_steps = 3
        
        for step in range(n_beta_steps):
            if _time_left() < 3.0 or (time.time() - start_t) > remaining - 3.0:
                break
                
            beta = beta_start * (beta_end / beta_start) ** (step / (n_beta_steps - 1))
            
            x0 = best_centers.flatten()
            
            try:
                max_iter = 50
                res = minimize(
                    _soft_min_objective,
                    x0,
                    args=(beta, iu),
                    method='L-BFGS-B',
                    bounds=[(-0.4999, 0.4999)] * (N * 3),
                    options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 10}
                )
                
                new_centers = res.x.reshape(-1, 3)
                new_centers = np.clip(new_centers, -0.4999, 0.4999)
                
                r = _compute_radius(new_centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = new_centers
                    
            except Exception:
                pass

    # Phase 2: Force-directed annealing polish
    # This is the main workhorse for maximizing radius
    if _time_left() > 3.0 and (time.time() - start_t) < remaining - 2.0:
        polish_deadline = min(deadline, start_t + remaining - 2.0)
        polish_centers, polish_radius = _polish_annealed(best_centers, polish_deadline, iu)
        if polish_radius > best_radius:
            best_radius = polish_radius
            best_centers = polish_centers
            
    return best_centers, best_radius

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 145.0
    
    iu = np.triu_indices(N, k=1)
    
    best_centers = None
    best_radius = 0.0
    
    def try_init(centers, time_budget=18.0):
        nonlocal best_centers, best_radius
        r_init = _compute_radius(centers)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = centers.copy()
        
        if _time_left() > time_budget + 3.0:
            opt_centers, opt_radius = _optimize_from_initial(centers, _DEADLINE, iu, time_budget)
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers
        else:
            if _time_left() > 4.0:
                polish_centers, polish_radius = _polish_annealed(centers, _DEADLINE, iu)
                if polish_radius > best_radius:
                    best_radius = polish_radius
                    best_centers = polish_centers
    
    # Initializations
    # Prioritize diverse, good starting points
    
    # 1. BCC lattice (often good for packing)
    if _time_left() > 30.0:
        centers = _build_bcc_init()
        try_init(centers, time_budget=28.0)
    
    # 2. Grid 5x5x4
    if _time_left() > 25.0:
        centers = _build_grid_init(5, 5, 4, 0.0, seed=111)
        try_init(centers, time_budget=22.0)
        
    # 3. Grid 4x5x5
    if _time_left() > 20.0:
        centers = _build_grid_init(4, 5, 5, 0.0, seed=222)
        try_init(centers, time_budget=18.0)
        
    # 4. Halton sequence
    if _time_left() > 15.0:
        centers = _build_halton_init()
        try_init(centers, time_budget=15.0)
        
    # 5. Sobol sequence
    if _time_left() > 15.0:
        centers = _build_sobol_init()
        try_init(centers, time_budget=15.0)

    # 6. Random initializations with different seeds
    rng = np.random.default_rng(999)
    for seed in [999, 1000]:
        if _time_left() < 12.0:
            break
        centers = rng.uniform(-0.45, 0.45, (N, 3))
        try_init(centers, time_budget=10.0)
    
    # Final polish with remaining time on the best candidate
    if best_centers is not None and _time_left() > 5.0:
        polish_centers, polish_radius = _polish_annealed(best_centers, _DEADLINE, iu)
        if polish_radius > best_radius:
            best_radius = polish_radius
            best_centers = polish_centers
        
    if best_centers is None:
        best_centers = _build_grid_init(5, 5, 4, 0.0)
        
    return best_centers

