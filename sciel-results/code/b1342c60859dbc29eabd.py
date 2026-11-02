import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

N = 100
_start_time = time.time()
DEADLINE = 145.0  # Safety margin before 150s kill

def _time_left():
    return DEADLINE - (time.time() - _start_time)

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
    
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _soft_min_grad(centers_flat, beta):
    """
    Analytical gradient of the soft-min objective.
    """
    centers = centers_flat.reshape(-1, 3)
    
    # 1. Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    # 2. Pairwise distances / 2
    dists = cdist(centers, centers, metric='euclidean')
    pair_vals = dists / 2.0
    pair_vals_unique = pair_vals[np.triu_indices(N, k=1)]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Soft-min value
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    # Weights: w_i = exp(-beta * x_i) / sum(exp(-beta * x_j))
    # This is the probability distribution over the "active" constraints
    log_w = neg_beta_x - (max_val + np.log(sum_exp))
    w = np.exp(log_w)
    
    grad = np.zeros_like(centers_flat)
    
    # Wall gradients
    # d/dx_i (0.5 - |x_ik|) = -sign(x_ik)
    wall_grad = np.zeros_like(centers)
    signs = np.sign(centers)
    signs[signs == 0] = 1.0  # Handle zero case arbitrarily
    wall_grad = -signs * w[:N*3].reshape(N, 3)
    grad[:N*3] = wall_grad.flatten()
    
    # Pair gradients
    # d/dx_i (||x_i - x_j|| / 2) = (x_i - x_j) / (2 * ||x_i - x_j||)
    # d/dx_j (||x_i - x_j|| / 2) = -(x_i - x_j) / (2 * ||x_i - x_j||)
    pair_w = w[N*3:]
    pair_grad = np.zeros_like(centers)
    
    # Vectorized computation of pair gradients
    # dists is (N, N)
    # For each pair (i, j) with i < j, weight is pair_w[k]
    # We need to accumulate gradients
    
    # Efficiently:
    # grad_i += sum_j w_ij * (x_i - x_j) / (2 * d_ij)
    # grad_j -= sum_i w_ij * (x_i - x_j) / (2 * d_ij)
    
    # d/dx_i (d_ij/2) = (x_i - x_j) / (2 * d_ij)
    # Let's compute the force matrix F where F[i,j] = w_ij * (x_i - x_j) / (2 * d_ij)
    # Note: d_ij = ||x_i - x_j||. So (x_i - x_j) / d_ij is the unit vector.
    # So term is 0.5 * unit_vec.
    
    # Avoid division by zero
    dists_safe = np.where(dists < 1e-10, 1e-10, dists)
    unit_vecs = (centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) / dists_safe[:, :, np.newaxis]
    
    # Weight matrix for pairs (i, j)
    # We only have weights for i < j in pair_w.
    # Let's build the full weight matrix W_pair (N, N)
    W_pair = np.zeros((N, N))
    iu = np.triu_indices(N, k=1)
    W_pair[iu] = pair_w
    W_pair[iu[1], iu[0]] = pair_w  # Symmetric
    
    # Force on i from j: W_pair[i, j] * 0.5 * unit_vec(i, j)
    # unit_vec(i, j) = (x_i - x_j) / d_ij
    # So Force_i = sum_j W_pair[i, j] * 0.5 * (x_i - x_j) / d_ij
    # This is equivalent to: 0.5 * sum_j W_pair[i, j] * unit_vec(i, j)
    
    # We can compute this via matrix multiplication if we reshape?
    # Or just loop over i? N=100, so 100 iterations is fine.
    # Or vectorize:
    # F[i, k] = sum_j W_pair[i, j] * 0.5 * (x_i[k] - x_j[k]) / d_ij
    
    # Let's use einsum or broadcasting.
    # term[i, j, k] = W_pair[i, j] * 0.5 * (x_i[k] - x_j[k]) / d_ij[i, j]
    # grad[i, k] = sum_j term[i, j, k]
    
    # This might be memory heavy (100*100*3 = 30000 floats, fine).
    terms = W_pair[:, :, np.newaxis] * 0.5 * unit_vecs
    pair_grad = np.sum(terms, axis=1)
    
    grad[:N*3] += pair_grad.flatten()
    
    return grad

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
    
    # Add slight noise to break symmetry
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_bcc_init():
    """
    Build a Body-Centered Cubic style initialization.
    """
    nx, ny, nz = 5, 5, 2
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

def _build_sobol_init(seed=42):
    """
    Build a Sobol sequence initialization.
    """
    from scipy.stats import qmc
    sobol = qmc.Sobol(d=3, scramble=True, seed=seed)
    points = sobol.random(N)
    centers = points - 0.5
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.005, centers.shape)
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
    no_improve_count = 0
    max_no_improve = 150
    
    while _time_left() > 1.0 and step_size > min_step and no_improve_count < max_no_improve:
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Wall forces
        for k in range(3):
            dists = 0.5 - np.abs(centers[:, k])
            # If a sphere is too close to the wall, push it in
            limit = r_current
            mask = dists < limit
            if np.any(mask):
                dirs = -np.sign(centers[mask, k])
                dirs[dirs == 0] = 1.0
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
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        step_size *= decay
        
    return best_centers, best_r

def _simulated_annealing(centers, deadline, iu):
    """
    Simulated annealing to escape local minima.
    """
    centers = centers.copy()
    best_centers = centers.copy()
    best_r = _compute_radius(best_centers)
    current_r = best_r
    T_start = 0.05
    T_end = 0.0001
    alpha = 0.95
    T = T_start
    rng = np.random.default_rng(42)
    
    while _time_left() > 2.0 and T > T_end:
        step_size = T * 0.1
        noise = rng.normal(0, step_size, centers.shape)
        new_centers = centers + noise
        new_centers = np.clip(new_centers, -0.4999, 0.4999)
        
        new_r = _compute_radius(new_centers)
        
        if new_r >= current_r:
            accept = True
        else:
            delta = current_r - new_r
            p = np.exp(-delta / T)
            accept = rng.random() < p
        
        if accept:
            centers = new_centers
            current_r = new_r
            if current_r > best_r + 1e-12:
                best_r = current_r
                best_centers = centers.copy()
        
        T *= alpha
        
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
        if _time_left() < 10.0:
            break
            
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta,),
                jac=_soft_min_grad,
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
    if _time_left() > 1.0:
        polish_centers, polish_radius = _polish_repulsion(best_centers, deadline, best_radius)
        if polish_radius > best_radius:
            best_radius = polish_radius
            best_centers = polish_centers
            
    return best_centers, best_radius

def construct():
    start_time = time.time()
    
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
        if _time_left() < 15.0:
            break
        
        # Try with and without noise
        for noise in [0.0, 0.01, 0.02]:
            if _time_left() < 5.0:
                break
            
            centers = _build_grid_init(nx, ny, nz, noise, seed=nx*100+ny*10+nz)
            r_init = _compute_radius(centers)
            
            if r_init > best_radius:
                best_radius = r_init
                best_centers = centers.copy()
            
            opt_centers, opt_radius = _optimize_from_initial(centers, start_time + DEADLINE)
            
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers

    # 2. Halton
    if _time_left() > 15.0:
        centers = _build_halton_init()
        r_init = _compute_radius(centers)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = centers.copy()
            
        opt_centers, opt_radius = _optimize_from_initial(centers, start_time + DEADLINE)
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers
            
    # 3. BCC
    if _time_left() > 15.0:
        centers = _build_bcc_init()
        r_init = _compute_radius(centers)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = centers.copy()
            
        opt_centers, opt_radius = _optimize_from_initial(centers, start_time + DEADLINE)
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers
            
    # 4. Sobol
    if _time_left() > 15.0:
        centers = _build_sobol_init()
        r_init = _compute_radius(centers)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = centers.copy()
            
        opt_centers, opt_radius = _optimize_from_initial(centers, start_time + DEADLINE)
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers

    # 5. Random
    if _time_left() > 15.0:
        rng = np.random.default_rng(123)
        for _ in range(3):
            if _time_left() < 5.0:
                break
            centers = rng.uniform(-0.45, 0.45, (N, 3))
            r_init = _compute_radius(centers)
            if r_init > best_radius:
                best_radius = r_init
                best_centers = centers.copy()
                
            opt_centers, opt_radius = _optimize_from_initial(centers, start_time + DEADLINE)
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers

    if best_centers is None:
        best_centers = _build_grid_init(5, 5, 4, 0.0)
        
    return best_centers
