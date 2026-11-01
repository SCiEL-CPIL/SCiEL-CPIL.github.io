import numpy as np
import time
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

def _soft_min_objective(centers, beta):
    """
    Smooth surrogate for the negative minimum radius using log-sum-exp.
    Returns (objective_value, gradient) where lower is better.
    objective = -soft_min(min_wall_dist, min_pair_dist/2)
    We maximize r = min(min_wall, min_pair/2), so we minimize -r.
    
    Uses: soft_min(a, b) ≈ -(1/beta) * log(exp(-beta*a) + exp(-beta*b))
    For multiple terms: soft_min(x_1,...,x_k) ≈ -(1/beta)*log(sum(exp(-beta*x_i)))
    """
    # Wall distances: 0.5 - |c_ik| for each i,k
    abs_c = np.abs(centers)  # (N, 3)
    wall_dists = 0.5 - abs_c  # (N, 3)
    
    # Pair distances / 2
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_dists_half = dists / 2.0  # (N, N)
    
    # Combine all constraints into a single soft-min
    # All wall distances and all pair distances/2
    all_dists = np.concatenate([wall_dists.ravel(), pair_dists_half.ravel()])
    
    # Soft min: -(1/beta) * log(sum(exp(-beta * d_i)))
    # For numerical stability, subtract max
    max_d = np.max(all_dists)
    shifted = -beta * (all_dists - max_d)
    # Clip for numerical stability
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = max_d - log_sum / beta
    
    # Gradient with respect to centers
    # d(soft_min)/d(centers) = -sum_i [exp(-beta*d_i) / sum_j exp(-beta*d_j)] * d(d_i)/d(centers)
    weights = np.exp(shifted) / np.sum(np.exp(shifted))  # (num_constraints,)
    
    grad = np.zeros_like(centers)
    
    # Wall constraint gradients: d(0.5 - |c_ik|)/dc_ik = -sign(c_ik)
    wall_weights = weights[:N*3].reshape(N, 3)
    grad -= wall_weights * np.sign(centers)
    
    # Pair constraint gradients: d(|c_i - c_j|/2)/dc_i = (c_i - c_j) / |c_i - c_j|
    pair_weights = weights[N*3:].reshape(N, N)
    # For each pair (i,j), contribution to grad[i] and grad[j]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (N, N, 3)
    norms = np.linalg.norm(diff, axis=2)  # (N, N)
    norms = np.where(norms > 1e-12, norms, 1.0)
    
    # grad[i] gets sum_j w_ij * (c_i - c_j)/|c_i - c_j|
    # grad[j] gets -sum_i w_ij * (c_i - c_j)/|c_i - c_j|
    unit_diff = diff / norms[:, :, np.newaxis]  # (N, N, 3)
    grad += np.sum(pair_weights[:, :, np.newaxis] * unit_diff, axis=1)
    
    objective = -soft_min_val  # We want to maximize r, so minimize -r
    return objective, grad

def _soft_min_optimize(centers, deadline, beta_start=50.0, beta_end=500.0, lr_start=0.01, lr_end=0.001):
    """
    Gradient-based optimization using soft-min surrogate.
    """
    centers = centers.copy()
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    
    n_steps = 2000
    rng = np.random.default_rng(42)
    
    for step in range(n_steps):
        if time.time() > deadline - 1.0:
            break
        
        # Anneal beta and learning rate
        t = step / max(n_steps - 1, 1)
        beta = beta_start * (beta_end / beta_start) ** t
        lr = lr_start * (lr_end / lr_start) ** t
        
        # Add small noise to escape local minima
        if step % 100 == 0 and t < 0.5:
            noise_scale = 0.01 * (1 - t)
            centers += rng.normal(0, noise_scale, centers.shape)
            centers = np.clip(centers, -0.499, 0.499)
        
        obj, grad = _soft_min_objective(centers, beta)
        
        # Gradient descent step
        centers -= lr * grad
        centers = np.clip(centers, -0.499, 0.499)
        
        # Check improvement
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    return best_centers, best_r

def _polish(centers, deadline):
    """
    High-precision polish: iteratively push apart closest pairs and away from walls.
    """
    centers = centers.copy()
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    
    max_iters = 5000
    
    for it in range(max_iters):
        if time.time() > deadline - 0.5:
            break
        
        # Compute current radius
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
        
        # Find closest pair
        tree = cKDTree(centers)
        dists, idxs = tree.query(centers, k=2)
        min_pair_dist = np.min(dists[:, 1])
        min_idx = np.argmin(dists[:, 1])
        i = min_idx
        j = idxs[min_idx, 1]
        
        # Find closest wall
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        wall_flat_idx = np.argmin(wall_dists)
        i_wall = wall_flat_idx // 3
        k_wall = wall_flat_idx % 3
        
        # Determine which constraint is active
        pair_r = min_pair_dist / 2.0
        if pair_r < min_wall:
            # Move pair apart
            diff = centers[i] - centers[j]
            norm = np.linalg.norm(diff)
            if norm > 1e-10:
                direction = diff / norm
            else:
                direction = rng.normal(0, 1, 3)
                direction /= np.linalg.norm(direction)
            
            # Adaptive step size
            step = min(0.002, (min_pair_dist - 2.0 * best_r) / 4.0 + 0.0005)
            step = max(step, 0.0001)
            
            centers[i] += step * direction
            centers[j] -= step * direction
        else:
            # Move away from wall
            if centers[i_wall, k_wall] > 0:
                direction = -1.0
            elif centers[i_wall, k_wall] < 0:
                direction = 1.0
            else:
                direction = 1.0
            
            step = min(0.002, (min_wall - best_r) / 2.0 + 0.0005)
            step = max(step, 0.0001)
            
            centers[i_wall, k_wall] += step * direction
        
        centers = np.clip(centers, -0.499, 0.499)
    
    return best_centers, best_r

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
    Build a BCC (body-centered cubic) inspired initialization.
    """
    centers = _build_grid_init(5, 5, 4, noise_scale=0.02, seed=999)
    return centers

def construct():
    start_time = time.time()
    deadline = _get_deadline()
    
    best_centers = None
    best_radius = 0.0
    
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
    total_time = _get_deadline() - time.time() - 3.0
    time_per_init = total_time / n_inits
    
    for centers in initializations:
        if time.time() > deadline - 3.0:
            break
            
        # Phase 1: Soft-min gradient optimization
        sm_deadline = min(time.time() + time_per_init * 0.5, deadline - 3.0)
        sm_centers, sm_radius = _soft_min_optimize(centers, sm_deadline)
        
        if sm_radius > best_radius:
            best_radius = sm_radius
            best_centers = sm_centers
            
        # Phase 2: Polish
        if time.time() < deadline - 3.0:
            polish_deadline = min(time.time() + time_per_init * 0.5, deadline - 3.0)
            polish_centers, polish_radius = _polish(sm_centers, polish_deadline)
            
            if polish_radius > best_radius:
                best_radius = polish_radius
                best_centers = polish_centers

    # Final polish on best result
    if best_centers is not None and time.time() < deadline - 1.0:
        final_centers, final_radius = _polish(best_centers, deadline - 0.5)
        if final_radius > best_radius:
            best_radius = final_radius
            best_centers = final_centers

    if best_centers is None:
        best_centers = _build_grid_init(5, 5, 4, 0.0)
        
    return best_centers
