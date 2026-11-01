import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

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

    # Pairwise distances using cdist for efficiency
    dists = cdist(centers, centers, metric='euclidean')
    
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)

    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(centers_flat, beta, iu):
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
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    # Using log-sum-exp trick for numerical stability
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _build_grid(nx, ny, nz, noise_scale=0.005, seed=42):
    """Build a grid of points with noise."""
    margin = 0.02
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers[:N]

def _build_bcc_init(seed=42):
    """
    Build a body-centered cubic (BCC) initialization.
    BCC has 2 points per cube cell: corners + center.
    For 100 points, use a 4x4x4 grid of BCC cells = 64 cells, 2 pts each = 128 pts, take 100.
    """
    rng = np.random.default_rng(seed)
    
    nx, ny, nz = 4, 4, 4
    points = []
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Corner point
                x = -0.45 + i * (0.9 / (nx - 1)) if nx > 1 else 0.0
                y = -0.45 + j * (0.9 / (ny - 1)) if ny > 1 else 0.0
                z = -0.45 + k * (0.9 / (nz - 1)) if nz > 1 else 0.0
                points.append([x, y, z])
                # Body center
                points.append([x + 0.9/(2*(nx-1)), y + 0.9/(2*(ny-1)), z + 0.9/(2*(nz-1))])
    
    centers = np.array(points[:N])
    centers += rng.normal(0, 0.003, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_halton_init(n_dims=3, seed=42):
    """
    Build a Halton sequence initialization for low-discrepancy distribution.
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(1, N + 1):
        point = [halton(i, base) for base in [2, 3, 5]]
        # Map from [0,1]^3 to [-0.5, 0.5]^3 with margin
        point = [p * 0.94 - 0.47 for p in point]
        points.append(point)
    
    centers = np.array(points)
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.002, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _fast_radius(centers):
    """
    Fast radius computation for annealing.
    """
    centers = np.asarray(centers, dtype=np.float64)
    
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances - only compute upper triangle for speed
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _anneal(centers, deadline, iu, n_iter=500, t_start=0.008, t_end=0.0005):
    """
    Simulated annealing to escape local minima.
    Uses fast radius computation for speed.
    Adaptive step size based on recent improvements.
    """
    best_centers = centers.copy()
    best_radius = _fast_radius(best_centers)
    current_radius = best_radius
    
    rng = np.random.default_rng(int(time.time() * 1000) % 100000)
    
    # Track recent improvements to adapt step size
    recent_improvements = []
    base_step = 0.004
    
    for it in range(n_iter):
        if time.time() > deadline - 25.0:
            break
        
        # Temperature schedule
        t = t_start * (t_end / t_start) ** (it / n_iter)
        
        # Adaptive step size: reduce if too many rejections, increase if stuck
        if len(recent_improvements) > 10:
            avg_imp = np.mean(recent_improvements[-10:])
            if avg_imp < 1e-6:
                base_step *= 0.95
            elif avg_imp > 1e-4:
                base_step = min(base_step * 1.05, 0.01)
        
        step_size = base_step * (t / t_start)
        new_centers = centers + rng.normal(0, step_size, centers.shape)
        new_centers = np.clip(new_centers, -0.499, 0.499)
        
        new_radius = _fast_radius(new_centers)
        delta = new_radius - current_radius
        
        # Acceptance criterion
        if delta > 0 or (t > 0 and np.log(rng.uniform()) < delta / t):
            centers = new_centers
            current_radius = new_radius
            recent_improvements.append(delta)
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
        else:
            recent_improvements.append(0.0)
        
        # Keep only last 20
        if len(recent_improvements) > 20:
            recent_improvements = recent_improvements[-20:]
    
    return best_centers, best_radius

def _optimize_sequence(centers, deadline, iu, beta_start=5.0, beta_end=50000.0, n_steps=12):
    """
    Run L-BFGS-B optimization with increasing beta.
    Returns (best_centers, best_radius).
    More aggressive beta schedule and tighter convergence.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Geometric progression of beta - more aggressive increase
    betas = np.geomspace(beta_start, beta_end, n_steps)
    
    for i, beta in enumerate(betas):
        if time.time() > deadline - 3.0:
            break
        
        x0 = best_centers.flatten()
        
        # Increase maxiter for higher beta values
        maxiter = 300 + int(200 * (i / n_steps))
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 150}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
            
    return best_centers, best_radius

def _final_polish(centers, deadline, iu):
    """
    High-precision polish pass with very high beta and tight tolerances.
    Also tries SLSQP for better handling of the non-smooth min.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Try a few very high beta values for final tightening
    high_betas = [10000, 20000, 50000, 100000, 200000]
    
    for beta in high_betas:
        if time.time() > deadline - 2.0:
            break
            
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 300}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
            
    # Try SLSQP with the actual (non-smooth) objective for final polish
    if time.time() < deadline - 1.5:
        def _neg_radius(centers_flat):
            centers = centers_flat.reshape(-1, 3)
            r = _compute_radius(centers)
            return -r
        
        x0 = best_centers.flatten()
        try:
            res = minimize(
                _neg_radius,
                x0,
                method='SLSQP',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 300, 'ftol': 1e-16, 'disp': False}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
            
    return best_centers, best_radius

def construct():
    start_time = time.time()
    # Safety margin: The limit is 150s. We aim to finish by 148s to be safe.
    deadline = start_time + 148.0
    
    # Precompute upper triangle indices for soft-min
    iu = np.triu_indices(N, k=1)
    
    best_centers = None
    best_radius = 0.0
    
    # Phase 1: Generate diverse initializations
    inits = []
    
    # BCC initialization
    bcc_centers = _build_bcc_init(seed=42)
    inits.append(bcc_centers)
    
    # Halton initialization
    halton_centers = _build_halton_init(seed=42)
    inits.append(halton_centers)
    
    # Grid initializations with different configurations
    grid_configs = [
        (5, 5, 4, 0.005, 42),
        (5, 4, 5, 0.005, 43),
        (4, 5, 5, 0.005, 44),
        (5, 5, 4, 0.01, 45),
        (5, 5, 4, 0.002, 46),
    ]
    
    for nx, ny, nz, noise, seed in grid_configs:
        centers = _build_grid(nx, ny, nz, noise_scale=noise, seed=seed)
        inits.append(centers)
    
    # Phase 2: For each initialization, run annealing + optimization
    # Use time budget wisely: allocate time based on remaining budget
    n_inits = len(inits)
    polish_time = 20.0  # Reserve for final polish
    
    for init_idx, init_centers in enumerate(inits):
        remaining = deadline - time.time() - polish_time
        if remaining < 5.0:
            break
        
        # Quick check initial radius
        r_init = _compute_radius(init_centers)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = init_centers.copy()
        
        # Allocate time: first init gets more, later ones less
        # Annealing: ~40% of remaining, Optimization: ~60%
        anneal_budget = min(15.0, remaining * 0.4)
        opt_budget = remaining - anneal_budget - 2.0
        
        if anneal_budget > 2.0:
            anneal_deadline = time.time() + anneal_budget
            annealed_centers, anneal_radius = _anneal(
                init_centers, 
                anneal_deadline, 
                iu, 
                n_iter=300, 
                t_start=0.008, 
                t_end=0.0005
            )
            
            if anneal_radius > best_radius:
                best_radius = anneal_radius
                best_centers = annealed_centers.copy()
        else:
            annealed_centers = init_centers
        
        # Optimization phase
        if opt_budget > 3.0:
            opt_deadline = time.time() + opt_budget
            opt_centers, opt_radius = _optimize_sequence(
                annealed_centers, 
                opt_deadline, 
                iu,
                beta_start=5.0, 
                beta_end=50000.0, 
                n_steps=12
            )
            
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers
            
    # If we haven't found anything (shouldn't happen), fallback
    if best_centers is None:
        best_centers = _build_grid(5, 5, 4, 0.005, seed=42)
        best_radius = _compute_radius(best_centers)
        
    # Final high-precision polish on the best candidate found
    if time.time() < deadline - 5.0:
        polished_centers, polished_radius = _final_polish(best_centers, deadline, iu)
        if polished_radius > best_radius:
            best_radius = polished_radius
            best_centers = polished_centers
            
    return best_centers

# EVOLVE-BLOCK-END
