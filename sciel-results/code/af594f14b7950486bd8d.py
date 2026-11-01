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
    For 100 points, use a 5x4x5 grid of BCC cells (5*4*5=100 cells, 2 pts each = 200, take 100).
    Actually, let's use a simpler approach: create a BCC lattice and take first 100 points.
    """
    rng = np.random.default_rng(seed)
    
    # BCC lattice: points at (i, j, k) and (i+0.5, j+0.5, k+0.5) for integer i,j,k
    # Scale to fit in [-0.5, 0.5]^3
    # Let's use a 4x4x4 grid of BCC cells = 64 cells, 2 pts each = 128 pts, take 100
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

def _anneal(centers, deadline, iu, n_iter=500, t_start=0.01, t_end=0.001):
    """
    Simulated annealing to escape local minima.
    Moves are random perturbations; accept if radius improves or with probability exp(-delta/T).
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_radius = best_radius
    
    rng = np.random.default_rng(int(time.time() * 1000) % 100000)
    
    for it in range(n_iter):
        if time.time() > deadline - 30.0:
            break
        
        # Temperature schedule
        t = t_start * (t_end / t_start) ** (it / n_iter)
        
        # Random perturbation
        step_size = 0.01 * (t / t_start)
        new_centers = centers + rng.normal(0, step_size, centers.shape)
        new_centers = np.clip(new_centers, -0.499, 0.499)
        
        new_radius = _compute_radius(new_centers)
        delta = new_radius - current_radius
        
        # Acceptance criterion
        if delta > 0 or (t > 0 and np.log(rng.uniform()) < delta / t):
            centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
    
    return best_centers, best_radius

def _optimize_sequence(centers, deadline, iu, beta_start=10.0, beta_end=20000.0, n_steps=10):
    """
    Run L-BFGS-B optimization with increasing beta.
    Returns (best_centers, best_radius).
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Geometric progression of beta
    betas = np.geomspace(beta_start, beta_end, n_steps)
    
    for beta in betas:
        if time.time() > deadline - 5.0:
            break
            
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
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
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Try a few very high beta values for final tightening
    high_betas = [5000, 10000, 20000, 50000]
    
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
                options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 100}
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
    # Safety margin: The limit is 150s. We aim to finish by 145s to be safe.
    deadline = start_time + 145.0
    
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
    for init_centers in inits:
        if time.time() > deadline - 20.0:
            break
        
        # Quick check initial radius
        r_init = _compute_radius(init_centers)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = init_centers.copy()
        
        # Annealing phase (limited time)
        anneal_deadline = min(deadline - 10.0, time.time() + 25.0)
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
        
        # Optimization phase
        opt_centers, opt_radius = _optimize_sequence(
            annealed_centers, 
            deadline, 
            iu,
            beta_start=10.0, 
            beta_end=5000.0, 
            n_steps=8
        )
        
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers
            
    # If we haven't found anything (shouldn't happen), fallback
    if best_centers is None:
        best_centers = _build_grid(5, 5, 4, 0.005, seed=42)
        best_radius = _compute_radius(best_centers)
        
    # Final high-precision polish on the best candidate found
    if time.time() < deadline - 10.0:
        polished_centers, polished_radius = _final_polish(best_centers, deadline, iu)
        if polished_radius > best_radius:
            best_radius = polished_radius
            best_centers = polished_centers
            
    return best_centers

# EVOLVE-BLOCK-END
