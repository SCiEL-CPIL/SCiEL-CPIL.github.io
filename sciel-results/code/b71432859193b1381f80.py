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
    BCC has higher packing density than simple cubic.
    For 100 points, create a BCC lattice and take the first 100 points.
    """
    rng = np.random.default_rng(seed)
    
    # BCC lattice: points at integer positions and half-integer positions
    # We'll create a grid of cells, each with 2 points (corner + center)
    # Use a 5x4x4 grid of BCC cells = 80 cells, 2 pts each = 160 pts, take 100
    nx, ny, nz = 5, 4, 4
    points = []
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Corner point (scaled to fit in [-0.5, 0.5]^3)
                x = -0.48 + i * (0.96 / (nx - 1))
                y = -0.48 + j * (0.96 / (ny - 1))
                z = -0.48 + k * (0.96 / (nz - 1))
                points.append([x, y, z])
                
                # Center point (offset by half the cell size)
                dx = 0.96 / (nx - 1) / 2
                dy = 0.96 / (ny - 1) / 2
                dz = 0.96 / (nz - 1) / 2
                points.append([x + dx, y + dy, z + dz])
    
    centers = np.array(points)
    
    # Add small random perturbation to break symmetry
    centers += rng.normal(0, 0.003, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers[:N]

def _build_halton_init(seed=42):
    """
    Build a Halton sequence initialization for well-distributed points.
    """
    rng = np.random.default_rng(seed)
    
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
        x = halton(i, 2)
        y = halton(i, 3)
        z = halton(i, 5)
        points.append([x - 0.5, y - 0.5, z - 0.5])
    
    centers = np.array(points)
    
    # Add small random perturbation
    centers += rng.normal(0, 0.002, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _anneal(centers, deadline, iu, n_restarts=3):
    """
    Simple simulated annealing to escape local minima.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    T = 0.05  # Initial temperature
    T_min = 0.001
    cooling = 0.95
    
    while T > T_min and time.time() < deadline - 10.0:
        # Generate random perturbation
        rng = np.random.default_rng()
        perturbation = rng.normal(0, T, (N, 3))
        new_centers = best_centers + perturbation
        new_centers = np.clip(new_centers, -0.499, 0.499)
        
        new_radius = _compute_radius(new_centers)
        
        # Accept if better or with probability exp(-(r_old - r_new)/T)
        if new_radius >= best_radius or np.random.random() < np.exp(-(best_radius - new_radius) / T):
            best_centers = new_centers
            best_radius = new_radius
        
        T *= cooling
    
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

def _bottleneck_polish(centers, deadline, iu, n_iterations=5):
    """
    Targeted polishing: identify the bottleneck constraint (closest pair or wall)
    and apply focused optimization to improve it.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    for _ in range(n_iterations):
        if time.time() > deadline - 3.0:
            break
        
        # Identify the bottleneck
        dists_to_walls = 0.5 - np.abs(best_centers)
        min_wall_idx = np.argmin(dists_to_walls)
        min_wall_val = np.min(dists_to_walls)
        
        dists = cdist(best_centers, best_centers, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        min_pair_idx = np.argmin(dists)
        min_pair_val = np.min(dists) / 2.0
        
        # Use very high beta for focused optimization
        beta = 100000.0
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 200}
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
    
    # Multiple diverse initializations
    # 1. Grid-based initializations
    grid_configs = [
        (5, 5, 4, 0.005),
        (5, 4, 5, 0.005),
        (4, 5, 5, 0.005),
        (5, 5, 4, 0.01),
        (5, 5, 4, 0.002),
    ]
    
    for nx, ny, nz, noise in grid_configs:
        if time.time() > deadline - 20.0:
            break
            
        centers = _build_grid(nx, ny, nz, noise_scale=noise, seed=int(time.time()*1000) % 10000)
        
        # Quick check
        r_init = _compute_radius(centers)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = centers.copy()
            
        # Optimize this specific initialization
        opt_centers, opt_radius = _optimize_sequence(
            centers, 
            deadline, 
            iu,
            beta_start=10.0, 
            beta_end=5000.0, 
            n_steps=8
        )
        
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers
    
    # 2. BCC initialization (higher packing density)
    if time.time() < deadline - 15.0:
        bcc_centers = _build_bcc_init(seed=int(time.time()*1000) % 10000)
        r_init = _compute_radius(bcc_centers)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = bcc_centers.copy()
            
        opt_centers, opt_radius = _optimize_sequence(
            bcc_centers, 
            deadline, 
            iu,
            beta_start=10.0, 
            beta_end=5000.0, 
            n_steps=8
        )
        
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers
    
    # 3. Halton initialization (well-distributed)
    if time.time() < deadline - 15.0:
        halton_centers = _build_halton_init(seed=int(time.time()*1000) % 10000)
        r_init = _compute_radius(halton_centers)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = halton_centers.copy()
            
        opt_centers, opt_radius = _optimize_sequence(
            halton_centers, 
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
        best_centers = _build_grid(5, 5, 4, 0.005)
        best_radius = _compute_radius(best_centers)
    
    # Bottleneck polishing: focus on the tightest constraint
    if time.time() < deadline - 10.0:
        polished_centers, polished_radius = _bottleneck_polish(best_centers, deadline, iu, n_iterations=3)
        if polished_radius > best_radius:
            best_radius = polished_radius
            best_centers = polished_centers
    
    # Final high-precision polish on the best candidate found
    if time.time() < deadline - 5.0:
        polished_centers, polished_radius = _final_polish(best_centers, deadline, iu)
        if polished_radius > best_radius:
            best_radius = polished_radius
            best_centers = polished_centers
            
    return best_centers

# EVOLVE-BLOCK-END
