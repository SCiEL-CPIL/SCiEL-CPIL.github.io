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
    if len(centers) == 0:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
        return 0.0
        
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances
    # Using cdist is efficient for N=100
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

def _optimize_local(centers, deadline, iu, time_budget=5.0):
    """
    Run a local L-BFGS-B optimization from initial centers.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    start_t = time.time()
    current_centers = centers.copy()
    
    # Use a few betas for quick local polish
    betas = [100.0, 500.0]
    
    for beta in betas:
        if _time_left() < 1.0 or (time.time() - start_t) > time_budget - 1.0:
            break
            
        x0 = current_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success:
                new_centers = res.x.reshape(-1, 3)
                new_radius = _compute_radius(new_centers)
                
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = new_centers.copy()
                    current_centers = new_centers.copy()
                        
        except Exception:
            pass
            
    return best_centers, best_radius

def _simulated_annealing(centers, deadline, iu, time_budget=100.0):
    """
    Simulated Annealing with periodic local polishing.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_centers = centers.copy()
    current_radius = best_radius
    
    start_t = time.time()
    
    # Temperature schedule
    T_start = 0.1
    T_end = 0.001
    cooling = 0.999
    
    T = T_start
    rng = np.random.default_rng(42)
    
    step_count = 0
    polish_interval = 5000
    
    while _time_left() > 2.0 and (time.time() - start_t) < time_budget - 1.0 and T > T_end:
        # Periodic polish
        if step_count > 0 and step_count % polish_interval == 0:
            if _time_left() > 5.0:
                polished, r_polished = _optimize_local(current_centers, deadline, iu, time_budget=5.0)
                if r_polished > current_radius:
                    current_centers = polished
                    current_radius = r_polished
                    if current_radius > best_radius:
                        best_radius = current_radius
                        best_centers = current_centers.copy()
        
        # Pick a random point
        i = rng.integers(0, N)
        
        # Generate a random move
        # Step size proportional to T
        step_size = 0.05 * T / 0.1 # Scale step with temperature
        move = rng.normal(0, step_size, 3)
        
        new_centers = current_centers.copy()
        new_centers[i] += move
        
        # Clip to cube
        new_centers[i] = np.clip(new_centers[i], -0.4999, 0.4999)
        
        # Calculate new radius
        new_radius = _compute_radius(new_centers)
        
        # Delta E = E_new - E_old = -new_radius - (-current_radius) = current_radius - new_radius
        delta_E = current_radius - new_radius
        
        # Metropolis criterion
        if delta_E < 0:
            # Improvement
            current_centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = current_centers.copy()
        else:
            # Worse, accept with probability exp(-delta_E / T)
            if rng.random() < np.exp(-delta_E / T):
                current_centers = new_centers
                current_radius = new_radius
        
        # Cool down
        T *= cooling
        step_count += 1
        
    # Final polish
    if _time_left() > 2.0:
        polished, r_polished = _optimize_local(current_centers, deadline, iu, time_budget=_time_left() - 2.0)
        if r_polished > best_radius:
            best_radius = r_polished
            best_centers = polished.copy()
            
    return best_centers, best_radius

def construct():
    start_time = time.time()
    iu = np.triu_indices(N, k=1)
    
    # Strategy:
    # 1. Get a few good local minima quickly.
    # 2. Run SA from the best one.
    
    initializations = [
        _build_grid_init(5, 5, 4, noise_scale=0.01, seed=42),
        _build_halton_init(seed=42),
        np.random.default_rng(1).uniform(-0.45, 0.45, (N, 3)),
    ]
    
    best_centers_overall = None
    best_radius_overall = -1.0
    
    # Phase 1: Quick local optimizations to find a good basin
    for init_centers in initializations:
        if _time_left() < 10.0:
            break
        centers, radius = _optimize_local(init_centers, _get_deadline(), iu, time_budget=5.0)
        if radius > best_radius_overall:
            best_radius_overall = radius
            best_centers_overall = centers.copy()
            
    if best_centers_overall is None:
        best_centers_overall = initializations[0]
        best_radius_overall = _compute_radius(best_centers_overall)
        
    # Phase 2: Simulated Annealing to escape local minima
    if _time_left() > 20.0:
        centers, radius = _simulated_annealing(best_centers_overall, _get_deadline(), iu, time_budget=_time_left() - 5.0)
        if radius > best_radius_overall:
            best_radius_overall = radius
            best_centers_overall = centers.copy()
            
    return best_centers_overall
