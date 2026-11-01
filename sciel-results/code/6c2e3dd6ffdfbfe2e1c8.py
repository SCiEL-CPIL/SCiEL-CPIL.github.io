import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

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
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(x, beta):
    """
    Soft-min objective function for L-BFGS-B.
    We want to maximize r = min(min_wall, min_pair/2).
    This is equivalent to minimizing -r.
    Using soft-min: -min(A) approx -log(sum(exp(-beta*A))) / beta.
    Here A consists of wall distances and half-pair distances.
    """
    centers = x.reshape(-1, 3)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    
    # Pairwise distances (half)
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Combine all constraints into a single vector
    # We use a large negative value for diagonal to ignore them in soft min
    # But cdist fills diagonal with 0. We set them to inf.
    # For soft min, we need finite values.
    # Let's just take the min of the off-diagonals.
    # Actually, standard soft min over a vector.
    # Let's create a vector of all relevant distances.
    
    # Wall distances (N values)
    # Pair distances (N*(N-1)/2 values)
    # To avoid constructing huge vectors repeatedly, we can approximate.
    # But for N=100, N^2 is 10k. It's fine.
    
    # Let's compute soft min of the minimum of (wall_dists, pair_dists)
    # Soft min of min(A, B) is not simply soft_min(A) + soft_min(B).
    # We need soft_min(min(all_constraints)).
    # Let C be the vector of all constraints.
    # C = [wall_dists, pair_dists_upper]
    
    # To be efficient, we can just compute the min directly if beta is high enough?
    # No, L-BFGS-B needs gradients.
    
    # Let's compute the soft min of the set of all distances.
    # We'll flatten the upper triangle of pair distances.
    iu = np.triu_indices(N, 1)
    pair_dists_flat = pair_dists[iu]
    
    all_dists = np.concatenate([wall_dists.ravel(), pair_dists_flat])
    
    # Soft min: -1/beta * log(sum(exp(-beta * all_dists)))
    # To avoid overflow/underflow, subtract max (which is min of dists? No, exp is decreasing).
    # We want to approximate min(d).
    # log-sum-exp trick:
    # Let m = min(all_dists)
    # sum(exp(-beta*d)) = exp(-beta*m) * sum(exp(-beta*(d-m)))
    # log(sum) = -beta*m + log(sum(exp(-beta*(d-m))))
    # SoftMin = -1/beta * (-beta*m + log(...)) = m - 1/beta * log(...)
    
    m = np.min(all_dists)
    # If m is very small, -beta*m might be large positive? No, d > 0.
    # exp(-beta * (d - m)) <= 1. So sum is <= N_constraints.
    # This is numerically stable.
    
    exp_terms = np.exp(-beta * (all_dists - m))
    log_sum = np.log(np.sum(exp_terms))
    
    soft_min_val = m - (1.0 / beta) * log_sum
    
    # We want to maximize soft_min_val, so minimize -soft_min_val
    return -soft_min_val

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

def _optimize_with_lbfgsb(centers, deadline):
    """
    Optimize centers using L-BFGS-B on the soft-min objective.
    """
    x0 = centers.ravel()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Adaptive beta
    # Start with low beta (smoother), increase over time (sharper)
    beta_start = 10.0
    beta_end = 1000.0
    
    best_x = x0
    best_val = _compute_radius(centers)
    
    # Run optimization
    # We might run multiple iterations with increasing beta
    # Or just one run with a reasonably high beta.
    # L-BFGS-B can be sensitive to beta.
    # Let's try a few betas.
    
    betas = [10.0, 50.0, 200.0, 1000.0]
    
    current_x = x0
    for beta in betas:
        if time.time() > deadline - 2.0:
            break
            
        def func(x):
            return _soft_min_objective(x, beta)
            
        try:
            res = minimize(func, current_x, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5})
            if res.success and res.fun < func(current_x):
                current_x = res.x
        except Exception:
            pass
        
        # Update best
        curr_centers = current_x.reshape(-1, 3)
        curr_r = _compute_radius(curr_centers)
        if curr_r > best_val:
            best_val = curr_r
            best_x = current_x
            
    return best_x.reshape(-1, 3), best_val

def construct():
    start_time = time.time()
    deadline = _get_deadline()
    
    best_centers = None
    best_radius = 0.0
    
    # Initializations
    initializations = []
    
    # Halton
    initializations.append(_build_halton_init(seed=42))
    initializations.append(_build_halton_init(seed=123))
    
    # Grids
    grid_configs = [
        (5, 5, 4),
        (4, 5, 5),
        (5, 4, 5),
    ]
    for nx, ny, nz in grid_configs:
        initializations.append(_build_grid_init(nx, ny, nz, noise_scale=0.01, seed=nx*100+ny*10+nz))
        
    # Random
    rng = np.random.default_rng(123)
    for _ in range(3):
        centers = rng.uniform(-0.45, 0.45, (N, 3))
        initializations.append(centers)
        
    n_inits = len(initializations)
    # Allocate time
    time_per_init = (_get_deadline() - time.time() - 5.0) / n_inits
    
    for centers in initializations:
        if time.time() > deadline - 5.0:
            break
            
        # Optimize
        opt_centers, opt_radius = _optimize_with_lbfgsb(centers, time.time() + time_per_init)
        
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers

    if best_centers is None:
        best_centers = _build_halton_init()
        
    # Final Polish: One more L-BFGS-B run on the best result with high beta
    if time.time() < deadline - 2.0:
        final_centers, final_radius = _optimize_with_lbfgsb(best_centers, deadline)
        if final_radius > best_radius:
            best_radius = final_radius
            best_centers = final_centers

    return best_centers
