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

    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dists_sq))

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
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    pair_vals = dists / 2.0
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
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

def _build_bcc_init(n=N, noise_scale=0.003, seed=42):
    """
    Build a BCC (Body-Centered Cubic) inspired initialization.
    BCC is known to be a good lattice for sphere packing.
    """
    nx, ny, nz = 5, 5, 4
    margin = 0.02
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    
    all_centers = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                all_centers.append([xs[i], ys[j], zs[k]])
                if i < nx - 1 and j < ny - 1 and k < nz - 1:
                    cx = (xs[i] + xs[i+1]) / 2
                    cy = (ys[j] + ys[j+1]) / 2
                    cz = (zs[k] + zs[k+1]) / 2
                    all_centers.append([cx, cy, cz])
    
    all_centers = np.array(all_centers)
    rng = np.random.default_rng(seed)
    all_centers += rng.normal(0, noise_scale, all_centers.shape)
    all_centers = np.clip(all_centers, -0.499, 0.499)
    
    # Select n points with maximum minimum distance (greedy)
    if len(all_centers) <= n:
        return all_centers
    
    selected = [all_centers[0]]
    min_dists = np.linalg.norm(all_centers - all_centers[0], axis=1)
    for _ in range(n - 1):
        idx = np.argmax(min_dists)
        selected.append(all_centers[idx])
        new_dists = np.linalg.norm(all_centers - all_centers[idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)
    
    return np.array(selected)

def _build_halton_init(n=N, seed=42):
    """
    Build a Halton sequence initialization for quasi-random distribution.
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    centers = np.zeros((n, 3))
    for i in range(n):
        for j, base in enumerate([2, 3, 5]):
            centers[i, j] = -0.5 + halton(i + 1, base)
    
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _fast_radius_update(centers, idx_changed, old_centers):
    """
    Fast radius computation when only a few points changed.
    Returns (min_wall, min_pair_half).
    """
    # Wall distances for changed points
    wall_changed = 0.5 - np.abs(centers[idx_changed])
    min_wall_changed = np.min(wall_changed)
    
    # Wall distances for unchanged points (precomputed would be better, but for now)
    unchanged_mask = np.ones(N, dtype=bool)
    unchanged_mask[idx_changed] = False
    if np.any(unchanged_mask):
        wall_unchanged = 0.5 - np.abs(centers[unchanged_mask])
        min_wall_unchanged = np.min(wall_unchanged)
        min_wall = min(min_wall_changed, min_wall_unchanged)
    else:
        min_wall = min_wall_changed
    
    # Pairwise: only need to check pairs involving changed points
    min_pair_half = np.inf
    for i in idx_changed:
        # Distance from i to all j != i
        diffs = centers - centers[i]
        dists_sq = np.sum(diffs ** 2, axis=1)
        dists_sq[i] = np.inf
        min_d = np.sqrt(np.min(dists_sq))
        min_pair_half = min(min_pair_half, min_d / 2.0)
    
    # Also need min among unchanged-unchanged pairs (expensive, but we can cache)
    # For now, do full check if few changes
    if len(idx_changed) < 10:
        # Full pairwise for safety (still fast enough for small N)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_pair_half = np.sqrt(np.min(dists_sq)) / 2.0
    
    return min(min_wall, min_pair_half)

def _simulated_annealing_fast(centers, deadline, initial_temp=0.01, cooling_rate=0.9997, min_temp=0.0001):
    """
    Faster simulated annealing with incremental updates.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_radius = best_radius
    temp = initial_temp
    
    rng = np.random.default_rng(123)
    
    while time.time() < deadline and temp > min_temp:
        n_perturb = rng.integers(1, 4)
        idx = rng.choice(N, size=n_perturb, replace=False)
        
        new_centers = best_centers.copy()
        step_size = temp * 0.5
        new_centers[idx] += rng.normal(0, step_size, (n_perturb, 3))
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_radius = _compute_radius(new_centers)
        
        delta = new_radius - current_radius
        if delta > 0:
            best_centers = new_centers
            best_radius = new_radius
            current_radius = new_radius
        else:
            prob = np.exp(delta / temp)
            if rng.random() < prob:
                best_centers = new_centers
                current_radius = new_radius
        
        temp *= cooling_rate
        
        if current_radius < best_radius * 0.95 and temp < 0.005:
            break
            
    return best_centers, best_radius

def _coordinate_polish(centers, deadline, tol=1e-10):
    """
    Coordinate descent polish: optimize each coordinate direction.
    For each point, try moving it along each axis to maximize the minimum radius.
    """
    centers = centers.copy()
    current_r = _compute_radius(centers)
    
    for coord in range(3):
        for _ in range(50):  # multiple passes
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                # Try to optimize coordinate i, coord
                # Binary search for optimal position along this axis
                # Current value
                val = centers[i, coord]
                
                # Try small perturbations
                for delta in [0.001, -0.001, 0.0005, -0.0005, 0.0001, -0.0001]:
                    new_val = val + delta
                    if new_val < -0.5 or new_val > 0.5:
                        continue
                    old_val = centers[i, coord]
                    centers[i, coord] = new_val
                    new_r = _compute_radius(centers)
                    if new_r > current_r + tol:
                        current_r = new_r
                        improved = True
                    else:
                        centers[i, coord] = old_val
            
            if not improved:
                break
    
    return centers, current_r

def _penalty_objective(centers_flat, r, iu):
    """
    Penalty objective for maximizing radius r.
    """
    centers = centers_flat.reshape(-1, 3)
    
    dists_to_walls = 0.5 - np.abs(centers)
    wall_viol = np.maximum(0, r - dists_to_walls)
    wall_penalty = np.sum(wall_viol ** 2)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    pair_viol = np.maximum(0, 2 * r - dists)
    pair_vals_unique = pair_viol[iu]
    pair_penalty = np.sum(pair_vals_unique ** 2)
    
    return wall_penalty + pair_penalty

def _penalty_optimize(centers, r_start, deadline, iu):
    """
    Iteratively optimize the penalty objective with increasing r.
    """
    best_centers = centers.copy()
    best_r = _compute_radius(best_centers)
    r = r_start
    
    r_step = 0.001
    max_r = 0.12
    
    while time.time() < deadline - 3.0 and r < max_r:
        try:
            res = minimize(
                _penalty_objective,
                best_centers.flatten(),
                args=(r, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 30}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            penalty = _penalty_objective(new_centers.flatten(), r, iu)
            if penalty < 1e-8:
                best_centers = new_centers
                actual_r = _compute_radius(best_centers)
                if actual_r > best_r:
                    best_r = actual_r
                r += r_step
            else:
                r += r_step * 0.3
        except Exception:
            r += r_step * 0.3
    
    return best_centers, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Precompute upper triangle indices
    iu = np.triu_indices(N, k=1)
    
    best_centers = None
    best_radius = 0.0
    
    # Multiple initializations
    inits = []
    
    # Grids
    grids = [
        (5, 5, 4),
        (5, 4, 5),
        (4, 5, 5),
        (4, 4, 7),
        (7, 4, 4),
        (4, 7, 4),
    ]
    for nx, ny, nz in grids:
        if time.time() > deadline - 60.0:
            break
        centers = _build_grid(nx, ny, nz, noise_scale=0.005)
        inits.append(centers)
    
    # BCC
    if time.time() < deadline - 55.0:
        inits.append(_build_bcc_init())
    
    # Halton
    if time.time() < deadline - 50.0:
        inits.append(_build_halton_init())
    
    # Evaluate all inits and pick best
    for centers in inits:
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
    
    if best_centers is None:
        best_centers = _build_grid(5, 5, 4)
        best_radius = _compute_radius(best_centers)
        
    # Phase 1: Soft-min optimization with increasing beta
    beta = 10.0
    max_beta = 10000.0
    
    while time.time() < deadline - 40.0 and beta < max_beta:
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
            
        beta *= 1.5
        
        # If we're converging, increase beta faster
        if beta > 1000:
            beta *= 2.0
    
    # Phase 2: Penalty-based optimization
    if time.time() < deadline - 30.0:
        try:
            p_centers, p_radius = _penalty_optimize(
                best_centers,
                best_radius,
                deadline - 30.0,
                iu
            )
            if p_radius > best_radius:
                best_radius = p_radius
                best_centers = p_centers
        except Exception:
            pass
            
    # Phase 3: Simulated Annealing
    if time.time() < deadline - 20.0:
        try:
            sa_centers, sa_radius = _simulated_annealing_fast(
                best_centers, 
                deadline - 20.0,
                initial_temp=0.005,
                cooling_rate=0.9998
            )
            if sa_radius > best_radius:
                best_radius = sa_radius
                best_centers = sa_centers
        except Exception:
            pass
            
    # Phase 4: Coordinate polish
    if time.time() < deadline - 10.0:
        try:
            polished, pol_r = _coordinate_polish(best_centers, deadline - 10.0)
            if pol_r > best_radius:
                best_radius = pol_r
                best_centers = polished
        except Exception:
            pass
            
    # Phase 5: Final high-precision soft-min polish
    if time.time() < deadline - 2.0:
        try:
            x0 = best_centers.flatten()
            res = minimize(
                _soft_min_objective,
                x0,
                args=(20000.0, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
            )
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
    
    return best_centers

# EVOLVE-BLOCK-END
