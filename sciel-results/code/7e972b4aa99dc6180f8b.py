"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

def _compute_radius(c):
    """
    Compute the exact maximum radius for the given centers.
    r = min( min_{i<j} ||c_i - c_j|| / 2, min_i min_k (0.5 - |c_ik|) )
    """
    if c.shape[0] < 2:
        return 0.5 - np.max(np.abs(c))
    
    # Wall clearance
    wall = 0.5 - np.max(np.abs(c))
    
    # Pairwise distances
    # Using cdist is efficient for N=50
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    
    return min(wall, min_pair)

def _soft_min_objective_and_grad(x, beta):
    """
    Compute the soft-min objective and its analytic gradient.
    f = -soft_min(vals) where vals are wall clearances and pairwise half-distances.
    Returns (f, g) where g is the gradient w.r.t. x.
    """
    c = x.reshape(N, 3)
    
    # Wall clearances: w_ik = 0.5 - |c_ik|
    wall = 0.5 - np.abs(c)  # shape (N, 3)
    
    # Pairwise half-distances: d_ij / 2 for i < j
    # Compute all pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]  # (N, N, 3)
    dists = np.sqrt(np.sum(diff ** 2, axis=2))  # (N, N)
    np.fill_diagonal(dists, np.inf)
    d_half = dists / 2.0  # (N, N)
    
    # Collect all values
    wall_flat = wall.ravel()  # (3N,)
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_half_upper = d_half[i_idx, j_idx]  # (N*(N-1)/2,)
    
    vals = np.concatenate([wall_flat, d_half_upper])
    n_vals = len(vals)
    
    # Soft-min: s = m - log(sum(exp(-beta*(v - m)))) / beta
    m = np.min(vals)
    shifted = -beta * (vals - m)
    # Clip for numerical stability
    shifted = np.clip(shifted, -500, 500)
    sum_exp = np.sum(np.exp(shifted))
    log_sum_exp = np.log(sum_exp)
    soft_min_val = m - log_sum_exp / beta
    
    f = -soft_min_val
    
    # Gradient: d(soft_min)/dv = -exp(-beta*(v-m)) / sum_exp
    # So d(f)/dv = +exp(-beta*(v-m)) / sum_exp
    weights = np.exp(shifted) / sum_exp  # (n_vals,)
    
    # Now compute gradient w.r.t. x
    g = np.zeros_like(x)
    
    # Wall part: w_ik = 0.5 - |c_ik|
    # d(w_ik)/dx = -sign(c_ik) for the corresponding coordinate
    # d(f)/dx = sum_k weights_wall * d(w_ik)/dx
    wall_weights = weights[:3 * N]  # (3N,)
    wall_w = wall_weights.reshape(N, 3)
    # d(w_ik)/d(c_ik) = -sign(c_ik), but at c_ik=0 it's 0 (use 0)
    signs = np.sign(c)
    signs[np.abs(c) < 1e-12] = 0.0
    g_wall = -wall_w * signs  # (N, 3)
    g += g_wall
    
    # Pair part: d_half_ij = ||c_i - c_j|| / 2
    # d(d_half_ij)/d(c_i) = (c_i - c_j) / (2 * ||c_i - c_j||) = (c_i - c_j) / (2 * dists_ij)
    # d(d_half_ij)/d(c_j) = -(c_i - c_j) / (2 * dists_ij)
    # d(f)/dx = sum_{i<j} weights_pair * [d(d_half_ij)/d(c_i) + d(d_half_ij)/d(c_j)]
    pair_weights = weights[3 * N:]  # (n_pairs,)
    
    # Vectorized gradient computation
    # For each pair (i,j), contribution to g[i] is +w * (c_i - c_j) / (2 * dist)
    # and to g[j] is -w * (c_i - c_j) / (2 * dist)
    # Use np.add.at for accumulation
    
    diff_ij = c[i_idx] - c[j_idx]  # (n_pairs, 3)
    dists_ij = dists[i_idx, j_idx]  # (n_pairs,)
    
    # Guard against zero distance
    safe_dists = np.where(dists_ij > 1e-12, dists_ij, 1.0)
    pair_grad = pair_weights[:, np.newaxis] * diff_ij / (2.0 * safe_dists[:, np.newaxis])
    
    # Accumulate: g[i] += pair_grad, g[j] -= pair_grad
    np.add.at(g, i_idx, pair_grad)
    np.add.at(g, j_idx, -pair_grad)
    
    return f, g

def _generate_bcc_lattice(n_points):
    """
    Generate points on a BCC lattice scaled to fit in [-0.5, 0.5]^3.
    BCC has points at corners and center of each cube.
    For a cubic region of side L, with n divisions per axis:
    - Corner points: (i, j, k) for i,j,k in 0..n
    - Center points: (i+0.5, j+0.5, k+0.5) for i,j,k in 0..n-1
    
    We'll generate a grid and select n_points that are well-distributed.
    """
    # Generate BCC points in [0, 1]^3
    n_div = 5  # 5 divisions gives 6^3 = 216 corner points + 5^3 = 125 center points = 341 total
    
    points = []
    
    # Corner points
    for i in range(n_div + 1):
        for j in range(n_div + 1):
            for k in range(n_div + 1):
                points.append([i / n_div, j / n_div, k / n_div])
    
    # Center points
    for i in range(n_div):
        for j in range(n_div):
            for k in range(n_div):
                points.append([(i + 0.5) / n_div, (j + 0.5) / n_div, (k + 0.5) / n_div])
    
    points = np.array(points)
    
    # Scale to [-0.5, 0.5]
    points = points - 0.5
    
    # Select n_points using farthest point sampling
    rng = np.random.default_rng(123)
    indices = np.arange(len(points))
    rng.shuffle(indices)
    points_shuffled = points[indices]
    
    selected = [points_shuffled[0]]
    for i in range(1, len(points_shuffled)):
        if len(selected) >= n_points:
            break
        p = points_shuffled[i]
        dists = np.linalg.norm(np.array(selected) - p, axis=1)
        if np.min(dists) > 0.08:
            selected.append(p)
    
    # If we didn't get enough, add random points
    while len(selected) < n_points:
        p = rng.uniform(-0.45, 0.45, size=3)
        selected.append(p)
    
    return np.array(selected[:n_points])

def _optimize_from_initial(c_init, deadline, seed):
    """
    Run continuation optimization from an initial configuration.
    Returns (best_c, best_val)
    """
    best_c = c_init.copy()
    best_val = _compute_radius(best_c)
    
    # Continuation schedule: ramp up beta
    betas = [3, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072]
    
    for beta in betas:
        if time.time() > deadline - 15:
            break
        
        def obj_and_grad(x):
            return _soft_min_objective_and_grad(x, beta)
        
        try:
            res = minimize(obj_and_grad, best_c.ravel(), method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': 200, 'ftol': 1e-11, 'gtol': 1e-7})
            
            new_c = res.x.reshape(N, 3)
            new_c = np.clip(new_c, -0.4995, 0.4995)
            
            new_val = _compute_radius(new_c)
            
            if new_val > best_val:
                best_val = new_val
                best_c = new_c.copy()
                
        except Exception:
            pass
    
    # Final polish with high beta
    if time.time() < deadline - 8:
        final_beta = 5000
        def obj_and_grad_final(x):
            return _soft_min_objective_and_grad(x, final_beta)
        try:
            res = minimize(obj_and_grad_final, best_c.ravel(), method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-9})
            new_c = res.x.reshape(N, 3)
            new_c = np.clip(new_c, -0.49995, 0.49995)
            new_val = _compute_radius(new_c)
            if new_val > best_val:
                best_val = new_val
                best_c = new_c.copy()
        except Exception:
            pass
    
    return best_c, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Generate multiple initial configurations
    rng = np.random.default_rng(42)
    
    # 1. BCC lattice initialization
    c_bcc = _generate_bcc_lattice(N)
    
    # 2. Add some random perturbations to create variants
    initial_configs = [c_bcc.copy()]
    
    # Create 3 variants with different perturbations
    for seed in [1, 2, 3]:
        c_variant = c_bcc.copy()
        noise = rng.normal(0, 0.02, size=(N, 3))
        c_variant = c_variant + noise
        c_variant = np.clip(c_variant, -0.45, 0.45)
        initial_configs.append(c_variant)
    
    # 3. Also try a grid-based initialization as backup
    grid_size = 10
    ticks = (np.arange(grid_size) + 0.5) / grid_size - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    indices = np.arange(len(grid))
    rng.shuffle(indices)
    grid_shuffled = grid[indices]
    
    selected = [grid_shuffled[0]]
    for i in range(1, len(grid_shuffled)):
        if len(selected) >= N:
            break
        p = grid_shuffled[i]
        dists = np.linalg.norm(np.array(selected) - p, axis=1)
        if np.min(dists) > 0.1:
            selected.append(p)
    
    while len(selected) < N:
        p = grid_shuffled[rng.integers(len(grid_shuffled))]
        dists = np.linalg.norm(np.array(selected) - p, axis=1)
        if len(dists) == 0 or np.min(dists) > 0.05:
            selected.append(p)
        else:
            p = rng.uniform(-0.45, 0.45, size=3)
            selected.append(p)
    
    initial_configs.append(np.array(selected[:N]))
    
    # Optimize from each initial configuration
    best_c = initial_configs[0].copy()
    best_val = _compute_radius(best_c)
    
    for idx, c_init in enumerate(initial_configs):
        if time.time() > deadline - 25:
            break
        
        try:
            c_opt, val = _optimize_from_initial(c_init, deadline, idx)
            if val > best_val:
                best_val = val
                best_c = c_opt.copy()
        except Exception:
            pass
    
    # Additional restarts with random perturbations of best solution if time permits
    if time.time() < deadline - 30:
        for restart in range(3):
            if time.time() > deadline - 20:
                break
            c_perturbed = best_c + rng.normal(0, 0.01, size=(N, 3))
            c_perturbed = np.clip(c_perturbed, -0.48, 0.48)
            try:
                c_opt, val = _optimize_from_initial(c_perturbed, deadline, 100 + restart)
                if val > best_val:
                    best_val = val
                    best_c = c_opt.copy()
            except Exception:
                pass
    
    # 4. Ensure valid output
    best_c = np.clip(best_c, -0.5, 0.5)
    best_c = np.asarray(best_c, dtype=float)
    
    if best_c.shape != (N, 3) or not np.all(np.isfinite(best_c)):
        # Fallback
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]
        
    return best_c

# EVOLVE-BLOCK-END
