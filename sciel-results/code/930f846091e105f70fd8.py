"""Improved packing: Hybrid Force-Directed Relaxation and Smooth Surrogate Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
deadline_global = 0.0

def _score(centers):
    """Compute the maximum common radius for given centers (exact, for evaluation)."""
    centers = np.asarray(centers)
    wall_min = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_min)
    n = len(centers)
    if n < 2:
        return r_wall
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    return min(r_wall, r_pair)

def _soft_min(values, beta=50.0):
    """Smooth approximation to min via log-sum-exp (negative log-sum-exp)."""
    values = np.asarray(values)
    m = np.min(values)
    exponent = -beta * (values - m)
    exponent = np.clip(exponent, -700, 700)
    log_sum = np.log(np.sum(np.exp(exponent)))
    return m - log_sum / beta

def _smooth_objective(x, beta=50.0):
    """
    Smooth surrogate: minimize -soft_min(all constraints).
    """
    c = x.reshape(-1, 3)
    
    # Wall constraints: 0.5 - |c_ik| for each coordinate
    wall_constraints = 0.5 - np.abs(c).ravel()
    
    # Pairwise half-distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_constraints = dists / 2.0
    pair_constraints = pair_constraints[np.triu_indices_from(pair_constraints, k=1)]
    
    # Combine all constraints
    all_constraints = np.concatenate([wall_constraints, pair_constraints])
    
    # Soft-min of all constraints (smooth approximation to the true min)
    s = _soft_min(all_constraints, beta)
    
    # Penalty for going outside bounds (soft)
    out_of_bounds = np.maximum(0, np.abs(c) - 0.4999)
    penalty = 10.0 * np.sum(out_of_bounds ** 2)
    
    return -s + penalty

def _repulsion_anneal(centers, max_iters=500, lr_start=0.02, lr_end=0.001):
    """
    Anneal the positions using repulsive forces to maximize minimum distance.
    """
    c = centers.copy()
    n = len(c)
    soft = 1e-4
    
    for i in range(max_iters):
        if time.time() > deadline_global - 10:
            break
            
        progress = i / max_iters
        lr = lr_start * (lr_end / lr_start) ** progress
        
        forces = np.zeros_like(c)
        
        # Pairwise repulsion
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Force magnitude: 1 / dist^2
        forces_mag = 1.0 / (dists ** 2 + soft)
        
        # Force direction: (c_i - c_j) / dist
        safe_dists = np.where(dists < 1e-9, 1.0, dists)
        force_dirs = diff / safe_dists[:, :, np.newaxis]
        
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        forces_mag = forces_mag * mask
        force_dirs = force_dirs * mask[:, :, np.newaxis]
        
        forces += np.sum(forces_mag[:, :, np.newaxis] * force_dirs, axis=1)
        
        # Wall repulsion
        wall_dist = 0.5 - np.abs(c)
        for k in range(3):
            w = wall_dist[:, k]
            f_w = np.zeros(n)
            idx = w < 0.2
            if np.any(idx):
                mag = 1.0 / (w[idx] ** 2 + soft)
                dir = -np.sign(c[idx, k])
                f_w[idx] = mag * dir
            forces[:, k] += f_w
        
        c += lr * forces
        c = np.clip(c, -0.5, 0.5)
        
    return c

def _initial_grid():
    """FCC-inspired grid initialization."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(grid))
    return grid[perm[:N]].copy()

def _initial_fcc_like():
    """Initialize with FCC-like structure: layers of hexagonal packing."""
    centers = np.zeros((N, 3))
    n_layers = 5
    per_layer = N // n_layers
    remaining = N - per_layer * n_layers
    
    for layer in range(n_layers):
        z = (layer + 0.5) / n_layers - 0.5
        n_in_layer = per_layer + (1 if layer < remaining else 0)
        
        for i in range(n_in_layer):
            row = i // 4
            col = i % 4
            x = (col + 0.5) / 4.0 - 0.5
            y = (row + 0.5) / 3.0 - 0.5
            if row % 2 == 1:
                x += 0.25 / 4.0
            centers[layer * per_layer + i] = [x, y, z]
    
    idx = per_layer * n_layers
    while idx < N:
        centers[idx] = [0, 0, 0]
        idx += 1
        
    return centers

def construct():
    global deadline_global
    t_start = time.time()
    deadline_global = t_start + 130  # Safety margin
    
    inits = [
        ("grid", _initial_grid()),
        ("fcc", _initial_fcc_like()),
    ]
    
    best_centers = None
    best_score = -np.inf
    
    for name, init in inits:
        if time.time() > deadline_global - 30:
            break
        
        cur = init.copy()
        cur_score = _score(cur)
        
        if best_centers is None or cur_score > best_score:
            best_centers = cur.copy()
            best_score = cur_score
            
        # Run repulsion annealing
        if time.time() < deadline_global - 20:
            annealed = _repulsion_anneal(cur, max_iters=300)
            annealed_score = _score(annealed)
            if annealed_score > best_score:
                best_centers = annealed.copy()
                best_score = annealed_score
                cur = annealed.copy()
            else:
                cur = best_centers.copy()
                
        # Polish with smooth optimization
        if time.time() > deadline_global - 15:
            break
            
        for beta in [50.0, 100.0]:
            if time.time() > deadline_global - 5:
                break
            try:
                res = minimize(
                    _smooth_objective,
                    cur.ravel(),
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * 3 * N,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                new_centers = res.x.reshape(-1, 3)
                new_score = _score(new_centers)
                if new_score > best_score:
                    best_centers = new_centers.copy()
                    best_score = new_score
                    cur = new_centers.copy()
            except Exception:
                pass
                
    return best_centers
