"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the exact maximum radius for given centers."""
    dist_bound = np.min(np.minimum(
        np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
        np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    ))
    n = len(centers)
    if n < 2:
        return dist_bound
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dists_sq))
    return min(min_pair_dist / 2.0, dist_bound)


def _soft_radius(centers, temp=0.005):
    """Smooth surrogate radius using soft-min via log-sum-exp."""
    # Boundary distances
    bx = np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5)
    by = np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    bdists = np.minimum(bx, by)
    
    # Soft min of boundary distances
    log_sum_exp_bd = temp * np.log(np.sum(np.exp(-bdists / temp)))
    soft_bd = -log_sum_exp_bd
    
    # Pairwise squared distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Soft min of pairwise distances (in sqrt space for better conditioning)
    # Use soft min on dists_sq then take sqrt, with a floor to avoid log(0)
    floor = 1e-12
    dists_sq_clipped = np.maximum(dists_sq, floor)
    log_sum_exp_pd = temp * np.log(np.sum(np.exp(-dists_sq_clipped / temp)))
    soft_pd_sq = -log_sum_exp_pd
    soft_pd = np.sqrt(max(soft_pd_sq, 0.0))
    
    # Soft min of (pairwise_dist/2, boundary_dist)
    # Combine: soft_min(a, b) = -temp * log(exp(-a/temp) + exp(-b/temp))
    a = soft_pd / 2.0
    b = soft_bd
    log_sum = np.log(np.exp(-a / temp) + np.exp(-b / temp))
    soft_r = -temp * log_sum
    
    return soft_r


def _negative_soft_radius(centers, temp=0.005):
    return -_soft_radius(centers, temp)


def _init_grid(n):
    """Initialize with a square grid."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]


def _init_hex(n, seed=42):
    """Hexagonal packing initialization."""
    rng = np.random.RandomState(seed)
    # Hexagonal lattice points
    points = []
    spacing = 0.14
    rows = int(np.ceil(1.0 / (spacing * np.sqrt(3)/2))) + 2
    for i in range(-1, rows):
        y = i * spacing * np.sqrt(3) / 2
        offset = (spacing / 2) if (i % 2) else 0
        for j in range(-1, int(1.0 / spacing) + 2):
            x = j * spacing + offset
            if -0.48 <= x <= 0.48 and -0.48 <= y <= 0.48:
                points.append((x, y))
    
    points = np.array(points)
    if len(points) < n:
        # Pad with random
        extra = rng.uniform(-0.45, 0.45, size=(n - len(points), 2))
        points = np.vstack([points, extra])
    
    # Shuffle and take first n
    idx = rng.permutation(len(points))[:n]
    return points[idx]


def _init_random(n, seed=42):
    """Initialize with random positions."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 2))


def _init_hybrid(n, seed=42):
    """Hybrid: grid + perturbation."""
    grid = _init_grid(n)
    rng = np.random.RandomState(seed)
    perturb = rng.normal(0, 0.015, size=(n, 2))
    centers = grid + perturb
    return np.clip(centers, -0.49, 0.49)


def _soft_optimize(centers, temp_start=0.02, temp_end=0.001, max_iters=800, 
                   deadline=None):
    """Optimize using smooth surrogate with annealing temperature."""
    n = len(centers)
    x0 = centers.flatten()
    
    def objective(x):
        c = x.reshape(-1, 2)
        return _negative_soft_radius(c, temp)
    
    def gradient(x):
        c = x.reshape(-1, 2)
        # Numerical gradient for robustness
        eps = 1e-6
        g = np.zeros_like(x)
        for i in range(len(x)):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            g[i] = (objective(xp) - objective(xm)) / (2 * eps)
        return g
    
    # Anneal temperature
    num_anneal_steps = 5
    temps = np.geomspace(temp_start, temp_end, num_anneal_steps)
    
    for temp in temps:
        if deadline is not None and time.time() > deadline:
            break
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                            jac=gradient,
                            options={'maxiter': max_iters // num_anneal_steps, 
                                     'gtol': 1e-5, 'ftol': 1e-8})
            x0 = result.x
        except Exception:
            pass
    
    c_final = x0.reshape(-1, 2)
    c_final = np.clip(c_final, -0.4999, 0.4999)
    return c_final, _compute_radius(c_final)


def _polish_exact(centers, max_iters=500):
    """High-precision polish using exact radius with small steps."""
    n = len(centers)
    c = centers.copy()
    current_r = _compute_radius(c)
    
    # Find the constraining pairs/boundaries
    # Do a few rounds of targeted moves
    for it in range(max_iters):
        improved = False
        # Compute distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        
        # Boundary distances
        bx = np.minimum(0.5 - c[:, 0], c[:, 0] + 0.5)
        by = np.minimum(0.5 - c[:, 1], c[:, 1] + 0.5)
        bdists = np.minimum(bx, by)
        
        # For each circle, try to move it away from its nearest neighbor/boundary
        for i in range(n):
            # Find nearest neighbor
            min_pair_dist = np.min(dists[i])
            nearest_j = np.argmin(dists[i])
            
            # Direction away from nearest neighbor
            away_dir = c[i] - c[nearest_j]
            away_norm = np.linalg.norm(away_dir)
            if away_norm > 1e-10:
                away_dir /= away_norm
            
            # Direction away from boundary
            bound_dir = np.zeros(2)
            if bx[i] < by[i]:
                bound_dir[0] = 1 if c[i, 0] > 0 else -1
            else:
                bound_dir[1] = 1 if c[i, 1] > 0 else -1
            
            # Try moving in combined direction
            for step in [0.0005, 0.0002, 0.0001]:
                new_c = c.copy()
                new_c[i] = c[i] + step * (away_dir + bound_dir)
                new_c[i] = np.clip(new_c[i], -0.4999, 0.4999)
                new_r = _compute_radius(new_c)
                if new_r > current_r + 1e-10:
                    c = new_c
                    current_r = new_r
                    improved = True
                    break
        
        if not improved:
            break
    
    return c, current_r


def construct():
    """Construct centers of 50 equal circles in unit square."""
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Diverse initializations
    inits = [
        _init_grid(N),
        _init_hex(N, seed=1),
        _init_random(N, seed=42),
        _init_hybrid(N, seed=42),
        _init_random(N, seed=123),
        _init_hybrid(N, seed=7),
        _init_hex(N, seed=99),
    ]
    
    for init in inits:
        if time.time() > deadline:
            break
        
        # Soft surrogate optimization with annealing
        c_soft, r_soft = _soft_optimize(init, temp_start=0.03, temp_end=0.002, 
                                        max_iters=600, deadline=deadline - 10)
        
        if r_soft > best_radius:
            best_radius = r_soft
            best_centers = c_soft.copy()
        
        # Exact polish
        if time.time() < deadline - 5:
            c_polish, r_polish = _polish_exact(c_soft, max_iters=300)
            if r_polish > best_radius:
                best_radius = r_polish
                best_centers = c_polish.copy()
    
    # Final high-precision polish on best
    if best_centers is not None and time.time() < deadline - 2:
        c_final, r_final = _polish_exact(best_centers, max_iters=500)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = c_final
    
    if best_centers is None:
        best_centers = _init_grid(N)
    
    # Ensure validity
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    # Verify no coincident centers
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(best_centers[i] - best_centers[j]) < 1e-9:
                best_centers[j, 0] += 1e-6
    
    return best_centers
# EVOLVE-BLOCK-END
