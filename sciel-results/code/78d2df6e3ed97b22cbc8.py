"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import random


def _compute_radius(c):
    """Compute the common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    dists = cdist(c, c)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)


def _softmin_radius(c, beta=50.0):
    """Smooth surrogate for min radius using soft-min via log-sum-exp."""
    # Pairwise distances
    dists = cdist(c, c)
    np.fill_diagonal(dists, np.inf)
    # Soft-min of pairwise distances / 2
    d_pair = dists / 2.0
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    all_dists = np.concatenate([d_pair.flatten(), wall_dists.flatten()])
    # Soft-min: -beta * log(sum(exp(-beta * d)))
    shifted = -beta * all_dists
    max_shifted = np.max(shifted)
    log_sum = np.log(np.sum(np.exp(shifted - max_shifted))) + max_shifted
    return -log_sum / beta


def _softmin_radius_grad(c, beta=50.0):
    """Gradient of soft-min radius w.r.t. center positions."""
    n = len(c)
    dists = cdist(c, c)
    np.fill_diagonal(dists, np.inf)
    d_pair = dists / 2.0
    wall_dists = 0.5 - np.abs(c)
    all_dists = np.concatenate([d_pair.flatten(), wall_dists.flatten()])
    
    # Compute weights for soft-min
    shifted = -beta * all_dists
    max_shifted = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    
    grad = np.zeros_like(c)
    
    # Pairwise contribution
    pair_weights = weights[:n * n].reshape(n, n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = pair_weights[i, j]
            if w > 1e-10:
                diff = c[i] - c[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-10:
                    grad[i] += w * diff / dist
                    grad[j] -= w * diff / dist
    
    # Wall contribution
    wall_weights = weights[n * n:]
    for i in range(n):
        for dim in range(3):
            w = wall_weights[i * 3 + dim]
            if w > 1e-10:
                # d(0.5 - |c[i,dim]|)/dc[i,dim] = -sign(c[i,dim])
                grad[i, dim] += -w * np.sign(c[i, dim])
    
    return -grad


def _epigraph_objective(x, beta=50.0):
    """Objective: -t + penalty for constraints, where x = [centers..., t]."""
    c = x[:N * 3].reshape(N, 3)
    t = x[N * 3]
    
    # Pairwise distances
    dists = cdist(c, c)
    np.fill_diagonal(dists, np.inf)
    d_pair = dists / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    all_dists = np.concatenate([d_pair.flatten(), wall_dists.flatten()])
    
    # Soft-min approximation of min(all_dists)
    shifted = -beta * all_dists
    max_shifted = np.max(shifted)
    log_sum = np.log(np.sum(np.exp(shifted - max_shifted))) + max_shifted
    soft_min = -log_sum / beta
    
    # Objective: maximize t, penalize if t > soft_min
    penalty = np.maximum(0, t - soft_min) ** 2
    return -t + penalty


def _epigraph_grad(x, beta=50.0):
    """Gradient of epigraph objective."""
    c = x[:N * 3].reshape(N, 3)
    t = x[N * 3]
    
    dists = cdist(c, c)
    np.fill_diagonal(dists, np.inf)
    d_pair = dists / 2.0
    wall_dists = 0.5 - np.abs(c)
    all_dists = np.concatenate([d_pair.flatten(), wall_dists.flatten()])
    
    shifted = -beta * all_dists
    max_shifted = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    
    grad_c = np.zeros_like(c)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            w = weights[i * N + j]
            if w > 1e-10:
                diff = c[i] - c[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-10:
                    grad_c[i] += w * diff / dist
                    grad_c[j] -= w * diff / dist
    
    for i in range(N):
        for dim in range(3):
            w = weights[N * N + i * 3 + dim]
            if w > 1e-10:
                grad_c[i, dim] += -w * np.sign(c[i, dim])
    
    soft_min = -np.log(np.sum(exp_shifted)) / beta + max_shifted / beta
    penalty_factor = 2 * np.maximum(0, t - soft_min)
    
    grad_t = -1 + penalty_factor
    grad_c = grad_c * penalty_factor
    
    grad = np.concatenate([grad_c.flatten(), [grad_t]])
    return grad


def _lloyd_step(c, beta=30.0):
    """Lloyd's algorithm step: move each center toward centroid of its Voronoi cell."""
    # Approximate using nearest neighbors
    n = len(c)
    dists = cdist(c, c)
    np.fill_diagonal(dists, np.inf)
    
    new_c = c.copy()
    for i in range(n):
        # Find 8 nearest neighbors
        nn_idx = np.argsort(dists[i])[:8]
        nn_points = c[nn_idx]
        # Weighted average (closer neighbors have more weight)
        weights = 1.0 / (dists[i][nn_idx] + 1e-10)
        weights /= np.sum(weights)
        target = np.sum(nn_points * weights[:, None], axis=0)
        # Move toward target but keep within bounds
        step = 0.1 * (target - c[i])
        new_c[i] = c[i] + step
        new_c[i] = np.clip(new_c[i], -0.45, 0.45)
    
    return new_c


def _lloyd_optimize(c, max_iters=50, beta=30.0):
    """Run Lloyd's algorithm for several iterations."""
    best_c = c.copy()
    best_r = _compute_radius(best_c)
    
    for it in range(max_iters):
        c_new = _lloyd_step(best_c, beta)
        r_new = _compute_radius(c_new)
        if r_new > best_r:
            best_c = c_new
            best_r = r_new
        else:
            # Try smaller step
            c_small = best_c + 0.05 * (c_new - best_c)
            c_small = np.clip(c_small, -0.45, 0.45)
            r_small = _compute_radius(c_small)
            if r_small > best_r:
                best_c = c_small
                best_r = r_small
            else:
                break
    
    return best_c, best_r


def _repulsion_anneal(c, time_budget=30.0, beta_start=20.0, beta_end=100.0):
    """Repulsion-based annealing with increasing softness."""
    start_time = time.time()
    best_c = c.copy()
    best_r = _compute_radius(best_c)
    
    n_steps = 200
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        beta = beta_start + (beta_end - beta_start) * step / n_steps
        T = 0.02 * (1 - step / n_steps) + 0.001
        
        # Compute gradient of soft-min
        grad = _softmin_radius_grad(best_c, beta)
        
        # Add random noise
        noise = np.random.randn(N, 3) * T
        
        # Move in gradient direction
        lr = 0.01 * (1 - step / n_steps) + 0.001
        c_try = best_c + lr * grad + noise
        c_try = np.clip(c_try, -0.48, 0.48)
        
        r_try = _compute_radius(c_try)
        if r_try > best_r:
            best_c = c_try
            best_r = r_try
    
    return best_c, best_r


def _local_search(c, max_iters=100, step_sizes=None):
    """Local search with adaptive step sizes."""
    if step_sizes is None:
        step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    best_c = c.copy()
    best_r = _compute_radius(best_c)
    
    for it in range(max_iters):
        improved = False
        for i in range(N):
            for dim in range(3):
                for step in step_sizes:
                    for sign in [1, -1]:
                        c_try = best_c.copy()
                        c_try[i, dim] += sign * step
                        if c_try[i, dim] < -0.5 or c_try[i, dim] > 0.5:
                            continue
                        r_try = _compute_radius(c_try)
                        if r_try > best_r + 1e-12:
                            best_c = c_try
                            best_r = r_try
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_c, best_r


def _seed_fcc():
    """FCC-like seed configuration."""
    # Face-centered cubic lattice points
    points = []
    m = 4
    a = 1.0 / m
    for i in range(m):
        for j in range(m):
            for k in range(m):
                # Corner
                points.append([i * a - 0.5 + a/2, j * a - 0.5 + a/2, k * a - 0.5 + a/2])
                # Face centers
                if i < m-1:
                    points.append([(i + 0.5) * a - 0.5 + a/2, j * a - 0.5 + a/2, k * a - 0.5 + a/2])
                if j < m-1:
                    points.append([i * a - 0.5 + a/2, (j + 0.5) * a - 0.5 + a/2, k * a - 0.5 + a/2])
                if k < m-1:
                    points.append([i * a - 0.5 + a/2, j * a - 0.5 + a/2, (k + 0.5) * a - 0.5 + a/2])
    
    random.seed(42)
    random.shuffle(points)
    return np.array(points[:N])


def _seed_grid():
    """Simple grid seed."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    points = []
    for z in ticks:
        for y in ticks:
            for x in ticks:
                points.append([x, y, z])
    random.seed(42)
    random.shuffle(points)
    return np.array(points[:N])


def _seed_random():
    """Random seed with rejection sampling."""
    random.seed(42)
    points = []
    for _ in range(N):
        for attempt in range(1000):
            p = np.array([random.uniform(-0.4, 0.4) for _ in range(3)])
            if len(points) == 0:
                points.append(p)
                break
            if all(np.linalg.norm(p - q) > 0.15 for q in points):
                points.append(p)
                break
    while len(points) < N:
        p = np.array([random.uniform(-0.4, 0.4) for _ in range(3)])
        points.append(p)
    return np.array(points[:N])


def construct():
    """Construct the optimal packing of 50 spheres in a unit cube."""
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    # Try multiple seeds
    seeds = [_seed_fcc(), _seed_grid(), _seed_random()]
    
    best_c = None
    best_r = 0
    
    for seed in seeds:
        if time.time() > deadline - 30:
            break
        
        c = seed.copy()
        
        # Phase 1: Lloyd's algorithm
        if time.time() < deadline - 25:
            c, r = _lloyd_optimize(c, max_iters=30)
            if r > best_r:
                best_c = c.copy()
                best_r = r
        
        # Phase 2: Repulsion annealing
        if time.time() < deadline - 15:
            c, r = _repulsion_anneal(c, time_budget=min(15.0, deadline - time.time() - 10))
            if r > best_r:
                best_c = c.copy()
                best_r = r
        
        # Phase 3: Local search polish
        if time.time() < deadline - 5:
            c, r = _local_search(c, max_iters=50, step_sizes=[0.02, 0.01, 0.005, 0.002, 0.001, 0.0005])
            if r > best_r:
                best_c = c.copy()
                best_r = r
    
    # Final fine polish
    if best_c is not None and time.time() < deadline - 2:
        c, r = _local_search(best_c, max_iters=30, step_sizes=[0.005, 0.002, 0.001, 0.0005, 0.0002])
        if r > best_r:
            best_c = c
            best_r = r
    
    # Ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
# EVOLVE-BLOCK-END
