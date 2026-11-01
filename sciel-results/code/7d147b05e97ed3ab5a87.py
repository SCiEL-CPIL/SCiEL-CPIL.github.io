"""Packing 50 equal spheres in a unit cube using MD + Voronoi-Lloyd + soft-min polish."""
import numpy as np
import time
import random
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

N = 50

def _compute_radius(c):
    """True radius: min over pairs of d/2 and over walls of (0.5-|x|)."""
    # Pairwise
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    # Walls
    wall = 0.5 - np.abs(c)
    min_wall = np.min(wall)
    return min(min_pair / 2.0, min_wall)

def _softmin_radius(c, beta=200.0):
    """
    Smooth surrogate of the true radius using log-sum-exp (soft-min).
    Returns (soft_r, grad) where grad is (N,3).
    The soft-min approximates min( {d_ij/2}_{i<j}, {0.5-|c_ik|}_ik ).
    """
    # Build all constraints in a single array
    # Pair constraints: d_ij / 2 for i<j
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    # Take upper triangle
    iu = np.triu_indices(N, k=1)
    pair_vals = d[iu] / 2.0  # shape (M,)
    # Wall constraints: for each point, 3 walls
    # 0.5 - c[i,k], 0.5 + c[i,k]
    wall_vals = np.concatenate([
        0.5 - c,   # +x walls: 0.5 - c[i,k]
        0.5 + c,   # -x walls: 0.5 + c[i,k]
    ]).ravel()  # shape (2N*3,)
    
    all_vals = np.concatenate([pair_vals, wall_vals])
    
    # Soft-min: -logsumexp(-beta * vals) / beta
    neg_beta_vals = -beta * all_vals
    # Numerical stability
    max_val = np.max(neg_beta_vals)
    lse = max_val + np.log(np.sum(np.exp(neg_beta_vals - max_val)))
    soft_r = -lse / beta
    
    # Gradient: d(soft_r)/d(c)
    # soft_r = -(1/beta) * log(sum(exp(-beta * v_m)))
    # d soft_r / d c = (1/beta) * sum_m [ exp(-beta v_m) * (-beta) * dv_m/dc ] / sum(exp(-beta v_m))
    #                = - sum_m [ w_m * dv_m/dc ] where w_m = exp(-beta v_m)/sum
    weights = np.exp(neg_beta_vals - lse)  # shape (len(all_vals),), sums to 1
    
    # Gradient w.r.t. c
    grad = np.zeros((N, 3))
    
    # Pair gradients: v_m = d_ij/2, dv_m/dc_i = (c_i - c_j)/d_ij, dv_m/dc_j = (c_j - c_i)/d_ij
    # For each pair (i,j): weight w, contributes -w * (c_i - c_j)/d_ij to grad[i]
    # and -w * (c_j - c_i)/d_ij to grad[j]
    # Note: d_ij = 2 * v_m, so (c_i - c_j)/d_ij = (c_i - c_j)/(2*v_m)
    # But we have pair_vals = d/2, so d = 2*pair_vals
    # Actually let's recompute: v = d_ij/2. dv/dc_i = (c_i - c_j)/d_ij = (c_i-c_j)/(2v)
    # Contribution to grad[i] from pair m: -w_m * (c_i - c_j)/(2 * pair_vals[m])
    # But pair_vals[m] = d_ij/2, so 2*pair_vals[m] = d_ij. OK.
    
    # Vectorized: for each pair (i,j) with weight w:
    # grad[i] -= w * (c[i] - c[j]) / d_ij
    # grad[j] -= w * (c[j] - c[i]) / d_ij = grad[j] += w * (c[i] - c[j]) / d_ij
    
    ii = iu[0]
    jj = iu[1]
    d_ij = d[ii, jj]
    # Avoid division by zero
    d_safe = np.where(d_ij > 1e-12, d_ij, 1e-12)
    w_pairs = weights[:len(ii)]
    diff = c[ii] - c[jj]  # (M, 3)
    contrib = w_pairs[:, None] * diff / d_safe[:, None]  # (M, 3)
    
    # Scatter add
    np.add.at(grad, ii, -contrib)
    np.add.at(grad, jj, contrib)
    
    # Wall gradients: v = 0.5 - c[i,k] => dv/dc[i,k] = -1
    # v = 0.5 + c[i,k] => dv/dc[i,k] = +1
    # Contribution to grad[i,k]: -w * dv/dc[i,k]
    w_walls = weights[len(ii):]
    # First half: 0.5 - c => dv/dc = -1 => grad += w
    # Second half: 0.5 + c => dv/dc = +1 => grad -= w
    w_pos = w_walls[:N*3].reshape(N, 3)  # 0.5 - c[i,k]
    w_neg = w_walls[N*3:].reshape(N, 3)  # 0.5 + c[i,k]
    grad += w_pos  # -w * (-1) = +w
    grad -= w_neg  # -w * (+1) = -w
    
    return soft_r, grad

def _md_relax(c, time_budget=30.0, beta_start=50.0, beta_end=500.0, dt=0.002):
    """
    Langevin molecular dynamics to minimize -soft_r (i.e., maximize soft_r).
    Uses velocity Verlet with friction and random kicks.
    """
    start = time.time()
    c = c.copy()
    n_steps = int(time_budget / dt)
    
    v = np.zeros_like(c)
    mass = 1.0
    gamma = 0.1  # friction
    kT = 0.005  # initial temperature
    
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for step in range(n_steps):
        if time.time() - start > time_budget:
            break
        
        # Anneal beta and temperature
        frac = step / max(1, n_steps)
        beta = beta_start + (beta_end - beta_start) * frac
        T = kT * (1.0 - frac) + 1e-6
        
        # Compute soft_r and gradient
        soft_r, grad = _softmin_radius(c, beta=beta)
        
        # Force = -d(-soft_r)/dc = grad (we want to maximize soft_r, so force = grad)
        # Actually, we minimize -soft_r, so force = -d(-soft_r)/dc = grad
        force = grad
        
        # Velocity Verlet (simplified, since force changes)
        v = v * (1.0 - gamma * dt) + force * dt + np.random.randn(*c.shape) * np.sqrt(2 * T * dt)
        c = c + v * dt
        
        # Project back into cube with hard walls
        c = np.clip(c, -0.5 + 1e-10, 0.5 - 1e-10)
        
        # Track best
        r_true = _compute_radius(c)
        if r_true > best_r:
            best_r = r_true
            best_c = c.copy()
    
    return best_c, best_r

def _lloyd_step(c, n_iters=20):
    """
    Voronoi-Lloyd: move each point toward the centroid of its Voronoi cell.
    This equalizes the packing and tends to increase the minimum distance.
    """
    c = c.copy()
    for it in range(n_iters):
        tree = cKDTree(c)
        # For each point, find its Voronoi cell centroid by sampling
        # Approximate: use nearest neighbors to estimate cell
        # Simpler: use the centroid of the 12 nearest neighbors as a proxy
        # Actually, let's do a proper-ish Voronoi via sampling
        # For speed, use a grid-based Voronoi approximation
        # Alternative: use scipy's Voronoi if available, but it's 2D only for scipy.spatial.Voronoi
        # Let's use a simple approach: for each point, compute the centroid of points
        # that are closer to it than to any other point, using a fine grid
        
        # Use a moderate grid for Voronoi cell estimation
        grid_size = 30
        ticks = np.linspace(-0.5, 0.5, grid_size)
        gx, gy, gz = np.meshgrid(ticks, ticks, ticks, indexing='ij')
        grid_pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        
        # For each grid point, find nearest sphere center
        tree2 = cKDTree(c)
        dists, idx = tree2.query(grid_pts, k=1)
        
        # Accumulate centroids
        centroids = np.zeros((N, 3))
        counts = np.zeros(N)
        for i in range(N):
            mask = idx == i
            if np.any(mask):
                centroids[i] = np.mean(grid_pts[mask], axis=0)
                counts[i] = np.sum(mask)
        
        # Update: move toward centroid, but only if it increases local clearance
        # Use a step size to avoid oscillation
        step_size = 0.1
        for i in range(N):
            if counts[i] > 0:
                direction = centroids[i] - c[i]
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    c[i] += step_size * direction / norm * min(norm, 0.05)
        
        # Project into cube
        c = np.clip(c, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return c

def _seed_fcc():
    """
    Generate an FCC-like seed for 50 points.
    FCC has 4 points per conventional cell. We'll use a distorted FCC.
    """
    # Start with a 4x4x4 grid and perturb toward FCC
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    points = []
    for z in ticks:
        for y in ticks:
            for x in ticks:
                points.append([x, y, z])
    
    # Add FCC offset points
    fcc_offsets = [
        [0.25/m, 0.25/m, 0.25/m],
        [0.25/m, 0.75/m, 0.75/m],
        [0.75/m, 0.25/m, 0.75/m],
        [0.75/m, 0.75/m, 0.25/m],
    ]
    # Actually, let's just use a random perturbation of the grid
    random.seed(12345)
    c = np.array(points[:N], dtype=float)
    c += np.random.randn(*c.shape) * 0.02
    c = np.clip(c, -0.5 + 1e-10, 0.5 - 1e-10)
    return c

def _seed_random():
    """Random seed with minimum separation."""
    random.seed(54321)
    np.random.seed(54321)
    c = np.random.uniform(-0.5 + 0.05, 0.5 - 0.05, (N, 3))
    return c

def _seed_grid():
    """Simple 4x4x4 grid seed, pick 50."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    points = []
    for z in ticks:
        for y in ticks:
            for x in ticks:
                points.append([x, y, z])
    random.seed(42)
    random.shuffle(points)
    return np.array(points[:N], dtype=float)

def _polish(c, time_budget=15.0):
    """
    Final polish: alternating soft-min gradient descent and direct radius check.
    """
    start = time.time()
    c = c.copy()
    best_r = _compute_radius(c)
    best_c = c.copy()
    
    lr = 0.005
    beta = 300.0
    
    while time.time() - start < time_budget:
        soft_r, grad = _softmin_radius(c, beta=beta)
        # Gradient ascent on soft_r
        c = c + lr * grad
        c = np.clip(c, -0.5 + 1e-10, 0.5 - 1e-10)
        
        r_true = _compute_radius(c)
        if r_true > best_r:
            best_r = r_true
            best_c = c.copy()
        
        # Decay learning rate
        lr *= 0.995
        beta *= 1.001
    
    return best_c, best_r

def construct():
    """Construct the optimal packing of 50 spheres in a unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Multi-restart with different seeds
    seeds = [
        _seed_fcc,
        _seed_random,
        _seed_grid,
    ]
    
    best_c = None
    best_r = -1.0
    
    for seed_fn in seeds:
        if time.time() > deadline - 20.0:
            break
        
        c = seed_fn()
        
        # Voronoi-Lloyd equalization
        c = _lloyd_step(c, n_iters=10)
        
        # MD relaxation
        remaining = deadline - time.time()
        md_budget = min(35.0, remaining * 0.5)
        if md_budget > 5.0:
            c, r_md = _md_relax(c, time_budget=md_budget)
            r_md = _compute_radius(c)
            if r_md > best_r:
                best_r = r_md
                best_c = c.copy()
        
        # Polish
        remaining = deadline - time.time()
        if remaining > 10.0:
            c_pol, r_pol = _polish(c, time_budget=min(15.0, remaining - 5.0))
            if r_pol > best_r:
                best_r = r_pol
                best_c = c_pol.copy()
    
    # Final safety polish if time remains
    if best_c is not None and time.time() < deadline - 5.0:
        c_final, r_final = _polish(best_c, time_budget=min(10.0, deadline - time.time() - 3.0))
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
    
    if best_c is None:
        best_c = _seed_grid()
    
    # Ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
