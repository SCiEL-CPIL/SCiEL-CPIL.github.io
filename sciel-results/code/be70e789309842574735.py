"""Rewritten program: hard-core repulsive gas with differentiable soft-min objective."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _pairwise_dists(centers):
    """Compute all pairwise distances between centers."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    return dists

def _boundary_dists(centers):
    """Compute distance from each center to the nearest cube face."""
    # Cube is [-0.5, 0.5]^3
    d = 0.5 - np.abs(centers)
    return np.min(d, axis=1)

def _soft_min_obj(params, beta):
    """
    Objective: minimize -log-sum-exp of scaled distances.
    This is a smooth approximation to maximizing the minimum distance.
    params: flat array of shape (3*N,)
    beta: temperature parameter (higher = smoother, lower = sharper)
    """
    centers = params.reshape(-1, 3)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
    
    # Upper triangle only (avoid self-distances and double counting)
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Boundary distances
    bnd_dists = 0.5 - np.abs(centers)
    bnd_dists = np.min(bnd_dists, axis=1)
    
    # Combine all distances
    all_dists = np.concatenate([pair_dists, bnd_dists])
    
    # Soft min via log-sum-exp: min(d) ≈ -(1/beta) * log(sum(exp(-beta * d)))
    # We want to MAXIMIZE min distance, so MINIMIZE -min distance
    # Equivalent: minimize (1/beta) * log(sum(exp(-beta * d)))
    
    # For numerical stability, use log-sum-exp trick
    scaled = -beta * all_dists
    max_scaled = np.max(scaled)
    if not np.isfinite(max_scaled):
        return 1e10
    
    sum_exp = np.sum(np.exp(scaled - max_scaled))
    if sum_exp <= 0 or not np.isfinite(sum_exp):
        return 1e10
    
    log_sum_exp = max_scaled + np.log(sum_exp)
    
    # The soft min value is: -(1/beta) * log_sum_exp
    # We want to maximize this, so we minimize its negative
    obj = -(1.0 / beta) * log_sum_exp
    
    return obj

def _soft_min_obj_and_grad(params, beta):
    """Objective and gradient for soft-min packing."""
    centers = params.reshape(-1, 3)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Boundary distances
    bnd_dists = 0.5 - np.abs(centers)
    bnd_dists = np.min(bnd_dists, axis=1)
    
    all_dists = np.concatenate([pair_dists, bnd_dists])
    
    # Soft min computation
    scaled = -beta * all_dists
    max_scaled = np.max(scaled)
    if not np.isfinite(max_scaled):
        return 1e10, np.zeros_like(params)
    
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    if sum_exp <= 0 or not np.isfinite(sum_exp):
        return 1e10, np.zeros_like(params)
    
    log_sum_exp = max_scaled + np.log(sum_exp)
    obj = -(1.0 / beta) * log_sum_exp
    
    # Gradient computation
    # d/dx [-(1/beta) * log(sum(exp(-beta * d)))]
    # = -(1/beta) * [sum(-beta * exp(-beta * d) * dd/dx) / sum(exp(-beta * d))]
    # = [sum(exp(-beta * d) * dd/dx) / sum(exp(-beta * d))]
    
    # Weights
    weights = exp_scaled / sum_exp  # shape (num_constraints,)
    
    grad = np.zeros_like(params)
    
    # Pairwise gradient contributions
    # For each pair (i,j), d = |c_i - c_j|
    # dd/dc_i = (c_i - c_j) / d
    # dd/dc_j = (c_j - c_i) / d = -(c_i - c_j) / d
    
    # We need to accumulate contributions
    for idx, (i, j) in enumerate(zip(*iu)):
        w = weights[idx]
        d = pair_dists[idx]
        if d < 1e-10:
            continue
        diff_ij = centers[i] - centers[j]
        grad[3*i:3*i+3] += w * diff_ij / d
        grad[3*j:3*j+3] -= w * diff_ij / d
    
    # Boundary gradient contributions
    # d = 0.5 - |c_k|, so dd/dc_k = -sign(c_k)
    for idx in range(N):
        w = weights[N + idx]  # offset by number of pairs
        c = centers[idx]
        grad[3*idx:3*idx+3] += w * (-np.sign(c))
    
    return obj, grad

def _exact_radius(centers):
    """Compute the exact packing radius."""
    if len(centers) == 0:
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(N, k=1)
    min_pair = np.min(dists[iu]) if N > 1 else 1e10
    
    # Boundary distances
    bnd = 0.5 - np.abs(centers)
    min_bnd = np.min(np.min(bnd, axis=1))
    
    r = min(min_pair / 2.0, min_bnd)
    
    # Check validity
    if not np.all(np.isfinite(centers)):
        return 0.0
    if not np.all(np.abs(centers) <= 0.5 + 1e-12):
        return 0.0
    
    return max(0.0, r)

def _init_random(rng, n=50):
    """Random initialization inside the cube."""
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def _init_lattice(rng, n=50):
    """Lattice-based initialization with jitter."""
    # Try a simple grid
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    if len(grid) >= n:
        # Add small random perturbation
        jitter = rng.normal(0, 0.01, size=grid[:n].shape)
        return grid[:n] + jitter
    else:
        return _init_random(rng, n)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    best_radius = 0.0
    best_centers = None
    
    # Multiple restarts with different initializations
    n_restarts = 8
    restarts_done = 0
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Choose initialization
        if restart % 3 == 0:
            centers = _init_lattice(rng, N)
        elif restart % 3 == 1:
            centers = _init_random(rng, N)
        else:
            # BCC-like lattice
            m = 3
            bcc_points = []
            for i in range(m):
                for j in range(m):
                    for k in range(m):
                        x = (i + 0.5) / m - 0.5
                        y = (j + 0.5) / m - 0.5
                        z = (k + 0.5) / m - 0.5
                        bcc_points.append([x, y, z])
                        # BCC interior point
                        if i < m-1 and j < m-1 and k < m-1:
                            bcc_points.append([x + 0.5/m, y + 0.5/m, z + 0.5/m])
            bcc_arr = np.array(bcc_points[:N])
            if len(bcc_arr) < N:
                bcc_arr = _init_random(rng, N)
            else:
                jitter = rng.normal(0, 0.005, size=bcc_arr.shape)
                bcc_arr = bcc_arr + jitter
            centers = bcc_arr
        
        # Clamp to valid region
        centers = np.clip(centers, -0.49, 0.49)
        
        # Optimize using L-BFGS-B with soft-min objective
        params = centers.flatten().copy()
        
        # Adaptive beta: start smooth, get sharper
        beta_schedule = [10.0, 30.0, 100.0, 300.0]
        
        for beta in beta_schedule:
            if time.time() > deadline - 3.0:
                break
            
            try:
                result = minimize(
                    _soft_min_obj,
                    params,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-10}
                )
                params = result.x
                centers = params.reshape(-1, 3)
                centers = np.clip(centers, -0.499, 0.499)
            except Exception:
                break
        
        # Final polish with exact radius check
        centers = np.clip(centers, -0.499, 0.499)
        r = _exact_radius(centers)
        
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
        
        restarts_done += 1
    
    if best_centers is None:
        # Fallback: simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
