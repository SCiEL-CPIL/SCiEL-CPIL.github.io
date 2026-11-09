"""
Sphere packing in a unit cube: 100 equal spheres.
Uses KMC (Kinetic Monte Carlo) with constraint-targeted push-apart,
combined with a soft-min surrogate gradient polish for final refinement.
"""

import time
import numpy as np
from scipy.spatial.distance import cdist

N = 100
CUBE_HALF = 0.5

def _compute_radius(c):
    """Compute the maximum common radius for centres c of shape (N,3)."""
    if c.shape != (N, 3) or not np.all(np.isfinite(c)):
        return 0.0
    # Wall distance
    wall = CUBE_HALF - np.max(np.abs(c), axis=1)
    min_wall = np.min(wall)
    if min_wall < 0:
        return 0.0
    # Pairwise distances
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    return min(min_wall, min_pair)

def _soft_min_radius(c, beta=50.0):
    """
    Smooth surrogate for the minimum radius using log-sum-exp.
    Returns (value, gradient) of a smooth approximation to min(pairwise/2, wall).
    """
    # Wall constraints: r_i = CUBE_HALF - max_k |c_ik|
    # Use soft-max for the max over coordinates
    abs_c = np.abs(c)  # (N, 3)
    # soft max along axis 1
    sm = np.max(abs_c, axis=1)  # hard max as baseline
    # For smoothness, use log-sum-exp
    beta_sm = 30.0
    shifted = abs_c * beta_sm
    max_shifted = np.max(shifted, axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted - max_shifted), axis=1, keepdims=True)) + max_shifted
    soft_max_coord = log_sum_exp / beta_sm  # (N, 1)
    wall_i = CUBE_HALF - soft_max_coord[:, 0]  # (N,)

    # Pairwise: for each pair i<j, d_ij/2
    # We'll compute a soft-min over all pairwise distances
    d = cdist(c, c)  # (N, N)
    np.fill_diagonal(d, np.inf)
    # Take upper triangle
    iu = np.triu_indices(N, k=1)
    pairs = d[iu] / 2.0  # (M,)

    # Soft-min of all constraints: walls and pairs
    # Combine into one array
    all_constraints = np.concatenate([wall_i, pairs])  # (N + M,)
    # soft-min = -soft-max(-x)
    beta_min = beta
    neg = -all_constraints * beta_min
    max_neg = np.max(neg)
    log_sum = np.log(np.sum(np.exp(neg - max_neg))) + max_neg
    soft_min_val = -log_sum / beta_min

    # Gradient
    # d(soft_min)/d(c) 
    # For wall part: d(wall_i)/d(c_i) = -d(soft_max_coord_i)/d(c_i)
    # soft_max_coord_i = (1/beta_sm) * log(sum_k exp(beta_sm * |c_ik|))
    # d/dc_ik = sign(c_ik) * exp(beta_sm * |c_ik|) / sum_k exp(beta_sm * |c_ik|)
    # For pair part: d(pair_ij)/d(c_i) = -(c_i - c_j) / (2 * |c_i - c_j|) ... but we use soft-min

    # Compute weights for soft-min
    exp_neg = np.exp(neg - max_neg)
    weights = exp_neg / np.sum(exp_neg)  # (N+M,)

    grad = np.zeros_like(c)

    # Wall contributions
    w_wall = weights[:N]  # (N,)
    # d(wall_i)/d(c_i) = -d(soft_max)/d(c_i)
    # soft_max_i = (1/beta_sm)*log(sum_k exp(beta_sm*|c_ik|))
    # d/dc_ik = sign(c_ik) * p_k where p_k = exp(beta_sm*|c_ik|)/sum
    exp_abs = np.exp(abs_c * beta_sm - max_shifted[:, 0])  # (N,3)
    probs = exp_abs / np.sum(exp_abs, axis=1, keepdims=True)  # (N,3)
    d_softmax_dc = probs * np.sign(c)  # (N,3)
    # d(wall_i)/d(c_i) = -d_softmax_dc
    grad += -w_wall[:, np.newaxis] * d_softmax_dc

    # Pair contributions
    w_pair = weights[N:]  # (M,)
    # For pair (i,j), constraint = |c_i - c_j|/2
    # d/dc_i = (c_i - c_j) / (2*|c_i-c_j|)  ... but with soft-min weighting
    # Actually d(pair_ij)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|)
    # d(pair_ij)/d(c_j) = (c_j - c_i) / (2 * |c_i - c_j|)
    
    # Build mapping from pair index to (i,j)
    # This is expensive for large N, but N=100 so M=4950
    # We can vectorize
    ii, jj = iu  # (M,)
    diffs = c[ii] - c[jj]  # (M, 3)
    dists = np.linalg.norm(diffs, axis=1, keepdims=True)  # (M, 1)
    dists = np.where(dists < 1e-12, 1e-12, dists)
    # d(pair_ij)/d(c_i) = diffs / (2 * dists)
    # d(pair_ij)/d(c_j) = -diffs / (2 * dists)
    pair_grad_i = diffs / (2 * dists)  # (M, 3)
    pair_grad_j = -pair_grad_i

    # Scatter
    np.add.at(grad, ii, w_pair[:, np.newaxis] * pair_grad_i)
    np.add.at(grad, jj, w_pair[:, np.newaxis] * pair_grad_j)

    return soft_min_val, grad

def _kmc_step(c, T, dt, rng):
    """
    One KMC step: pick a random sphere, give it a random kick,
    then do a few iterations of constraint-targeted push-apart.
    Returns new configuration.
    """
    c_new = c.copy()
    idx = rng.integers(0, N)
    # Random kick
    kick = rng.normal(0, dt, size=3)
    c_new[idx] += kick

    # Constraint-targeted push-apart: find the tightest constraints and resolve them
    for _ in range(3):
        # Compute current radius
        wall = CUBE_HALF - np.max(np.abs(c_new), axis=1)
        d = cdist(c_new, c_new)
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
        r = min(np.min(wall), min_pair)
        
        if r <= 0:
            # Sphere is outside or overlapping; push it back
            # Find which constraint is violated
            min_wall_idx = np.argmin(wall)
            min_pair_iu = np.argmin(d)
            ii, jj = np.unravel_index(min_pair_iu, (N, N))
            
            if wall[min_wall_idx] < min_pair:
                # Wall constraint
                ci = c_new[min_wall_idx]
                # Push towards center
                c_new[min_wall_idx] = ci * 0.9 + rng.normal(0, 0.01, size=3)
            else:
                # Pair constraint
                ci = c_new[ii]
                cj = c_new[jj]
                diff = ci - cj
                dist = np.linalg.norm(diff)
                if dist > 1e-12:
                    push = diff / dist * 0.01
                    c_new[ii] += push
                    c_new[jj] -= push
                else:
                    c_new[ii] += rng.normal(0, 0.01, size=3)
        else:
            # Find the K tightest constraints and try to expand
            # Get all constraints
            all_constr = np.concatenate([wall, d[np.triu_indices(N, k=1)] / 2.0])
            # Find the 5 tightest
            K = 5
            tightest_idx = np.argpartition(all_constr, K)[:K]
            # For each tightest constraint, apply a small push apart
            for ti in tightest_idx:
                if ti < N:
                    # Wall constraint for sphere ti
                    ci = c_new[ti]
                    # Find which wall is closest
                    abs_ci = np.abs(ci)
                    wall_dir = np.sign(ci) * np.eye(3)[np.argmax(abs_ci)]
                    # Push inward
                    c_new[ti] -= wall_dir * 0.005 * rng.random()
                else:
                    # Pair constraint
                    pair_idx = ti - N
                    ii, jj = np.unravel_index(np.triu_indices(N, k=1)[0][pair_idx] * N + np.triu_indices(N, k=1)[1][pair_idx], (N, N))
                    # Actually let's just use the indices directly
                    ii = np.triu_indices(N, k=1)[0][pair_idx]
                    jj = np.triu_indices(N, k=1)[1][pair_idx]
                    ci = c_new[ii]
                    cj = c_new[jj]
                    diff = ci - cj
                    dist = np.linalg.norm(diff)
                    if dist > 1e-12:
                        push = diff / dist * 0.005 * rng.random()
                        c_new[ii] += push
                        c_new[jj] -= push
    return c_new

def _gradient_polish(c, max_iter=200, lr=0.001, beta=50.0):
    """Polish using gradient ascent on soft-min radius."""
    c = c.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for it in range(max_iter):
        val, grad = _soft_min_radius(c, beta)
        c_new = c + lr * grad
        # Project back into cube
        c_new = np.clip(c_new, -CUBE_HALF, CUBE_HALF)
        # Check if improved
        r_new = _compute_radius(c_new)
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
        c = c_new
        # Adaptive learning rate
        if it % 50 == 49:
            lr *= 0.5
    
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Initialize with FCC lattice structure
    # FCC has packing fraction ~0.74, cubic ~0.52
    # Use a subset of FCC lattice points
    # FCC lattice: points at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in unit cell
    # Scale to fit in [-0.5, 0.5]^3
    
    # Generate FCC lattice points
    fcc_offsets = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ])
    
    # Use a 4x4x4 supercell of FCC
    m = 4
    lattice_pts = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                for off in fcc_offsets:
                    pt = np.array([ix, iy, iz]) + off
                    lattice_pts.append(pt)
    
    lattice_pts = np.array(lattice_pts)  # (64,)
    
    # Scale to fit in [-0.5, 0.5]^3
    # Current range is [0, 4+0.5] = [0, 4.5]
    # Scale by 1/4.5 and shift
    scale = 1.0 / (m + 0.5)
    lattice_pts = lattice_pts * scale - 0.5 - 0.25 * scale  # Center it
    # Actually let's just normalize
    lattice_pts = (lattice_pts - lattice_pts.min(axis=0)) / (lattice_pts.max(axis=0) - lattice_pts.min(axis=0))
    lattice_pts = lattice_pts * 0.9 - 0.45  # Slightly inset from walls
    
    # We have 64 points, need 100. Add more by using a denser grid
    # Or just use a 5x5x5 cubic grid as fallback and mix
    # Let's use a 5x5x5 grid = 125 points, take first 100
    ticks5 = (np.arange(5) + 0.5) / 5 - 0.5
    grid5 = np.array([(x, y, z) for z in ticks5 for y in ticks5 for x in ticks5])
    
    # Mix: use FCC for first 64, then add from grid5
    c = np.vstack([lattice_pts, grid5[64:100]])
    c = c[:N]
    
    # Initial radius
    r_current = _compute_radius(c)
    best_c = c.copy()
    best_r = r_current
    
    # KMC phase
    T = 0.01
    dt = 0.02
    kmc_iter = 0
    
    while time.time() < deadline - 30.0:  # Leave 30s for polish
        c_new = _kmc_step(c, T, dt, rng)
        r_new = _compute_radius(c_new)
        
        # Metropolis accept/reject
        if r_new > r_current or rng.random() < np.exp((r_new - r_current) / T):
            c = c_new
            r_current = r_new
            if r_new > best_r:
                best_r = r_new
                best_c = c.copy()
        
        # Cool down
        T *= 0.999
        dt *= 0.9995
        kmc_iter += 1
        
        # Every 500 iterations, try a gradient polish
        if kmc_iter % 500 == 0 and time.time() < deadline - 60.0:
            polished, r_polished = _gradient_polish(c, max_iter=100, lr=0.0005, beta=30.0)
            if r_polished > best_r:
                best_r = r_polished
                best_c = polished.copy()
                c = polished.copy()
                r_current = r_polished
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        polished, r_polished = _gradient_polish(best_c, max_iter=300, lr=0.0001, beta=100.0)
        if r_polished > best_r:
            best_r = r_polished
            best_c = polished.copy()
    
    # Ensure validity
    best_c = np.clip(best_c, -CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)
    
    return best_c
