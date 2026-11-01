"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial.distance import cdist
import time

N = 100
DEADLINE = 140.0  # seconds, safety margin before 150s kill


def _score(centers):
    """Compute the radius for a given set of centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check bounds
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    # Pairwise distances
    D = cdist(centers, centers)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    # Distance to walls
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    r = min(min_pair / 2.0, min_wall)
    return r


def _logsumexp_softmin(values, beta):
    """Compute soft-min via log-sum-exp."""
    # soft-min(x) = -1/beta * log(sum(exp(-beta * x)))
    neg_beta_vals = -beta * values
    max_val = np.max(neg_beta_vals)
    if np.isinf(max_val):
        return -np.inf
    exp_sum = np.sum(np.exp(neg_beta_vals - max_val))
    return -(max_val + np.log(exp_sum)) / beta


def _soft_min_objective(centers, beta):
    """
    Objective to maximize: soft approximation of min(min_pair/2, min_wall).
    We maximize this, which is equivalent to minimizing its negative.
    """
    D = cdist(centers, centers)
    np.fill_diagonal(D, np.inf)
    pair_terms = D / 2.0  # we want min of these
    
    wall_terms = 0.5 - np.abs(centers)
    wall_terms = np.min(wall_terms, axis=1)  # shape (N,)
    
    # Combine all terms into one vector
    all_terms = np.concatenate([pair_terms.flatten(), wall_terms.flatten()])
    
    # Soft-min
    return _logsumexp_softmin(all_terms, beta)


def _initial_grid():
    """Create an initial grid-based configuration."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _random_initial():
    """Random initial configuration within bounds."""
    return np.random.uniform(-0.45, 0.45, size=(N, 3))


def _gradient_ascent(centers, beta, max_iter=200, lr=0.01, deadline=None):
    """
    Gradient ascent on the soft-min objective.
    Returns improved centers and the final soft objective value.
    """
    centers = centers.copy()
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute objective
        obj = _soft_min_objective(centers, beta)
        
        # Compute gradient numerically or analytically
        # Let's use analytical gradient for speed
        D = cdist(centers, centers)
        np.fill_diagonal(D, np.inf)
        pair_terms = D / 2.0
        
        wall_terms = 0.5 - np.abs(centers)
        wall_terms_min_idx = np.argmin(wall_terms, axis=1)
        wall_vals = np.min(wall_terms, axis=1)
        
        all_terms = np.concatenate([pair_terms.flatten(), wall_vals.flatten()])
        
        # Soft-min gradient: d/dx softmin = -exp(-beta*x) / sum(exp(-beta*x))
        neg_beta_vals = -beta * all_terms
        max_val = np.max(neg_beta_vals)
        exps = np.exp(neg_beta_vals - max_val)
        sum_exps = np.sum(exps)
        weights = exps / sum_exps
        
        # Gradient of softmin w.r.t. x is -weights (because softmin = -1/beta * log(sum(exp(-beta*x))))
        # Actually: d/dx_i [-1/beta * log(sum_j exp(-beta*x_j))] = -1/beta * (-beta * exp(-beta*x_i) / sum) = exp(-beta*x_i) / sum = weights_i
        # So gradient of softmin w.r.t. x_i is +weights_i
        # But we want to MAXIMIZE softmin, so we move in direction of +gradient
        
        grad = np.zeros_like(centers)
        
        # Pair terms: each pair (i,j) contributes to both center i and j
        pair_weights = weights[:N*N].reshape(N, N)
        # For pair term (i,j), the term is |c_i - c_j|/2
        # d/dc_i (|c_i - c_j|/2) = (c_i - c_j) / (2 * |c_i - c_j|) = (c_i - c_j) / D[i,j]
        for i_idx in range(N):
            for j_idx in range(N):
                if i_idx == j_idx:
                    continue
                w = pair_weights[i_idx, j_idx]
                if w > 0 and D[i_idx, j_idx] > 1e-15:
                    diff = centers[i_idx] - centers[j_idx]
                    grad[i_idx] += w * diff / D[i_idx, j_idx]
        
        # Wall terms: for each center i, the min wall distance
        wall_weights = weights[N*N:]
        for i_idx in range(N):
            w = wall_weights[i_idx]
            if w > 0:
                # d/dc_i (0.5 - |c_i_k|) where k is the min coordinate
                k = wall_terms_min_idx[i_idx]
                grad[i_idx, k] -= w * np.sign(centers[i_idx, k])
        
        # Gradient ascent step
        centers += lr * grad
        
        # Project back to valid region
        centers = np.clip(centers, -0.499999, 0.499999)
    
    return centers


def _local_search(centers, deadline):
    """
    Local search: iteratively move each center to improve the minimum distance.
    """
    centers = centers.copy()
    best_r = _score(centers)
    
    max_rounds = 50
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            
            # Try to find a better position for center i
            current_center = centers[i].copy()
            current_r = _score(centers)
            
            # Perturb and try
            best_pos = current_center
            best_local_r = current_r
            
            for _ in range(20):
                trial = centers.copy()
                trial[i] = current_center + np.random.normal(0, 0.01, size=3)
                trial[i] = np.clip(trial[i], -0.499999, 0.499999)
                trial_r = _score(trial)
                if trial_r > best_local_r + 1e-12:
                    best_local_r = trial_r
                    best_pos = trial[i].copy()
            
            if best_local_r > current_r + 1e-12:
                centers[i] = best_pos
                improved = True
        
        if not improved:
            break
    
    return centers


def construct():
    start_time = time.time()
    
    # Start with a good initial configuration
    centers = _initial_grid()
    best_centers = centers.copy()
    best_score = _score(best_centers)
    
    # Phase 1: Gradient ascent with increasing beta
    betas = [5, 10, 20, 50, 100]
    for beta in betas:
        if time.time() > deadline_guard():
            break
        centers = _gradient_ascent(centers, beta, max_iter=100, lr=0.005, deadline=deadline_guard())
        s = _score(centers)
        if s > best_score:
            best_score = s
            best_centers = centers.copy()
    
    # Phase 2: Local search refinement
    if time.time() < deadline_guard():
        centers = _local_search(best_centers, deadline=deadline_guard())
        s = _score(centers)
        if s > best_score:
            best_score = s
            best_centers = centers.copy()
    
    # Phase 3: Final polish with high-beta gradient ascent
    if time.time() < deadline_guard():
        centers = _gradient_ascent(best_centers, 200, max_iter=50, lr=0.001, deadline=deadline_guard())
        s = _score(centers)
        if s > best_score:
            best_score = s
            best_centers = centers.copy()
    
    # Final validation
    final_score = _score(best_centers)
    if final_score > 0 and np.all(np.isfinite(best_centers)) and np.all(np.abs(best_centers) <= 0.5):
        return best_centers
    else:
        return _initial_grid()


def deadline_guard():
    return DEADLINE
