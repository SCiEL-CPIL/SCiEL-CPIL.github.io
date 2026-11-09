"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(centers):
    """Compute the radius for a set of centers in the unit cube [-0.5, 0.5]^3."""
    # Distance between all pairs
    # Use broadcasting for vectorized computation
    # For 50 points, this is 50x50 = 2500 distances, very fast
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    # Get minimum pairwise distance (excluding self)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    wall_dists = np.minimum(0.5 - np.abs(centers), axis=0) if centers.ndim > 1 else None
    # Actually need min over both axes
    min_wall_dist = np.min(0.5 - np.abs(centers))
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _soft_min_score(centers, beta):
    """
    Soft-min surrogate objective to minimize.
    We want to maximize the minimum distance, so we minimize a soft-min of the negative distances.
    Actually, let's think differently.
    
    We want to maximize r = min(min_pair_dist/2, min_wall_dist)
    
    Let's define the objective as the negative of a soft-min of the relevant distances.
    The relevant constraints are:
    1. For each pair (i,j): |c_i - c_j| >= 2r
    2. For each point i and wall k: 0.5 - |c_ik| >= r
    
    A common approach: minimize the log-sum-exp of the "violations" or use a soft-min.
    
    Let d_min be the minimum of all pairwise distances and wall distances.
    We want to maximize d_min.
    
    Soft approach: minimize -log(sum(exp(-beta * d_k))) where d_k are all the distances.
    This is equivalent to maximizing the soft-min of the distances.
    
    Actually, let's use: f(centers) = -logsumexp(-beta * d_k)
    where d_k are all pairwise distances and wall distances.
    
    Minimizing f is equivalent to maximizing the soft-min of d_k.
    """
    centers = centers.reshape(-1, 3)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists[np.triu_indices_from(dists, k=1)]
    
    # Wall distances
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)
    
    # Combine all distances
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Soft-min: -logsumexp(-beta * d)
    # To avoid overflow, use the max trick
    beta_d = beta * all_dists
    max_beta_d = np.max(beta_d)
    log_sum_exp = max_beta_d + np.log(np.sum(np.exp(beta_d - max_beta_d)))
    
    return -log_sum_exp

def _soft_min_grad(centers, beta):
    """
    Gradient of the soft-min objective.
    f = -logsumexp(-beta * d_k)
    df/dx_i = -sum_k (exp(-beta*d_k) / sum_j exp(-beta*d_j)) * (-beta * dd_k/dx_i)
             = beta * sum_k (p_k * dd_k/dx_i)
    where p_k = exp(-beta*d_k) / sum_j exp(-beta*d_j)
    """
    centers = centers.reshape(-1, 3)
    n = len(centers)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists[np.triu_indices_from(dists, k=1)]
    
    # Wall distances
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)
    
    # Combine
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Compute softmax weights
    beta_d = beta * all_dists
    max_beta_d = np.max(beta_d)
    exp_beta_d = np.exp(beta_d - max_beta_d)
    sum_exp = np.sum(exp_beta_d)
    p = exp_beta_d / sum_exp  # p_k = exp(-beta*d_k)/sum... wait, sign
    
    # Let me redo this carefully.
    # f = -logsumexp(-beta * d_k)
    # Let u_k = -beta * d_k
    # f = -log(sum(exp(u_k)))
    # df/du_k = -exp(u_k)/sum(exp(u_k)) = -q_k where q_k = exp(u_k)/sum(exp(u_j))
    # du_k/dx = -beta * dd_k/dx
    # df/dx = sum_k (df/du_k) * (du_k/dx) = sum_k (-q_k) * (-beta * dd_k/dx) = beta * sum_k q_k * dd_k/dx
    
    u = -beta * all_dists
    max_u = np.max(u)
    exp_u = np.exp(u - max_u)
    sum_exp_u = np.sum(exp_u)
    q = exp_u / sum_exp_u  # q_k = exp(u_k)/sum(exp(u_j))
    
    # Now compute gradient
    grad = np.zeros_like(centers)
    
    # Pairwise part
    i_idx, j_idx = np.triu_indices_from(dists, k=1)
    for k in range(len(pair_dists)):
        i = i_idx[k]
        j = j_idx[k]
        w = beta * q[k]
        # d(dist_ij)/d(c_i) = (c_i - c_j)/dist_ij
        # d(dist_ij)/d(c_j) = (c_j - c_i)/dist_ij
        d_ij = diff[i, j]
        dist_ij = dists[i, j]
        if dist_ij > 1e-12:
            grad[i] += w * d_ij / dist_ij
            grad[j] += w * (-d_ij) / dist_ij
    
    # Wall part
    for k in range(n):
        # Which wall is closest for point k?
        # wall_dists[k] = min(0.5 - |c_kx|, 0.5 - |c_ky|, 0.5 - |c_kz|)
        abs_c = np.abs(centers[k])
        wall_vals = 0.5 - abs_c
        min_wall_idx = np.argmin(wall_vals)
        w = beta * q[len(pair_dists) + k]
        # d(0.5 - |c_ik|)/d(c_ik) = -sign(c_ik)
        sign_c = np.sign(centers[k])
        d_wall = -sign_c
        d_wall[min_wall_idx] = -sign_c[min_wall_idx]
        # Only the closest wall contributes
        grad[k] += w * d_wall * 0  # This isn't right, only the min wall
        # Actually, for the min, only the argmin term has a gradient
        # Let me redo: wall_dist_k = 0.5 - |c_{k,m}| where m = argmin
        # d(wall_dist_k)/d(c_{k,m}) = -sign(c_{k,m})
        # d(wall_dist_k)/d(c_{k,n}) = 0 for n != m
        grad[k] *= 0  # Reset, this is getting messy
    
    # Let me use a simpler approach: numerical gradient or just use scipy's numerical gradient
    return grad

def construct():
    """
    Find the optimal placement of 50 equal spheres in a unit cube.
    Uses annealed soft-min surrogate with L-BFGS-B optimization.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial guess: simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    best_centers = centers.copy()
    best_score = _score(centers)
    
    # Annealing schedule
    betas = [10, 20, 50, 100, 200, 500, 1000]
    
    for beta in betas:
        if time.time() > deadline - 10.0:
            break
        
        # Optimize using L-BFGS-B
        # We minimize the soft-min objective
        def objective(x):
            return _soft_min_score(x, beta)
        
        # Use numerical gradient for simplicity and robustness
        try:
            result = minimize(
                objective,
                centers,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * 3 * N,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < objective(centers):
                new_centers = result.x
                new_score = _score(new_centers)
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_centers.copy()
                centers = new_centers
        except Exception:
            pass
    
    # Final local refinement with higher beta
    if time.time() < deadline - 5.0:
        for beta in [2000, 5000]:
            if time.time() > deadline - 5.0:
                break
            
            def objective(x):
                return _soft_min_score(x, beta)
            
            try:
                result = minimize(
                    objective,
                    best_centers,
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * 3 * N,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                
                if result.fun < objective(best_centers):
                    new_centers = result.x
                    new_score = _score(new_centers)
                    if new_score > best_score:
                        best_score = new_score
                        best_centers = new_centers.copy()
            except Exception:
                pass
    
    return best_centers
# EVOLVE-BLOCK-END
