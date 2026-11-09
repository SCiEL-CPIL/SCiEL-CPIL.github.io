"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import random

DEADLINE_SECONDS = 115.0  # safety margin before 120s kill

def _score(centers):
    """Compute the packing score: min of (min pairwise dist / 2, min distance to boundary)."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check inside cube [-0.5, 0.5]^3
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    # Pairwise distances
    d = pdist(centers, metric='euclidean')
    min_pair = d.min() if d.size > 0 else 1.0
    # Distance to boundary
    dist_to_boundary = 0.5 - np.abs(centers).max(axis=1)
    min_boundary = dist_to_boundary.min()
    return min(min_pair / 2.0, min_boundary)

def _compute_violations(centers, r_target):
    """Compute violations for all pairs and boundaries."""
    # Pairwise
    d = pdist(centers, metric='euclidean')
    v_pairs = np.maximum(0.0, 2.0 * r_target - d)
    
    # Boundary
    dist_to_boundary = 0.5 - np.abs(centers)
    v_boundary = np.maximum(0.0, r_target - dist_to_boundary)
    
    return v_pairs, v_boundary

def _soft_max_objective(centers_flat, r_target, beta):
    centers = centers_flat.reshape(N, 3)
    v_pairs, v_boundary = _compute_violations(centers, r_target)
    
    # Combine violations
    v_all = np.concatenate([v_pairs, v_boundary.flatten()])
    if v_all.size == 0:
        return 0.0
        
    v_max = np.max(v_all)
    if v_max == 0.0:
        return 0.0
        
    # Soft max
    exp_terms = np.exp(beta * (v_all - v_max))
    log_sum_exp = np.log(np.sum(exp_terms)) + beta * v_max
    return log_sum_exp / beta

def _soft_max_gradient(centers_flat, r_target, beta):
    centers = centers_flat.reshape(N, 3)
    v_pairs, v_boundary = _compute_violations(centers, r_target)
    
    v_all = np.concatenate([v_pairs, v_boundary.flatten()])
    n_pairs = v_pairs.size
    n_boundary = v_boundary.size
    
    if np.max(v_all) == 0.0:
        return np.zeros_like(centers_flat)
        
    v_max = np.max(v_all)
    exp_terms = np.exp(beta * (v_all - v_max))
    sum_exp = np.sum(exp_terms)
    
    # Probabilities
    p_all = exp_terms / sum_exp
    
    # Split probabilities back
    p_pairs = p_all[:n_pairs]
    p_boundary = p_all[n_pairs:]
    
    grad = np.zeros_like(centers_flat)
    
    # Pairwise gradients
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist_mat, 1.0)
    
    # Map p_pairs (condensed) to full matrix
    p_mat = np.zeros((N, N))
    i_idx, j_idx = np.triu_indices(N, k=1)
    p_mat[i_idx, j_idx] = p_pairs
    p_mat[j_idx, i_idx] = p_pairs
    
    # Compute factor for gradient
    factor = np.zeros_like(dist_mat)
    active_mask = (dist_mat > 0) & (dist_mat < 2.0 * r_target)
    factor[active_mask] = p_mat[active_mask] * (2.0 * r_target - dist_mat[active_mask]) / dist_mat[active_mask]
    
    # grad[i] = sum_j factor[i,j] * (c_i - c_j)
    sum_factor = factor.sum(axis=1, keepdims=True)
    weighted_c = factor @ centers
    grad_pairs = centers * sum_factor - weighted_c
    
    # Boundary gradients
    p_boundary_mat = p_boundary.reshape(N, 3)
    active_boundary = v_boundary > 0
    grad_boundary = p_boundary_mat * np.sign(centers)
    grad_boundary[~active_boundary] = 0.0
    
    grad = (grad_pairs + grad_boundary).flatten()
    return grad

def _local_search_softmax(centers, r_target, beta_start=1.0, beta_end=50.0, max_iter=200):
    x0 = centers.flatten()
    current_x = x0
    num_steps = 5
    
    for step in range(num_steps):
        beta = beta_start * (beta_end / beta_start) ** (step / max(1, num_steps - 1))
        
        def obj(x):
            return _soft_max_objective(x, r_target, beta)
        
        def grad(x):
            return _soft_max_gradient(x, r_target, beta)
            
        try:
            result = minimize(obj, current_x, jac=grad, method='L-BFGS-B',
                              bounds=[(-0.5, 0.5)] * (N * 3),
                              options={'maxiter': max_iter // num_steps, 'ftol': 1e-10, 'gtol': 1e-8})
            current_x = result.x
        except Exception:
            break
            
    return current_x.reshape(N, 3)

def construct():
    start_time = time.time()
    best_score = 0.0
    best_centers = None
    
    rng = np.random.default_rng(42)
    n_starts = 10
    
    for i in range(n_starts):
        if time.time() - start_time > DEADLINE_SECONDS:
            break
            
        centers = rng.uniform(-0.4, 0.4, size=(N, 3))
        r_target = 0.115
        
        try:
            improved = _local_search_softmax(centers, r_target, beta_start=1.0, beta_end=50.0, max_iter=200)
            s = _score(improved)
            if s > best_score:
                best_score = s
                best_centers = improved.copy()
        except Exception:
            pass
        
    if best_centers is None:
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N].copy()
        best_score = _score(best_centers)
        
    if time.time() - start_time < DEADLINE_SECONDS - 5:
         try:
             r_final = best_score * 1.01
             improved = _local_search_softmax(best_centers, r_final, beta_start=10.0, beta_end=100.0, max_iter=300)
             s = _score(improved)
             if s > best_score:
                 best_score = s
                 best_centers = improved.copy()
         except Exception:
             pass

    best_centers = np.clip(best_centers, -0.5 + 1e-9, 0.5 - 1e-9)
    return best_centers
# EVOLVE-BLOCK-END
