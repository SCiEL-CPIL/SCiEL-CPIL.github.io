import numpy as np
import time
from scipy.optimize import minimize

N = 300
SEED = 12345
T_START = time.time()
DEADLINE = 110.0

def _score(points):
    """Return the feasible common radius (0.0 if invalid)."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    norm = np.linalg.norm(points, axis=1)
    r_bound = 1.0 - norm
    if np.min(r_bound) < 0.0:
        return 0.0
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    r_pair = 0.5 * np.min(d)
    r = min(r_bound.min(), r_pair)
    return float(r) if r > 0.0 else 0.0

def _lattice_seed(spacing):
    """Generate a hexagonal lattice and pick N points inside the disk."""
    while True:
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        keep = rad <= 1.0 - spacing / 2.0
        pts = pts[keep]
        if len(pts) >= N:
            order = np.argsort(np.hypot(pts[:, 0], pts[:, 1]), kind="stable")
            return pts[order[:N]].copy()
        spacing *= 0.97

def _soft_min_obj_grad(points, beta):
    """
    Compute the soft-min objective and its gradient.
    Objective: -softmin( concat(0.5 * dist_ij, 1 - norm_i) )
    We want to MINIMIZE this objective, which maximizes the soft-min radius.
    """
    N = points.shape[0]
    
    # 1. Pairwise distances
    diff = points[:, None, :] - points[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    dists = dists[mask]
    # We need to map back to indices or compute gradient directly
    # Let's compute gradient using the formula:
    # grad_p = sum_k w_k * dS_k/dp
    # S_k = 0.5 * dist_ij for pairs
    # S_k = 1 - norm_i for boundaries
    
    # Compute weights for pairs
    # exp(-beta * 0.5 * dists)
    # To avoid overflow/underflow, we can shift by max
    # But beta is moderate initially.
    
    # Vectorized gradient calculation for pairs
    # diff is (N, N, 2)
    # dists_full is (N, N)
    dists_full = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists_full, np.inf)
    
    # Weights for pairs
    # log_weights = -beta * 0.5 * dists_full
    # We need sum of exp over all terms (pairs + boundaries)
    
    # Boundary terms
    norms = np.linalg.norm(points, axis=1)
    bound_terms = 1.0 - norms
    
    # Combine all terms for the denominator of the weight calculation
    # We can compute the log-sum-exp for stability
    # Terms: 0.5*dists (N^2) and 1-norms (N)
    
    # For gradient, we need w_k = exp(-beta * S_k) / Sum(exp(-beta * S_j))
    
    # Let's compute the log-sum-exp of -beta * S
    # S_pairs = 0.5 * dists_full (with inf on diagonal)
    # S_bounds = 1 - norms
    
    # We'll compute the gradient by iterating over the terms? No, vectorize.
    
    # Gradient of S_pair_ij wrt p_i: 0.5 * (p_i - p_j) / dist_ij
    # Gradient of S_pair_ij wrt p_j: 0.5 * (p_j - p_i) / dist_ij
    
    # Gradient of S_bound_i wrt p_i: -p_i / norm_i
    
    # Let's compute the weights w_pair and w_bound
    # To be efficient, we can compute the "force" directly.
    
    # Let Z = Sum(exp(-beta * S))
    # w_pair_ij = exp(-beta * 0.5 * d_ij) / Z
    # w_bound_i = exp(-beta * (1 - r_i)) / Z
    
    # grad_p_i = Sum_j w_pair_ij * 0.5 * (p_i - p_j)/d_ij + w_bound_i * (-p_i/r_i)
    #           - Sum_j w_pair_ji * 0.5 * (p_j - p_i)/d_ji  (Note: w_pair_ji corresponds to term j,i which is same as i,j)
    # Actually, the sum over k in grad includes all k.
    # For a fixed point i, it receives contributions from all pairs (i,j) and (j,i) and its own boundary.
    # Since dist_ij = dist_ji, the pair term (i,j) and (j,i) are identical in value S, so weights are identical.
    # But in the sum over k, we sum over all unique pairs?
    # The objective is softmin of the set of values. The set contains 0.5*d_ij for i<j.
    # So there are N(N-1)/2 pair terms.
    
    # Let's stick to i<j pairs.
    # Indices for i<j
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_ij = dists_full[i_idx, j_idx]
    S_pairs = 0.5 * d_ij
    
    # Boundary terms
    S_bounds = 1.0 - norms
    
    # All S values
    S_all = np.concatenate([S_pairs, S_bounds])
    
    # Calculate weights
    # Use log-sum-exp for stability
    max_S = np.max(S_all)
    # We want exp(-beta * S). Let u = -beta * S.
    u = -beta * S_all
    max_u = np.max(u)
    exp_u = np.exp(u - max_u)
    Z = np.sum(exp_u)
    weights = exp_u / Z
    
    w_pairs = weights[:len(S_pairs)]
    w_bounds = weights[len(S_pairs):]
    
    # Objective value
    obj = -1.0 / beta * (max_u + np.log(Z))
    
    # Gradient
    grad = np.zeros_like(points)
    
    # Pair contributions
    # For each pair (i, j) with weight w:
    # grad[i] += w * 0.5 * (p_i - p_j) / d_ij
    # grad[j] += w * 0.5 * (p_j - p_i) / d_ij
    
    # Vectorized:
    # diff_ij = points[i_idx] - points[j_idx]
    # unit_vec = diff_ij / d_ij[:, None]
    # contrib = w_pairs[:, None] * 0.5 * unit_vec
    
    diff_ij = points[i_idx] - points[j_idx]
    # Handle zero distance (shouldn't happen with init, but safe)
    d_safe = np.where(d_ij < 1e-9, 1e-9, d_ij)
    unit_vec = diff_ij / d_safe[:, None]
    
    contrib = w_pairs[:, None] * 0.5 * unit_vec
    
    # Accumulate to grad
    # np.add.at is slow. Use scatter or bincount?
    # Since N is small (300), we can just loop over unique indices? No.
    # We can use:
    # grad[i_idx] += contrib
    # grad[j_idx] -= contrib
    # This works with advanced indexing?
    # In NumPy, `grad[i_idx] += contrib` does NOT accumulate if indices repeat.
    # We need `np.add.at(grad, i_idx, contrib)` and `np.add.at(grad, j_idx, -contrib)`.
    
    np.add.at(grad, i_idx, contrib)
    np.add.at(grad, j_idx, -contrib)
    
    # Boundary contributions
    # grad[i] += w_bound_i * (-p_i / r_i)
    r_safe = np.where(norms < 1e-9, 1e-9, norms)
    bound_grad = -w_bounds[:, None] * points / r_safe[:, None]
    grad += bound_grad
    
    return obj, grad

def _optimize(points, beta, t_budget, rng):
    """Run L-BFGS-B optimization for a given beta."""
    start = time.time()
    
    def obj_grad(x):
        p = x.reshape(N, 2)
        val, g = _soft_min_obj_grad(p, beta)
        return val, g.flatten()
    
    # Bounds: keep points within [-1, 1] roughly? 
    # The soft-min penalizes going out, but bounds help L-BFGS.
    # Actually, if we bound to [-1,1], we might restrict optimal positions near boundary?
    # No, optimal positions are inside.
    # But if we bound strictly to [-1,1], and the true optimum is at 0.99, it's fine.
    # However, the "1 - norm" term becomes 0 at norm=1.
    # If we allow norm > 1, the term becomes negative, soft-min decreases, objective increases (bad).
    # So the optimizer will naturally stay inside.
    # We can set loose bounds like [-1.5, 1.5] to allow exploration but prevent divergence.
    
    bounds = [(-1.5, 1.5)] * (N * 2)
    
    # Use L-BFGS-B
    # maxiter depends on time budget
    # Estimate time per iter: ~1ms?
    # Let's allow 200 iterations.
    
    res = minimize(
        obj_grad, 
        points.flatten(), 
        method='L-BFGS-B', 
        jac=True, 
        bounds=bounds,
        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    return res.x.reshape(N, 2)

def construct():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # Initial seed
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    pts = _lattice_seed(spacing)
    best = pts.copy()
    best_r = _score(best)
    
    # Schedule of betas
    # Start low (smooth) to end high (sharp)
    betas = [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    for beta in betas:
        if time.time() - t0 > DEADLINE - 10.0:
            break
            
        # Small perturbation to escape symmetry?
        # Or just use current best.
        # Let's try a few restarts per beta if time permits?
        # No, L-BFGS is deterministic. 
        # To get diversity, we can jitter the start.
        
        # Actually, let's just run L-BFGS from the current best.
        # If it gets stuck, the next beta (sharper) might not help if we are in a local min.
        # So we should do multi-start.
        
        # Let's do 3 restarts per beta if time allows.
        # But L-BFGS is fast.
        
        for restart in range(3):
            if time.time() - t0 > DEADLINE - 5.0:
                break
                
            # Jitter
            jitter = 0.01 * rng.standard_normal((N, 2))
            start_pts = best + jitter
            # Clip to disk roughly
            rad = np.linalg.norm(start_pts, axis=1)
            mask = rad > 1.0
            if np.any(mask):
                start_pts[mask] = start_pts[mask] / rad[mask, None] * 0.99
                
            # Optimize
            try:
                opt_pts = _optimize(start_pts, beta, DEADLINE - (time.time() - t0), rng)
                r = _score(opt_pts)
                if r > best_r:
                    best_r = r
                    best = opt_pts
            except Exception:
                pass
                
    return best
