import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # Safety margin

def _score(centers):
    """Compute the exact packing score."""
    n = len(centers)
    # Pairwise distances
    min_pair = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d / 2.0 < min_pair:
                min_pair = d / 2.0
    # Boundary
    min_boundary = np.min(0.5 - np.abs(centers))
    return min(min_pair, min_boundary)

def _soft_min_obj_grad(centers, beta):
    """
    Vectorized soft-min objective and gradient.
    Objective: -soft_min(margins)
    Gradient: -d(soft_min)/dc
    """
    n = centers.shape[0]
    
    # 1. Pairwise margins
    # diffs[i, j] = centers[i] - centers[j]
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diffs**2, axis=2)
    # Avoid division by zero
    dists = np.sqrt(dists_sq + 1e-12)
    
    # Only use upper triangle
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    pair_margins = pair_dists / 2.0
    
    # Unit vectors for gradients
    # unit_diffs[i, j] = (centers[i] - centers[j]) / dists[i, j]
    unit_diffs = diffs / (dists[:, :, np.newaxis] + 1e-12)
    # Extract relevant unit vectors
    unit_diffs_pairs = unit_diffs[i_idx, j_idx] # shape (P, 3)
    
    # 2. Boundary margins
    bound_margins = 0.5 - np.abs(centers) # shape (N, 3)
    bound_margins_flat = bound_margins.flatten()
    
    # 3. Combine margins
    all_margins = np.concatenate([pair_margins, bound_margins_flat])
    
    # 4. Soft-min calculation
    # f = -1/beta * log(sum(exp(-beta * m)))
    neg_beta_m = -beta * all_margins
    max_val = np.max(neg_beta_m)
    exp_vals = np.exp(neg_beta_m - max_val)
    log_sum_exp = max_val + np.log(np.sum(exp_vals))
    soft_min_val = -log_sum_exp / beta
    
    # Objective is -soft_min_val
    obj_val = -soft_min_val
    
    # 5. Gradient calculation
    # Weights w_i = exp(-beta*m_i) / sum(exp(...))
    weights = exp_vals / np.sum(exp_vals)
    
    # Split weights
    n_pairs = len(pair_margins)
    w_pairs = weights[:n_pairs]
    w_bounds = weights[n_pairs:].reshape(n, 3)
    
    # Gradient for pairs
    # d(pair_margin)/d(c_i) = 0.5 * unit_diffs[i, j]
    # d(pair_margin)/d(c_j) = -0.5 * unit_diffs[i, j]
    # We need to accumulate these weighted by w_pairs
    
    # Create a tensor to hold contributions
    # contrib[i, j, :] will be the contribution to gradient of c_i from pair (i,j)
    # But we have flattened indices.
    
    # Efficient accumulation using np.add.at or similar
    # grad_pairs = np.zeros((n, 3))
    # For each pair k (i, j):
    #   grad_pairs[i] += 0.5 * w_pairs[k] * unit_diffs_pairs[k]
    #   grad_pairs[j] -= 0.5 * w_pairs[k] * unit_diffs_pairs[k]
    
    # Vectorized approach:
    # We can't easily do this with pure broadcasting without a loop or advanced indexing.
    # Given N=50, P=1225, a Python loop over P is fast enough (~1ms).
    # But let's try to vectorize if possible.
    # We can use np.add.at
    grad_pairs = np.zeros((n, 3))
    
    # Contributions to i
    # w_pairs[:, np.newaxis] * unit_diffs_pairs
    contrib_i = 0.5 * w_pairs[:, np.newaxis] * unit_diffs_pairs
    contrib_j = -0.5 * w_pairs[:, np.newaxis] * unit_diffs_pairs
    
    np.add.at(grad_pairs, i_idx, contrib_i)
    np.add.at(grad_pairs, j_idx, contrib_j)
    
    # Gradient for boundaries
    # d(bound_margin)/d(c_ik) = -sign(c_ik)
    grad_bounds = -np.sign(centers) * w_bounds
    
    total_grad = grad_pairs + grad_bounds
    
    # We want to minimize -soft_min, so gradient of objective is -total_grad
    grad_obj = -total_grad
    
    return obj_val, grad_obj

def _optimize(centers, beta, max_iter=200, time_limit=5.0):
    t0 = time.time()
    
    def objective(x):
        c = x.reshape(-1, 3)
        # Clip to stay slightly inside to avoid numerical issues with abs/sign
        c = np.clip(c, -0.4999, 0.4999)
        val, _ = _soft_min_obj_grad(c, beta)
        return val
        
    def gradient(x):
        c = x.reshape(-1, 3)
        c = np.clip(c, -0.4999, 0.4999)
        _, g = _soft_min_obj_grad(c, beta)
        return g
        
    x0 = centers.flatten()
    res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
    
    elapsed = time.time() - t0
    best_centers = np.clip(res.x.reshape(-1, 3), -0.5, 0.5)
    
    return best_centers

def _generate_grid_init(n=N):
    # 4x4x4 grid = 64 points. Take 50.
    ticks = np.linspace(-0.45, 0.45, 4)
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n].copy()

def _generate_bcc_init(n=N):
    # Generate BCC lattice points
    # M=3 cubes per side
    M = 3
    # Corners
    corners = []
    for x in range(M + 1):
        for y in range(M + 1):
            for z in range(M + 1):
                corners.append([x, y, z])
    # Centers
    centers = []
    for x in range(M):
        for y in range(M):
            for z in range(M):
                centers.append([x + 0.5, y + 0.5, z + 0.5])
    
    pts = np.array(corners + centers)
    # Scale to [-0.5, 0.5]
    # Range is 0 to M. Scale factor 1/M. Shift -0.5.
    pts = pts / M - 0.5
    
    # Select n points with max min-distance (greedy)
    # Start with a random point or corner
    # To be deterministic, pick the first point
    selected = [0]
    remaining = list(range(1, len(pts)))
    
    while len(selected) < n and remaining:
        # Find point in remaining with max min distance to selected
        best_idx = -1
        best_dist = -1
        for idx in remaining:
            p = pts[idx]
            min_d = np.min(np.linalg.norm(p - pts[selected], axis=1))
            if min_d > best_dist:
                best_dist = min_d
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)
        
    return pts[selected]

def construct():
    t_start = time.time()
    rng = np.random.RandomState(42)
    
    best_centers = None
    best_score = -np.inf
    
    # Initial candidates
    inits = []
    inits.append(_generate_grid_init(N))
    inits.append(_generate_bcc_init(N))
    
    for _ in range(5):
        # Random init
        c = rng.uniform(-0.45, 0.45, size=(N, 3))
        inits.append(c)
        
    betas = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    
    for init in inits:
        current_centers = init.copy()
        
        for beta in betas:
            if time.time() - t_start > DEADLINE - 10:
                break
            
            try:
                # Reduce max_iter for higher beta to save time
                max_iter = 200 if beta < 1000 else 100
                optimized = _optimize(current_centers, beta, max_iter=max_iter)
                current_centers = optimized
            except Exception:
                break
            
            score = _score(current_centers)
            if score > best_score:
                best_score = score
                best_centers = current_centers.copy()
                
        if time.time() - t_start > DEADLINE - 10:
            break
    
    # Final polish
    if best_centers is not None and time.time() - t_start < DEADLINE - 5:
        try:
            # Very high beta, tight tolerance
            polished = _optimize(best_centers, 20000, max_iter=300)
            score = _score(polished)
            if score > best_score:
                best_score = score
                best_centers = polished
        except Exception:
            pass
    
    if best_centers is None:
        best_centers = _generate_grid_init(N)
        
    # Ensure valid output
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
