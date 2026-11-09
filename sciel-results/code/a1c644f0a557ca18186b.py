"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

# Safety margin for the 150s limit (120s per task contract, but evaluator kills at 150s)
# We aim to finish within 110s to be safe.
TIME_LIMIT = 110.0
_start_time = time.time()

def _time_left():
    return TIME_LIMIT - (time.time() - _start_time)

def _compute_radius(centers):
    """
    Compute the exact score: largest common radius for given centers.
    r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|))
    """
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5 + 1e-10):
        return 0.0
    
    # Pairwise distances
    # Use cdist for efficiency if available, otherwise manual vectorization
    try:
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(centers, centers)
    except ImportError:
        # Fallback vectorized distance calculation
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    
    # Exclude self-distances
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair_dist = np.min(dist_matrix)
    
    # Wall distances
    # Distance to nearest wall: min(0.5 - x, 0.5 - y, x + 0.5, y + 0.5)
    # Since square is [-0.5, 0.5], distance to wall for point (x,y) is min(0.5-x, 0.5-y, 0.5+x, 0.5+y)
    # which is 0.5 - max(|x|, |y|)
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _softmin_obj_and_grad(centers, beta=10.0):
    """
    Compute the smooth negative radius objective and its gradient.
    Objective: -softmin(D) where D contains pair_dists/2 and wall_dists.
    softmin(D) = -1/beta * ln(sum(exp(-beta * d_k)))
    -softmin(D) = 1/beta * ln(sum(exp(-beta * d_k)))
    
    Gradient:
    Let S = sum(exp(-beta * d_k))
    f = 1/beta * ln(S)
    df/dx = 1/beta * (1/S) * sum(-beta * exp(-beta * d_k) * grad_d_k)
    df/dx = - sum(exp(-beta * d_k) * grad_d_k) / S
    """
    # 1. Compute distances
    try:
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(centers, centers)
    except ImportError:
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    
    # We only need i < j for pairs to save memory/time, but cdist gives full matrix.
    # Let's extract upper triangle indices
    iu = np.triu_indices(N, k=1)
    pair_dists = dist_matrix[iu]
    
    # Wall distances
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    
    # Combine into a single vector of distances
    # Note: Pair distance contribution to radius is d/2. Wall is d.
    # So D_vector = [pair_dists / 2.0, wall_dists]
    D = np.concatenate([pair_dists / 2.0, wall_dists])
    
    # Compute weights w_k = exp(-beta * D_k)
    # To avoid overflow/underflow, subtract max
    D_max = np.max(D)
    exp_terms = np.exp(-beta * (D - D_max))
    S = np.sum(exp_terms)
    
    # Objective value: 1/beta * ln(S) + D_max (since we shifted)
    # Actually, softmin(D) = D_max - 1/beta * ln(sum(exp(-beta*(D-D_max))))
    # We want to MINIMIZE -softmin(D) = -D_max + 1/beta * ln(S)
    # But for gradient, the constant shift doesn't matter.
    # Let's just compute the gradient.
    
    grad_centers = np.zeros_like(centers)
    
    # Weights normalized
    weights = exp_terms / S
    
    # Split weights back into pair and wall parts
    n_pairs = len(pair_dists)
    w_pairs = weights[:n_pairs]
    w_walls = weights[n_pairs:]
    
    # Gradient of wall distances:
    # d_i = 0.5 - max(|x_i|, |y_i|)
    # This is non-differentiable at |x_i|=|y_i|. 
    # Subgradient: if |x|>|y|, grad = (-sign(x), 0). If |y|>|x|, grad = (0, -sign(y)).
    # If equal, average.
    abs_x = np.abs(centers[:, 0])
    abs_y = np.abs(centers[:, 1])
    
    # For simplicity and robustness, we can use a smooth approximation for wall distance too?
    # Or just use the subgradient. The soft-min smooths the selection, but the individual d_k gradients are still non-smooth at diagonals.
    # However, the probability of hitting exactly |x|=|y| is low.
    
    grad_walls = np.zeros_like(centers)
    # Vectorized subgradient for wall distance
    # mask_x = abs_x > abs_y
    # mask_y = abs_y > abs_x
    # mask_eq = abs_x == abs_y
    # This is tricky to vectorize perfectly without branching.
    # Alternative: Use a smooth approximation for the "distance to boundary" term?
    # Let's stick to the standard subgradient.
    
    for i in range(N):
        if abs_x[i] > abs_y[i]:
            grad_walls[i, 0] = -np.sign(centers[i, 0])
            grad_walls[i, 1] = 0.0
        elif abs_y[i] > abs_x[i]:
            grad_walls[i, 0] = 0.0
            grad_walls[i, 1] = -np.sign(centers[i, 1])
        else:
            # Equal, use average
            grad_walls[i, 0] = -0.5 * np.sign(centers[i, 0])
            grad_walls[i, 1] = -0.5 * np.sign(centers[i, 1])
            
    # Accumulate wall gradient contribution
    # grad_total += sum( w_walls[i] * grad_walls[i] )
    # Note: The objective is -softmin. 
    # f = -softmin(D)
    # grad f = - sum( w_k * grad d_k ) ... wait.
    # softmin(D) = min_k d_k (approx)
    # -softmin(D) = -min_k d_k
    # If we increase d_k, we decrease -d_k. So we want to move in direction of -grad d_k?
    # Let's re-verify signs.
    # We want to MAXIMIZE radius r.
    # r = min(D).
    # We want to move x to increase min(D).
    # Gradient of min(D) is grad d_k where k is argmin.
    # So we should move in direction of grad d_k.
    # Our objective function for minimization is f = -softmin(D).
    # grad f = - grad(softmin(D)) = - sum(w_k * grad d_k).
    # So to minimize f, we do x_new = x - step * grad_f = x + step * sum(w_k * grad d_k).
    # This matches "move in direction of weighted average of gradients of distances".
    
    grad_centers += np.sum(w_walls[:, np.newaxis] * grad_walls, axis=0)
    
    # Gradient of pair distances:
    # d_ij = ||c_i - c_j||
    # grad_{c_i} d_ij = (c_i - c_j) / d_ij
    # grad_{c_j} d_ij = (c_j - c_i) / d_ij
    # We need to accumulate these weighted by w_pairs.
    
    # This part is expensive if done in a loop.
    # We have iu indices.
    i_idx, j_idx = iu
    # diffs = centers[i_idx] - centers[j_idx]
    # dists = pair_dists
    # unit_vecs = diffs / dists[:, np.newaxis]
    # 
    # For each pair (i,j) with weight w:
    # grad_c[i] += w * unit_vec
    # grad_c[j] -= w * unit_vec
    
    # Vectorized accumulation using np.add.at
    diffs = centers[i_idx] - centers[j_idx]
    # Avoid division by zero (though centers should be distinct)
    safe_dists = np.where(pair_dists < 1e-8, 1e-8, pair_dists)
    unit_vecs = diffs / safe_dists[:, np.newaxis]
    
    # Weighted vectors
    weighted_vecs = w_pairs[:, np.newaxis] * unit_vecs
    
    np.add.at(grad_centers, i_idx, weighted_vecs)
    np.add.at(grad_centers, j_idx, -weighted_vecs)
    
    # The gradient of the objective f = -softmin is -sum(w * grad_d)
    # So we return -grad_centers for the optimizer to minimize f.
    # Or we can just return the "descent direction" for maximizing radius.
    # Let's return the gradient of the objective function f(x) = -softmin(D).
    # grad_f = - grad_centers
    return -grad_centers

def _anneal_optimize(centers, beta_start=20.0, beta_end=1.0, n_steps=500, step_size=0.01):
    """
    Local optimization using annealing of the soft-min temperature.
    """
    centers = centers.copy()
    current_radius = _compute_radius(centers)
    
    for step in range(n_steps):
        if _time_left() < 5.0:
            break
        
        # Anneal beta
        t = step / max(1, n_steps - 1)
        beta = beta_start * (beta_end / beta_start) ** t
        
        # Compute gradient
        try:
            grad = _softmin_obj_and_grad(centers, beta)
        except Exception:
            break
            
        if np.max(np.abs(grad)) < 1e-8:
            break
            
        # Line search
        found_step = False
        for _ in range(20):
            new_centers = centers + step_size * grad
            # Clip to square
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            new_radius = _compute_radius(new_centers)
            
            if new_radius > current_radius + 1e-9:
                centers = new_centers
                current_radius = new_radius
                found_step = True
                break
            else:
                step_size *= 0.5
                if step_size < 1e-6:
                    break
        
        if not found_step:
            # If we can't improve, try a smaller step or break
            step_size *= 0.5
            if step_size < 1e-6:
                break
                
    return centers, current_radius

def _grid_initial(m_cols):
    """Generate a grid initialization."""
    m_rows = int(np.ceil(N / m_cols))
    xs = (np.arange(m_cols) + 0.5) / m_cols - 0.5
    ys = (np.arange(m_rows) + 0.5) / m_rows - 0.5
    
    centers = []
    for y in ys:
        for x in xs:
            centers.append([x, y])
            if len(centers) >= N:
                break
        if len(centers) >= N:
            break
            
    # Pad if necessary (shouldn't happen with ceil, but just in case)
    while len(centers) < N:
        centers.append([np.random.uniform(-0.45, 0.45), np.random.uniform(-0.45, 0.45)])
        
    return np.array(centers[:N])

def _hex_initial():
    """Generate a hexagonal initialization."""
    centers = []
    # Approximate spacing for N circles in unit square
    # Area per circle ~ 1/N. Hex area ~ sqrt(3)/2 * s^2.
    # s ~ sqrt(2/(sqrt(3)*N))
    s = np.sqrt(2.0 / (np.sqrt(3.0) * N))
    
    y = -0.5 + s * 0.5
    row = 0
    while y < 0.5 and len(centers) < N:
        offset = (row % 2) * s * 0.5
        x = -0.5 + s * 0.5 + offset
        while x < 0.5 and len(centers) < N:
            if abs(x) <= 0.5 and abs(y) <= 0.5:
                centers.append([x, y])
            x += s
        y += s * np.sqrt(3.0) / 2.0
        row += 1
        
    # If we have too many, trim. If too few, pad.
    if len(centers) > N:
        centers = centers[:N]
    else:
        while len(centers) < N:
            centers.append([np.random.uniform(-0.45, 0.45), np.random.uniform(-0.45, 0.45)])
            
    return np.array(centers)

def construct():
    """
    Construct the centers of 200 equal circles inside [-1/2, 1/2]^2
    to maximize the common radius.
    """
    global _start_time
    _start_time = time.time()
    
    best_centers = None
    best_r = -1.0
    
    # List of initializations to try
    initializations = []
    
    # 1. Standard Grid (10x20, 14x15, etc.)
    for m in [10, 11, 12, 14, 15]:
        initializations.append(_grid_initial(m))
        
    # 2. Hexagonal
    initializations.append(_hex_initial())
    
    # 3. Random (a few)
    for _ in range(3):
        initializations.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # Process each initialization
    for init_idx, init_centers in enumerate(initializations):
        if _time_left() < 10.0:
            break
            
        # Evaluate initial
        r = _compute_radius(init_centers)
        if r > best_r:
            best_r = r
            best_centers = init_centers.copy()
        
        # Local optimization with annealing
        try:
            opt_centers, opt_r = _anneal_optimize(init_centers, beta_start=20.0, beta_end=1.0, n_steps=300)
            if opt_r > best_r:
                best_r = opt_r
                best_centers = opt_centers.copy()
        except Exception:
            pass
        
        # If time allows, do a second pass with smaller beta for higher precision
        if _time_left() > 20.0 and best_centers is not None:
            try:
                # Optimize the best found so far with tighter beta
                # Note: We are optimizing the best_centers from the previous iteration?
                # No, we should optimize the current init_centers result again?
                # Let's optimize the best_centers found globally so far.
                if opt_r > best_r - 1e-6: # If this init was good
                    pass # Already done above
                    
            except Exception:
                pass

    # Final Polish on the best result
    if best_centers is not None and _time_left() > 15.0:
        try:
            # Use a smaller beta for final precision
            final_centers, final_r = _anneal_optimize(best_centers, beta_start=5.0, beta_end=1.0, n_steps=500)
            if final_r > best_r:
                best_r = final_r
                best_centers = final_centers
        except Exception:
            pass
            
    # Ensure valid output
    if best_centers is None:
        return _grid_initial(10)
        
    # Final validation check
    if _compute_radius(best_centers) <= 0:
        return _grid_initial(10)
        
    return best_centers

# EVOLVE-BLOCK-END
