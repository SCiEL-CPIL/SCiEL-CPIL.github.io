import numpy as np
import time
from scipy.spatial.distance import cdist

N = 200
HALF = 0.5
_TIME_LIMIT = 140  # Leave 10s margin for 150s kill


def _score(c):
    """Return the common radius for a configuration (higher is better)."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # inside square
    if np.any(np.abs(c) > HALF + 1e-12):
        return 0.0
    # pairwise min distance
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    # wall distance
    wall = np.min([
        np.min(c[:, 0]) + HALF,
        HALF - np.max(c[:, 0]),
        np.min(c[:, 1]) + HALF,
        HALF - np.max(c[:, 1]),
    ])
    r = min(wall, min_pair / 2.0)
    return r if r > 0 else 0.0


def _hex_grid_points(offset_x=0.0, offset_y=0.0, scale=1.0):
    """
    Generate a hexagonal lattice of points covering the unit square plus margin.
    Returns array of shape (M, 2) for M points.
    """
    # Hex lattice spacing: for packing circles of radius r, the hex spacing is 2r.
    # We generate a dense hex grid and let greedy select the best N.
    # Base spacing ~ 1/15 ≈ 0.0667, so r ~ 0.0333. We want slightly denser.
    spacing = 1.0 / 16.0 * scale
    dx = spacing
    dy = spacing * np.sqrt(3) / 2.0
    
    # Generate grid over a range that covers [-0.5, 0.5] plus margin
    margin = 0.1
    x_min, x_max = -HALF - margin, HALF + margin
    y_min, y_max = -HALF - margin, HALF + margin
    
    nx = int(np.ceil((x_max - x_min) / dx)) + 1
    ny = int(np.ceil((y_max - y_min) / dy)) + 1
    
    points = []
    for j in range(ny):
        y = y_min + j * dy + offset_y
        x_offset = 0.0 if j % 2 == 0 else dx / 2.0
        for i in range(nx):
            x = x_min + i * dx + x_offset + offset_x
            points.append([x, y])
    
    return np.array(points)


def _greedy_select_fast(candidate_points, n_select=N):
    """
    Faster greedy: use a priority-queue-like approach with vectorized updates.
    At each step, pick the candidate with the largest min(dist_to_selected/2, wall_dist).
    """
    mask = np.all(np.abs(candidate_points) <= HALF, axis=1)
    candidates = candidate_points[mask]
    
    if len(candidates) < n_select:
        extra = np.random.uniform(-HALF, HALF, (n_select - len(candidates), 2))
        candidates = np.vstack([candidates, extra])
    
    n_cand = len(candidates)
    
    wall_dists = np.min([
        candidates[:, 0] + HALF,
        HALF - candidates[:, 0],
        candidates[:, 1] + HALF,
        HALF - candidates[:, 1],
    ], axis=0)
    
    # current_min_dist[i] = min distance from candidate i to any selected point
    current_min_dist = np.full(n_cand, np.inf)
    selected = np.zeros(n_select, dtype=int)
    used = np.zeros(n_cand, dtype=bool)
    
    for step in range(n_select):
        # Score for each unused candidate: min(current_min_dist[i]/2, wall_dists[i])
        score = np.minimum(current_min_dist / 2.0, wall_dists)
        score[used] = -1.0
        
        best_idx = np.argmax(score)
        selected[step] = best_idx
        used[best_idx] = True
        
        # Update distances for all unused candidates
        if step < n_select - 1:
            d_new = np.linalg.norm(candidates - candidates[best_idx], axis=1)
            current_min_dist = np.minimum(current_min_dist, d_new)
            current_min_dist[used] = np.inf
    
    return candidates[selected]


def _gradient_polish(pts, max_iters=200, beta=50.0, lr_init=0.01):
    """
    Gradient ascent on the soft-min of constraints (pairwise dists and wall dists).
    """
    pts = pts.copy()
    n = len(pts)
    
    for it in range(max_iters):
        # Compute pairwise distances
        d_mat = cdist(pts, pts)
        np.fill_diagonal(d_mat, np.inf)
        
        # Compute wall distances for each point
        w_left = pts[:, 0] + HALF
        w_right = HALF - pts[:, 0]
        w_bottom = pts[:, 1] + HALF
        w_top = HALF - pts[:, 1]
        w_mat = np.stack([w_left, w_right, w_bottom, w_top], axis=1) # shape (N, 4)
        
        # Constraints:
        # 1. Pairwise: d_mat / 2.0 (shape N x N)
        # 2. Wall: w_mat (shape N x 4)
        
        # We want to maximize min of all these.
        # Soft min: -1/beta * log(sum(exp(-beta * x)))
        # Gradient of soft min w.r.t x_k is prob_k = exp(-beta x_k) / sum(exp(-beta x))
        
        # Flatten constraints for weights
        # Note: d_mat is symmetric, we only need to consider i<j or just use full matrix 
        # but gradients will double count if not careful. 
        # Actually, the objective is min over ALL pairs. 
        # If we use the full matrix, the gradient for point i from pair (i,j) and (j,i) 
        # will be added. Since d_ij = d_ji, and the force on i from j is opposite to force on j from i,
        # using the full matrix is fine if we sum correctly.
        
        # Let's construct the list of constraints and their gradients.
        # This is memory intensive if we store all. 
        # Alternative: Compute weights for pairs and walls separately.
        
        # Weights for pairs:
        # x_pair = d_mat / 2
        # We need P_pair[i,j] = exp(-beta * x_pair[i,j]) / Z_pair
        # But Z_pair is sum over all pairs.
        # And we also have walls.
        # The total soft min is -1/beta log (Sum_pairs exp(-beta d/2) + Sum_walls exp(-beta w))
        # This is not exactly the min of the union, but a good surrogate.
        
        # Let's compute the log-sum-exp for stability.
        # log_sum_exp(x) = m + log(sum(exp(x - m)))
        
        # Values:
        v_pairs = d_mat / 2.0
        v_walls = w_mat
        
        # To avoid overflow in exp, subtract max.
        # But we have two groups.
        # Let's just compute the gradient directly using probabilities.
        
        # P(i,j) proportional to exp(-beta * d_ij/2)
        # P(i,w) proportional to exp(-beta * w_iw)
        
        # We want to move point i in direction:
        # sum_{j != i} P(i,j) * grad_{x_i} (d_ij/2) + sum_{w} P(i,w) * grad_{x_i} (w_iw)
        
        # Note: The denominator Z is the same for all terms in the sum for the gradient of the log-sum-exp.
        # grad_{x} ( -1/beta log sum exp(-beta x_k) ) = sum (exp(-beta x_k) / sum exp(-beta x_j)) * grad_{x} x_k
        # So we need the normalized weights.
        
        # Compute unnormalized weights
        # exp(-beta * v)
        # To prevent underflow/overflow, we can scale.
        # Since we only need ratios, we can compute relative weights.
        
        # However, computing exp for 200x200 + 200x4 is fine.
        
        # Mask diagonal for pairs
        mask_diag = np.eye(n, dtype=bool)
        
        # Weights for pairs
        # v_pairs has inf on diagonal. exp(-beta * inf) = 0.
        w_pairs = np.exp(-beta * v_pairs)
        w_pairs[mask_diag] = 0.0
        
        # Weights for walls
        w_walls = np.exp(-beta * v_walls)
        
        # Total weight for each point i is sum of w_pairs[i,:] + sum of w_walls[i,:]
        # But the normalization Z is global: sum(w_pairs) + sum(w_walls)
        # The gradient is (1/Z) * [sum_j w_pairs[i,j] * grad(d_ij/2) + sum_w w_walls[i,w] * grad(w_iw)]
        
        # Since Z is constant for the update direction (it doesn't depend on the specific direction of x_i in the gradient step? 
        # No, Z depends on all x. But for the gradient at a specific point, Z is just a scalar factor.
        # We can ignore Z for the direction, or include it in the learning rate.
        
        # Compute gradients of distances
        # grad_{x_i} d_ij = (x_i - x_j) / d_ij
        # grad_{x_i} (d_ij / 2) = (x_i - x_j) / (2 * d_ij)
        
        # Vectorized calculation:
        # diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :] # (N, N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        
        # d_mat is (N, N). 
        # grad_pair = diff / (2 * d_mat)[:,:,np.newaxis]
        # Handle d=0 (diagonal) which is masked out in weights.
        with np.errstate(divide='ignore', invalid='ignore'):
            grad_pair = diff / (2.0 * d_mat[:, :, np.newaxis])
        
        # Where d is inf or 0, grad is 0 or NaN. We masked weights to 0, so 0*NaN=NaN. 
        # Better to set grad to 0 where mask is true.
        grad_pair[mask_diag, :] = 0.0
        
        # Sum weighted gradients for pairs:
        # grad_i = sum_j w_pairs[i,j] * grad_pair[i,j]
        # w_pairs is (N, N), grad_pair is (N, N, 2)
        # grad_pair_sum = np.sum(w_pairs[:, :, np.newaxis] * grad_pair, axis=1)
        
        # Wall gradients
        # w_left = x + 0.5, grad = (1, 0)
        # w_right = 0.5 - x, grad = (-1, 0)
        # w_bottom = y + 0.5, grad = (0, 1)
        # w_top = 0.5 - y, grad = (0, -1)
        
        # grad_walls shape (N, 4, 2)
        grad_walls = np.zeros((n, 4, 2))
        grad_walls[:, 0, 0] = 1.0   # left
        grad_walls[:, 1, 0] = -1.0  # right
        grad_walls[:, 2, 1] = 1.0   # bottom
        grad_walls[:, 3, 1] = -1.0  # top
        
        # Sum weighted gradients for walls:
        # grad_wall_sum = np.sum(w_walls[:, :, np.newaxis] * grad_walls, axis=1)
        
        # Combine
        # Total gradient direction for point i:
        # g_i = grad_pair_sum[i] + grad_wall_sum[i]
        
        # Note: The global Z is sum(w_pairs) + sum(w_walls).
        # We can normalize the gradient by Z to get the true gradient of the soft-min.
        Z = np.sum(w_pairs) + np.sum(w_walls)
        if Z < 1e-10:
            break
            
        grad_total = (np.sum(w_pairs[:, :, np.newaxis] * grad_pair, axis=1) + 
                      np.sum(w_walls[:, :, np.newaxis] * grad_walls, axis=1)) / Z
        
        # Update
        lr = lr_init / (1.0 + it / 20.0) # Decay
        pts += lr * grad_total
        
        # Project to square
        pts = np.clip(pts, -HALF, HALF)
        
        # Check time
        if time.time() > _TIME_LIMIT - 5:
            break
            
    return pts


def construct():
    t0 = time.time()
    best_r = 0.0
    best_pts = None
    
    # Strategy: hex-grid greedy with multiple offsets, then gradient polish.
    
    np.random.seed(42)
    
    # Generate hex grids with different offsets and scales
    offsets_x = [0.0, 0.02, 0.04, 0.01, 0.03]
    offsets_y = [0.0, 0.02, 0.04, 0.01, 0.03]
    scales = [1.0, 0.95, 1.05]
    
    attempts = 0
    max_attempts = 30
    
    for scale in scales:
        for ox in offsets_x:
            for oy in offsets_y:
                if time.time() - t0 > _TIME_LIMIT - 20:
                    break
                if attempts >= max_attempts:
                    break
                
                # Generate hex grid
                grid = _hex_grid_points(offset_x=ox, offset_y=oy, scale=scale)
                
                # Greedy select
                pts = _greedy_select_fast(grid)
                attempts += 1
                
                # Gradient polish
                if time.time() - t0 < _TIME_LIMIT - 10:
                    pts = _gradient_polish(pts, max_iters=150, beta=50.0, lr_init=0.01)
                
                r = _score(pts)
                if r > best_r:
                    best_r = r
                    best_pts = pts.copy()
                
                # If we have time, do a second polish on the best
                if time.time() - t0 < _TIME_LIMIT - 5 and best_pts is not None:
                    refined = _gradient_polish(best_pts, max_iters=100, beta=100.0, lr_init=0.005)
                    r_refined = _score(refined)
                    if r_refined > best_r:
                        best_r = r_refined
                        best_pts = refined.copy()
        
        if time.time() - t0 > _TIME_LIMIT - 20:
            break
    
    # Final polish on the best result if time permits
    if best_pts is not None and time.time() - t0 < _TIME_LIMIT - 2:
        final = _gradient_polish(best_pts, max_iters=100, beta=200.0, lr_init=0.002)
        r_final = _score(final)
        if r_final > best_r:
            best_r = r_final
            best_pts = final.copy()
    
    # Fallback
    if best_pts is None:
        best_pts = np.random.uniform(-HALF, HALF, (N, 2))
    
    return best_pts
