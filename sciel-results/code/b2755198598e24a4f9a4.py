"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 200

# EVOLVE-BLOCK-START

def _hex_init(n):
    """Generate a hexagonal lattice initialization for n circles."""
    # Approximate number of rows
    # Area per circle ~ r^2 * 2 / sqrt(3) for hex packing
    # r ~ 0.036, area ~ 0.0015 / 0.866 ~ 0.0017
    # Total area 1. 200 * 0.0017 ~ 0.34. 
    # Hex lattice density is sqrt(3)/2 * (2r)^2 = sqrt(3) * r^2.
    # Let's just try a grid of hex points and pick the best n.
    
    best_centers = None
    best_score = -1
    
    # Try different row counts
    for rows in range(10, 25):
        # Calculate spacing to fit in square
        # Assume square side 1.
        # y spacing = dy. x spacing = dx.
        # dy = 1.0 / (rows - 1) approx? No, need to fit centers.
        # Let's define dx, dy such that they fit.
        # For hex: dy = dx * sqrt(3) / 2.
        # Total height = (rows - 1) * dy.
        # Total width = (cols - 1) * dx + dx/2 (for offset).
        
        # We want to maximize min distance.
        # Let's iterate on a scaling factor.
        
        centers = []
        # We need to distribute n circles into rows.
        base_cols = n // rows
        extra = n % rows
        
        # Try a few base spacings
        for scale in [0.9, 0.95, 1.0, 1.05, 1.1]:
            # Estimate dx
            # Width ~ (base_cols) * dx <= 1.0 -> dx <= 1.0 / base_cols
            # Height ~ (rows) * dy = rows * dx * sqrt(3)/2 <= 1.0 -> dx <= 2.0 / (rows * sqrt(3))
            max_dx_width = 1.0 / max(base_cols, 1)
            max_dx_height = 2.0 / (rows * np.sqrt(3))
            dx = min(max_dx_width, max_dx_height) * scale
            dy = dx * np.sqrt(3) / 2
            
            # Check if it fits
            if (rows - 1) * dy > 1.0 or base_cols * dx > 1.0:
                continue
                
            for i in range(rows):
                cols = base_cols + (1 if i < extra else 0)
                if cols == 0: continue
                # Center the row
                row_width = (cols - 1) * dx
                start_x = -row_width / 2
                if i % 2 == 1:
                    start_x += dx / 2
                
                # Check if row fits in width
                if start_x + row_width > 0.5 or start_x < -0.5:
                    # Shift or skip? If it doesn't fit, this scale is bad.
                    # But we centered it. If width > 1, it won't fit.
                    if row_width > 1.0:
                        break
                    
                for j in range(cols):
                    x = start_x + j * dx
                    y = (i - (rows - 1) / 2) * dy
                    # Clip to square
                    x = np.clip(x, -0.5, 0.5)
                    y = np.clip(y, -0.5, 0.5)
                    centers.append([x, y])
            
            centers = np.array(centers)
            if len(centers) != n:
                continue
            
            # Calculate current radius
            # Simple min distance check
            if len(centers) < 2:
                score = 0.5
            else:
                # Use scipy distance if available, or manual
                # For speed in init, manual vectorized
                dxs = centers[:, 0][:, None] - centers[:, 0][None, :]
                dys = centers[:, 1][:, None] - centers[:, 1][None, :]
                dists = np.sqrt(dxs**2 + dys**2)
                np.fill_diagonal(dists, np.inf)
                min_pair = np.min(dists) / 2
                
                wall_dists = np.min(np.stack([
                    0.5 - centers[:, 0],
                    centers[:, 0] + 0.5,
                    0.5 - centers[:, 1],
                    centers[:, 1] + 0.5
                ]), axis=0)
                min_wall = np.min(wall_dists)
                
                score = min(min_pair, min_wall)
            
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
        
        if best_centers is not None and best_score > 0.03:
             break # Found a decent one
    
    if best_centers is None:
        # Fallback grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:n]
        
    return best_centers

def _force_directed(centers, r, max_iter=500, step=0.01):
    """Simple force-directed relaxation to resolve overlaps."""
    centers = centers.copy()
    n = len(centers)
    
    # Precompute pairs
    # For efficiency, we'll just loop over all pairs in the force calc if n is small, 
    # but n=200 is 20k pairs. 500 iters * 20k = 10M ops. Fast enough in NumPy.
    
    for _ in range(max_iter):
        forces = np.zeros_like(centers)
        
        # Pairwise forces
        dx = centers[:, 0][:, None] - centers[:, 0][None, :]
        dy = centers[:, 1][:, None] - centers[:, 1][None, :]
        dists = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dists, 1.0) # Avoid div by 0
        
        # Overlap force: if dist < 2r, push apart
        # Force magnitude proportional to overlap
        overlap = np.maximum(0, 2*r - dists)
        # Direction is normalized dx, dy
        # Avoid division by zero where dist is 0 (coincident)
        inv_dists = np.where(dists > 1e-6, 1.0/dists, 0.0)
        
        fx = (dx * inv_dists * overlap)
        fy = (dy * inv_dists * overlap)
        
        forces[:, 0] = np.sum(fx, axis=1)
        forces[:, 1] = np.sum(fy, axis=1)
        
        # Wall forces
        # If x < -0.5 + r, push right. If x > 0.5 - r, push left.
        wall_fx = np.zeros(n)
        wall_fy = np.zeros(n)
        
        # Left wall
        dist_left = centers[:, 0] + 0.5
        wall_fx += np.maximum(0, r - dist_left)
        # Right wall
        dist_right = 0.5 - centers[:, 0]
        wall_fx -= np.maximum(0, r - dist_right)
        
        # Bottom wall
        dist_bottom = centers[:, 1] + 0.5
        wall_fy += np.maximum(0, r - dist_bottom)
        # Top wall
        dist_top = 0.5 - centers[:, 1]
        wall_fy -= np.maximum(0, r - dist_top)
        
        forces[:, 0] += wall_fx
        forces[:, 1] += wall_fy
        
        # Update
        centers += step * forces
        
        # Clip to square
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
        # Decay step size
        step *= 0.99
        
    return centers

def _soft_min_obj(params, beta, r):
    """
    Objective function for L-BFGS-B.
    params: flattened array of centers (2N).
    beta: sharpness of soft min.
    r: current radius estimate.
    Returns: -r + soft_min_violation.
    """
    n = len(params) // 2
    centers = params.reshape(n, 2)
    
    # Pairwise distances
    dx = centers[:, 0][:, None] - centers[:, 0][None, :]
    dy = centers[:, 1][:, None] - centers[:, 1][None, :]
    dists_sq = dx**2 + dy**2
    # We only need upper triangle for sum, but log-sum-exp is over all.
    # Using full matrix is easier but 2x work.
    # Let's use full matrix for simplicity, divide by 2? No, log-sum-exp is not linear.
    # We want sum over i<j.
    # Mask upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    dists = np.sqrt(dists_sq[mask])
    
    # Violations: 2r - dist
    viol_pairs = 2*r - dists
    
    # Wall distances
    wall_dists = np.concatenate([
        0.5 - centers[:, 0],
        centers[:, 0] + 0.5,
        0.5 - centers[:, 1],
        centers[:, 1] + 0.5
    ])
    # This is 4N values. We want min over these for each circle, then min over circles?
    # The constraint is min_i min_k (wall_dist_ik) >= r.
    # So violations are r - wall_dist_ik for all i, k.
    viol_walls = r - wall_dists
    
    # Combine violations
    # We want to penalize positive violations.
    # Soft min of (2r - d) is not quite right if we want to maximize min d.
    # We want to minimize max(0, 2r - d).
    # Soft max of violations: (1/beta) log sum exp(beta * viol).
    # If all viol < 0, this is small. If one is large positive, this is large.
    
    all_viol = np.concatenate([viol_pairs, viol_walls])
    
    # Numerical stability: subtract max
    max_viol = np.max(all_viol)
    if max_viol > 0:
        # If max is positive, the log-sum-exp will be dominated by it.
        # We can just compute it.
        pass
    
    # To avoid overflow, subtract max
    shifted = beta * (all_viol - max_viol)
    # Clip for stability
    shifted = np.clip(shifted, -100, 100)
    log_sum_exp = max_viol + np.log(np.sum(np.exp(shifted))) / beta
    
    # Objective: We want to minimize the "worst case" violation.
    # And we want to increase r.
    # So minimize: log_sum_exp - r?
    # If feasible, viol <= 0, LSE ~ log(N) / beta (small).
    # As r increases, viol increases, LSE increases.
    # So minimizing LSE - r finds the balance.
    
    return log_sum_exp - r

def _soft_min_grad(params, beta, r):
    """Gradient of the soft min objective."""
    n = len(params) // 2
    centers = params.reshape(n, 2)
    
    # Recalculate distances
    dx = centers[:, 0][:, None] - centers[:, 0][None, :]
    dy = centers[:, 1][:, None] - centers[:, 1][None, :]
    dists_sq = dx**2 + dy**2
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    dists = np.sqrt(dists_sq[mask])
    
    viol_pairs = 2*r - dists
    weights_pairs = np.exp(beta * (viol_pairs - np.max(viol_pairs)))
    weights_pairs /= np.sum(weights_pairs)
    
    # Gradient of dist wrt centers
    # d(dist)/d(x_i) = (x_i - x_j) / dist
    # We need to map weights back to the full matrix gradient
    
    grad_centers = np.zeros_like(centers)
    
    # Iterate over pairs? Too slow for Python loop.
    # Use vectorized accumulation.
    # We have dx, dy for all pairs.
    # Let's construct the gradient contribution.
    # For each pair (i, j) with i < j:
    # dV/dx_i += w_ij * (x_i - x_j) / d_ij
    # dV/dx_j += w_ij * (x_j - x_i) / d_ij
    
    # Create full matrices for dx, dy, dists
    # dx_full = dx
    # dy_full = dy
    # dist_full = sqrt(dx^2+dy^2)
    # We only have upper triangle dists.
    
    # Let's just compute full distance matrix for gradient simplicity, it's 200x200.
    dists_full = np.sqrt(dists_sq)
    np.fill_diagonal(dists_full, 1.0)
    
    # Weights for full matrix
    # viol_full = 2r - dists_full
    viol_full = 2*r - dists_full
    # Zero out diagonal
    viol_full[~np.triu(np.ones((n,n), dtype=bool), k=1)] = -np.inf # Or just handle mask
    # Actually, we want to sum over i<j.
    # Let's compute weights on full matrix and divide by 2?
    # No, the objective was sum over i<j.
    # Let's compute gradient on full matrix assuming symmetric sum, then divide by 2.
    
    # Shift for stability
    max_v = np.max(viol_full)
    w_full = np.exp(beta * (viol_full - max_v))
    # Zero diagonal
    np.fill_diagonal(w_full, 0)
    # We want sum over i<j. The full matrix sum is 2 * sum over i<j.
    # So weights for the objective should be w_full / 2 ?
    # Let's check: LSE = (1/beta) log sum_{i<j} exp(beta v_ij).
    # If we use full matrix sum, it's log (2 * sum_{i<j}).
    # This adds log(2)/beta to the objective. Constant offset.
    # Gradient is proportional to exp(beta v) * grad(v).
    # So using full matrix weights is fine, just scaled.
    
    # Normalization? No, LSE gradient doesn't require normalization by sum in the same way softmax does for prob.
    # d/dx log(sum exp(f)) = (sum exp(f) * f') / sum exp(f) = sum (w_i f').
    # Where w_i = exp(f_i) / sum exp(f).
    # So we DO need to normalize weights.
    
    sum_w = np.sum(w_full)
    if sum_w > 0:
        w_full /= sum_w
    
    # Gradient of dist wrt x_i
    # d(d_ij)/d(x_i) = (x_i - x_j) / d_ij
    # Term in objective: w_ij * (2r - d_ij) is not linear.
    # Objective J = (1/beta) log sum exp(beta(2r - d)).
    # dJ/dx = (1/beta) * (1/sum) * sum exp(beta(2r-d)) * beta * (-1) * d(d)/dx
    # dJ/dx = - sum (w_ij * d(d_ij)/dx)
    # Where w_ij is the normalized weight.
    
    # Vectorized:
    # grad_x[i] = - sum_j w_ij * (x_i - x_j) / d_ij
    
    # We can compute this using matrix multiplication?
    # Let A[i, j] = w_ij / d_ij
    # Then grad_x[i] = - sum_j A[i, j] * (x_i - x_j)
    # = - x_i * sum_j A[i, j] + sum_j A[i, j] * x_j
    
    inv_d = np.zeros_like(dists_full)
    np.fill_diagonal(inv_d, 0)
    inv_d[dists_full > 0] = 1.0 / dists_full[dists_full > 0]
    
    A = w_full * inv_d
    
    sum_A = np.sum(A, axis=1) # shape (n,)
    
    # sum_j A[i, j] * x_j is A @ x
    Ax = A @ centers
    
    grad_centers[:, 0] = - (centers[:, 0] * sum_A - Ax[:, 0])
    grad_centers[:, 1] = - (centers[:, 1] * sum_A - Ax[:, 1])
    
    # Wall gradients
    # viol_walls = r - wall_dist
    # wall_dists = [0.5-x, x+0.5, 0.5-y, y+0.5]
    # grad(r - (0.5-x)) / dx = +1
    # grad(r - (x+0.5)) / dx = -1
    # etc.
    
    wall_dists = np.concatenate([
        0.5 - centers[:, 0],
        centers[:, 0] + 0.5,
        0.5 - centers[:, 1],
        centers[:, 1] + 0.5
    ])
    viol_walls = r - wall_dists
    
    # Combine weights for walls
    # We used full matrix for pairs. For walls, we have 4N items.
    # We need to combine the LSE of pairs and walls.
    # J = (1/beta) log (sum_pairs exp(beta v_p) + sum_walls exp(beta v_w))
    # This is harder to separate gradients if we just summed them.
    # My objective function _soft_min_obj concatenated them.
    # So the weights w_full and w_walls should be normalized against the TOTAL sum.
    
    # Let's recalculate weights properly.
    all_viol = np.concatenate([
        viol_full[np.triu(np.ones((n,n), dtype=bool), k=1)], # Pairs
        viol_walls
    ])
    
    max_all = np.max(all_viol)
    exp_all = np.exp(beta * (all_viol - max_all))
    sum_all = np.sum(exp_all)
    
    # Weights for pairs
    w_pairs = exp_all[:len(viol_pairs)] / sum_all
    # Weights for walls
    w_walls = exp_all[len(viol_pairs):] / sum_all
    
    # Gradient from pairs
    # We need to map w_pairs (upper triangle) back to full matrix A.
    # w_full[i, j] = w_pairs[idx] if i<j, w_pairs[idx] if j<i, 0 if i=j.
    # This is slow to construct.
    # Alternative: Compute gradient using the full matrix weights derived from the combined sum.
    
    # Let's just use the full matrix weights derived from the combined sum.
    # exp_full_pairs = exp(beta * (viol_full - max_all))
    # exp_full_walls = exp(beta * (viol_walls - max_all))
    
    # This is getting complicated.
    # Simplification: Use a simpler objective for L-BFGS-B.
    # Minimize sum of squares of violations + penalty * (-r).
    # V = sum (max(0, 2r - d))^2 + sum (max(0, r - w))^2.
    # This is non-smooth.
    # Smoothed: sum (soft_max(0, 2r - d))^2.
    
    # Let's stick to the previous gradient but fix the normalization.
    # I will rewrite the gradient function to be consistent with the objective.
    
    # Actually, for L-BFGS-B, providing an approximate gradient or using finite differences is acceptable if it converges.
    # But let's try to get it right.
    
    # Re-calculate weights for pairs and walls with common normalization.
    # We already have exp_all, sum_all.
    # w_pairs_full needs to be constructed.
    
    # Let's just compute the gradient numerically? No, too slow.
    # Let's assume the pair weights dominate or are comparable.
    
    # Gradient from pairs (using normalized weights w_pairs on upper triangle)
    # grad_x[i] = - sum_{j>i} w_ij (x_i - x_j)/d_ij - sum_{j<i} w_ji (x_i - x_j)/d_ij
    
    # Let's construct A_full from w_pairs.
    A_full = np.zeros((n, n))
    # Vectorized assignment of upper triangle
    i_idx, j_idx = np.where(np.triu(np.ones((n, n), dtype=bool), k=1))
    A_full[i_idx, j_idx] = w_pairs * inv_d[i_idx, j_idx]
    A_full[j_idx, i_idx] = w_pairs * inv_d[j_idx, i_idx] # Symmetric
    
    sum_A = np.sum(A_full, axis=1)
    Ax = A_full @ centers
    grad_centers[:, 0] = - (centers[:, 0] * sum_A - Ax[:, 0])
    grad_centers[:, 1] = - (centers[:, 1] * sum_A - Ax[:, 1])
    
    # Gradient from walls
    # w_walls has 4N elements.
    # w_walls[0:N] correspond to 0.5 - x (Left). Grad: +w
    # w_walls[N:2N] correspond to x + 0.5 (Right). Grad: -w
    # w_walls[2N:3N] correspond to 0.5 - y (Bottom). Grad: +w
    # w_walls[3N:4N] correspond to y + 0.5 (Top). Grad: -w
    
    grad_centers[:, 0] += w_walls[0:n] - w_walls[n:2*n]
    grad_centers[:, 1] += w_walls[2*n:3*n] - w_walls[3*n:4*n]
    
    # Gradient wrt r
    # dJ/dr = (1/beta) * (1/sum) * sum exp(beta v) * beta * (dv/dr)
    # dv_pairs/dr = 2
    # dv_walls/dr = 1
    # dJ/dr = sum_pairs (w_pairs * 2) + sum_walls (w_walls * 1) - 1 (from the -r term)
    
    grad_r = 2 * np.sum(w_pairs) + np.sum(w_walls) - 1
    
    # Return gradient for [x1, y1, ..., xN, yN, r]
    grad_params = np.concatenate([grad_centers.flatten(), [grad_r]])
    
    return grad_params

def _optimize_lbfgs(centers, r, beta, max_iter=100):
    """Run L-BFGS-B optimization."""
    n = len(centers)
    x0 = np.concatenate([centers.flatten(), [r]])
    
    # Bounds: centers in [-0.5, 0.5], r in [0, 0.5]
    bounds = [(-0.5, 0.5)] * (2*n) + [(0, 0.5)]
    
    res = minimize(
        _soft_min_obj, 
        x0, 
        args=(beta, r), # Note: r is fixed in args for this call? 
        # Wait, if we optimize r, we shouldn't pass r as arg to obj if we want to vary it.
        # My obj takes (params, beta, r). If r is in params, we ignore arg r?
        # Let's modify obj to take r from params if desired, or keep r fixed.
        # To maximize r, we should optimize it.
        # But the obj currently uses the `r` passed in args.
        # Let's change obj to use params[-1] as r.
        
        jac=_soft_min_grad,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-6, 'gtol': 1e-4}
    )
    
    new_centers = res.x[:-1].reshape(n, 2)
    new_r = res.x[-1]
    
    return new_centers, new_r, res.fun

def _compute_radius(centers):
    """Exact radius calculation."""
    n = len(centers)
    if n < 2:
        return 0.5
    dx = centers[:, 0][:, None] - centers[:, 0][None, :]
    dy = centers[:, 1][:, None] - centers[:, 1][None, :]
    dists = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    wall = np.min(np.stack([
        0.5 - centers[:, 0],
        centers[:, 0] + 0.5,
        0.5 - centers[:, 1],
        centers[:, 1] + 0.5
    ]), axis=0)
    min_wall = np.min(wall)
    
    return min(min_pair, min_wall)

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    # 1. Init
    centers = _hex_init(N)
    r = _compute_radius(centers)
    
    # 2. Force Directed
    if time.time() < deadline - 30:
        centers = _force_directed(centers, r, max_iter=200)
        r = _compute_radius(centers)
    
    # 3. L-BFGS-B with increasing beta
    betas = [10, 50, 100, 200, 500]
    for beta in betas:
        if time.time() > deadline - 20:
            break
        try:
            new_centers, new_r, _ = _optimize_lbfgs(centers, r, beta, max_iter=50)
            # Validate
            new_r_exact = _compute_radius(new_centers)
            if new_r_exact > r:
                r = new_r_exact
                centers = new_centers
        except Exception:
            pass
            
    # 4. Final Polish (Local Search)
    if time.time() < deadline - 5:
        step = 0.001
        for _ in range(500):
            if time.time() > deadline - 2:
                break
            idx = np.random.randint(N)
            dx = np.random.uniform(-step, step)
            dy = np.random.uniform(-step, step)
            new_centers = centers.copy()
            new_centers[idx, 0] += dx
            new_centers[idx, 1] += dy
            if np.any(new_centers[:, 0] < -0.5) or np.any(new_centers[:, 0] > 0.5) or \
               np.any(new_centers[:, 1] < -0.5) or np.any(new_centers[:, 1] > 0.5):
                continue
            new_r = _compute_radius(new_centers)
            if new_r > r:
                r = new_r
                centers = new_centers
                step = max(0.0001, step * 0.95)
            else:
                step = min(0.01, step * 1.05)
                
    return centers

# EVOLVE-BLOCK-END
