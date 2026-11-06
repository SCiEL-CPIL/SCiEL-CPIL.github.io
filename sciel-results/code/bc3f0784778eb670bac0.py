"""Pack 200 equal circles in the unit square [-1/2, 1/2]^2 to maximize common radius."""
import time
import numpy as np
from scipy.spatial.distance import pdist

N = 200
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s kill


def _remaining_time():
    return TIME_LIMIT - (time.time() - START_TIME)


def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    if len(centers) == 0:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall distances
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    wall_min = min(np.min(dx), np.min(dy))
    if wall_min < 0:
        return 0.0
    # Pairwise distances
    dists = pdist(centers)
    pair_min = np.min(dists) / 2.0
    return min(wall_min, pair_min)


def _pdist_index_to_pair(idx, n):
    """Convert pdist index to (i, j) pair where i < j."""
    # pdist uses upper triangle in row-major order
    # For n points, the index k corresponds to (i, j) where:
    # k = i*(n-1) - i*(i-1)/2 + (j - i - 1) for i < j
    # We need to invert this.
    # Let m = n - 1
    # k = i*m - i*(i+1)/2 + (j - i - 1) ... this is getting complicated
    # Simpler: precompute or use a different approach
    # For 200 points, we can just search
    # Actually, let's use a cleaner formula
    # The k-th element in pdist (0-indexed) corresponds to:
    # i = floor((n - 0.5) - sqrt((n - 0.5)^2 - 4*k))
    # j = k - i*(n - 1 - i)/2 + i
    # Let me verify: for n=4, pdist has 6 elements:
    # (0,1)=0, (0,2)=1, (0,3)=2, (1,2)=3, (1,3)=4, (2,3)=5
    # For k=0: i = floor(3.5 - sqrt(12.25)) = floor(3.5-3.5) = 0, j = 0 - 0 + 0 = 0? No, j should be 1
    # Let me just use a lookup or iterative approach
    # For efficiency, let's precompute the mapping
    # Actually, for 200 points, 19900 pairs, we can afford a simple loop or use np.triu_indices
    # Let's just use np.triu_indices to get all pairs
    return None  # We'll handle this differently in the optimization


def _hexagonal_grid(n):
    """Generate a hexagonal grid of n points scaled to fit in [-0.5, 0.5]^2."""
    # For n circles in a hexagonal arrangement, we need to figure out the grid dimensions
    # A hexagonal grid with rows of varying length
    # For 200 points, let's use a grid that's roughly square
    # Number of rows: approximately sqrt(n)
    # In a hexagonal grid, each row has either floor(sqrt(2n/3)) or similar points
    
    # Simpler approach: use a square grid with hexagonal offset
    m = int(np.ceil(np.sqrt(n)))
    # Create a grid of m x m points
    # Hexagonal spacing: horizontal spacing = s, vertical spacing = s * sqrt(3)/2
    # For a square region, we want to fit m points horizontally and m points vertically
    # Width: (m-1) * s = 1.0 (for the interior, we'll adjust)
    # Height: (m-1) * s * sqrt(3)/2
    
    # Let's scale so the grid fits in [-0.5, 0.5]^2
    # Horizontal: m points, spacing s_x
    # Vertical: m points, spacing s_y = s_x * sqrt(3)/2 for hexagonal
    # But we want to fit in a square, so we might need to adjust
    
    # For a proper hexagonal packing in a square, the optimal aspect ratio is different
    # Let's just use a square grid with small random perturbation as initial guess
    # The optimizer will find the better arrangement
    
    ticks_x = (np.arange(m) + 0.5) / m - 0.5
    ticks_y = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
    
    # If we have more grid points than needed, take the first n
    if len(grid) > n:
        grid = grid[:n]
    elif len(grid) < n:
        # Pad with random points
        extra = np.random.randn(n - len(grid), 2) * 0.1
        extra[:, 0] = np.clip(extra[:, 0], -0.4, 0.4)
        extra[:, 1] = np.clip(extra[:, 1], -0.4, 0.4)
        grid = np.vstack([grid, extra])
    
    return grid


def _soft_min_objective_and_grad(x, t=0.01):
    """
    Compute the soft-min objective and its gradient.
    x is a flat array of length 2*N.
    """
    pts = x.reshape(-1, 2)
    
    # Wall distances
    dx = 0.5 - np.abs(pts[:, 0])
    dy = 0.5 - np.abs(pts[:, 1])
    wall_dists = np.concatenate([dx, dy])
    
    # Pairwise distances / 2
    dists = pdist(pts)
    pair_dists = dists / 2.0
    
    all_dists = np.concatenate([wall_dists, pair_dists])
    
    # Soft min
    m = np.max(all_dists)
    exp_vals = np.exp(-(all_dists - m) / t)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    soft_min_val = m - t * np.log(sum_exp)
    
    # Gradient of soft_min with respect to pts
    grad = np.zeros_like(pts)
    
    # Wall gradients
    # d(0.5 - |x|)/dx = -sign(x) for x != 0
    gx_wall = -np.sign(pts[:, 0])
    gy_wall = -np.sign(pts[:, 1])
    
    grad[:, 0] += weights[:N] * gx_wall
    grad[:, 1] += weights[N:2*N] * gy_wall
    
    # Pairwise gradients
    # For each pair (i, j), the contribution to grad[i] is weights_pair * (pts_i - pts_j) / (2 * dist_ij)
    # and to grad[j] is weights_pair * (pts_j - pts_i) / (2 * dist_ij)
    
    # We need to map pdist indices back to (i, j) pairs
    # For N=200, we can use np.triu_indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)
    
    # weights_pair = weights[2*N : 2*N + num_pairs]
    weights_pair = weights[2*N:2*N + num_pairs]
    
    # Compute differences
    diffs = pts[i_idx] - pts[j_idx]  # shape (num_pairs, 2)
    dists_matrix = dists  # shape (num_pairs,)
    
    # Avoid division by zero
    safe_dists = np.where(dists_matrix > 1e-10, dists_matrix, 1.0)
    
    # grad[i] += sum_j weights[i,j] * (pts_i - pts_j) / (2 * dist_ij)
    # grad[j] += sum_i weights[i,j] * (pts_j - pts_i) / (2 * dist_ij)
    
    # term_k = weights_pair[k] * diffs[k] / (2 * safe_dists[k])
    terms = weights_pair[:, np.newaxis] * diffs / (2.0 * safe_dists[:, np.newaxis])
    
    # Accumulate into grad
    # For each pair k, add terms[k] to grad[i_idx[k]] and -terms[k] to grad[j_idx[k]]
    np.add.at(grad, i_idx, terms)
    np.add.at(grad, j_idx, -terms)
    
    # The objective is -soft_min, so the gradient of the objective is -grad(soft_min)
    obj_val = -soft_min_val
    grad_val = -grad.flatten()
    
    return obj_val, grad_val


def _optimize_l_bfgs(x0, max_iter=300, t=0.01):
    """Optimize using L-BFGS-B on the soft-min objective."""
    from scipy.optimize import minimize
    
    def objective_and_grad(x):
        val, grad = _soft_min_objective_and_grad(x, t)
        return val, grad
    
    res = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 10000})
    return res.x.reshape(-1, 2)


def _refine(pts, max_iter=200):
    """Refine a configuration using L-BFGS-B."""
    x0 = pts.flatten()
    pts_opt = _optimize_l_bfgs(x0, max_iter=max_iter)
    return pts_opt


def construct():
    best_pts = None
    best_r = 0.0
    
    np.random.seed(42)
    
    # Initial guess: hexagonal-like grid
    pts0 = _hexagonal_grid(N)
    
    # Optimize the initial guess
    pts_opt = _refine(pts0, max_iter=300)
    r = _compute_radius(pts_opt)
    if r > best_r:
        best_r = r
        best_pts = pts_opt.copy()
    
    # Random restarts with different initial perturbations
    max_restarts = 30
    for i in range(max_restarts):
        if _remaining_time() < 5.0:
            break
        
        # Generate a random initial guess based on the grid
        noise_scale = 0.005 + 0.01 * (i % 5)
        pts_rand = _hexagonal_grid(N) + np.random.randn(N, 2) * noise_scale
        pts_rand[:, 0] = np.clip(pts_rand[:, 0], -0.49, 0.49)
        pts_rand[:, 1] = np.clip(pts_rand[:, 1], -0.49, 0.49)
        
        # Optimize
        pts_opt = _refine(pts_rand, max_iter=150)
        r = _compute_radius(pts_opt)
        if r > best_r:
            best_r = r
            best_pts = pts_opt.copy()
    
    # Final polish on the best solution
    if best_pts is not None and _remaining_time() > 3.0:
        pts_polish = _refine(best_pts, max_iter=200)
        r = _compute_radius(pts_polish)
        if r > best_r:
            best_r = r
            best_pts = pts_polish.copy()
    
    return best_pts
