"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist


def _score(pts):
    """Compute the radius for a set of points."""
    # Wall constraints
    walls = np.minimum(np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1])))
    # Pairwise distances
    d = pdist(pts)
    min_dist = d.min() if len(d) > 0 else 1.0
    return min(min_dist / 2.0, walls.min())


def _neg_score(pts):
    """Negative score for minimization."""
    return -_score(pts)


def _grad_neg_score(pts):
    """
    Approximate gradient of the negative score.
    The score is min over all constraints. We can approximate the gradient
    by finding which constraint is active and taking its gradient.
    
    For a more robust approach, we use a smooth approximation.
    """
    pts = np.asarray(pts).reshape(-1, 2)
    n = len(pts)
    
    # Wall constraints: for point i, constraint is min(0.5 - |x_i|, 0.5 - |y_i|)
    # The gradient of the minimum is the gradient of the active constraint.
    
    # Pairwise: d_ij / 2 for i < j
    # We'll compute all distances and find the minimum
    
    # For efficiency, compute all pairwise distances
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    
    # Find the minimum pairwise distance
    min_idx = np.unravel_index(np.argmin(d), d.shape)
    min_pair_dist = d[min_idx]
    
    # Wall distances
    walls = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
    min_wall_idx = np.argmin(walls)
    min_wall = walls[min_wall_idx]
    
    # Determine which is the binding constraint
    grad = np.zeros_like(pts)
    
    if min_pair_dist / 2.0 <= min_wall:
        # Pairwise constraint is active
        i, j = min_idx
        diff = pts[i] - pts[j]
        norm = np.linalg.norm(diff)
        if norm > 1e-12:
            unit = diff / norm
            grad[i] = -0.5 * unit  # d/dpts[i] of -d_ij/2 = -unit/2... 
            # Actually score = d_ij/2, so -score = -d_ij/2
            # d(-d_ij/2)/dpts[i] = -0.5 * (pts[i]-pts[j])/||pts[i]-pts[j]||
            grad[i] = -0.5 * unit
            grad[j] = 0.5 * unit
    else:
        # Wall constraint is active
        i = min_wall_idx
        x, y = pts[i]
        wall_x = 0.5 - abs(x)
        wall_y = 0.5 - abs(y)
        
        if wall_x <= wall_y:
            # x-wall is active
            if x >= 0:
                grad[i, 0] = 0.5  # d(0.5-x)/dx = -1, so d(-score)/dx = -(-1) = 1? 
                # score = 0.5 - x (if x > 0), -score = x - 0.5, d/dx = 1
                grad[i, 0] = 1.0
            else:
                # score = 0.5 + x, -score = -0.5 - x, d/dx = -1
                grad[i, 0] = -1.0
        else:
            # y-wall is active
            if y >= 0:
                grad[i, 1] = 1.0
            else:
                grad[i, 1] = -1.0
    
    return grad.flatten()


def _smooth_neg_score(pts, beta=50.0):
    """
    Smooth approximation of the negative score using log-sum-exp.
    score = min(c_1, c_2, ..., c_M)
    -score = -min(c_k) = min(-c_k)
    We use a smooth minimum: softmin(c_k) = -1/beta * log(sum(exp(-beta * c_k)))
    Then -softmin = 1/beta * log(sum(exp(-beta * c_k)))
    """
    pts = np.asarray(pts).reshape(-1, 2)
    n = len(pts)
    
    # Collect all constraint values
    # Wall constraints
    walls = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
    
    # Pairwise constraints
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    pair_dists = d[d < np.inf] / 2.0
    
    # Combine all constraints
    all_constraints = np.concatenate([walls, pair_dists])
    
    # Smooth minimum
    # softmin = -1/beta * log(sum(exp(-beta * c)))
    # We want -softmin = 1/beta * log(sum(exp(-beta * c)))
    # For numerical stability, use log-sum-exp trick
    max_c = np.max(all_constraints)
    exp_terms = np.exp(-beta * (all_constraints - max_c))
    log_sum = np.log(np.sum(exp_terms)) + (-beta * max_c)
    smooth_min = log_sum / beta
    
    return smooth_min


def _smooth_grad(pts, beta=50.0):
    """Gradient of the smooth negative score."""
    pts = np.asarray(pts).reshape(-1, 2)
    n = len(pts)
    
    # Wall constraints
    walls = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
    
    # Pairwise constraints
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    pair_dists = d[d < np.inf] / 2.0
    
    all_constraints = np.concatenate([walls, pair_dists])
    
    # Compute weights for softmin
    max_c = np.max(all_constraints)
    exp_terms = np.exp(-beta * (all_constraints - max_c))
    weights = exp_terms / np.sum(exp_terms)
    
    # Gradient of softmin w.r.t. each constraint is -weight_k
    # So gradient of -softmin w.r.t. constraint k is weight_k
    # We need d(-softmin)/dpts = sum_k weight_k * d(c_k)/dpts
    
    grad = np.zeros_like(pts)
    
    # Wall constraints: first n entries
    for i in range(n):
        w = weights[i]
        x, y = pts[i]
        wall_x = 0.5 - abs(x)
        wall_y = 0.5 - abs(y)
        
        if wall_x <= wall_y:
            if x >= 0:
                grad[i, 0] += w * (-1.0)  # d(0.5-x)/dx = -1
            else:
                grad[i, 0] += w * (1.0)   # d(0.5+x)/dx = 1
        else:
            if y >= 0:
                grad[i, 1] += w * (-1.0)
            else:
                grad[i, 1] += w * (1.0)
    
    # Pairwise constraints: remaining entries
    # We need to map back to pairs
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            w = weights[n + idx]
            diff = pts[i] - pts[j]
            norm = np.linalg.norm(diff)
            if norm > 1e-12:
                unit = diff / norm
                # d(d_ij/2)/dpts[i] = unit/2, d/dpts[j] = -unit/2
                grad[i] += w * 0.5 * unit
                grad[j] += w * (-0.5 * unit)
            idx += 1
    
    return grad.flatten()


def _initial_guess():
    """Generate an initial grid-like arrangement."""
    m = int(np.ceil(np.sqrt(N)))
    # Use a slightly perturbed grid to avoid symmetry
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pts = grid[:N].copy()
    # Add small random perturbation
    pts += 0.001 * np.random.randn(N, 2)
    # Clip to stay inside
    pts[:, 0] = np.clip(pts[:, 0], -0.49, 0.49)
    pts[:, 1] = np.clip(pts[:, 1], -0.49, 0.49)
    return pts


def construct():
    """Construct the circle packing."""
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    np.random.seed(42)
    
    best_pts = _initial_guess()
    best_score = _score(best_pts)
    
    # Multi-start optimization
    n_starts = 5
    
    for start_idx in range(n_starts):
        if time.time() > deadline - 20.0:
            break
        
        if start_idx == 0:
            pts = best_pts.copy()
        else:
            pts = _initial_guess()
        
        # Use L-BFGS-B with smooth objective
        options = {
            'maxiter': 2000,
            'ftol': 1e-12,
            'gtol': 1e-8,
            'maxls': 50
        }
        
        try:
            res = minimize(
                _smooth_neg_score,
                pts.flatten(),
                jac=_smooth_grad,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options=options,
                args=(50.0,)
            )
            new_pts = res.x.reshape(-1, 2)
            new_score = _score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        except Exception as e:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5.0:
        try:
            options = {
                'maxiter': 5000,
                'ftol': 1e-14,
                'gtol': 1e-10,
                'maxls': 100
            }
            res = minimize(
                _smooth_neg_score,
                best_pts.flatten(),
                jac=_smooth_grad,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options=options,
                args=(100.0,)
            )
            new_pts = res.x.reshape(-1, 2)
            new_score = _score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        except Exception as e:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
