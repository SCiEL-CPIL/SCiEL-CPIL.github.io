"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 115.0  # safety margin before 120s kill


def _hex_lattice_init(n):
    """Generate hexagonal lattice points and select n closest to center."""
    # Estimate spacing for ~n points in unit disk
    # Area of hex lattice per point ~ s^2 * sqrt(3)/2
    # Area of unit disk = pi
    # n * s^2 * sqrt(3)/2 ~ pi  =>  s ~ sqrt(2*pi / (n*sqrt(3)))
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    # Add margin
    s *= 0.95
    
    k = int(1.0 / s) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0:
                points.append((x, y, r))
    
    points = np.array(points)
    # Sort by distance from center, take closest n
    idx = np.argsort(points[:, 2])
    selected = points[idx[:n]][:, :2]
    
    if len(selected) < n:
        # Fallback: fill remaining randomly
        remaining = n - len(selected)
        extra = np.random.randn(remaining, 2) * 0.3
        selected = np.vstack([selected, extra])
    
    return selected


def _compute_score(pts):
    """Compute the packing score (minimum radius)."""
    n = len(pts)
    if n < 2:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Distance to boundary
    radii = np.sqrt(np.sum(pts**2, axis=1))
    min_boundary = 1.0 - np.max(radii)
    
    return min(min_dist / 2.0, min_boundary)


def _soft_min_objective(pts, beta):
    """
    Negative soft-min of pairwise distances and boundary distances.
    We want to maximize the minimum, so minimize the negative.
    """
    n = len(pts)
    if n < 2:
        return 0.0
    
    # Pairwise distances (vectorized)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Boundary distances
    radii_sq = np.sum(pts**2, axis=1)
    boundary_dists = 1.0 - np.sqrt(radii_sq)
    
    # Soft-min over all constraints
    # For pairwise: we use distances (not squared) for the min
    # But for numerical stability, work with squared then convert
    # Actually, let's work directly with distances
    
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, 1e10)
    
    # Collect all constraint values
    # Pairwise distances / 2 (since radius = min_dist/2)
    pair_vals = dists / 2.0
    # Boundary distances
    bound_vals = boundary_dists
    
    all_vals = np.concatenate([pair_vals.ravel(), bound_vals])
    all_vals = all_vals[all_vals > 1e-10]  # filter out zero/invalid
    
    if len(all_vals) == 0:
        return 0.0
    
    # Log-sum-exp for soft-min: -softmin(x) = logsumexp(-x) / beta
    # softmin_beta(x) = -1/beta * log(sum(exp(-beta * x)))
    # So negative soft-min = 1/beta * log(sum(exp(-beta * x)))
    
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    if max_val > 700:
        # Overflow protection
        return -1.0 / beta * (max_val + np.log(np.sum(np.exp(neg_beta_x - max_val))))
    
    log_sum = np.log(np.sum(np.exp(neg_beta_x)))
    neg_softmin = log_sum / beta
    
    return neg_softmin


def _soft_min_grad(pts, beta):
    """Gradient of the negative soft-min objective."""
    n = len(pts)
    if n < 2:
        return np.zeros_like(pts)
    
    # Recompute values for weights
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, 1e10)
    dists = np.sqrt(dists_sq)
    
    radii_sq = np.sum(pts**2, axis=1)
    boundary_dists = 1.0 - np.sqrt(radii_sq)
    
    pair_vals = dists / 2.0
    bound_vals = boundary_dists
    
    all_vals = np.concatenate([pair_vals.ravel(), bound_vals])
    all_vals = np.maximum(all_vals, 1e-10)
    
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    exp_vals = np.exp(neg_beta_x - max_val)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    # Gradient of -softmin = (1/beta) * sum(weights * grad(log(exp(-beta*x))) )
    # = (1/beta) * sum(weights * (-beta * grad_x))
    # = -sum(weights * grad_x)
    
    grad = np.zeros_like(pts)
    
    # Pairwise contributions
    pair_weights = weights[:n * (n-1)].reshape(n, n-1)
    # Need to handle the upper/lower triangle properly
    # Actually, let's recompute more carefully
    
    # For each pair (i,j) with i<j, the constraint is dist(i,j)/2
    # grad_{pts[i]} of dist(i,j)/2 = (pts[i] - pts[j]) / (2 * dist(i,j))
    # grad_{pts[j]} of dist(i,j)/2 = (pts[j] - pts[i]) / (2 * dist(i,j))
    
    # Rebuild pairwise part
    n_pairs = n * (n - 1) // 2
    pair_weights_2d = weights[:n_pairs]
    
    # Build index arrays for pairs
    ii, jj = np.triu_indices(n, k=1)
    pair_dists = dists[ii, jj]
    pair_dists_safe = np.maximum(pair_dists, 1e-10)
    
    for idx in range(n_pairs):
        i, j = ii[idx], jj[idx]
        w = pair_weights_2d[idx]
        d = pair_dists_safe[idx]
        # grad of dist/2 w.r.t. pts[i] = (pts[i]-pts[j])/(2*d)
        # grad of dist/2 w.r.t. pts[j] = (pts[j]-pts[i])/(2*d)
        # Objective is -softmin, so grad_obj = -sum(w * grad_constraint)
        grad[i] -= w * (pts[i] - pts[j]) / (2.0 * d)
        grad[j] -= w * (pts[j] - pts[i]) / (2.0 * d)
    
    # Boundary contributions
    bound_weights = weights[n_pairs:]
    radii_safe = np.sqrt(np.maximum(radii_sq, 1e-10))
    for i in range(n):
        w = bound_weights[i]
        # grad of (1 - |pts[i]|) w.r.t. pts[i] = -pts[i]/|pts[i]|
        # grad_obj = -sum(w * grad_constraint) = -w * (-pts[i]/|pts[i]|) = w * pts[i]/|pts[i]|
        if radii_sq[i] > 1e-10:
            grad[i] += w * pts[i] / radii_safe[i]
    
    return grad


def construct():
    """Construct 300 circle centers in unit disk to maximize minimum radius."""
    start_time = time.time()
    
    # Initialize with hexagonal lattice
    pts = _hex_lattice_init(N)
    
    # Ensure valid initial points (inside unit disk)
    radii = np.sqrt(np.sum(pts**2, axis=1))
    mask = radii < 1.0
    if np.sum(mask) < N:
        # Rescale
        pts = pts * (0.95 / np.max(radii))
    
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    # Adaptive beta schedule: start low, increase over time
    beta_schedule = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    
    for beta in beta_schedule:
        if time.time() > _DEADLINE - 10:
            break
        
        # Define objective and gradient
        def obj(x):
            if time.time() > _DEADLINE - 5:
                raise ValueError("timeout")
            x = x.reshape(N, 2)
            return _soft_min_objective(x, beta)
        
        def grad(x):
            if time.time() > _DEADLINE - 5:
                raise ValueError("timeout")
            x = x.reshape(N, 2)
            return _soft_min_grad(x, beta).ravel()
        
        try:
            result = minimize(
                obj,
                pts.ravel(),
                jac=grad,
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxfun': 1000
                }
            )
            
            new_pts = result.x.reshape(N, 2)
            
            # Check validity
            radii = np.sqrt(np.sum(new_pts**2, axis=1))
            if np.all(radii < 1.0) and np.all(np.isfinite(new_pts)):
                new_score = _compute_score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
                    pts = new_pts.copy()
            else:
                # Invalid, keep previous
                pass
                
        except Exception:
            # Timeout or other error, keep best
            pass
    
    # Final polish: try to improve with higher beta
    if time.time() < _DEADLINE - 15:
        for beta in [5000, 10000, 20000]:
            if time.time() > _DEADLINE - 5:
                break
            
            def obj(x):
                if time.time() > _DEADLINE - 5:
                    raise ValueError("timeout")
                x = x.reshape(N, 2)
                return _soft_min_objective(x, beta)
            
            def grad(x):
                if time.time() > _DEADLINE - 5:
                    raise ValueError("timeout")
                x = x.reshape(N, 2)
                return _soft_min_grad(x, beta).ravel()
            
            try:
                result = minimize(
                    obj,
                    best_pts.ravel(),
                    jac=grad,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 300,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                        'maxfun': 500
                    }
                )
                
                new_pts = result.x.reshape(N, 2)
                radii = np.sqrt(np.sum(new_pts**2, axis=1))
                if np.all(radii < 1.0) and np.all(np.isfinite(new_pts)):
                    new_score = _compute_score(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_pts = new_pts.copy()
                        
            except Exception:
                pass
    
    return best_pts
# EVOLVE-BLOCK-END
