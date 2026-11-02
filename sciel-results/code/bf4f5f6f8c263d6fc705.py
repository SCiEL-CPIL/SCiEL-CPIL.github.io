"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 145.0  # safety margin before 150s kill


def _compute_score(pts):
    """Compute the packing score (minimum radius)."""
    n = len(pts)
    if n < 2:
        return 0.0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_dist_sq = np.min(dists_sq)
    min_dist = np.sqrt(min_dist_sq)
    
    radii_sq = np.sum(pts**2, axis=1)
    min_boundary = 1.0 - np.sqrt(np.max(radii_sq))
    
    return min(min_dist / 2.0, min_boundary)


def _get_critical_constraints(pts, tol=0.005):
    """
    Get the critical constraints: pairs and boundary distances
    that are within tol of the minimum.
    Returns (pair_indices, boundary_indices, min_val)
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    radii = np.sqrt(np.sum(pts**2, axis=1))
    boundary_dists = 1.0 - radii
    
    ii, jj = np.triu_indices(n, k=1)
    pair_vals = dists[ii, jj] / 2.0
    all_vals = np.concatenate([pair_vals, boundary_dists])
    min_val = np.min(all_vals)
    
    # Critical pairs: within tol of min
    pair_crit = np.where(pair_vals <= min_val + tol)[0]
    # Critical boundaries: within tol of min
    bound_crit = np.where(boundary_dists <= min_val + tol)[0]
    
    return pair_crit, bound_crit, min_val


def _soft_min_active(pts, active_pairs, active_bounds, beta):
    """
    Soft-min over only the active (critical) constraints.
    This focuses optimization on the tightest constraints.
    """
    n = len(pts)
    
    if len(active_pairs) == 0 and len(active_bounds) == 0:
        return 0.0, np.zeros_like(pts)
    
    # Compute distances for active pairs
    ii = np.array([active_pairs[k] for k in range(len(active_pairs))])
    jj = np.array([active_pairs[k] for k in range(len(active_pairs))])
    # Actually need to unpack from triu indices
    all_ii, all_jj = np.triu_indices(n, k=1)
    pair_i = all_ii[active_pairs]
    pair_j = all_jj[active_pairs]
    
    diff = pts[pair_i] - pts[pair_j]
    pair_dists = np.sqrt(np.sum(diff**2, axis=1))
    pair_vals = pair_dists / 2.0
    
    # Boundary distances for active bounds
    bound_radii = np.sqrt(np.sum(pts[active_bounds]**2, axis=1))
    bound_vals = 1.0 - bound_radii
    
    all_vals = np.concatenate([pair_vals, bound_vals])
    all_vals = np.maximum(all_vals, 1e-15)
    
    # Log-sum-exp
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    exp_vals = np.exp(neg_beta_x - max_val)
    sum_exp = np.sum(exp_vals)
    log_sum_exp = max_val + np.log(sum_exp)
    
    obj_val = log_sum_exp / beta
    
    # Weights
    weights = exp_vals / sum_exp
    w_pairs = weights[:len(active_pairs)]
    w_bounds = weights[len(active_pairs):]
    
    # Gradient
    grad = np.zeros_like(pts)
    
    # Pair contributions
    safe_dists = np.maximum(pair_dists, 1e-10)
    w_over_d = w_pairs / (2.0 * safe_dists)
    
    row_sum = np.zeros(n)
    weighted_sum = np.zeros((n, 2))
    
    np.add.at(row_sum, pair_i, w_over_d)
    np.add.at(row_sum, pair_j, w_over_d)
    
    np.add.at(weighted_sum, pair_i, w_over_d[:, np.newaxis] * pts[pair_j])
    np.add.at(weighted_sum, pair_j, w_over_d[:, np.newaxis] * pts[pair_i])
    
    grad_pairs = -pts * row_sum[:, np.newaxis] + weighted_sum
    
    # Boundary contributions
    safe_radii = np.maximum(bound_radii, 1e-10)
    grad_bounds = np.zeros_like(pts)
    grad_bounds[active_bounds] = w_bounds[:, np.newaxis] * pts[active_bounds] / safe_radii[:, np.newaxis]
    
    grad = grad_pairs + grad_bounds
    
    return obj_val, grad


def _local_search_active(pts, max_iters=200, tol=1e-12):
    """
    Local search that focuses on the active (tightest) constraints.
    Uses a direct approach: identify the single tightest constraint
    and move points to increase it.
    """
    pts = pts.copy()
    n = len(pts)
    best_score = _compute_score(pts)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
        
        # Find the single tightest constraint
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        radii = np.sqrt(np.sum(pts**2, axis=1))
        boundary_dists = 1.0 - radii
        
        ii, jj = np.triu_indices(n, k=1)
        pair_vals = dists[ii, jj] / 2.0
        all_vals = np.concatenate([pair_vals, boundary_dists])
        min_idx = np.argmin(all_vals)
        min_val = all_vals[min_idx]
        
        if min_val < 0:
            break
        
        # Compute subgradient direction to increase the minimum
        grad = np.zeros_like(pts)
        
        if min_idx < len(ii):
            i, j = ii[min_idx], jj[min_idx]
            d = dists[i, j]
            if d > 1e-10:
                diff_ij = pts[i] - pts[j]
                grad[i] -= diff_ij / (2 * d)
                grad[j] += diff_ij / (2 * d)
        else:
            i = min_idx - len(ii)
            r = radii[i]
            if r > 1e-10:
                grad[i] -= pts[i] / r
        
        # Adaptive step size
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-15:
            break
        
        # Try a few step sizes
        best_new_score = best_score
        best_new_pts = pts.copy()
        
        for lr in [0.05, 0.02, 0.01, 0.005, 0.002]:
            pts_try = pts + lr * grad
            # Project to unit disk
            radii_try = np.sqrt(np.sum(pts_try**2, axis=1))
            mask = radii_try >= 1.0
            if np.any(mask):
                pts_try[mask] *= (0.9999 / radii_try[mask])[:, np.newaxis]
            
            new_score = _compute_score(pts_try)
            if new_score > best_new_score:
                best_new_score = new_score
                best_new_pts = pts_try.copy()
        
        if best_new_score > best_score + tol:
            pts = best_new_pts
            best_score = best_new_score
        else:
            # Try smaller steps
            improved = False
            for lr in [0.001, 0.0005, 0.0002, 0.0001]:
                pts_try = pts + lr * grad
                radii_try = np.sqrt(np.sum(pts_try**2, axis=1))
                mask = radii_try >= 1.0
                if np.any(mask):
                    pts_try[mask] *= (0.9999 / radii_try[mask])[:, np.newaxis]
                
                new_score = _compute_score(pts_try)
                if new_score > best_score + tol:
                    pts = pts_try
                    best_score = new_score
                    improved = True
                    break
            
            if not improved:
                break
    
    return pts, best_score


def _hex_init(n):
    """Hexagonal lattice initialization."""
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0))) * 0.95
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
    idx = np.argsort(points[:, 2])
    selected = points[idx[:n]][:, :2]
    
    if len(selected) < n:
        remaining = n - len(selected)
        extra = np.random.randn(remaining, 2) * 0.3
        selected = np.vstack([selected, extra])
    
    noise = np.random.randn(n, 2) * 0.002
    selected = selected + noise
    
    radii = np.sqrt(np.sum(selected**2, axis=1))
    mask = radii >= 1.0
    if np.any(mask):
        selected[mask] *= (0.99 / radii[mask])[:, np.newaxis]
    
    return selected


def _ring_init(n):
    """Concentric ring initialization."""
    pts = []
    # Center point
    pts.append([0.0, 0.0])
    
    # Rings
    ring_radius = 0.15
    while len(pts) < n:
        # How many points fit on this ring?
        circumference = 2 * np.pi * ring_radius
        # Approximate spacing
        spacing = ring_radius * 0.9
        n_on_ring = int(circumference / spacing)
        if n_on_ring < 3:
            n_on_ring = 3
        
        angles = np.linspace(0, 2 * np.pi, n_on_ring, endpoint=False)
        for a in angles:
            if len(pts) >= n:
                break
            x = ring_radius * np.cos(a)
            y = ring_radius * np.sin(a)
            if np.hypot(x, y) < 1.0:
                pts.append([x, y])
        
        ring_radius += 0.12
    
    pts = np.array(pts[:n])
    noise = np.random.randn(n, 2) * 0.001
    pts = pts + noise
    
    radii = np.sqrt(np.sum(pts**2, axis=1))
    mask = radii >= 1.0
    if np.any(mask):
        pts[mask] *= (0.99 / radii[mask])[:, np.newaxis]
    
    return pts


def construct():
    """Construct 300 circle centers in unit disk to maximize minimum radius."""
    start_time = time.time()
    
    # Try multiple initializations
    initializations = [
        _hex_init(N),
        _ring_init(N),
        _hex_init(N),  # Different random seed
    ]
    
    best_pts = None
    best_score = -np.inf
    
    for init_pts in initializations:
        if time.time() > _DEADLINE - 20:
            break
        
        pts = init_pts.copy()
        score = _compute_score(pts)
        
        # Phase 1: Soft-min optimization with increasing beta
        beta_schedule = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        
        for beta in beta_schedule:
            if time.time() > _DEADLINE - 15:
                break
            
            def obj(x):
                if time.time() > _DEADLINE - 5:
                    raise ValueError("timeout")
                x = x.reshape(N, 2)
                # Use full soft-min for stability
                diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
                dists_sq = np.sum(diff**2, axis=2)
                np.fill_diagonal(dists_sq, np.inf)
                dists = np.sqrt(dists_sq)
                
                radii = np.sqrt(np.sum(x**2, axis=1))
                boundary_dists = 1.0 - radii
                
                ii, jj = np.triu_indices(N, k=1)
                pair_vals = dists[ii, jj] / 2.0
                all_vals = np.concatenate([pair_vals, boundary_dists])
                all_vals = np.maximum(all_vals, 1e-15)
                
                neg_beta_x = -beta * all_vals
                max_val = np.max(neg_beta_x)
                exp_vals = np.exp(neg_beta_x - max_val)
                sum_exp = np.sum(exp_vals)
                log_sum_exp = max_val + np.log(sum_exp)
                
                return log_sum_exp / beta
            
            def grad(x):
                if time.time() > _DEADLINE - 5:
                    raise ValueError("timeout")
                x = x.reshape(N, 2)
                
                diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
                dists_sq = np.sum(diff**2, axis=2)
                np.fill_diagonal(dists_sq, np.inf)
                dists = np.sqrt(dists_sq)
                
                radii = np.sqrt(np.sum(x**2, axis=1))
                boundary_dists = 1.0 - radii
                
                ii, jj = np.triu_indices(N, k=1)
                pair_vals = dists[ii, jj] / 2.0
                all_vals = np.concatenate([pair_vals, boundary_dists])
                all_vals = np.maximum(all_vals, 1e-15)
                
                neg_beta_x = -beta * all_vals
                max_val = np.max(neg_beta_x)
                exp_vals = np.exp(neg_beta_x - max_val)
                sum_exp = np.sum(exp_vals)
                
                weights = exp_vals / sum_exp
                w_pairs = weights[:len(ii)]
                w_bounds = weights[len(ii):]
                
                grad = np.zeros_like(x)
                
                safe_dists = np.maximum(dists[ii, jj], 1e-10)
                w_over_d = w_pairs / (2.0 * safe_dists)
                
                row_sum = np.zeros(N)
                weighted_sum = np.zeros((N, 2))
                
                np.add.at(row_sum, ii, w_over_d)
                np.add.at(row_sum, jj, w_over_d)
                
                np.add.at(weighted_sum, ii, w_over_d[:, np.newaxis] * x[jj])
                np.add.at(weighted_sum, jj, w_over_d[:, np.newaxis] * x[ii])
                
                grad_pairs = -x * row_sum[:, np.newaxis] + weighted_sum
                
                safe_radii = np.maximum(radii, 1e-10)
                grad_bounds = w_bounds[:, np.newaxis] * x / safe_radii[:, np.newaxis]
                
                grad = grad_pairs + grad_bounds
                return grad.ravel()
            
            try:
                maxiter = 100 if beta >= 1000 else 200
                result = minimize(
                    obj,
                    pts.ravel(),
                    jac=grad,
                    method='L-BFGS-B',
                    options={
                        'maxiter': maxiter,
                        'ftol': 1e-13,
                        'gtol': 1e-10,
                        'maxfun': maxiter * 2
                    }
                )
                
                new_pts = result.x.reshape(N, 2)
                radii = np.sqrt(np.sum(new_pts**2, axis=1))
                if np.all(radii < 1.0) and np.all(np.isfinite(new_pts)):
                    new_score = _compute_score(new_pts)
                    if new_score > score:
                        score = new_score
                        pts = new_pts.copy()
                        
            except Exception:
                pass
        
        # Phase 2: Active-set local search
        if time.time() < _DEADLINE - 10:
            try:
                polished_pts, polished_score = _local_search_active(pts, max_iters=300)
                if polished_score > score:
                    score = polished_score
                    pts = polished_pts.copy()
            except Exception:
                pass
        
        # Phase 3: Another round of soft-min with very high beta
        if time.time() < _DEADLINE - 8:
            for beta in [20000, 50000]:
                if time.time() > _DEADLINE - 5:
                    break
                
                def obj2(x):
                    if time.time() > _DEADLINE - 5:
                        raise ValueError("timeout")
                    x = x.reshape(N, 2)
                    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
                    dists_sq = np.sum(diff**2, axis=2)
                    np.fill_diagonal(dists_sq, np.inf)
                    dists = np.sqrt(dists_sq)
                    
                    radii = np.sqrt(np.sum(x**2, axis=1))
                    boundary_dists = 1.0 - radii
                    
                    ii, jj = np.triu_indices(N, k=1)
                    pair_vals = dists[ii, jj] / 2.0
                    all_vals = np.concatenate([pair_vals, boundary_dists])
                    all_vals = np.maximum(all_vals, 1e-15)
                    
                    neg_beta_x = -beta * all_vals
                    max_val = np.max(neg_beta_x)
                    exp_vals = np.exp(neg_beta_x - max_val)
                    sum_exp = np.sum(exp_vals)
                    log_sum_exp = max_val + np.log(sum_exp)
                    
                    return log_sum_exp / beta
                
                def grad2(x):
                    if time.time() > _DEADLINE - 5:
                        raise ValueError("timeout")
                    x = x.reshape(N, 2)
                    
                    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
                    dists_sq = np.sum(diff**2, axis=2)
                    np.fill_diagonal(dists_sq, np.inf)
                    dists = np.sqrt(dists_sq)
                    
                    radii = np.sqrt(np.sum(x**2, axis=1))
                    boundary_dists = 1.0 - radii
                    
                    ii, jj = np.triu_indices(N, k=1)
                    pair_vals = dists[ii, jj] / 2.0
                    all_vals = np.concatenate([pair_vals, boundary_dists])
                    all_vals = np.maximum(all_vals, 1e-15)
                    
                    neg_beta_x = -beta * all_vals
                    max_val = np.max(neg_beta_x)
                    exp_vals = np.exp(neg_beta_x - max_val)
                    sum_exp = np.sum(exp_vals)
                    
                    weights = exp_vals / sum_exp
                    w_pairs = weights[:len(ii)]
                    w_bounds = weights[len(ii):]
                    
                    grad = np.zeros_like(x)
                    
                    safe_dists = np.maximum(dists[ii, jj], 1e-10)
                    w_over_d = w_pairs / (2.0 * safe_dists)
                    
                    row_sum = np.zeros(N)
                    weighted_sum = np.zeros((N, 2))
                    
                    np.add.at(row_sum, ii, w_over_d)
                    np.add.at(row_sum, jj, w_over_d)
                    
                    np.add.at(weighted_sum, ii, w_over_d[:, np.newaxis] * x[jj])
                    np.add.at(weighted_sum, jj, w_over_d[:, np.newaxis] * x[ii])
                    
                    grad_pairs = -x * row_sum[:, np.newaxis] + weighted_sum
                    
                    safe_radii = np.maximum(radii, 1e-10)
                    grad_bounds = w_bounds[:, np.newaxis] * x / safe_radii[:, np.newaxis]
                    
                    grad = grad_pairs + grad_bounds
                    return grad.ravel()
                
                try:
                    result = minimize(
                        obj2,
                        pts.ravel(),
                        jac=grad2,
                        method='L-BFGS-B',
                        options={
                            'maxiter': 80,
                            'ftol': 1e-14,
                            'gtol': 1e-11,
                            'maxfun': 160
                        }
                    )
                    
                    new_pts = result.x.reshape(N, 2)
                    radii = np.sqrt(np.sum(new_pts**2, axis=1))
                    if np.all(radii < 1.0) and np.all(np.isfinite(new_pts)):
                        new_score = _compute_score(new_pts)
                        if new_score > score:
                            score = new_score
                            pts = new_pts.copy()
                            
                except Exception:
                    pass
        
        # Final local polish
        if time.time() < _DEADLINE - 5:
            try:
                final_pts, final_score = _local_search_active(pts, max_iters=200)
                if final_score > score:
                    score = final_score
                    pts = final_pts.copy()
            except Exception:
                pass
        
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Final safety check
    if best_pts is None:
        best_pts = _hex_init(N)
    
    radii = np.sqrt(np.sum(best_pts**2, axis=1))
    mask = radii >= 1.0
    if np.any(mask):
        best_pts[mask] *= (0.999 / radii[mask])[:, np.newaxis]
    
    return best_pts
# EVOLVE-BLOCK-END
