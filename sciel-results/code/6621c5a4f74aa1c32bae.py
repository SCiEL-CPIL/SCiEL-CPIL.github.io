"""
Strategy: Deterministic Gradient-Based Optimization using Log-Sum-Exp Soft-Min.

The current program uses Simulated Annealing (SA), which is a stochastic local search.
This program replaces that with a deterministic, differentiable approach:
1. Approximate the "min" operation in the score function using a Log-Sum-Exp (soft-min)
   transformation. This creates a smooth, differentiable objective function.
2. Use L-BFGS-B (a quasi-Newton method) to maximize this smooth objective.
3. Perform multiple restarts from different high-quality initializations (hexagonal lattice
   variants) to explore different basins of attraction.

Why this is different:
- SA relies on random moves and temperature schedules. It can get stuck in local minima
  and requires many evaluations to converge.
- Gradient-based methods use the shape of the landscape to take large, directed steps.
  For smooth approximations of packing problems, these often converge to higher quality
  solutions faster and more reliably than random perturbations.
"""

import numpy as np
import time
import scipy.optimize as opt
from scipy.special import logsumexp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

N = 300
TIME_LIMIT = 118.0  # Safety margin for 150s kill switch

def _hex_init(rotation=0.0, jitter=0.0, spacing=None):
    """
    Generate a hexagonal lattice initialization.
    Returns an array of shape (N, 2).
    """
    if spacing is None:
        spacing = 0.105
    
    # Generate points in a grid large enough to cover the unit circle
    k = int(1.1 / spacing) + 2
    points = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            points.append([x, y])
    
    points = np.array(points)
    
    # Rotate points
    if rotation != 0.0:
        c, s = np.cos(rotation), np.sin(rotation)
        R = np.array([[c, -s], [s, c]])
        points = points @ R.T
        
    # Add jitter if requested
    if jitter > 0.0:
        points += np.random.normal(0, jitter, points.shape)
        
    # Filter points strictly inside unit circle (with small margin)
    radii = np.linalg.norm(points, axis=1)
    mask = radii < 0.99
    points = points[mask]
    radii = radii[mask]
    
    # Sort by radius (closest to center first) and take N
    idx = np.argsort(radii)
    points = points[idx]
    
    if len(points) < N:
        # Fallback: if we don't have enough points (shouldn't happen), 
        # pad with random points
        theta = np.random.uniform(0, 2 * np.pi, N - len(points))
        rho = np.sqrt(np.random.uniform(0, 0.5, N - len(points)))
        pad = np.column_stack((rho * np.cos(theta), rho * np.sin(theta)))
        points = np.vstack([points, pad])
    else:
        points = points[:N]
        
    return points


def _concentric_ring_init(rings=20, jitter=0.0):
    """
    Alternative initialization: concentric rings with points distributed evenly.
    """
    points = [[0.0, 0.0]]
    for ring in range(1, rings):
        # Number of points on this ring increases with ring number
        n_pts = 6 * ring
        # Radius of this ring
        r = ring * 0.053
        if r > 0.95:
            break
        for k in range(n_pts):
            theta = 2 * np.pi * k / n_pts + ring * np.pi / 6  # stagger rings
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y])
    
    points = np.array(points)
    
    if jitter > 0.0:
        points += np.random.normal(0, jitter, points.shape)
    
    # Filter and pad/truncate to N
    radii = np.linalg.norm(points, axis=1)
    mask = radii < 0.99
    points = points[mask]
    radii = radii[mask]
    idx = np.argsort(radii)
    points = points[idx]
    
    if len(points) < N:
        theta = np.random.uniform(0, 2 * np.pi, N - len(points))
        rho = np.sqrt(np.random.uniform(0, 0.9, N - len(points)))
        pad = np.column_stack((rho * np.cos(theta), rho * np.sin(theta)))
        points = np.vstack([points, pad])
    else:
        points = points[:N]
        
    return points

def _objective_and_grad(flat_x, beta=50.0):
    """
    Objective: Minimize negative soft-min of distances.
    Returns (objective_value, gradient_flat).
    """
    points = flat_x.reshape(N, 2)
    
    # Calculate pairwise distances
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(dists, np.inf)
    
    # Values for pairs: dists / 2
    v_pairs = dists / 2.0
    
    # Boundary
    radii = np.linalg.norm(points, axis=1)
    v_bounds = 1.0 - radii
    
    # Clip to avoid numerical issues
    v_pairs = np.clip(v_pairs, 1e-10, None)
    v_bounds = np.clip(v_bounds, 1e-10, None)
    
    # Log weights
    log_w_pairs = -beta * v_pairs
    log_w_bounds = -beta * v_bounds
    
    mask_diag = np.eye(N, dtype=bool)
    
    # Compute log_S using logsumexp
    log_w_pairs_valid = np.where(mask_diag, -np.inf, log_w_pairs)
    all_log_w = np.concatenate([log_w_pairs_valid.flatten(), log_w_bounds])
    log_S = logsumexp(all_log_w)
    
    # Weights
    w_pairs = np.exp(log_w_pairs - log_S)
    w_pairs[mask_diag] = 0.0
    w_bounds = np.exp(log_w_bounds - log_S)
    
    # Objective value: minimize log_S / beta
    obj = log_S / beta
    
    # Gradient
    grad = np.zeros_like(points)
    
    # Pair contributions: d/dx_i [w_ij * |x_i - x_j|/2]
    # = w_ij * (x_i - x_j) / (2 * |x_i - x_j|)
    with np.errstate(divide='ignore', invalid='ignore'):
        factor_pairs = np.where(dists > 1e-8, w_pairs * 0.5 / dists, 0.0)
        
    sum_factor = np.sum(factor_pairs, axis=1)
    sum_factor_c = factor_pairs @ points
    grad_pairs = points * sum_factor[:, np.newaxis] - sum_factor_c
    
    # Boundary contributions: d/dx_i [w_i * (1 - |x_i|)] = -w_i * x_i / |x_i|
    with np.errstate(divide='ignore', invalid='ignore'):
        grad_bound_factors = np.where(radii > 1e-8, w_bounds / radii, 0.0)
        
    grad_bounds = -points * grad_bound_factors[:, np.newaxis]
    
    grad = grad_pairs + grad_bounds
    grad_flat = grad.flatten()
    
    return obj, grad_flat


def _compute_true_score(points):
    """Compute the actual packing radius (hard min)."""
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    radii = np.linalg.norm(points, axis=1)
    min_bound_dist = np.min(1.0 - radii)
    return min(min_pair_dist / 2.0, min_bound_dist)


def _active_set_polish(points, max_iter=50, tol=1e-12):
    """
    Final polish using active-set method: identify binding constraints,
    solve the KKT linear system for the contact graph, and take a step.
    """
    current = points.copy()
    best_score = _compute_true_score(current)
    
    for iteration in range(max_iter):
        if time.time() > TIME_LIMIT - 3.0:
            break
        
        score = _compute_true_score(current)
        if score <= best_score:
            break
        best_score = score
        
        # Find the tightest constraints
        diffs = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        np.fill_diagonal(dists, np.inf)
        radii = np.linalg.norm(current, axis=1)
        bound_dists = 1.0 - radii
        
        # The bottleneck radius
        r_pair = np.min(dists) / 2.0
        r_bound = np.min(bound_dists)
        r_bottleneck = min(r_pair, r_bound)
        
        # Select active set: pairs with distance < r_bottleneck * 1.2 + small_margin
        # and boundary points with dist < r_bottleneck * 1.2 + small_margin
        margin = r_bottleneck * 0.15 + 1e-8
        active_pairs_mask = dists < (r_bottleneck + margin) * 2.0
        active_pairs_mask[np.diag_indices(N)] = False
        active_bounds_mask = bound_dists < (r_bottleneck + margin)
        
        # Get indices of active pairs
        active_pair_indices = np.argwhere(active_pairs_mask)
        n_active_pairs = len(active_pair_indices)
        n_active_bounds = np.sum(active_bounds_mask)
        
        if n_active_pairs == 0 and n_active_bounds == 0:
            break
        
        # Build the KKT system:
        # We want to find direction d such that:
        # 1. For each active pair (i,j): d_i - d_j is along (x_i - x_j)/|x_i - x_j|
        #    and the component along this direction is positive (separating)
        # 2. For each active boundary point i: d_i is along x_i/|x_i| (outward)
        # 3. Minimize ||d||^2 subject to these constraints
        #
        # This is a quadratic program. We solve it via KKT:
        # Minimize 0.5 * d^T d
        # Subject to: A * d >= b (where A encodes the active constraints)
        #
        # For simplicity, we use a gradient-based approach on the active set only.
        
        # Use high-beta soft-min on active constraints only
        beta = 300.0
        
        def active_obj_grad(flat_x):
            pts = flat_x.reshape(N, 2)
            
            # Recompute distances
            d = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            ds = np.linalg.norm(d, axis=-1)
            np.fill_diagonal(ds, np.inf)
            
            rad = np.linalg.norm(pts, axis=1)
            bd = 1.0 - rad
            
            # Only active constraints
            v_p = np.where(active_pairs_mask, ds / 2.0, np.inf)
            v_b = np.where(active_bounds_mask, bd, np.inf)
            
            # Clip to avoid log(0)
            v_p = np.clip(v_p, 1e-10, None)
            v_b = np.clip(v_b, 1e-10, None)
            
            log_wp = -beta * v_p
            log_wb = -beta * v_b
            
            # Set inactive to -inf
            log_wp[~active_pairs_mask] = -np.inf
            log_wb[~active_bounds_mask] = -np.inf
            log_wp[np.diag_indices(N)] = -np.inf
            
            all_lw = np.concatenate([log_wp.flatten(), log_wb])
            
            # Handle all-inf case
            finite_mask = np.isfinite(all_lw)
            if not np.any(finite_mask):
                return 0.0, np.zeros_like(flat_x)
            
            log_S = logsumexp(all_lw)
            
            wp = np.exp(log_wp - log_S)
            wp[~active_pairs_mask] = 0.0
            wp[np.diag_indices(N)] = 0.0
            wb = np.exp(log_wb - log_S)
            wb[~active_bounds_mask] = 0.0
            
            obj = log_S / beta
            
            grad = np.zeros_like(pts)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                fp = np.where(ds > 1e-8, wp * 0.5 / ds, 0.0)
            
            sf = np.sum(fp, axis=1)
            sfc = fp @ pts
            gp = pts * sf[:, np.newaxis] - sfc
            
            with np.errstate(divide='ignore', invalid='ignore'):
                gb_f = np.where(rad > 1e-8, wb / rad, 0.0)
            
            gb = -pts * gb_f[:, np.newaxis]
            
            grad = gp + gb
            return obj, grad.flatten()
        
        try:
            res = opt.minimize(
                active_obj_grad,
                current.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 30, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 20}
            )
            new_points = res.x.reshape(N, 2)
            
            new_score = _compute_true_score(new_points)
            if new_score > best_score + 1e-15:
                best_score = new_score
                current = new_points
            else:
                break
        except Exception:
            break
    
    return current, best_score


def _kkt_contact_polish(points, max_iter=30):
    """
    KKT-based polish: identify the contact graph and solve for the optimal
    perturbation direction that maximizes the minimum constraint margin.
    
    At the optimum, we have a set of active constraints (touching pairs and
    boundary contacts). The KKT conditions give us a linear system to solve.
    """
    current = points.copy()
    best_score = _compute_true_score(current)
    
    for iteration in range(max_iter):
        if time.time() > TIME_LIMIT - 2.0:
            break
        
        score = _compute_true_score(current)
        if score <= best_score:
            break
        best_score = score
        
        # Identify active constraints
        diffs = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        np.fill_diagonal(dists, np.inf)
        radii = np.linalg.norm(current, axis=1)
        bound_dists = 1.0 - radii
        
        r_bottleneck = min(np.min(dists) / 2.0, np.min(bound_dists))
        
        # Active pairs: within 2% of bottleneck
        margin = r_bottleneck * 0.02 + 1e-10
        active_pairs_mask = dists < (r_bottleneck + margin) * 2.0
        active_pairs_mask[np.diag_indices(N)] = False
        active_bounds_mask = bound_dists < (r_bottleneck + margin)
        
        n_active_pairs = np.sum(active_pairs_mask)
        n_active_bounds = np.sum(active_bounds_mask)
        
        if n_active_pairs == 0 and n_active_bounds == 0:
            break
        
        # Build the constraint Jacobian for the active set
        # For pair (i,j): constraint is |x_i - x_j| >= 2r
        # Gradient w.r.t. x_i: (x_i - x_j) / |x_i - x_j|
        # Gradient w.r.t. x_j: -(x_i - x_j) / |x_i - x_j|
        #
        # For boundary i: constraint is 1 - |x_i| >= r
        # Gradient w.r.t. x_i: -x_i / |x_i|
        
        # We want to find d that maximizes the minimum margin increase.
        # This is equivalent to: maximize t subject to A*d >= t, ||d|| <= 1
        # Or equivalently, solve for the direction that equalizes the active margins.
        
        # Simple approach: use the gradient of the min over active constraints
        # and take a step in that direction with line search.
        
        # Compute the "pressure" on each active constraint
        pair_dists = dists[active_pairs_mask]
        bound_vals = bound_dists[active_bounds_mask]
        
        # Normalize: how close each active constraint is to the bottleneck
        pair_proximity = 1.0 - (pair_dists / 2.0 - r_bottleneck) / (r_bottleneck + 1e-15)
        bound_proximity = 1.0 - (bound_vals - r_bottleneck) / (r_bottleneck + 1e-15)
        
        # Clip to [0, 1]
        pair_proximity = np.clip(pair_proximity, 0, 1)
        bound_proximity = np.clip(bound_proximity, 0, 1)
        
        # Compute gradient direction
        grad = np.zeros_like(current)
        
        # Pair contributions
        pair_indices = np.argwhere(active_pairs_mask)
        for idx in pair_indices:
            i, j = idx
            d_ij = current[i] - current[j]
            dist_ij = np.linalg.norm(d_ij)
            if dist_ij > 1e-10:
                direction = d_ij / dist_ij
                weight = pair_proximity[idx[0] * N + idx[1]] if idx[0] * N + idx[1] < len(pair_proximity) else 0.5
                grad[i] += weight * direction
                grad[j] -= weight * direction
        
        # Boundary contributions
        for i in np.where(active_bounds_mask)[0]:
            rad_i = radii[i]
            if rad_i > 1e-10:
                direction = -current[i] / rad_i
                weight = bound_proximity[np.where(active_bounds_mask)[0].tolist().index(i)] if i in np.where(active_bounds_mask)[0] else 0.5
                grad[i] += weight * direction
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        grad /= grad_norm
        
        # Line search: try a small step
        step_sizes = [0.001, 0.0005, 0.0002, 0.0001]
        improved = False
        for step in step_sizes:
            new_points = current + step * grad
            new_score = _compute_true_score(new_points)
            if new_score > best_score + 1e-15:
                best_score = new_score
                current = new_points
                improved = True
                break
        
        if not improved:
            break
    
    return current, best_score

def construct():
    start_time = time.time()
    best_points = None
    best_score = -np.inf
    
    # Define multiple different initializations to restart from
    np.random.seed(42)  # Reproducibility
    inits = [
        _hex_init(rotation=0.0, jitter=0.0),
        _hex_init(rotation=np.pi/6, jitter=0.0),
        _hex_init(rotation=0.0, jitter=0.008),
        _hex_init(rotation=np.pi/3, jitter=0.008),
        _hex_init(rotation=np.pi/12, jitter=0.005),
        _concentric_ring_init(rings=20, jitter=0.003),
    ]
    
    # Beta annealing schedule: start smooth, end sharp
    beta_schedule = [15.0, 40.0, 80.0, 150.0]
    
    for i, init_points in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 12.0:
            break
            
        x0 = init_points.flatten()
        
        # Multi-phase beta annealing
        current_x = x0.copy()
        
        for phase, beta in enumerate(beta_schedule):
            if time.time() - start_time > TIME_LIMIT - 12.0:
                break
                
            # Create closure with fixed beta
            def f_and_g(x, _beta=beta):
                return _objective_and_grad(x, _beta)
            
            try:
                # Scale maxiter based on remaining time
                remaining = TIME_LIMIT - 12.0 - (time.time() - start_time)
                maxiter = max(30, int(min(150, remaining * 5)))
                
                res = opt.minimize(
                    f_and_g,
                    current_x,
                    method='L-BFGS-B',
                    jac=True,
                    options={
                        'maxiter': maxiter,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxls': 30
                    }
                )
                current_x = res.x
            except Exception:
                break
            
            # Check time after each phase
            if time.time() - start_time > TIME_LIMIT - 12.0:
                break
        
        x_opt = current_x.reshape(N, 2)
        
        # Calculate true score
        score = _compute_true_score(x_opt)
        
        if score > best_score:
            best_score = score
            best_points = x_opt.copy()
            
        if time.time() - start_time > TIME_LIMIT - 18.0:
            break
    
    # Final active-set polish on the best solution
    if best_points is not None and time.time() - start_time < TIME_LIMIT - 8.0:
        polished, polished_score = _active_set_polish(best_points, max_iter=40)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished
    
    # KKT contact polish
    if best_points is not None and time.time() - start_time < TIME_LIMIT - 4.0:
        kkt_polished, kkt_score = _kkt_contact_polish(best_points, max_iter=20)
        if kkt_score > best_score:
            best_score = kkt_score
            best_points = kkt_polished
    
    # Final validity check
    if best_points is None:
        best_points = inits[0]
        
    radii = np.linalg.norm(best_points, axis=1)
    valid = np.isfinite(best_points).all(axis=1) & (radii < 1.0)
    if not np.all(valid):
        theta = np.random.uniform(0, 2 * np.pi, N)
        rho = np.sqrt(np.random.uniform(0, 0.9, N))
        best_points = np.column_stack((rho * np.cos(theta), rho * np.sin(theta)))
        
    return best_points

# Helper for testing
if __name__ == "__main__":
    pts = construct()
    # Calculate score
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    radii = np.linalg.norm(pts, axis=1)
    min_bound_dist = np.min(1.0 - radii)
    score = min(min_pair_dist / 2.0, min_bound_dist)
    print(f"Score: {score}")
