"""Rewritten program for 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _compute_radius(pts):
    """Compute the max common radius for the given centers."""
    if pts.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check inside square [-0.5, 0.5]^2
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
    # Wall constraint: distance from each center to nearest wall
    wall_dists = np.minimum(0.5 - pts, pts + 0.5)
    r_wall = np.min(np.min(wall_dists, axis=1))
    # Pairwise distances
    dists = cdist(pts, pts)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    r_pair = min_pair / 2.0
    return min(r_wall, r_pair)


def _smooth_energy_and_grad(pts, beta=50.0):
    """
    Smooth surrogate: minimize -softmin of all constraint distances.
    Constraints: wall distances (2N values) and half pairwise distances (N*(N-1)/2 values).
    softmin(a; beta) = -1/beta * log(sum(exp(-beta * a_i)))
    gradient w.r.t. pts is computed analytically.
    """
    # Wall distances: for each point i, distance to each of 4 walls
    # d_left_i = pts[i,0] + 0.5
    # d_right_i = 0.5 - pts[i,0]
    # d_bottom_i = pts[i,1] + 0.5
    # d_top_i = 0.5 - pts[i,1]
    d_left = pts[:, 0] + 0.5
    d_right = 0.5 - pts[:, 0]
    d_bottom = pts[:, 1] + 0.5
    d_top = 0.5 - pts[:, 1]
    
    # Pairwise half-distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 2)
    dist_sq = np.sum(diff**2, axis=2)  # (N, N)
    # Only upper triangle
    iu = np.triu_indices(N, k=1)
    pair_dists = np.sqrt(dist_sq[iu]) / 2.0  # (M,)
    
    # All constraint values
    all_dists = np.concatenate([d_left, d_right, d_bottom, d_top, pair_dists])
    
    # Soft-min with temperature
    # softmin = -1/beta * log(sum(exp(-beta * a_i)))
    # For numerical stability, shift by min
    m = np.min(all_dists)
    shifted = -beta * (all_dists - m)
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    softmin = m - log_sum / beta
    
    # Gradient of softmin w.r.t. all_dists:
    # d(softmin)/d(a_k) = -exp(-beta*(a_k - m)) / sum(exp(-beta*(a_j - m)))
    # = -w_k where w_k = exp(-beta*(a_k - m)) / sum(...)
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    # d(softmin)/d(a_k) = -weights[k]
    # Energy = -softmin, so d(E)/d(a_k) = weights[k]
    
    # Now chain rule: d(E)/d(pts) = sum_k weights[k] * d(a_k)/d(pts)
    grad = np.zeros_like(pts)
    
    # Wall contributions
    # d(d_left_i)/d(pts[i,0]) = 1, others 0
    # d(d_right_i)/d(pts[i,0]) = -1
    # d(d_bottom_i)/d(pts[i,1]) = 1
    # d(d_top_i)/d(pts[i,1]) = -1
    w_left = weights[0:N]
    w_right = weights[N:2*N]
    w_bottom = weights[2*N:3*N]
    w_top = weights[3*N:4*N]
    
    grad[:, 0] += w_left - w_right
    grad[:, 1] += w_bottom - w_top
    
    # Pairwise contributions
    # For pair (i,j) with i<j, a = |p_i - p_j|/2
    # d(a)/d(p_i) = (p_i - p_j) / (2 * |p_i - p_j|) = (p_i - p_j) / (2 * 2*a) = (p_i - p_j)/(4a)
    # Wait: a = |diff|/2, so |diff| = 2a
    # d(a)/d(p_i) = (p_i - p_j) / (2 * |p_i - p_j|) = (p_i - p_j) / (4a)
    # d(a)/d(p_j) = (p_j - p_i) / (2 * |p_i - p_j|) = (p_j - p_i) / (4a)
    
    w_pair = weights[4*N:]  # (M,)
    i_idx, j_idx = iu
    # For each pair (i,j), contribution to grad[i] and grad[j]
    # diff[i,j] = p_i - p_j
    pair_diff = diff[i_idx, j_idx]  # (M, 2)
    pair_dist = np.linalg.norm(pair_diff, axis=1, keepdims=True)  # (M, 1)
    # Avoid division by zero
    pair_dist = np.maximum(pair_dist, 1e-12)
    
    # d(a)/d(p_i) = pair_diff / (2 * pair_dist)
    # d(a)/d(p_j) = -pair_diff / (2 * pair_dist)
    grad_ij = pair_diff / (2.0 * pair_dist)  # (M, 2)
    
    # grad[i] += w_pair[m] * grad_ij[m]
    # grad[j] -= w_pair[m] * grad_ij[m]
    for m in range(len(w_pair)):
        i = i_idx[m]
        j = j_idx[m]
        grad[i] += w_pair[m] * grad_ij[m]
        grad[j] -= w_pair[m] * grad_ij[m]
    
    energy = -softmin
    return energy, grad


def _hex_grid_init(n):
    """Create a hexagonal grid initialization for n points."""
    # Hexagonal packing: rows offset by half spacing
    # Try to find a good hexagonal arrangement
    # For n points, use roughly sqrt(n) rows
    rows = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
    cols = int(np.ceil(n / rows))
    
    # Hexagonal spacing
    # In a hexagonal lattice, horizontal spacing = d, vertical spacing = d * sqrt(3)/2
    # Alternate rows are offset by d/2
    # We want to fit in [-0.5, 0.5]^2
    # Width needed: (cols-1)*d + d/2 (for offset rows)
    # Height needed: (rows-1)*d*sqrt(3)/2
    
    # Estimate d to fit
    # Width: (cols - 0.5) * d <= 1.0 => d <= 1.0 / (cols - 0.5)
    # Height: (rows - 1) * d * sqrt(3)/2 <= 1.0 => d <= 1.0 / ((rows-1)*sqrt(3)/2)
    
    d_w = 1.0 / max(cols - 0.5, 1)
    d_h = 1.0 / max((rows - 1) * np.sqrt(3) / 2, 1)
    d = min(d_w, d_h)
    
    pts = []
    for r in range(rows):
        offset = (r % 2) * d / 2
        for c in range(cols):
            x = -0.5 + d/2 + c * d + offset
            y = -0.5 + d * np.sqrt(3) / 4 + r * d * np.sqrt(3) / 2
            if len(pts) < n:
                pts.append([x, y])
    
    pts = np.array(pts[:n])
    # Center the arrangement
    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])
    pts[:, 0] -= cx
    pts[:, 1] -= cy
    # Clip to square
    pts = np.clip(pts, -0.5, 0.5)
    return pts


def _initial_grid():
    """Create initial grid placement."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _random_initial():
    """Create random initial placement."""
    return np.random.uniform(-0.45, 0.45, (N, 2))


def _optimize_for_radius(pts0, deadline, beta_start=50.0, beta_end=500.0, max_iter=500):
    """
    Optimize using L-BFGS-B on the smooth surrogate.
    Returns (best_pts, best_radius).
    """
    pts = pts0.copy()
    best_pts = pts.copy()
    best_radius = _compute_radius(pts)
    
    # Schedule beta to increase over iterations
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Interpolate beta
        frac = it / max(max_iter - 1, 1)
        beta = beta_start * (beta_end / beta_start) ** frac
        
        def obj_and_grad(x):
            pts_2d = x.reshape(-1, 2)
            e, g = _smooth_energy_and_grad(pts_2d, beta=beta)
            return e, g.flatten()
        
        # Use L-BFGS-B
        res = minimize(
            obj_and_grad,
            pts.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 20, 'gtol': 1e-8}
        )
        
        new_pts = res.x.reshape(-1, 2)
        new_pts = np.clip(new_pts, -0.5, 0.5)
        
        new_r = _compute_radius(new_pts)
        if new_r > best_radius:
            best_radius = new_r
            best_pts = new_pts.copy()
        
        pts = new_pts
    
    return best_pts, best_radius


def _polish_centres(pts, deadline):
    """
    Final polishing step: iteratively increase a target radius and solve
    the feasibility problem.
    """
    best_pts = pts.copy()
    best_radius = _compute_radius(pts)
    
    # Try to increase radius in small steps
    current_r = best_radius
    for step in range(50):
        if time.time() > deadline:
            break
        
        target_r = current_r * 1.01  # Try 1% increase
        # Solve feasibility: find pts such that all pairwise dist >= 2*target_r
        # and all wall dist >= target_r
        # This is a non-convex problem; use the smooth surrogate with high beta
        
        def obj_and_grad(x):
            pts_2d = x.reshape(-1, 2)
            # Penalize violations of target radius
            # Wall violations
            wall_d = np.minimum(0.5 - pts_2d, pts_2d + 0.5)
            wall_viol = np.maximum(0, target_r - np.min(wall_d, axis=1))
            # Pairwise violations
            diff = pts_2d[:, np.newaxis, :] - pts_2d[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            min_dist = np.sqrt(np.min(dist_sq))
            pair_viol = max(0, 2 * target_r - min_dist)
            
            energy = np.sum(wall_viol**2) + pair_viol**2
            if energy < 1e-12:
                return energy, np.zeros_like(x)
            
            # Compute gradient numerically if needed, or use smooth surrogate
            # For simplicity, use smooth surrogate with high beta
            e, g = _smooth_energy_and_grad(pts_2d, beta=200.0)
            return e, g.flatten()
        
        res = minimize(
            obj_and_grad,
            best_pts.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 30, 'gtol': 1e-10}
        )
        
        new_pts = res.x.reshape(-1, 2)
        new_pts = np.clip(new_pts, -0.5, 0.5)
        new_r = _compute_radius(new_pts)
        
        if new_r > best_radius:
            best_radius = new_r
            best_pts = new_pts.copy()
            current_r = new_r
        else:
            # Can't increase further
            break
    
    return best_pts, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_pts = None
    best_radius = 0.0
    
    np.random.seed(42)
    
    # Try multiple initializations
    inits = [
        _initial_grid,
        _hex_grid_init,
        _random_initial,
        _random_initial,
        _random_initial,
    ]
    
    for init_fn in inits:
        if time.time() > deadline:
            break
        
        pts = init_fn()
        if pts.shape != (N, 2):
            continue
        
        # Optimize
        opt_pts, opt_r = _optimize_for_radius(pts, deadline)
        
        if opt_r > best_radius:
            best_radius = opt_r
            best_pts = opt_pts.copy()
    
    # Also try starting from best with perturbation
    if best_pts is not None and time.time() < deadline - 10:
        for trial in range(3):
            if time.time() > deadline:
                break
            pts = best_pts + np.random.randn(N, 2) * 0.005
            pts = np.clip(pts, -0.5, 0.5)
            opt_pts, opt_r = _optimize_for_radius(pts, deadline, beta_start=100.0, beta_end=1000.0, max_iter=200)
            if opt_r > best_radius:
                best_radius = opt_r
                best_pts = opt_pts.copy()
    
    # Final polish
    if best_pts is not None and time.time() < deadline - 5:
        best_pts, best_radius = _polish_centres(best_pts, deadline)
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _initial_grid()
    
    # Final safety check
    best_pts = np.clip(best_pts, -0.5, 0.5)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _initial_grid()
    
    return best_pts
# EVOLVE-BLOCK-END
