"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(pts):
    """Compute the packing radius: min of half-min-distance and min(1-|c_i|)."""
    n = len(pts)
    if n < 2:
        return 0.0
    # Pairwise distances using vectorized computation
    dx = pts[:, 0:1] - pts[:, 0:1].T
    dy = pts[:, 1:2] - pts[:, 1:2].T
    dist_sq = dx * dx + dy * dy
    np.fill_diagonal(dist_sq, np.inf)
    min_dist = np.sqrt(np.min(dist_sq))
    # Distance to boundary
    r = np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(1.0 - r)
    return min(min_dist / 2.0, min_boundary)

def _soft_min(values, beta):
    """
    Smooth approximation to min(values) using log-sum-exp.
    As beta -> infinity, soft_min -> true min.
    """
    # Use the log-sum-exp trick for numerical stability
    m = np.min(values)
    # If all values are very close, just return m
    if np.max(values) - m < 1e-12:
        return m
    # soft_min = -log(sum(exp(-beta * (v - m)))) / beta + m
    # = m - (1/beta) * log(sum(exp(-beta * (v - m))))
    exps = np.exp(-beta * (values - m))
    log_sum = np.log(np.sum(exps))
    return m - log_sum / beta

def _soft_min_grad(values, beta):
    """
    Gradient of soft_min with respect to values.
    Returns array of same shape as values.
    """
    m = np.min(values)
    if np.max(values) - m < 1e-12:
        return np.ones_like(values) / len(values)
    exps = np.exp(-beta * (values - m))
    sum_exps = np.sum(exps)
    # d(soft_min)/d(v_i) = -exp(-beta*(v_i - m)) / sum_exps
    return -exps / sum_exps

def _generate_hex_init(n):
    """Generate hexagonal lattice points clipped to unit disk, select n closest to center."""
    # Use a finer spacing to get more candidate points, then select n best
    spacing = 0.08
    k = int(1.1 / spacing) + 2
    points = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            points.append([x, y])
    points = np.array(points)
    r = np.hypot(points[:, 0], points[:, 1])
    mask = r <= 1.0 - spacing / 2.0
    points = points[mask]
    
    if len(points) < n:
        # Fallback: generate more points with smaller spacing
        spacing = 0.06
        k = int(1.1 / spacing) + 2
        points = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                points.append([x, y])
        points = np.array(points)
        r = np.hypot(points[:, 0], points[:, 1])
        mask = r <= 1.0 - spacing / 2.0
        points = points[mask]

    # Sort by distance to center and take the first n
    r = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(r, kind="stable")[:n]
    return points[idx]

def _compute_soft_min_objective_and_grad(pts, beta):
    """
    Compute the soft-min objective (to maximize) and its gradient.
    Objective = soft_min([dist_ij/2 for all i<j, (1-|c_i|) for all i])
    Uses vectorized gradient computation for speed.
    """
    n = len(pts)
    # Pairwise distances
    dx = pts[:, 0:1] - pts[:, 0:1].T
    dy = pts[:, 1:2] - pts[:, 1:2].T
    dist_sq = dx * dx + dy * dy
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    
    # All pairwise distances (upper triangle)
    iu, ju = np.triu_indices(n, k=1)
    pair_dists = dist[iu, ju]
    
    # Boundary gaps
    r = np.hypot(pts[:, 0], pts[:, 1])
    boundary_gaps = 1.0 - r
    
    # Combine all constraints: pair_dists/2 and boundary_gaps
    all_values = np.concatenate([pair_dists / 2.0, boundary_gaps])
    
    # Soft min value
    soft_min_val = _soft_min(all_values, beta)
    
    # Gradient of soft min w.r.t. all_values
    val_grad = _soft_min_grad(all_values, beta)
    
    # Now compute gradient w.r.t. pts using vectorized operations
    grad_pts = np.zeros_like(pts)
    
    # Pairwise gradient: for each pair (i,j), contribution to grad_pts[i] and grad_pts[j]
    # d(dist_ij)/d(c_i) = (c_i - c_j) / dist_ij
    # Weight for pair k is val_grad[k] * 0.5
    pair_weights = val_grad[:len(iu)] * 0.5
    
    # Compute vectors (c_i - c_j) / dist_ij for all pairs
    vecs_x = (pts[iu, 0] - pts[ju, 0]) / pair_dists
    vecs_y = (pts[iu, 1] - pts[ju, 1]) / pair_dists
    
    # Accumulate contributions using np.add.at for proper handling of repeated indices
    weighted_vecs_x = pair_weights * vecs_x
    weighted_vecs_y = pair_weights * vecs_y
    
    # For each pair, add to i and subtract from j
    np.add.at(grad_pts, iu, weighted_vecs_x[:, None])
    np.add.at(grad_pts, iu, weighted_vecs_y[:, None])
    np.add.at(grad_pts, ju, -weighted_vecs_x[:, None])
    np.add.at(grad_pts, ju, -weighted_vecs_y[:, None])
    
    # Boundary gradient: d(boundary_gap_i)/d(c_i) = -c_i / |c_i|
    boundary_weights = val_grad[len(pair_dists):]
    safe_r = np.where(r > 1e-10, r, 1.0)
    boundary_grad_x = -pts[:, 0] / safe_r * boundary_weights
    boundary_grad_y = -pts[:, 1] / safe_r * boundary_weights
    
    # Zero out where r is too small
    small_r = r < 1e-10
    boundary_grad_x[small_r] = 0
    boundary_grad_y[small_r] = 0
    
    grad_pts[:, 0] += boundary_grad_x
    grad_pts[:, 1] += boundary_grad_y
    
    return soft_min_val, grad_pts

def _optimize_soft_min(pts, time_limit, beta_start=100.0, beta_end=10000.0):
    """
    Optimize packing using smooth soft-min surrogate with gradient ascent.
    Multi-phase with adaptive beta and step size control.
    """
    start_time = time.time()
    n = len(pts)
    pts = pts.copy().astype(np.float64)
    
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    # Multi-phase optimization with increasing beta
    # Each phase: (beta, time_fraction, initial_step_size)
    phases = [
        (beta_start, 30.0, 0.010),      # Coarse: low beta, larger steps
        (beta_start * 5, 25.0, 0.006),   # Medium: moderate beta
        (beta_start * 20, 25.0, 0.004),  # Fine: high beta
        (beta_end, 20.0, 0.002),         # Very fine: highest beta
    ]
    
    for phase_idx, (beta, time_frac, base_step) in enumerate(phases):
        phase_time = time_limit * time_frac / 100.0
        phase_end = start_time + phase_time
        
        step_size = base_step
        step_decay = 0.9998
        min_step = base_step * 0.1
        max_step = base_step * 2.0
        consecutive_rejects = 0
        
        it = 0
        while time.time() < phase_end:
            elapsed = time.time() - start_time
            if elapsed > time_limit - 5.0:
                break
            
            # Compute soft-min objective and gradient
            soft_val, grad = _compute_soft_min_objective_and_grad(pts, beta)
            
            # Compute true score for tracking (every few iterations to save time)
            if it % 5 == 0:
                true_score = _compute_score(pts)
                if true_score > best_score:
                    best_score = true_score
                    best_pts = pts.copy()
            
            # Gradient ascent with adaptive step size
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-14:
                break
            
            grad_normalized = grad / grad_norm
            pts_new = pts + step_size * grad_normalized
            
            # Project inside unit circle with safety margin
            r_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
            mask_out = r_new >= 1.0
            if np.any(mask_out):
                pts_new[mask_out] *= (0.9995 / r_new[mask_out])
            
            # Evaluate new point
            soft_val_new, _ = _compute_soft_min_objective_and_grad(pts_new, beta)
            
            if soft_val_new > soft_val:
                pts = pts_new
                consecutive_rejects = 0
                # Increase step slightly on success
                step_size = min(step_size * 1.01, max_step)
            else:
                consecutive_rejects += 1
                # Decrease step on failure
                step_size *= 0.92
                if step_size < min_step:
                    step_size = min_step
                
                # Escape local minima: try perturbation
                if consecutive_rejects >= 10 and it % 30 == 0:
                    noise_scale = 0.002 + 0.001 * phase_idx
                    noise = np.random.normal(0, noise_scale, size=pts.shape)
                    pts_pert = pts + noise
                    r_p = np.hypot(pts_pert[:, 0], pts_pert[:, 1])
                    if np.all(r_p < 1.0):
                        score_p = _compute_score(pts_pert)
                        if score_p > best_score:
                            pts = pts_pert
                            best_score = score_p
                            best_pts = pts.copy()
                            consecutive_rejects = 0
                            step_size = base_step * 0.5
            
            step_size = max(step_size * step_decay, min_step)
            it += 1
    
    return best_pts, best_score

def _polish(pts, time_limit):
    """
    Final high-precision polish using very sharp soft-min.
    Focuses on the most constrained points for fine-tuning.
    """
    start_time = time.time()
    n = len(pts)
    pts = pts.copy().astype(np.float64)
    
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    # Use progressively sharper beta
    betas = [50000.0, 100000.0, 200000.0]
    step_sizes = [0.0008, 0.0005, 0.0003]
    
    for beta, base_step in zip(betas, step_sizes):
        phase_end = min(start_time + time_limit, time.time() + time_limit / len(betas))
        step_size = base_step
        step_decay = 0.99995
        min_step = base_step * 0.1
        
        it = 0
        while time.time() < phase_end:
            elapsed = time.time() - start_time
            if elapsed > time_limit - 3.0:
                break
            
            soft_val, grad = _compute_soft_min_objective_and_grad(pts, beta)
            
            # Track best
            true_score = _compute_score(pts)
            if true_score > best_score:
                best_score = true_score
                best_pts = pts.copy()
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-15:
                break
            
            grad_normalized = grad / grad_norm
            pts_new = pts + step_size * grad_normalized
            
            # Project inside unit circle
            r_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
            mask_out = r_new >= 1.0
            if np.any(mask_out):
                pts_new[mask_out] *= (0.9999 / r_new[mask_out])
            
            soft_val_new, _ = _compute_soft_min_objective_and_grad(pts_new, beta)
            
            if soft_val_new > soft_val:
                pts = pts_new
            else:
                step_size *= 0.95
                if step_size < min_step:
                    step_size = min_step
            
            step_size = max(step_size * step_decay, min_step)
            it += 1
    
    return best_pts, best_score

def construct():
    """
    Construct N=300 equal circles in unit disk.
    Uses hexagonal lattice init + multi-phase soft-min optimization + polish.
    """
    start_time = time.time()
    total_time_limit = 115.0  # Safety margin before 120s kill
    
    # Generate initial hexagonal lattice
    pts = _generate_hex_init(N)
    
    # Main optimization with soft-min (multi-phase)
    main_pts, main_score = _optimize_soft_min(pts, total_time_limit * 0.82)
    
    # Final polish with very sharp soft-min
    final_pts, final_score = _polish(main_pts, total_time_limit * 0.18)
    
    # Final validation: ensure all points are strictly inside
    r = np.hypot(final_pts[:, 0], final_pts[:, 1])
    for i in range(len(final_pts)):
        if r[i] >= 1.0:
            final_pts[i] *= 0.999 / r[i]
    
    # One last check and potential tiny adjustment
    final_score_check = _compute_score(final_pts)
    if final_score_check > final_score:
        final_score = final_score_check
    
    return final_pts
# EVOLVE-BLOCK-END
