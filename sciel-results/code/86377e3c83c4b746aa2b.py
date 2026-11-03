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
    # Pairwise distances
    dx = pts[:, 0:1] - pts[:, 0:1].T
    dy = pts[:, 1:2] - pts[:, 1:2].T
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    # Distance to boundary
    r = np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(1.0 - r)
    return min(min_dist / 2.0, min_boundary)


def _generate_hex_init(n):
    """Generate hexagonal lattice points clipped to unit disk, select n closest to center."""
    # Best known r ~ 0.053, spacing ~ 0.106. Use finer grid for good init.
    spacing = 0.09
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
    r = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(r, kind="stable")[:n]
    return points[idx]


def _soft_min(values, beta):
    """Soft minimum using log-sum-exp trick."""
    # -log-sum-exp(-values/beta) * beta
    scaled = -values / beta
    max_s = np.max(scaled)
    if not np.isfinite(max_s):
        return np.min(values)
    lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    return -lse * beta


def _optimize(pts, time_limit, n_iter=5000, lr_init=0.001, beta_init=0.01, beta_final=0.001):
    """
    Optimize packing using gradient ascent on soft-min of gaps.
    Uses smooth surrogate for the nonsmooth min objective.
    """
    start_time = time.time()
    n = len(pts)
    pts = pts.copy().astype(np.float64)
    
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    # Adaptive learning rate and beta schedule
    for it in range(n_iter):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 10.0:  # 10s safety margin
            break
        
        # Beta schedule: start large (smooth), decrease to sharp
        progress = it / max(n_iter, 1)
        beta = beta_init * (beta_final / beta_init) ** progress
        
        # Compute gaps: pairwise distances/2 and boundary distances
        dx = pts[:, 0:1] - pts[:, 0:1].T
        dy = pts[:, 1:2] - pts[:, 1:2].T
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, np.inf)
        pair_gaps = dist / 2.0
        
        r = np.hypot(pts[:, 0], pts[:, 1])
        boundary_gaps = 1.0 - r
        
        # Soft min of all gaps
        all_gaps = np.concatenate([pair_gaps[np.triu_indices(n, k=1)], boundary_gaps])
        target = _soft_min(all_gaps, beta)
        
        # Compute gradients
        # For pairwise gap d_ij/2, grad wrt c_i = (c_i - c_j)/d_ij, grad wrt c_j = (c_j - c_i)/d_ij
        # For boundary gap 1-|c_i|, grad wrt c_i = -c_i/|c_i|
        
        # Soft min weights
        scaled = -all_gaps / beta
        max_s = np.max(scaled)
        weights = np.exp(scaled - max_s)
        weights /= np.sum(weights)
        
        grad = np.zeros_like(pts)
        
        # Pairwise contributions
        n_pairs = n * (n - 1) // 2
        pair_weights = weights[:n_pairs]
        for idx in range(n_pairs):
            i, j = np.unravel_index(idx, (n, n), order='F')
            if i >= j:
                # Convert linear index to (i,j) with i<j
                # Actually np.triu_indices gives us the order
                pass
        
        # More efficient: compute gradient directly
        grad = np.zeros_like(pts)
        
        # Pairwise: for each pair (i,j) with i<j, weight w_ij
        # grad_i += w_ij * (c_i - c_j) / dist_ij
        # grad_j += w_ij * (c_j - c_i) / dist_ij
        
        # Use the upper triangle
        iu, ju = np.triu_indices(n, k=1)
        pair_dist = dist[iu, ju]
        pair_w = pair_weights[:len(iu)]
        
        # Avoid division by zero
        valid = pair_dist > 1e-10
        pair_dist_safe = np.where(valid, pair_dist, 1.0)
        pair_w_safe = np.where(valid, pair_w, 0.0)
        
        # Accumulate gradients
        for idx in range(len(iu)):
            i, j = iu[idx], ju[idx]
            w = pair_w_safe[idx]
            if w < 1e-15:
                continue
            dxij = pts[i, 0] - pts[j, 0]
            dyij = pts[i, 1] - pts[j, 1]
            d = pair_dist_safe[idx]
            grad[i, 0] += w * dxij / d
            grad[i, 1] += w * dyij / d
            grad[j, 0] -= w * dxij / d
            grad[j, 1] -= w * dyij / d
        
        # Boundary: for each i, weight w_i for gap (1-|c_i|)
        # grad_i += w_i * (-c_i / |c_i|)
        bound_w = weights[n_pairs:]
        for i in range(n):
            w = bound_w[i]
            if w < 1e-15:
                continue
            ri = r[i]
            if ri > 1e-10:
                grad[i, 0] -= w * pts[i, 0] / ri
                grad[i, 1] -= w * pts[i, 1] / ri
        
        # Normalize gradient to unit norm for stable steps
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad /= grad_norm
        
        # Adaptive learning rate
        lr = lr_init * (1.0 - 0.5 * progress)
        
        # Update
        pts_new = pts + lr * grad
        
        # Check validity: all points must be inside unit circle
        r_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
        if np.any(r_new >= 1.0):
            # Project back inside
            for i in range(n):
                if r_new[i] >= 1.0:
                    pts_new[i] *= 0.99 / r_new[i]
        
        # Check for overlaps (shouldn't happen with small steps, but be safe)
        score_new = _compute_score(pts_new)
        
        if score_new > best_score:
            best_score = score_new
            best_pts = pts_new.copy()
            pts = pts_new
        else:
            # Accept with some probability (simulated annealing style)
            # Or just reject
            pass
        
        # Every 100 iterations, try a small random perturbation to escape local optima
        if it % 100 == 99 and progress < 0.8:
            perturbation = np.random.normal(0, 0.001, size=pts.shape)
            pts_perturbed = pts + perturbation
            r_p = np.hypot(pts_perturbed[:, 0], pts_perturbed[:, 1])
            if np.all(r_p < 1.0):
                score_p = _compute_score(pts_perturbed)
                if score_p > best_score:
                    best_score = score_p
                    best_pts = pts_perturbed.copy()
                    pts = pts_perturbed
    
    return best_pts, best_score


def _polish(pts, time_limit):
    """
    Final high-precision polish: use smaller steps and tighter tolerance.
    """
    start_time = time.time()
    n = len(pts)
    pts = pts.copy().astype(np.float64)
    
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    n_iter = 2000
    lr_init = 0.0001
    beta = 0.001
    
    for it in range(n_iter):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 5.0:
            break
        
        progress = it / max(n_iter, 1)
        beta_cur = beta * (0.5) ** progress
        
        # Compute gaps
        dx = pts[:, 0:1] - pts[:, 0:1].T
        dy = pts[:, 1:2] - pts[:, 1:2].T
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, np.inf)
        pair_gaps = dist / 2.0
        
        r = np.hypot(pts[:, 0], pts[:, 1])
        boundary_gaps = 1.0 - r
        
        all_gaps = np.concatenate([pair_gaps[np.triu_indices(n, k=1)], boundary_gaps])
        target = _soft_min(all_gaps, beta_cur)
        
        # Weights
        scaled = -all_gaps / beta_cur
        max_s = np.max(scaled)
        weights = np.exp(scaled - max_s)
        weights /= np.sum(weights)
        
        grad = np.zeros_like(pts)
        
        iu, ju = np.triu_indices(n, k=1)
        pair_dist = dist[iu, ju]
        pair_w = weights[:len(iu)]
        
        valid = pair_dist > 1e-10
        pair_dist_safe = np.where(valid, pair_dist, 1.0)
        pair_w_safe = np.where(valid, pair_w, 0.0)
        
        for idx in range(len(iu)):
            i, j = iu[idx], ju[idx]
            w = pair_w_safe[idx]
            if w < 1e-15:
                continue
            dxij = pts[i, 0] - pts[j, 0]
            dyij = pts[i, 1] - pts[j, 1]
            d = pair_dist_safe[idx]
            grad[i, 0] += w * dxij / d
            grad[i, 1] += w * dyij / d
            grad[j, 0] -= w * dxij / d
            grad[j, 1] -= w * dyij / d
        
        n_pairs = n * (n - 1) // 2
        bound_w = weights[n_pairs:]
        for i in range(n):
            w = bound_w[i]
            if w < 1e-15:
                continue
            ri = r[i]
            if ri > 1e-10:
                grad[i, 0] -= w * pts[i, 0] / ri
                grad[i, 1] -= w * pts[i, 1] / ri
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad /= grad_norm
        
        lr = lr_init * (1.0 - 0.5 * progress)
        pts_new = pts + lr * grad
        
        r_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
        if np.any(r_new >= 1.0):
            for i in range(n):
                if r_new[i] >= 1.0:
                    pts_new[i] *= 0.99 / r_new[i]
        
        score_new = _compute_score(pts_new)
        if score_new > best_score:
            best_score = score_new
            best_pts = pts_new.copy()
            pts = pts_new
    
    return best_pts, best_score


def construct():
    """
    Construct N=300 equal circles in unit disk.
    Uses hexagonal lattice init + gradient optimization with smooth surrogate + polish.
    """
    start_time = time.time()
    total_time_limit = 115.0  # Safety margin before 120s kill
    
    # Generate initial hexagonal lattice
    pts = _generate_hex_init(N)
    init_score = _compute_score(pts)
    
    # Main optimization phase
    time_for_main = total_time_limit - 20.0  # Reserve 20s for polish
    main_pts, main_score = _optimize(pts, time_for_main, n_iter=3000, lr_init=0.001, beta_init=0.01, beta_final=0.001)
    
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed
    
    if remaining > 10.0:
        # Polish phase
        final_pts, final_score = _polish(main_pts, remaining)
    else:
        final_pts, final_score = main_pts, main_score
    
    # Final validation: ensure all points are strictly inside
    r = np.hypot(final_pts[:, 0], final_pts[:, 1])
    for i in range(len(final_pts)):
        if r[i] >= 1.0:
            final_pts[i] *= 0.999 / r[i]
    
    return final_pts
# EVOLVE-BLOCK-END
