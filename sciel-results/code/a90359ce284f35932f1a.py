import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree, Delaunay
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

N = 300

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0

    # Try multiple rotations and spacings to find a good subset
    for attempt in range(15):
        # Base spacing for packing N circles in radius 1
        # Area per circle ~ pi * r^2. Total area ~ pi.
        # N * pi * r^2 < pi => r < 1/sqrt(N).
        # Spacing d = 2r.
        base_spacing = 2.0 / np.sqrt(n_points) * 0.98
        spacing = base_spacing * (0.96 + 0.012 * attempt)
        
        k = int(1.0 / spacing) + 5
        rot_angle = rng.uniform(0, 2 * np.pi)
        cos_a = np.cos(rot_angle)
        sin_a = np.sin(rot_angle)
        
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                # Hexagonal lattice coordinates
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                
                # Rotate
                x_rot = x * cos_a - y * sin_a
                y_rot = x * sin_a + y * cos_a
                r = np.hypot(x_rot, y_rot)
                
                # Must fit inside unit circle (center distance < 1 - radius)
                if r <= 1.0 - spacing / 2.0:
                    rows.append((x_rot, y_rot, r))

        if len(rows) < n_points:
            continue

        pts = np.array([(r[0], r[1]) for r in rows])
        radii = np.array([r[2] for r in rows])
        
        # Select points closest to center to maximize symmetry and packing density
        order = np.argsort(radii, kind="stable")
        selected = pts[order[:n_points]]
        
        # Add small noise for subsequent attempts to break symmetry
        if attempt > 0:
            selected = selected + rng.normal(0, spacing * 0.015, selected.shape)
            norms = np.linalg.norm(selected, axis=1)
            mask = norms < 1.0
            selected = selected[mask]
            if len(selected) < n_points:
                continue
            selected = selected[:n_points]

        # Compute current radius
        d = cdist(selected, selected)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_bound = 1.0 - np.linalg.norm(selected, axis=1).max()
        r = min(min_pair, min_bound)

        if r > best_r:
            best_r = r
            best_pts = selected.copy()

    if best_pts is None:
        # Fallback: simple grid or hex without rotation
        spacing = 2.0 / np.sqrt(n_points) * 0.9
        k = int(1.0 / spacing) + 5
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    rows.append((x, y))
        pts = np.array(rows)
        order = np.argsort(np.linalg.norm(pts, axis=1), kind="stable")
        best_pts = pts[order[:n_points]]

    return best_pts

def _compute_radius(pts):
    """Compute the packing radius for given points."""
    norms = np.linalg.norm(pts, axis=1)
    min_bound = 1.0 - norms.max()
    if min_bound <= 0:
        return 0.0
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    return min(min_pair, min_bound)

def _softmin_obj_grad(pts, beta):
    """
    Compute soft-min radius and its gradient using logsumexp for numerical stability.
    r_soft = -1/beta * logsumexp(-beta * f_k)
    where f_k are the constraints (distances/2 and boundary distances).
    We want to maximize r_soft, so we minimize -r_soft.
    """
    n = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.linalg.norm(diff, axis=2)
    # Mask diagonal
    np.fill_diagonal(d, np.inf)
    
    # Boundary distances
    norms = np.linalg.norm(pts, axis=1)
    bound_dist = 1.0 - norms
    
    # Collect all f_k
    triu_idx = np.triu_indices(n, k=1)
    pair_dists = d[triu_idx] / 2.0
    
    # Combine
    all_f = np.concatenate([pair_dists, bound_dist])
    
    # Use scipy's logsumexp for numerical stability
    # r_soft = -1/beta * logsumexp(-beta * all_f)
    neg_beta_f = -beta * all_f
    log_sum = logsumexp(neg_beta_f)
    r_soft = -1.0 / beta * log_sum
    
    # Weights: w_k = exp(-beta * f_k) / sum(exp(-beta * f_m))
    # For numerical stability, use logsumexp result
    # log(w_k) = -beta * f_k - log_sum
    log_weights = neg_beta_f - log_sum
    weights = np.exp(log_weights)
    
    # Split weights back
    n_pairs = len(pair_dists)
    w_pairs = weights[:n_pairs]
    w_bound = weights[n_pairs:]
    
    grad = np.zeros_like(pts)
    
    # Gradient from pairs
    W = np.zeros((n, n))
    W[triu_idx] = w_pairs
    W.T[triu_idx] = w_pairs  # Symmetric
    np.fill_diagonal(W, 0.0)
    
    # d f_ij / d p_i = (p_i - p_j) / (2 * d_ij)
    valid_mask = np.isfinite(d) & (d > 1e-10)
    inv_2d = np.zeros_like(d)
    inv_2d[valid_mask] = 1.0 / (2.0 * d[valid_mask])
    
    M = W * inv_2d
    row_sums = M.sum(axis=1)
    M_pts = M.dot(pts)
    
    grad_pair = row_sums[:, np.newaxis] * pts - M_pts
    
    # Gradient from boundary
    norms_safe = np.where(norms > 1e-10, norms, 1.0)
    grad_bound = -w_bound[:, np.newaxis] * pts / norms_safe[:, np.newaxis]
    
    total_grad = grad_pair + grad_bound
    
    # We want to MINIMIZE -r_soft
    return -r_soft, -total_grad

def _local_polish_softmin(pts, t_init, deadline):
    """
    Polish using soft-min log-sum-exp formulation with aggressive annealing.
    """
    n_pts = N
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    # More aggressive annealing schedule with higher max beta
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    x0 = best_pts.copy()
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        def obj_func(x):
            pts_x = x.reshape(n_pts, 2)
            val, _ = _softmin_obj_grad(pts_x, beta)
            return val
        
        def grad_func(x):
            pts_x = x.reshape(n_pts, 2)
            _, grad = _softmin_obj_grad(pts_x, beta)
            return grad.ravel()
        
        try:
            # Use more iterations for higher beta values to converge better
            maxiter = 300 if beta < 100 else 500
            maxfun = 1500 if beta < 100 else 2500
            
            res = minimize(
                obj_func, 
                x0.ravel(), 
                jac=grad_func,
                method='L-BFGS-B',
                options={
                    'maxiter': maxiter,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                    'maxfun': maxfun,
                    'disp': False
                }
            )
            
            new_pts = res.x.reshape(n_pts, 2)
            
            # Check validity
            if np.all(np.isfinite(new_pts)):
                # Project back inside if slightly outside
                norms = np.linalg.norm(new_pts, axis=1)
                if np.any(norms > 1.0):
                    new_pts[norms > 1.0] = new_pts[norms > 1.0] / norms[norms > 1.0, np.newaxis] * 0.9999
                
                true_r = _compute_radius(new_pts)
                if true_r > best_r:
                    best_r = true_r
                    best_pts = new_pts.copy()
                    x0 = new_pts.copy()
        except Exception:
            pass
            
    return best_pts, best_r

def _force_relax(pts, n_steps=1000, deadline=None):
    """Fast force-directed relaxation using cKDTree for neighbor search."""
    pos = pts.copy().astype(np.float64)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    n = len(pos)
    r_est = best_r

    lr = 0.008
    cooldown = 0.998

    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break

        forces = np.zeros_like(pos)
        target = 2.0 * r_est

        try:
            tree = cKDTree(pos)
            pairs = tree.query_pairs(r=target, output_type='ndarray')
            for i, j in pairs:
                diff = pos[i] - pos[j]
                d = np.linalg.norm(diff)
                if d < 1e-10:
                    continue
                excess = target - d
                if excess > 0:
                    # Stronger repulsion for better convergence
                    force_mag = 1.5 * excess
                    unit_vec = diff / d
                    forces[i] += force_mag * unit_vec
                    forces[j] -= force_mag * unit_vec
        except Exception:
            pass

        # Boundary force
        norms = np.linalg.norm(pos, axis=1)
        wall_margin = 0.04
        close_to_wall = norms > (1.0 - wall_margin)
        if np.any(close_to_wall):
            excess_b = np.maximum(norms[close_to_wall] - (1.0 - wall_margin), 0)
            f_mags = 12.0 * excess_b
            directions = -pos[close_to_wall] / norms[close_to_wall, np.newaxis]
            forces[close_to_wall] += f_mags[:, np.newaxis] * directions

        pos = pos + lr * forces

        # Project back into unit disk
        norms = np.linalg.norm(pos, axis=1)
        mask_out = norms > 1.0
        if np.any(mask_out):
            pos[mask_out] = pos[mask_out] / norms[mask_out, np.newaxis] * 0.9999

        if step % 15 == 0:
            r = _compute_radius(pos)
            if r > best_r:
                best_r = r
                best_pos = pos.copy()
                r_est = r

        lr *= cooldown

    return best_pos, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_pts = None
    best_r = 0.0

    # Phase 1: Initialization and Coarse Relaxation
    # More diverse seeds for better exploration
    seeds = [42, 123, 7, 99, 256, 777, 1234, 5678, 1111, 2222, 3333, 4444, 5555, 6666, 7777]
    
    for seed in seeds:
        if time.time() > deadline - 45.0:
            break

        pts = _hex_init(N, seed=seed)
        r = _compute_radius(pts)

        # Quick relaxation to remove initial overlaps
        relaxed_pts, relaxed_r = _force_relax(pts, n_steps=400, deadline=deadline - 40.0)
        
        if relaxed_r > r:
            pts = relaxed_pts
            r = relaxed_r

        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
        best_r = _compute_radius(best_pts)

    # Phase 2: Soft-Min Polish with aggressive annealing
    if time.time() < deadline - 15.0:
        polished_pts, polished_r = _local_polish_softmin(best_pts, best_r, deadline=deadline - 5.0)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()
    
    # Phase 3: Final High-Precision Polish
    if time.time() < deadline - 2.0:
        polished_pts2, polished_r2 = _local_polish_softmin(best_pts, best_r, deadline=deadline - 0.5)
        if polished_r2 > best_r:
            best_r = polished_r2
            best_pts = polished_pts2.copy()

    # Final validity check
    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
    
    norms = np.linalg.norm(best_pts, axis=1)
    if not np.all(norms < 1.0) or not np.all(np.isfinite(best_pts)):
        # Fallback
        best_pts = _hex_init(N, seed=0)
        norms = np.linalg.norm(best_pts, axis=1)
        mask = norms < 1.0
        if not np.all(mask):
             # If even fallback fails (shouldn't happen), return zero
             best_pts = np.zeros((N, 2))

    return best_pts
