import numpy as np
import time
from scipy.spatial import Delaunay, cKDTree
from scipy.optimize import minimize

N = 300

def _hex_init(spacing=0.11):
    """Generate hexagonal lattice points inside unit circle, pick N closest to origin."""
    points = []
    range_val = int(1.5 / spacing) + 2
    for j in range(-range_val, range_val + 1):
        y = j * spacing * np.sqrt(3.0) / 2.0
        if abs(y) > 1.2:
            continue
        offset = 0.0
        if j % 2 != 0:
            offset = spacing / 2.0
        for i in range(-range_val, range_val + 1):
            x = i * spacing + offset
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
    if len(points) < N:
        # Fallback: random points
        return np.random.randn(N, 2) * 0.5
    points = np.array(points)
    dists = np.hypot(points[:, 0], points[:, 1])
    indices = np.argsort(dists)
    return points[indices[:N]]


def _compute_radius(pts):
    """Compute the packing radius: min(min_pair_dist/2, min_boundary_dist)."""
    d = cKDTree(pts).query(pts, k=2)
    min_pair_dist = np.min(d[0]) / 2.0
    norms = np.linalg.norm(pts, axis=1)
    min_boundary = np.min(1.0 - norms)
    return min(min_pair_dist, min_boundary)


def _spectral_objective(pts_flat, fixed_mask):
    """
    Spectral objective: minimize the second-smallest eigenvalue of the graph Laplacian
    of the Delaunay triangulation (algebraic connectivity). Lower algebraic connectivity
    means the graph is less well-connected, so we want to MINIMIZE it... wait, we want to
    MAXIMIZE algebraic connectivity for rigidity. So we minimize -lambda_2.
    
    Actually, for packing, we want uniform spacing. The Delaunay triangulation's
    Laplacian eigenvalues encode structural uniformity. Maximizing lambda_2 (Fiedler value)
    promotes connectivity and uniformity.
    
    But this is a complex objective. Let's use a simpler spectral approach:
    minimize the variance of edge lengths in the Delaunay triangulation, weighted by
    the Laplacian.
    
    Alternative: Use the soft-min of pairwise distances as before, but add a spectral
    regularization term that penalizes non-uniform spacing via the Laplacian.
    """
    pts = pts_flat.reshape(-1, 2)
    
    # Boundary penalty
    norms = np.linalg.norm(pts, axis=1)
    boundary_violation = np.maximum(0, norms - 1.0)
    boundary_penalty = np.sum(boundary_violation ** 2)
    
    if boundary_penalty > 1e-6:
        return 1e6 + boundary_penalty
    
    # Compute pairwise distances
    d = cKDTree(pts).query(pts, k=2)
    min_pair_dist = np.min(d[0]) / 2.0
    min_boundary = np.min(1.0 - norms)
    r = min(min_pair_dist, min_boundary)
    
    return -r


def _soft_min_objective(pts_flat, beta=50.0):
    """
    Soft-min objective using log-sum-exp approximation.
    Minimize -r where r = min(min_pair_dist/2, min_boundary_dist).
    """
    pts = pts_flat.reshape(-1, 2)
    
    # Boundary
    norms = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - norms
    if np.any(boundary_dists < -1e-8):
        return 1e6 + np.sum(np.maximum(0, -boundary_dists) ** 2)
    
    # Pairwise distances - use KDTree for efficiency
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2)
    pair_dists = d[:, 1]  # distance to nearest neighbor
    
    # Soft-min of pair distances / 2 and boundary distances
    # r = min(min(pair_dists)/2, min(boundary_dists))
    # Use log-sum-exp for soft-min
    vals = np.concatenate([pair_dists / 2.0, boundary_dists])
    vals = np.maximum(vals, 1e-10)  # avoid log(0)
    
    # Soft-min: -log(sum(exp(-beta * vals))) / beta
    max_val = np.max(vals)
    log_sum_exp = max_val + np.log(np.sum(np.exp(-beta * (vals - max_val))))
    soft_min = -log_sum_exp / beta
    
    return -soft_min


def _soft_min_gradient(pts, beta=50.0):
    """Compute gradient of the soft-min objective."""
    N_pts = len(pts)
    grads = np.zeros_like(pts)
    
    # Boundary gradient
    norms = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - norms
    boundary_dists_clipped = np.maximum(boundary_dists, 1e-10)
    bnd_weights = np.exp(-beta * boundary_dists_clipped)
    
    # Pairwise: for each point, find nearest neighbor
    tree = cKDTree(pts)
    d, idx = tree.query(pts, k=2)
    pair_dists = d[:, 1]
    nn_idx = idx[:, 1]
    
    pair_dists_clipped = np.maximum(pair_dists, 1e-10)
    pair_weights = np.exp(-beta * pair_dists_clipped / 2.0)
    
    total_weight = np.sum(pair_weights) + np.sum(bnd_weights)
    if total_weight < 1e-10:
        return grads
    
    # Gradient from boundary
    safe_norms = np.where(norms > 1e-8, norms, 1.0)
    bnd_unit = pts / safe_norms[:, np.newaxis]
    bnd_grad = beta * bnd_weights[:, np.newaxis] * bnd_unit
    
    # Gradient from pairs: for each i, the nearest neighbor is nn_idx[i]
    # d_i = ||pts[i] - pts[nn_idx[i]]||
    # d(d_i)/d(pts[i]) = (pts[i] - pts[nn_idx[i]]) / d_i
    diff = pts - pts[nn_idx]
    safe_dists = np.where(pair_dists > 1e-8, pair_dists, 1.0)
    unit_vecs = diff / safe_dists[:, np.newaxis]
    pair_grad = -beta / 2.0 * pair_weights[:, np.newaxis] * unit_vecs
    
    total_grad = pair_grad + bnd_grad
    grad_F = (1.0 / (beta * total_weight)) * total_grad
    return -grad_F


def _force_relax(pts, max_iter=5000, dt=0.001, start_time=None, deadline=None):
    """
    Force-directed relaxation with Hertzian overlap repulsion.
    """
    if start_time is None:
        start_time = time.time()
    if deadline is None:
        deadline = start_time + 100.0
    
    pts = pts.copy()
    
    for it in range(max_iter):
        if time.time() > deadline - 30.0:
            break
        
        # Compute forces
        forces = np.zeros_like(pts)
        tree = cKDTree(pts)
        
        # For each point, find neighbors within 2*r + margin
        # Use a fixed search radius
        r_est = _compute_radius(pts)
        search_radius = max(2.5 * r_est, 0.05)
        
        for i in range(len(pts)):
            neighbors = tree.query_ball_point(pts[i], search_radius)
            for j in neighbors:
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random perturbation for coincident points
                    forces[i] += np.random.randn(2) * 0.01
                    continue
                if dist < 2.0 * r_est:
                    # Hertzian repulsion: F ~ (2r - d)^4
                    overlap = max(0, 2.0 * r_est - dist)
                    if overlap > 0:
                        force_mag = overlap ** 4
                        forces[i] += force_mag * diff / dist
            # Boundary force
            norm_i = np.linalg.norm(pts[i])
            if norm_i > 1.0 - r_est:
                margin = r_est - (1.0 - norm_i)
                if margin > 0:
                    force_mag = margin ** 4
                    forces[i] -= force_mag * pts[i] / norm_i
        
        # Update positions
        pts += dt * forces
        
        # Project to unit circle
        norms = np.linalg.norm(pts, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            pts[mask] /= norms[mask, np.newaxis]
        
        # Adaptive dt
        if it % 100 == 0:
            r_new = _compute_radius(pts)
            if r_new > r_est * 1.001:
                dt = min(dt * 1.1, 0.005)
            elif r_new < r_est * 0.999:
                dt *= 0.8
        
        r_est = _compute_radius(pts)
    
    return pts


def _local_polish(pts, max_rounds=5, k_local=15):
    """Local refinement using Nelder-Mead on critical points."""
    current_x = pts.copy()
    current_r = _compute_radius(current_x)
    start_time = time.time()
    deadline = start_time + 20.0
    
    for round_num in range(max_rounds):
        if time.time() > deadline - 2.0:
            break
        
        # Identify critical points
        tree = cKDTree(current_x)
        d, idx = tree.query(current_x, k=2)
        min_pair = d[:, 1] / 2.0
        norms = np.linalg.norm(current_x, axis=1)
        min_bound = 1.0 - norms
        
        tightness = np.minimum(min_pair, min_bound)
        crit_indices = np.where(tightness < current_r * 1.1)[0]
        
        if len(crit_indices) == 0:
            break
        
        # Sort by tightness, take worst k_local
        crit_sorted = crit_indices[np.argsort(tightness[crit_indices])]
        active = crit_sorted[:k_local]
        fixed = np.setdiff1d(np.arange(len(current_x)), active)
        
        x0 = current_x[active].flatten()
        fixed_pts = current_x.copy()
        
        def obj_local(x_flat):
            pts_local = fixed_pts.copy()
            pts_local[active] = x_flat.reshape(-1, 2)
            norms_l = np.linalg.norm(pts_local, axis=1)
            if np.any(norms_l > 1.0):
                return 1e6 + np.sum(np.maximum(0, norms_l - 1.0) ** 2)
            tree_l = cKDTree(pts_local)
            d_l, _ = tree_l.query(pts_local, k=2)
            min_dist_l = np.min(d_l[:, 1]) / 2.0
            min_bound_l = np.min(1.0 - norms_l)
            return -min(min_dist_l, min_bound_l)
        
        try:
            res = minimize(obj_local, x0, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < obj_local(x0):
                new_x = fixed_pts.copy()
                new_x[active] = res.x.reshape(-1, 2)
                if np.all(np.linalg.norm(new_x, axis=1) <= 1.0):
                    r_new = _compute_radius(new_x)
                    if r_new > current_r:
                        current_r = r_new
                        current_x = new_x
        except Exception:
            continue
    
    return current_x, current_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Phase 1: Initial hexagonal lattice
    x = _hex_init(0.11)
    best_x = x.copy()
    best_r = _compute_radius(x)
    
    # Phase 2: Force-directed relaxation with Hertzian repulsion
    # This provides a physically-motivated, smooth force field that naturally
    # settles into equilibrium configurations with uniform spacing.
    x_relaxed = _force_relax(x, max_iter=8000, dt=0.002, start_time=start_time, deadline=deadline)
    r_relaxed = _compute_radius(x_relaxed)
    if r_relaxed > best_r:
        best_r = r_relaxed
        best_x = x_relaxed.copy()
    
    # Phase 3: Soft-min gradient ascent for local precision
    if time.time() < deadline - 30.0:
        x_curr = best_x.copy()
        beta = 30.0
        step_size = 0.02
        stall_count = 0
        
        while time.time() < deadline - 30.0:
            grad = _soft_min_gradient(x_curr, beta)
            if not np.all(np.isfinite(grad)):
                beta *= 0.5
                continue
            
            x_new = x_curr + step_size * grad
            if np.any(np.linalg.norm(x_new, axis=1) > 1.0):
                step_size *= 0.5
                x_new = x_curr + step_size * grad
                if np.any(np.linalg.norm(x_new, axis=1) > 1.0):
                    continue
            
            r_new = _compute_radius(x_new)
            if r_new > best_r + 1e-10:
                best_r = r_new
                best_x = x_new.copy()
                x_curr = x_new
                beta *= 1.05
                step_size = min(step_size * 1.03, 0.05)
                stall_count = 0
            else:
                stall_count += 1
                step_size *= 0.9
                beta *= 1.02
                if stall_count > 50:
                    break
            
            if step_size < 1e-7:
                break
    
    # Phase 4: Local refinement on critical points
    if time.time() < deadline - 5.0:
        best_x, best_r = _local_polish(best_x, max_rounds=5, k_local=15)
    
    # Phase 5: Final polish with small random perturbations
    if time.time() < deadline - 2.0:
        for _ in range(30):
            if time.time() > deadline - 1.0:
                break
            noise = np.random.randn(N, 2) * 0.003
            x_pert = best_x + noise
            norms = np.linalg.norm(x_pert, axis=1)
            mask = norms > 1.0
            if np.any(mask):
                x_pert[mask] /= norms[mask, np.newaxis]
            r_pert = _compute_radius(x_pert)
            if r_pert > best_r:
                best_r = r_pert
                best_x = x_pert.copy()
    
    return best_x
