"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(pts):
    """
    Compute the exact minimum common radius.
    r = min(min_pair_dist / 2, min_boundary_dist)
    """
    n = len(pts)
    if n < 2:
        return 0.0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    min_boundary = np.min(1.0 - np.linalg.norm(pts, axis=1))
    return min(min_pair_dist / 2.0, min_boundary)


def _hex_init(n):
    """Generate a hexagonal lattice packing of n points in the unit disk."""
    s = np.sqrt(2 * np.pi / (n * np.sqrt(3))) * 0.95
    points = []
    max_coord = int(1.0 / s) + 2
    for j in range(-max_coord, max_coord + 1):
        for i in range(-max_coord, max_coord + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2
            if x * x + y * y <= (1.0 - s / 2) ** 2:
                points.append([x, y])
    points = np.array(points)
    if len(points) < n:
        s *= 0.9
        points = []
        max_coord = int(1.0 / s) + 2
        for j in range(-max_coord, max_coord + 1):
            for i in range(-max_coord, max_coord + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2
                if x * x + y * y <= (1.0 - s / 2) ** 2:
                    points.append([x, y])
        points = np.array(points)
    if len(points) >= n:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n]
        return points[idx]
    else:
        angles = np.random.uniform(0, 2 * np.pi, n)
        radii = np.sqrt(np.random.uniform(0, 1, n))
        return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _active_set_soft_min(pts, deadline, eps_start=20.0, eps_end=500.0,
                         lr_start=0.005, lr_end=1e-5, max_iter=5000,
                         active_frac=0.15):
    """
    Gradient ascent on active-set soft-min surrogate.
    Only considers the closest pairs (active set) to focus gradient on binding constraints.
    """
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    n = len(best_pts)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        progress = it / max_iter
        eps = eps_start * (eps_end / eps_start) ** progress
        lr = lr_start * (lr_end / lr_start) ** progress
        
        # Compute pairwise distances
        diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find active set: closest pairs
        n_pairs = n * (n - 1) // 2
        n_active = max(50, int(n_pairs * active_frac))
        # Get indices of smallest distances (excluding diagonal)
        dists_flat = dists.copy()
        np.fill_diagonal(dists_flat, np.inf)
        flat_idx = np.argsort(dists_flat.ravel())[:n_active]
        row_idx = flat_idx // n
        col_idx = flat_idx % n
        
        # Build sparse-like weights for active pairs only
        weights_pair = np.zeros((n, n))
        active_dists = dists[row_idx, col_idx]
        weights_pair[row_idx, col_idx] = np.exp(-eps * active_dists)
        weights_pair[col_idx, row_idx] = weights_pair[row_idx, col_idx]
        
        safe_dists = np.where(dists > 1e-12, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        
        grad_pair = -eps * np.sum(weights_pair[:, :, np.newaxis] * unit_vecs, axis=1)
        
        # Boundary gradient
        radii = np.linalg.norm(best_pts, axis=1)
        b_dists = 1.0 - radii
        weights_bound = np.exp(-eps * b_dists)
        safe_norms = np.where(radii > 1e-12, radii, 1.0)
        unit_radii = best_pts / safe_norms[:, np.newaxis]
        grad_bound = eps * weights_bound[:, np.newaxis] * unit_radii
        
        total_grad = grad_pair + grad_bound
        
        # Adaptive step: scale by gradient magnitude
        grad_norm = np.linalg.norm(total_grad, axis=1)
        grad_norm[grad_norm < 1e-12] = 1.0
        total_grad /= grad_norm[:, np.newaxis]
        
        candidate = best_pts + lr * total_grad
        
        # Enforce boundary
        norms_cand = np.linalg.norm(candidate, axis=1)
        too_far = norms_cand > 0.9999
        if np.any(too_far):
            candidate[too_far] *= 0.9999 / norms_cand[too_far]
        
        cand_r = _compute_radius(candidate)
        if cand_r > best_r:
            best_r = cand_r
            best_pts = candidate.copy()
        else:
            lr *= 0.98
    
    return best_pts, best_r


def _newton_local_refine(pts, deadline, max_iter=200, tol=1e-12):
    """
    Newton-based local refinement for the tightest constraints.
    Identifies binding constraints (pairs and boundary at minimum distance)
    and solves the quadratic subproblem to maximize the minimum radius.
    """
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    n = len(best_pts)
    
    for outer in range(max_iter):
        if time.time() > deadline:
            break
        
        # Identify binding constraints
        diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        radii = np.linalg.norm(best_pts, axis=1)
        b_dists = 1.0 - radii
        
        min_pair = np.min(dists)
        min_boundary = np.min(b_dists)
        current_r = min(min_pair / 2.0, min_boundary)
        
        # Find all constraints within tolerance of minimum
        tol_bind = 1e-8
        # Binding pairs
        pair_mask = dists <= (min_pair + 2 * tol_bind)
        np.fill_diagonal(pair_mask, False)
        pair_i, pair_j = np.where(pair_mask)
        
        # Binding boundary
        bound_mask = b_dists <= (min_boundary + 2 * tol_bind)
        bound_idx = np.where(bound_mask)[0]
        
        if len(pair_i) == 0 and len(bound_idx) == 0:
            break
        
        # Build system: maximize t subject to constraints
        # For each binding pair (i,j): |p_i - p_j| >= 2t
        # For each binding boundary i: 1 - |p_i| >= t
        # Linearize around current point:
        # Pair: (p_i - p_j) · (p_i - p_j)/|p_i - p_j| >= 2t - |p_i - p_j| + 2*(current_r - t)
        # Simplified: use gradient of distance constraint
        
        # Build A matrix and b vector for QP-like solve
        # Variables: delta_p (2n) and delta_t (1)
        # We want to maximize delta_t such that constraints are satisfied
        
        # Use a simple iterative approach: move in direction that increases min constraint
        # Compute gradient of each binding constraint w.r.t. positions
        
        # For pair (i,j): gradient of dist_ij w.r.t. p_i is (p_i-p_j)/dist, w.r.t. p_j is -(p_i-p_j)/dist
        # For boundary i: gradient of (1-|p_i|) w.r.t. p_i is -p_i/|p_i|
        
        # We want to find delta that maximizes the minimum of all binding constraint changes
        # This is equivalent to: maximize min_k (g_k · delta + c_k) where c_k is current slack
        
        # Use subgradient method: average gradients of most violated constraints
        grads = np.zeros_like(best_pts)
        
        # Pair gradients
        for i, j in zip(pair_i, pair_j):
            d = best_pts[i] - best_pts[j]
            dist_ij = np.linalg.norm(d)
            if dist_ij > 1e-12:
                unit = d / dist_ij
                # To increase distance, move i away from j and j away from i
                grads[i] += unit
                grads[j] -= unit
        
        # Boundary gradients
        for i in bound_idx:
            r_i = np.linalg.norm(best_pts[i])
            if r_i > 1e-12:
                # To increase (1 - |p_i|), move inward
                grads[i] -= best_pts[i] / r_i
        
        # Normalize
        grad_norm = np.linalg.norm(grads, axis=1)
        grad_norm[grad_norm < 1e-12] = 1.0
        grads /= grad_norm[:, np.newaxis]
        
        # Line search
        for step in [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
            candidate = best_pts + step * grads
            norms_cand = np.linalg.norm(candidate, axis=1)
            too_far = norms_cand > 0.9999
            if np.any(too_far):
                candidate[too_far] *= 0.9999 / norms_cand[too_far]
            
            cand_r = _compute_radius(candidate)
            if cand_r > best_r + 1e-15:
                best_r = cand_r
                best_pts = candidate.copy()
                break
            else:
                # Try smaller step
                continue
        
        else:
            # No improvement with any step size
            break
        
        # Convergence check
        if current_r - best_r < -tol:
            break
    
    return best_pts, best_r


def _polish(pts, deadline):
    """
    Final high-precision polish with very small steps and high eps.
    """
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    n = len(best_pts)
    
    for lr in [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]:
        if time.time() > deadline:
            break
        
        for it in range(300):
            if time.time() > deadline:
                break
            
            eps = 300.0
            diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            radii = np.linalg.norm(best_pts, axis=1)
            b_dists = 1.0 - radii
            
            weights_pair = np.exp(-eps * dists)
            np.fill_diagonal(weights_pair, 0.0)
            
            safe_dists = np.where(dists > 1e-12, dists, 1.0)
            unit_vecs = diff / safe_dists[:, :, np.newaxis]
            
            grad_pair = -eps * np.sum(weights_pair[:, :, np.newaxis] * unit_vecs, axis=1)
            
            weights_bound = np.exp(-eps * b_dists)
            safe_norms = np.where(radii > 1e-12, radii, 1.0)
            unit_radii = best_pts / safe_norms[:, np.newaxis]
            
            grad_bound = eps * weights_bound[:, np.newaxis] * unit_radii
            
            total_grad = grad_pair + grad_bound
            grad_norm = np.linalg.norm(total_grad, axis=1)
            grad_norm[grad_norm < 1e-12] = 1.0
            total_grad /= grad_norm[:, np.newaxis]
            
            candidate = best_pts + lr * total_grad
            norms_cand = np.linalg.norm(candidate, axis=1)
            too_far = norms_cand > 0.9999
            if np.any(too_far):
                candidate[too_far] *= 0.9999 / norms_cand[too_far]
            
            cand_r = _compute_radius(candidate)
            if cand_r > best_r:
                best_r = cand_r
                best_pts = candidate.copy()
    
    return best_pts, best_r


def construct():
    """
    Construct 300 equal circle centers inside the unit disk to maximize the minimum
    common radius. Uses active-set soft-min gradient ascent with Newton local refinement.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Initialize with hexagonal lattice
    pts = _hex_init(N)
    norms = np.linalg.norm(pts, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        pts[mask] = pts[mask] / norms[mask] * 0.999
    
    # Phase 1: Active-set soft-min gradient ascent (main optimization)
    time_budget1 = deadline - 25
    pts, score = _active_set_soft_min(pts, time_budget1)
    best_pts = pts
    best_score = score
    
    # Phase 2: Newton local refinement for binding constraints
    if time.time() < deadline - 10:
        time_budget2 = deadline - 5
        pts2, score2 = _newton_local_refine(best_pts, time_budget2)
        if score2 > best_score:
            best_pts = pts2
            best_score = score2
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 3:
        time_budget3 = deadline - 1
        best_pts, best_score = _polish(best_pts, time_budget3)
    
    # Final validation
    if best_pts is None:
        best_pts = _hex_init(N)
    
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_pts[mask] = best_pts[mask] / norms[mask] * 0.99
    
    return best_pts
# EVOLVE-BLOCK-END
