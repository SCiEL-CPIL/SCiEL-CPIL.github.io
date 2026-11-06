"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time


def _pairwise_distances(pts):
    """Compute all pairwise distances, shape (N, N)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _compute_score(pts):
    """Compute the packing score: min(min_pair_dist/2, min_boundary_dist)."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check inside square
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
    # Boundary distance
    bd = 0.5 - np.abs(pts).min()
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pd = dists.min() / 2.0
    return min(bd, pd)


def _energy_and_grad(pts, r):
    """
    Compute smooth energy and gradient for the packing problem.
    Energy penalizes pair distances < 2r and boundary distances < r.
    """
    n = len(pts)
    # Pairwise
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    # Penalty: max(0, 2r - d)^2
    viol = np.maximum(0, 2 * r - dists)
    # Gradient contribution: -2 * viol / d * (diff) for each pair
    # But we need to be careful: d_ij = d_ji, so each pair counted twice
    # Energy = sum_{i<j} viol_ij^2 = 0.5 * sum_{i != j} viol_ij^2
    # dE/dp_i = sum_{j != i} 2 * viol_ij * (-1) * (d_ij / d_ij^2) * (p_i - p_j)
    #         = -2 * sum_{j != i} viol_ij * (p_i - p_j) / d_ij
    # But viol = max(0, 2r - d), so d(viol)/d(p_i) = -1 * (p_i - p_j)/d_ij when 2r > d_ij
    
    # Let's compute gradient properly
    # E_pair = 0.5 * sum_{i != j} max(0, 2r - d_ij)^2
    # dE/dp_i = sum_{j != i} max(0, 2r - d_ij) * (p_i - p_j) / d_ij  (when 2r > d_ij)
    # Note: the 0.5 and the double counting: each pair (i,j) and (j,i) both contribute
    # E = 0.5 * sum_{i != j} viol_ij^2
    # dE/dp_i = 0.5 * sum_{j != i} 2 * viol_ij * d(viol_ij)/d(p_i)
    #         = sum_{j != i} viol_ij * (-(p_i - p_j)/d_ij)  [when viol > 0]
    # Wait, let me redo: viol_ij = max(0, 2r - d_ij)
    # d(viol_ij)/d(p_i) = -d(d_ij)/d(p_i) = -(p_i - p_j)/d_ij  when d_ij < 2r
    # So dE/dp_i = sum_{j != i} viol_ij * (-(p_i - p_j)/d_ij)  for pairs where d_ij < 2r
    # = -sum_{j != i} viol_ij * (p_i - p_j)/d_ij
    
    # Vectorized: for each i, sum over j of viol_ij * (p_i - p_j)/d_ij
    # Create mask for violations
    mask = dists < 2 * r
    # Set diagonal to 0 to avoid self
    mask[np.diag_indices(n)] = False
    viol = np.where(mask, 2 * r - dists, 0.0)
    
    # grad_pair[i] = -sum_j viol_ij * (p_i - p_j) / d_ij
    # (p_i - p_j) / d_ij = diff[i,j] / dists[i,j]
    dir_mat = diff / dists[:, :, np.newaxis]  # (N, N, 2)
    # Set diagonal to 0
    dir_mat[np.diag_indices(n) + (np.arange(n),)] = 0
    # Actually, let's zero out diagonal properly
    idx = np.arange(n)
    dir_mat[idx, idx, :] = 0
    dir_mat[:, idx, :] = 0  # not needed, already 0
    
    # grad_pair = -sum_j viol[i,j] * dir_mat[i,j]
    grad_pair = -np.sum(viol[:, :, np.newaxis] * dir_mat, axis=1)
    
    # Boundary penalty
    # E_boundary = sum_i sum_k max(0, r - (0.5 - |c_ik|))^2
    # For axis k: viol_b = max(0, r - (0.5 - |c_ik|))
    # dE/dc_ik = 2 * viol_b * d(viol_b)/dc_ik = 2 * viol_b * (-sign(c_ik)) when |c_ik| < 0.5 - r
    
    bd = 0.5 - np.abs(pts)
    viol_b = np.maximum(0, r - bd)
    # d(viol_b)/dc_ik = -sign(c_ik) when |c_ik| < 0.5 - r, else 0
    signs = np.sign(pts)
    # Only where viol_b > 0, i.e., |c_ik| > 0.5 - r
    active = viol_b > 0
    grad_bound = np.zeros_like(pts)
    grad_bound[active] = -2 * viol_b[active] * signs[active]
    
    energy = 0.5 * np.sum(viol ** 2) + np.sum(viol_b ** 2)
    grad = grad_pair + grad_bound
    
    return energy, grad


def _local_search(pts0, r_target, max_iter=2000, lr_init=0.005, tol=1e-12, deadline=None):
    """
    Local search to find positions where all constraints are satisfied for radius r_target.
    Returns (pts, success, final_energy).
    """
    pts = pts0.copy()
    n = len(pts)
    lr = lr_init
    prev_energy = _energy_and_grad(pts, r_target)[0]
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        energy, grad = _energy_and_grad(pts, r_target)
        
        if energy < tol:
            return pts, True, energy
        
        # Adaptive learning rate
        # Simple momentum-like: if energy increased, reduce lr
        if energy > prev_energy * 1.01:
            lr *= 0.7
        else:
            lr = min(lr * 1.01, lr_init * 2)
        
        pts_new = pts - lr * grad
        # Clip to square
        pts_new = np.clip(pts_new, -0.4999, 0.4999)
        
        energy_new, _ = _energy_and_grad(pts_new, r_target)
        
        if energy_new < energy:
            pts = pts_new
            prev_energy = energy_new
        else:
            lr *= 0.5
        
        # Check convergence
        if np.max(np.abs(grad)) < 1e-10:
            break
    
    energy_final, _ = _energy_and_grad(pts, r_target)
    success = energy_final < 1e-8
    return pts, success, energy_final


def _init_grid():
    """Square grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _init_hexagonal():
    """Hexagonal lattice initialization, clipped to square."""
    pts = []
    # Generate hexagonal points in a larger area, then clip
    # Hexagonal spacing
    s = 1.0 / np.sqrt(N) * 1.2  # slightly larger to fit
    dx = s
    dy = s * np.sqrt(3) / 2
    # Generate points
    for row in range(-10, 10):
        for col in range(-10, 10):
            x = col * dx + (row % 2) * dx / 2
            y = row * dy
            if -0.45 <= x <= 0.45 and -0.45 <= y <= 0.45:
                pts.append([x, y])
    pts = np.array(pts)
    if len(pts) < N:
        # Fill with random
        while len(pts) < N:
            pts = np.vstack([pts, np.random.uniform(-0.45, 0.45, 2)])
    # Take first N
    return pts[:N]


def _init_random(repulsion_iters=50):
    """Random initialization with simple repulsion to spread points."""
    pts = np.random.uniform(-0.45, 0.45, (N, 2))
    # Simple repulsion: push points apart
    for _ in range(repulsion_iters):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-10)
        np.fill_diagonal(dists, np.inf)
        # Force: 1/d^2 directed away
        force = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i != j:
                    f = diff[i, j] / (dists[i, j] ** 3)
                    force[i] += f
        pts += 0.01 * force
        pts = np.clip(pts, -0.45, 0.45)
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Best result tracking
    best_pts = None
    best_score = 0.0
    
    # Generate initializations
    inits = []
    inits.append(_init_grid())
    inits.append(_init_hexagonal())
    for _ in range(3):
        inits.append(_init_random(20))
    
    # For each initialization, try to optimize
    for init_idx, pts0 in enumerate(inits):
        if time.time() > deadline:
            break
        
        # Try increasing radii
        # Start with a conservative estimate
        base_score = _compute_score(pts0)
        
        # Binary search / incremental increase on radius
        r_low = base_score * 0.9
        r_high = base_score * 1.5
        
        # First, try to improve from current position
        # Do a few rounds of local search at increasing radii
        for r_attempt in np.linspace(r_low, r_high, 8):
            if time.time() > deadline:
                break
            r_attempt = max(r_attempt, 0.01)
            pts, success, _ = _local_search(pts0, r_attempt, max_iter=500, lr_init=0.003, deadline=deadline)
            score = _compute_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
            # Use this as new starting point
            pts0 = pts
    
    # Final refinement: try to push the best solution a bit more
    if best_pts is not None and time.time() < deadline:
        r_target = best_score * 1.05
        pts, success, _ = _local_search(best_pts, r_target, max_iter=1000, lr_init=0.001, deadline=deadline)
        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    if best_pts is None:
        best_pts = _init_grid()
    
    # Ensure valid output
    best_pts = np.clip(best_pts, -0.4999, 0.4999)
    return best_pts

# EVOLVE-BLOCK-END

