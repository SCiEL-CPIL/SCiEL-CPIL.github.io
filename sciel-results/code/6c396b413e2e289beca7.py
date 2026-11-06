"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

def construct():
    """
    Place 100 equal non-overlapping spheres in the unit cube [-0.5, 0.5]^3
    to maximize the common radius.
    
    Strategy:
    1. BCC/FCC-inspired and jittered grid initializations for diverse starts.
    2. Multi-start soft-min gradient ascent with adaptive momentum.
    3. Critical-constraint equalization polish (iterative radius increase).
    4. Dynamic time allocation: 3 explorations, 1 long refinement, final polish.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    best_radius = 0.0
    best_pts = None
    
    # Time budget allocation
    num_starts = 3
    exploration_time = 22.0
    refinement_time = 45.0
    polish_time = 8.0
    
    seeds = [42, 123, 999]
    
    for i in range(num_starts):
        if time.time() > deadline - refinement_time - polish_time - 5.0:
            break
            
        # Different initialization strategies
        if i == 0:
            # Standard jittered 5x5x4 grid
            np.random.seed(seeds[0])
            current_pts = _init_grid_jittered(seed=seeds[0], jitter=0.03)
        elif i == 1:
            # BCC-inspired initialization
            np.random.seed(seeds[1])
            current_pts = _init_bcc_inspired(seed=seeds[1])
        elif i == 2:
            # FCC-inspired with jitter
            np.random.seed(seeds[2])
            current_pts = _init_fcc_inspired(seed=seeds[2])
        else:
            current_pts = _init_grid_jittered(seed=i*111, jitter=0.04)
        
        current_pts = np.clip(current_pts, -0.5, 0.5)
        
        # Run optimization
        pts, radius = _optimize(current_pts, time_limit=exploration_time, 
                                start_time=time.time(), fine=False)
        
        if radius > best_radius:
            best_radius = radius
            best_pts = pts.copy()
    
    # Refine the best found so far
    if best_pts is not None and time.time() < deadline - polish_time - 2.0:
        pts, radius = _optimize(best_pts, time_limit=refinement_time, 
                                start_time=time.time(), fine=True)
        if radius > best_radius:
            best_radius = radius
            best_pts = pts
    
    # Final Polish: critical-constraint equalization
    if best_pts is not None and time.time() < deadline - 1.0:
        best_pts, best_radius = _polish_critical(best_pts, best_radius, 
                                                 time_limit=polish_time, 
                                                 start_time=time.time())
        
    # Final verification
    if best_pts is None:
        best_pts = _init_grid_jittered(seed=0, jitter=0.0)
        best_radius = _compute_radius(best_pts)
        
    best_pts = np.clip(best_pts, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_pts

def _init_grid_jittered(seed=42, jitter=0.03):
    """Jittered 5x5x4 grid initialization."""
    np.random.seed(seed)
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    base_points = np.array([[x, y, z] for z in zs for y in ys for x in xs])
    return base_points + np.random.uniform(-jitter, jitter, (N, 3))

def _init_bcc_inspired(seed=123):
    """
    BCC-inspired initialization: 5x5x4 grid with alternating offset layers
    to mimic body-centered cubic packing geometry.
    """
    np.random.seed(seed)
    nx, ny, nz = 5, 5, 4
    base_spacing = 0.8 / max(nx, ny, nz)
    offset = base_spacing * 0.5
    
    points = []
    for k in range(nz):
        z = -0.4 + k * (0.8 / (nz - 1))
        for j in range(ny):
            y = -0.4 + j * (0.8 / (ny - 1))
            for i in range(nx):
                x = -0.4 + i * (0.8 / (nx - 1))
                # Alternate offset for BCC-like structure
                if (i + j + k) % 2 == 1:
                    x += offset * 0.3
                    y += offset * 0.3
                    z += offset * 0.3
                points.append([x, y, z])
    
    pts = np.array(points[:N])
    pts += np.random.uniform(-0.02, 0.02, (N, 3))
    return pts

def _init_fcc_inspired(seed=999):
    """
    FCC-inspired initialization: face-centered cubic layer arrangement.
    """
    np.random.seed(seed)
    nx, ny, nz = 5, 5, 4
    points = []
    layer_offset = 0.08
    
    for k in range(nz):
        z = -0.4 + k * (0.8 / (nz - 1))
        for j in range(ny):
            y = -0.4 + j * (0.8 / (ny - 1))
            for i in range(nx):
                x = -0.4 + i * (0.8 / (nx - 1))
                # FCC-like staggered layers
                if k % 2 == 1:
                    x += layer_offset
                if j % 2 == 1:
                    y += layer_offset * 0.5
                points.append([x, y, z])
    
    pts = np.array(points[:N])
    pts += np.random.uniform(-0.02, 0.02, (N, 3))
    return pts

def _compute_radius(pts):
    """Compute the maximum valid radius for given points (vectorized)."""
    if not np.all(np.abs(pts) <= 0.5):
        return 0.0
    
    n = len(pts)
    dists = cdist(pts, pts)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    boundary_dists = 0.5 - np.abs(pts)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist / 2.0, min_boundary)

def _compute_radius_and_crit(pts):
    """Compute radius and return the critical pair/boundary indices."""
    if not np.all(np.abs(pts) <= 0.5):
        return 0.0, None, None
    
    n = len(pts)
    dists = cdist(pts, pts)
    np.fill_diagonal(dists, np.inf)
    
    min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
    min_pair_dist = dists[min_pair_idx]
    
    boundary_dists = 0.5 - np.abs(pts)
    min_boundary_idx = np.unravel_index(np.argmin(boundary_dists), boundary_dists.shape)
    min_boundary = boundary_dists[min_boundary_idx]
    
    if min_pair_dist / 2.0 <= min_boundary:
        r = min_pair_dist / 2.0
        i_idx, j_idx = min_pair_idx
        if i_idx > j_idx:
            i_idx, j_idx = j_idx, i_idx
        crit_pair = (i_idx, j_idx)
        crit_boundary = None
    else:
        r = min_boundary
        crit_pair = None
        crit_boundary = min_boundary_idx
    
    return r, crit_pair, crit_boundary

def _soft_min_grad(pts, alpha):
    """
    Compute gradient of soft-min radius w.r.t. point positions.
    Uses log-sum-exp for numerical stability.
    """
    n = len(pts)
    dists = cdist(pts, pts)
    np.fill_diagonal(dists, np.inf)
    
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu] / 2.0
    pair_diffs = pts[iu[0]] - pts[iu[1]]
    
    boundary_terms = 0.5 - np.abs(pts)
    boundary_flat = boundary_terms.flatten()
    
    all_terms = np.concatenate([pair_dists, boundary_flat])
    
    neg_alpha_x = -alpha * all_terms
    max_val = np.max(neg_alpha_x)
    exp_vals = np.exp(neg_alpha_x - max_val)
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(pts)
    
    pair_weights = weights[:len(pair_dists)]
    safe_dists = np.maximum(dists[iu], 1e-10)
    unit_vecs = pair_diffs / safe_dists[:, np.newaxis]
    
    i_idx, j_idx = iu
    np.add.at(grad, i_idx, pair_weights[:, np.newaxis] * unit_vecs)
    np.add.at(grad, j_idx, -pair_weights[:, np.newaxis] * unit_vecs)
    
    boundary_weights = weights[len(pair_dists):]
    bw = boundary_weights.reshape(n, 3)
    signs = np.sign(pts)
    grad -= bw * signs
    
    return grad

def _optimize(pts, time_limit, start_time, fine=False):
    """
    Run the soft-min gradient ascent optimization with adaptive parameters.
    """
    current_pts = pts.copy()
    current_radius = _compute_radius(current_pts)
    best_radius = current_radius
    best_pts = current_pts.copy()
    
    if fine:
        alpha = 80.0
        lr = 0.005
        T = 0.001
    else:
        alpha = 30.0
        lr = 0.03
        T = 0.005
        
    momentum = np.zeros_like(current_pts)
    beta = 0.9
    no_improve = 0
    deadline = start_time + time_limit
    
    while time.time() < deadline:
        grad = _soft_min_grad(current_pts, alpha)
        
        momentum = beta * momentum + (1 - beta) * grad
        new_pts = current_pts + lr * momentum
        new_pts = np.clip(new_pts, -0.5 + 1e-9, 0.5 - 1e-9)
        
        new_radius = _compute_radius(new_pts)
        
        if new_radius >= current_radius:
            current_radius = new_radius
            current_pts = new_pts
            no_improve = 0
            T *= 0.998
            lr *= 0.9995
            alpha *= 1.001
            if lr < (0.001 if fine else 0.005):
                lr = 0.003 if fine else 0.015
        else:
            delta = new_radius - current_radius
            if np.random.random() < np.exp(delta / max(T, 1e-8)):
                current_radius = new_radius
                current_pts = new_pts
            no_improve += 1
            T *= 0.97
            lr *= 0.93
            if lr < (0.001 if fine else 0.005):
                if no_improve > (50 if fine else 80):
                    if fine:
                        idx = np.random.choice(N, 3, replace=False)
                        current_pts[idx] += np.random.uniform(-0.01, 0.01, (3, 3))
                    else:
                        idx = np.random.choice(N, 8, replace=False)
                        current_pts[idx] = np.random.uniform(-0.4, 0.4, (8, 3))
                    
                    current_pts = np.clip(current_pts, -0.5, 0.5)
                    current_radius = _compute_radius(current_pts)
                    no_improve = 0
                    lr = 0.005 if fine else 0.03
                    T = 0.001 if fine else 0.005
                    alpha = 80.0 if fine else 30.0
                    momentum = np.zeros_like(current_pts)
                else:
                    lr = 0.005 if fine else 0.03
                    T = 0.001 if fine else 0.005
                    alpha = 80.0 if fine else 30.0
                    momentum = np.zeros_like(current_pts)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_pts = current_pts.copy()
            
    return best_pts, best_radius

def _polish_critical(pts, r, time_limit, start_time):
    """
    High-precision polish: iteratively equalize binding constraints by
    progressively increasing the target radius.
    """
    pts = pts.copy()
    deadline = start_time + time_limit
    target_r = r
    stuck_count = 0
    
    for it in range(5000):
        if time.time() > deadline:
            break
        
        # Try to increase target radius slightly
        target_r += 0.00001
        
        # Find all violated constraints at target_r
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        boundary_dists = 0.5 - np.abs(pts)
        
        # Find pairs too close
        close_pairs = np.argwhere(dists < 2 * target_r)
        # Find boundary violations
        close_bound = np.argwhere(boundary_dists < target_r)
        
        if len(close_pairs) == 0 and len(close_bound) == 0:
            # All constraints satisfied at target_r, continue increasing
            r = target_r
            stuck_count = 0
            continue
        
        # Resolve the most violated constraint
        if len(close_pairs) > 0:
            # Pick the most violated pair
            min_dist = np.inf
            best_i, best_j = 0, 1
            for idx in close_pairs:
                i, j = idx
                if i > j:
                    i, j = j, i
                d = dists[i, j]
                if d < min_dist:
                    min_dist = d
                    best_i, best_j = i, j
            
            d_vec = pts[best_i] - pts[best_j]
            dist = np.linalg.norm(d_vec)
            if dist > 1e-12:
                unit = d_vec / dist
                deficit = 2 * target_r - dist
                if deficit > 0:
                    step = 0.5 * deficit * 0.9
                    pts[best_i] += step * unit
                    pts[best_j] -= step * unit
        elif len(close_bound) > 0:
            # Pick the most violated boundary
            min_bd = np.inf
            best_bi, best_bk = 0, 0
            for idx in close_bound:
                bi, bk = idx
                bd = boundary_dists[bi, bk]
                if bd < min_bd:
                    min_bd = bd
                    best_bi, best_bk = bi, bk
            
            sign = np.sign(pts[best_bi, best_bk])
            if sign == 0:
                sign = 1
            deficit = target_r - boundary_dists[best_bi, best_bk]
            if deficit > 0:
                pts[best_bi, best_bk] += sign * deficit * 0.9
        
        pts = np.clip(pts, -0.5 + 1e-12, 0.5 - 1e-12)
        
        # Check if we made progress
        new_r = _compute_radius(pts)
        if new_r < r - 1e-10:
            # We got worse, revert target_r
            target_r -= 0.00002
            stuck_count += 1
            if stuck_count > 50:
                break
        else:
            if new_r > r:
                r = new_r
            stuck_count = 0
    
    return pts, r

# EVOLVE-BLOCK-END
