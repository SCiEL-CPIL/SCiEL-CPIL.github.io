"""Direct geometric construction for 300 equal circles in a unit circle."""
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START


def _score(points):
    """Feasible common radius using KDTree for nearest-neighbor distance."""
    if points is None or len(points) < 2:
        return 0.0
    norms = np.linalg.norm(points, axis=1)
    min_boundary = float(1.0 - np.max(norms))
    if min_boundary <= 0:
        return 0.0
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    min_pair = float(np.min(dists[:, 1])) / 2.0
    return min(min_boundary, min_pair)


def _build_hex_lattice(spacing):
    """Hex lattice points inside unit disk, pick N closest to center."""
    k = int(1.0 / spacing) + 3
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - spacing * 0.5:
                pts.append((x, y))
    if len(pts) < N:
        return None
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    return pts[idx]


def _build_hex_rings(spacing):
    """
    Build N points in concentric rings with given spacing.
    Each ring holds points at a given radius, placed uniformly in angle.
    """
    pts = []
    n_rings = int(1.0 / spacing) + 1
    ring_radii = np.linspace(spacing * 0.8, 1.0 - spacing * 0.5, n_rings)
    counts = []
    total = 0
    for k, r in enumerate(ring_radii):
        if k == 0 and r < 0.1:
            c = 1
        else:
            c = max(6, int(2 * np.pi * r / spacing))
        counts.append(c)
        total += c
    if total < N:
        deficit = N - total
        for i in range(len(counts) - 1, -1, -1):
            add = min(deficit, max(1, counts[i] // 2))
            counts[i] += add
            deficit -= add
            if deficit <= 0:
                break
    elif total > N:
        excess = total - N
        for i in range(len(counts) - 1, -1, -1):
            rem = min(excess, counts[i] - (1 if i == 0 and ring_radii[i] < 0.1 else 6))
            counts[i] -= rem
            excess -= rem
            if excess <= 0:
                break
    for k, r in enumerate(ring_radii):
        c = counts[k]
        if c <= 0:
            continue
        if c == 1 and r < 0.1:
            pts.append([0.0, 0.0])
        else:
            angles = 2 * np.pi * np.arange(c) / c + (np.pi / c) * (k % 2)
            pts.append(np.column_stack([r * np.cos(angles), r * np.sin(angles)]))
    pts = np.vstack(pts)
    if len(pts) != N:
        if len(pts) < N:
            extra = N - len(pts)
            angles = 2 * np.pi * np.arange(extra) / extra
            rr = 0.9
            pts = np.vstack([pts, np.column_stack([rr * np.cos(angles), rr * np.sin(angles)])])
        else:
            pts = pts[:N]
    return pts


def _soft_min_objective_fast(x, n_points, beta=150.0, k_neighbors=6):
    """
    Fast soft-min objective using KDTree for nearest neighbors only.
    x is flattened (N,2) array. Returns negative of soft-min radius (to minimize).
    """
    pts = x.reshape(n_points, 2)
    
    # Boundary margins
    norms = np.linalg.norm(pts, axis=1)
    boundary_margins = 1.0 - norms
    
    # Use KDTree to get nearest k neighbors for each point
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=k_neighbors + 1)  # +1 for self
    # Exclude self-distance (first column)
    pair_dists = dists[:, 1:] / 2.0  # distance / 2 = radius contribution
    
    # Soft min of pair distances (only over nearest neighbors)
    min_pair = np.min(pair_dists)
    # Shift for numerical stability
    exp_terms = np.exp(-beta * (pair_dists - min_pair))
    soft_min_pair = min_pair - np.log(np.sum(exp_terms)) / beta
    
    # Soft min of boundary margins
    min_boundary = np.min(boundary_margins)
    exp_terms_b = np.exp(-beta * (boundary_margins - min_boundary))
    soft_min_boundary = min_boundary - np.log(np.sum(exp_terms_b)) / beta
    
    # Soft min of the two components
    m = min(soft_min_pair, soft_min_boundary)
    exp_p = np.exp(-beta * (soft_min_pair - m))
    exp_b = np.exp(-beta * (soft_min_boundary - m))
    soft_min_radius = m - np.log(exp_p + exp_b) / beta
    
    return -soft_min_radius


def _local_refine(points, deadline, max_iters=3000, base_step=0.012):
    """
    Fast local refinement: move one point at a time, using cached neighbor info.
    Returns (refined_points, refined_score).
    """
    if points is None:
        return None, 0.0
    points = points.copy()
    best_points = points.copy()
    best_r = _score(points)
    rng = np.random.default_rng(42)
    step = base_step
    
    # Precompute neighbor indices
    tree = cKDTree(points)
    neighbor_indices = tree.query_ball_point(points, r=step * 4)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        idx = rng.integers(0, N)
        original = points[idx].copy()
        best_local = original
        best_local_r = best_r
        
        for _ in range(15):
            if time.time() > deadline:
                break
            s = step * rng.uniform(0.1, 1.0)
            ang = rng.uniform(0, 2 * np.pi)
            cand = original + s * np.array([np.cos(ang), np.sin(ang)])
            points[idx] = cand
            
            # Fast local check: boundary + distances to precomputed neighbors
            boundary_margin = 1.0 - np.linalg.norm(cand)
            if boundary_margin <= 0:
                points[idx] = original
                continue
            nbs = neighbor_indices[idx]
            if len(nbs) > 0:
                diffs = points[nbs] - cand
                d2 = np.sum(diffs * diffs, axis=1)
                min_pair = np.sqrt(np.min(d2)) / 2.0
            else:
                min_pair = boundary_margin
            r_local = min(boundary_margin, min_pair)
            
            if r_local > best_local_r + 1e-12:
                # Verify with full score
                r_full = _score(points)
                if r_full > best_local_r + 1e-12:
                    best_local_r = r_full
                    best_local = cand.copy()
                else:
                    points[idx] = original
            else:
                points[idx] = original
        points[idx] = best_local
        if best_local_r > best_r + 1e-12:
            best_r = best_local_r
            best_points = points.copy()
            step = max(0.001, step * 0.995)
            if it % 100 == 0:
                tree = cKDTree(points)
                neighbor_indices = tree.query_ball_point(points, r=step * 4)
    
    return best_points, best_r


def _polish_precise(points, deadline, max_iters=500):
    """
    Final high-precision polish: coordinate descent with small steps and exact scoring.
    """
    if points is None:
        return None, 0.0
    points = points.copy()
    best_points = points.copy()
    best_r = _score(points)
    rng = np.random.default_rng(123)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        idx = rng.integers(0, N)
        original = points[idx].copy()
        best_local = original
        best_local_r = best_r
        
        # Try small perturbations in both axes
        for _ in range(8):
            if time.time() > deadline:
                break
            dx = rng.uniform(-0.002, 0.002)
            dy = rng.uniform(-0.002, 0.002)
            cand = original + np.array([dx, dy])
            points[idx] = cand
            
            r_full = _score(points)
            if r_full > best_local_r + 1e-14:
                best_local_r = r_full
                best_local = cand.copy()
            else:
                points[idx] = original
        
        points[idx] = best_local
        if best_local_r > best_r + 1e-14:
            best_r = best_local_r
            best_points = points.copy()
    
    return best_points, best_r


def _soft_min_full(x, alpha=80.0):
    """
    Smooth surrogate objective using full pairwise distances.
    Uses log-sum-exp soft-min to approximate the minimum of:
      - pairwise distances (contribution to radius)
      - boundary margins
    Returns negative of the soft-min radius (to minimize).
    """
    pts = x.reshape(N, 2)
    
    # Boundary margins
    norms = np.linalg.norm(pts, axis=1)
    boundary_margins = 1.0 - norms
    
    # Pairwise distances using cdist
    from scipy.spatial.distance import cdist
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    
    # We want soft-min of all (d_ij / 2) and boundary_margins
    # Combine into one array
    d_flat = d[d < np.inf] / 2.0  # radius contributions from pairs
    all_vals = np.concatenate([d_flat, boundary_margins])
    
    if len(all_vals) == 0:
        return 0.0
    
    min_val = np.min(all_vals)
    # Soft-min via log-sum-exp
    exp_terms = np.exp(-alpha * (all_vals - min_val))
    soft_min = min_val - np.log(np.sum(exp_terms)) / alpha
    
    return -soft_min


def _soft_min_full_grad(x, alpha=80.0):
    """
    Gradient of _soft_min_full.
    """
    pts = x.reshape(N, 2)
    
    # Boundary margins
    norms = np.linalg.norm(pts, axis=1)
    boundary_margins = 1.0 - norms
    
    # Pairwise distances
    from scipy.spatial.distance import cdist
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    
    d_flat = d[d < np.inf] / 2.0
    all_vals = np.concatenate([d_flat, boundary_margins])
    
    if len(all_vals) == 0:
        return np.zeros_like(x)
    
    min_val = np.min(all_vals)
    exp_terms = np.exp(-alpha * (all_vals - min_val))
    w_sum = np.sum(exp_terms)
    
    if w_sum == 0:
        return np.zeros_like(x)
    
    # Gradient of soft_min w.r.t. each value
    # d(soft_min)/d(v_i) = w_i / w_sum
    # where w_i = exp(-alpha * (v_i - min_val))
    
    # For pairwise distances: d_ij = ||p_i - p_j|| / 2
    # grad_{p_i} d_ij = (p_i - p_j) / (2 * d_ij_full) where d_ij_full = ||p_i - p_j||
    # But we're working with d_ij/2, so grad_{p_i} (d_ij/2) = (p_i - p_j) / (2 * ||p_i - p_j||)
    
    # Build weight matrix for pairs
    d_full = cdist(pts, pts)
    np.fill_diagonal(d_full, np.inf)
    
    # Weights for pairs
    w_pair = np.exp(-alpha * (d_full / 2.0 - min_val))
    np.fill_diagonal(w_pair, 0.0)
    
    # Weights for boundary
    w_boundary = np.exp(-alpha * (boundary_margins - min_val))
    
    # Gradient of soft_min w.r.t. points
    grad_pts = np.zeros_like(pts)
    
    # Pairwise contribution
    # For each pair (i,j), weight w_ij, gradient of (d_ij/2) w.r.t. p_i is (p_i - p_j)/(2*||p_i-p_j||)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            w_ij = w_pair[i, j]
            d_ij = d_full[i, j]
            if d_ij > 1e-10:
                diff = pts[i] - pts[j]
                # Gradient of (d_ij/2) w.r.t. p_i
                grad_pts[i] += (w_ij / w_sum) * diff / (2.0 * d_ij)
                # Gradient w.r.t. p_j is negative
                grad_pts[j] -= (w_ij / w_sum) * diff / (2.0 * d_ij)
    
    # Boundary contribution
    # d(boundary_margin_i)/d(p_i) = -p_i / ||p_i||
    for i in range(N):
        w_bi = w_boundary[i]
        if norms[i] > 1e-10:
            grad_pts[i] += (w_bi / w_sum) * (-pts[i] / norms[i])
    
    # Objective is -soft_min, so gradient is -grad(soft_min)
    grad_obj = -grad_pts
    
    return grad_obj.flatten()


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin

    candidates = []

    # Generate diverse initial candidates from hex lattices and rings
    for trial in range(30):
        spacing = 0.052 + trial * 0.0025
        pts = _build_hex_lattice(spacing)
        if pts is not None:
            r = _score(pts)
            if r > 0:
                candidates.append((pts, r))

    for trial in range(20):
        spacing = 0.054 + trial * 0.0025
        pts = _build_hex_rings(spacing)
        if pts is not None:
            r = _score(pts)
            if r > 0:
                candidates.append((pts, r))

    if not candidates:
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_points, best_r = candidates[0]

    # Phase 1: Fast local refinement on top candidates
    # Allocate ~30s for local refinement
    local_deadline = start_time + 30.0
    for cand_idx in range(min(8, len(candidates))):
        if time.time() > local_deadline:
            break
        cand_pts, cand_r = candidates[cand_idx]
        refined_pts, refined_r = _local_refine(cand_pts, local_deadline, max_iters=3000,
                                               base_step=0.010)
        if refined_pts is not None and refined_r > best_r:
            best_points, best_r = refined_pts, refined_r

    # Phase 2: L-BFGS-B polish using full soft-min surrogate with gradient
    # This is the key improvement: use a smooth full-pairwise soft-min with proper gradients
    # Allocate remaining time for gradient optimization
    opt_deadline = deadline - 20.0
    
    # Try L-BFGS-B on the current best and a couple of diverse starts
    opt_starts = [best_points.copy()]
    for cand_pts, cand_r in candidates[:6]:
        if np.linalg.norm(cand_pts - best_points) > 0.03:
            opt_starts.append(cand_pts.copy())
        if len(opt_starts) >= 4:
            break

    for i, start_pts in enumerate(opt_starts):
        if time.time() > opt_deadline:
            break
        
        x0 = start_pts.flatten()
        
        try:
            # Use the full soft-min objective with gradient
            # Start with moderate alpha, then increase for finer tuning
            alphas = [60.0, 100.0, 150.0]
            for alpha in alphas:
                if time.time() > opt_deadline:
                    break
                
                res = minimize(
                    _soft_min_full,
                    x0,
                    jac=_soft_min_full_grad,
                    args=(alpha,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 1000}
                )
                if res.x is not None:
                    polished = res.x.reshape(N, 2)
                    r_polished = _score(polished)
                    if r_polished > best_r:
                        best_points = polished
                        best_r = r_polished
                        x0 = polished.flatten()
        except Exception:
            pass

    # Phase 3: Final high-precision polish
    polish_deadline = deadline - 2.0
    if time.time() < polish_deadline:
        polished_pts, polished_r = _polish_precise(best_points, polish_deadline, max_iters=300)
        if polished_pts is not None and polished_r > best_r:
            best_points, best_r = polished_pts, polished_r

    return best_points


# EVOLVE-BLOCK-END
