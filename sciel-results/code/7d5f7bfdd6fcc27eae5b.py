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


def _soft_min_vec(flat_pts, beta=80.0):
    """
    Vectorized soft-min objective using KDTree batch queries.
    Returns (value, gradient) for L-BFGS-B.
    """
    pts = flat_pts.reshape(N, 2)
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms  # shape (N,)
    
    # Batch KDTree query for all points at once
    tree = cKDTree(pts)
    dists, nn_idx = tree.query(pts, k=2)
    pair_half = dists[:, 1] / 2.0  # shape (N,)
    
    # Combine all values
    all_vals = np.concatenate([pair_half, boundary])
    all_vals = np.maximum(all_vals, 1e-12)
    
    # Soft-min via log-sum-exp
    scaled = -beta * all_vals
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -log_sum / beta
    
    # Weights for gradient
    weights = np.exp(scaled - max_scaled)
    weights /= np.sum(weights)
    
    # Gradient computation (vectorized)
    grad_flat = np.zeros(2 * N)
    
    # Boundary part: d(1-|c_i|)/d(c_i) = -c_i/|c_i|
    w_b = weights[N:]  # shape (N,)
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    dir_x = pts[:, 0] / safe_norms
    dir_y = pts[:, 1] / safe_norms
    mask = norms > 1e-10
    grad_flat[0::2] += -w_b[mask] * (-dir_x[mask])
    grad_flat[1::2] += -w_b[mask] * (-dir_y[mask])
    
    # Pair part: for each i, nearest neighbor j
    w_p = weights[:N]  # shape (N,)
    j_idx = nn_idx[:, 1]
    d = dists[:, 1]
    safe_d = np.where(d > 1e-10, d, 1.0)
    
    # d(pair_half_i)/d(c_i) = (c_i - c_j) / (2*d)
    diff_x = pts[:, 0] - pts[j_idx, 0]
    diff_y = pts[:, 1] - pts[j_idx, 1]
    grad_ri_x = diff_x / (2.0 * safe_d)
    grad_ri_y = diff_y / (2.0 * safe_d)
    
    # Contribution to c_i
    valid = d > 1e-10
    grad_flat[0::2] += -w_p[valid] * grad_ri_x[valid]
    grad_flat[1::2] += -w_p[valid] * grad_ri_y[valid]
    
    # Contribution to c_j (scatter)
    grad_rj_x = -grad_ri_x
    grad_rj_y = -grad_ri_y
    np.add.at(grad_flat[0::2], j_idx[valid], -w_p[valid] * grad_rj_x[valid])
    np.add.at(grad_flat[1::2], j_idx[valid], -w_p[valid] * grad_rj_y[valid])
    
    return -soft_min_val, grad_flat


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


def _build_hex_rings(spacing, ring_radii=None):
    """
    Build N points in concentric hexagonal rings with given spacing.
    """
    pts = []
    if ring_radii is None:
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


def _soft_min_obj(flat_pts, beta=80.0):
    """
    Soft-min objective: minimize -r where r is approximated by soft-min of
    all pairwise half-distances and boundary distances.
    Returns (value, gradient) for L-BFGS-B.
    """
    pts = flat_pts.reshape(N, 2)
    # Boundary terms: 1 - |c_i|
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms  # shape (N,)
    
    # Pairwise distances via KDTree for speed
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    pair_half = dists[:, 1] / 2.0  # shape (N,)
    
    # Combine: we want min of all (pair_half_i, boundary_i)
    # Use log-sum-exp trick for soft-min
    all_vals = np.concatenate([pair_half, boundary])
    # Clip to avoid log(0)
    all_vals = np.maximum(all_vals, 1e-12)
    
    m = all_vals.shape[0]
    # soft_min = -log(sum(exp(-beta * x))) / beta
    scaled = -beta * all_vals
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -log_sum / beta
    
    # Gradient: d(soft_min)/dx_i = -sum_j (exp(-beta*x_j) / sum_k exp(-beta*x_k)) * (-beta * dx_j/dx_i)
    # Actually: soft_min = -1/beta * log(sum(exp(-beta * x_j)))
    # d(soft_min)/d(x_j) = -1/beta * (1/sum) * (-beta * exp(-beta*x_j)) = exp(-beta*x_j) / sum(exp(-beta*x_k))
    # So gradient of soft_min w.r.t. x_j is p_j where p_j = exp(-beta*x_j)/sum
    # We minimize -soft_min, so gradient is -p_j w.r.t. x_j
    
    weights = np.exp(scaled - max_scaled)
    weights /= np.sum(weights)
    
    # Now compute gradient of -soft_min w.r.t. flat_pts
    # soft_min depends on pair_half_i (for i=0..N-1) and boundary_i (for i=N..2N-1)
    # grad of -soft_min = -sum_j p_j * d(x_j)/d(flat_pts)
    
    grad_flat = np.zeros(2 * N)
    
    # Boundary part: x_{N+i} = 1 - |c_i|, d(1-|c_i|)/d(c_i) = -c_i/|c_i|
    for i in range(N):
        w = weights[N + i]
        if norms[i] > 1e-10:
            grad_flat[2*i] += -w * (-pts[i, 0] / norms[i])
            grad_flat[2*i+1] += -w * (-pts[i, 1] / norms[i])
        else:
            grad_flat[2*i] += 0.0
            grad_flat[2*i+1] += 0.0
    
    # Pair part: x_i = dist(c_i, nearest_j) / 2
    # This is tricky because nearest neighbor changes. Approximate:
    # For each i, find nearest j, then d(dist(c_i,c_j)/2)/d(c_i) = (c_i - c_j) / (2 * dist)
    # But we need to know which j is nearest for each i
    _, nn_idx = tree.query(pts, k=2)
    for i in range(N):
        w = weights[i]
        j = nn_idx[i, 1]
        d = dists[i, 1]
        if d > 1e-10:
            # d(pair_half_i)/d(c_i) = (c_i - c_j) / (2*d)
            grad_flat[2*i] += -w * (pts[i, 0] - pts[j, 0]) / (2.0 * d)
            grad_flat[2*i+1] += -w * (pts[i, 1] - pts[j, 1]) / (2.0 * d)
            # Also affects c_j: d(pair_half_i)/d(c_j) = (c_j - c_i) / (2*d)
            grad_flat[2*j] += -w * (pts[j, 0] - pts[i, 0]) / (2.0 * d)
            grad_flat[2*j+1] += -w * (pts[j, 1] - pts[i, 1]) / (2.0 * d)
    
    return -soft_min_val, grad_flat


def _lbfgs_refine(points, deadline, max_iters=500):
    """Refine using L-BFGS-B on vectorized soft-min objective."""
    flat = points.flatten()
    best_flat = flat.copy()
    best_r = _score(points)
    
    t0 = time.time()
    remaining = deadline - t0
    if remaining < 5:
        return points, best_r
    
    # Use multiple beta values for multi-start on the soft surface
    for beta in [50.0, 100.0, 200.0, 400.0]:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(
                _soft_min_vec,
                flat.copy(),
                args=(beta,),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': max_iters, 'maxfun': 8000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None:
                cand = res.x.reshape(N, 2)
                r = _score(cand)
                if r > best_r:
                    best_r = r
                    best_flat = res.x.copy()
        except Exception:
            pass
    
    return best_flat.reshape(N, 2), best_r


def _local_refine_fast(points, deadline, max_iters=8000, base_step=0.02):
    """Fast local refinement using only neighbor checks for moved point."""
    points = points.copy()
    best_points = points.copy()
    best_r = _score(points)
    rng = np.random.default_rng(42)
    step = base_step
    
    # Precompute KDTree
    tree = cKDTree(points)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        idx = rng.integers(0, N)
        original = points[idx].copy()
        best_local = original
        best_local_r = best_r
        
        # Find current neighbors of idx (k=8 to be safe)
        _, nn_idx = tree.query(original, k=8)
        
        for attempt in range(6):
            if time.time() > deadline:
                break
            s = step * rng.uniform(0.3, 1.0)
            ang = rng.uniform(0, 2 * np.pi)
            cand = original + s * np.array([np.cos(ang), np.sin(ang)])
            
            # Check boundary
            cnorm = np.linalg.norm(cand)
            if cnorm >= 1.0:
                continue
            
            # Check against neighbors only
            min_dist = 1.0 - cnorm  # boundary constraint
            for j in nn_idx:
                if j == idx:
                    continue
                d = np.linalg.norm(cand - points[j])
                if d < min_dist:
                    min_dist = d
            # Also check a few more random points for safety
            extra_idx = rng.integers(0, N, size=5)
            for j in extra_idx:
                if j == idx:
                    continue
                d = np.linalg.norm(cand - points[j])
                if d < min_dist:
                    min_dist = d
            
            cand_r = min_dist / 2.0
            if cand_r > best_local_r:
                best_local_r = cand_r
                best_local = cand.copy()
        
        points[idx] = best_local
        if best_local_r > best_r:
            best_r = best_local_r
            best_points = points.copy()
            # Update tree periodically
            if it % 50 == 0:
                tree = cKDTree(points)
            step = max(0.003, step * 0.997)
    
    # Final exact score
    final_r = _score(best_points)
    return best_points, final_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s

    candidates = []

    # Hex lattice over a range of spacings
    for trial in range(30):
        spacing = 0.055 + trial * 0.004
        pts = _build_hex_lattice(spacing)
        if pts is not None:
            r = _score(pts)
            if r > 0:
                candidates.append((pts, r))

    # Hex rings with tuned radii
    for trial in range(15):
        spacing = 0.058 + trial * 0.004
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
    for cand_idx in range(min(5, len(candidates))):
        if time.time() > deadline - 40.0:
            break
        cand_pts, cand_r = candidates[cand_idx]
        refined_pts, refined_r = _local_refine_fast(cand_pts, deadline - 40.0,
                                                     max_iters=5000, base_step=0.02)
        if refined_pts is not None and refined_r > best_r:
            best_points, best_r = refined_pts, refined_r

    # Phase 2: L-BFGS-B refinement on the best candidate (more time allocated)
    if time.time() < deadline - 20.0:
        lbfgs_pts, lbfgs_r = _lbfgs_refine(best_points, deadline - 15.0, max_iters=500)
        if lbfgs_pts is not None and lbfgs_r > best_r:
            best_points, best_r = lbfgs_pts, lbfgs_r

    # Phase 3: Final high-precision polish with tighter tolerances
    if time.time() < deadline - 5.0:
        # First a fast local polish
        polished_pts, polished_r = _local_refine_fast(best_points, deadline - 3.0,
                                                       max_iters=3000, base_step=0.005)
        if polished_pts is not None and polished_r > best_r:
            best_points, best_r = polished_pts, polished_r
        
        # Then a final L-BFGS-B pass with very high beta for precision
        if time.time() < deadline - 2.0:
            final_pts, final_r = _lbfgs_refine(best_points, deadline - 1.0, max_iters=200)
            if final_pts is not None and final_r > best_r:
                best_points, best_r = final_pts, final_r

    return best_points


# EVOLVE-BLOCK-END
