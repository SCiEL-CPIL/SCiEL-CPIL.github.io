"""Packing 50 equal circles in a unit circle - SDR-based QCQP with L-BFGS-B polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0):
        return 0.0
    r_cont = 1.0 - np.max(dists)
    if r_cont < 0:
        return 0.0
    if N < 2:
        return r_cont
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dmat = np.sqrt(dx ** 2 + dy ** 2)
    np.fill_diagonal(dmat, np.inf)
    min_pair = np.min(dmat)
    r_pair = min_pair / 2.0
    return min(r_cont, r_pair)


def _soft_min(values, beta):
    """Compute the soft minimum of an array of values."""
    v = np.array(values)
    m = np.min(v)
    exp_terms = np.exp(-beta * (v - m))
    return m - (1.0 / beta) * np.log(np.sum(exp_terms))


def _objective_and_gradient(params, n, beta, symmetry_penalty=0.0):
    """
    Objective to minimize: -soft_min(constraints) + symmetry_penalty * symmetry_breaker
    params: (n, 2) array of centers
    Returns (objective_value, gradient)
    Vectorized gradient computation for speed.
    """
    centers = params.reshape(n, 2)
    
    # Pairwise distances
    i_idx, j_idx = np.triu_indices(n, k=1)
    diff = centers[i_idx, :] - centers[j_idx, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    
    # Boundary distances
    norms = np.linalg.norm(centers, axis=1)
    boundary_dists = 1.0 - norms
    
    # Combine all constraints
    vals = np.concatenate([dists / 2.0, boundary_dists])
    
    obj = -_soft_min(vals, beta)
    
    # Compute gradient
    m = np.min(vals)
    exp_vals = np.exp(-beta * (vals - m))
    S = np.sum(exp_vals)
    weights = exp_vals / S
    
    grad = np.zeros_like(centers)
    
    # Pairwise part - vectorized
    pair_weights = weights[:len(i_idx)]
    safe_dists = np.where(dists > 1e-8, dists, 1.0)
    unit_vecs = diff / safe_dists[:, np.newaxis]
    weighted_vecs = pair_weights[:, np.newaxis] * 0.5 * unit_vecs
    
    np.add.at(grad, i_idx, weighted_vecs)
    np.add.at(grad, j_idx, -weighted_vecs)
    
    # Boundary part - vectorized
    bound_weights = weights[len(i_idx):]
    safe_norms = np.where(norms > 1e-8, norms, 1.0)
    grad -= bound_weights[:, np.newaxis] * centers / safe_norms[:, np.newaxis]
    
    # Symmetry-breaking penalty: penalize points that are too close to high-symmetry positions
    # This encourages exploration of asymmetric local minima
    if symmetry_penalty > 0:
        # Penalize points near the origin (high symmetry)
        origin_penalty = symmetry_penalty * np.sum(norms ** 2)
        obj += origin_penalty
        grad += 2.0 * symmetry_penalty * centers
        
        # Penalize points that lie on axes (x-axis, y-axis, diagonals)
        # This breaks 4-fold and 6-fold symmetries
        axis_penalty = symmetry_penalty * 0.1
        # x-axis: y close to 0
        axis_penalty += axis_penalty * np.sum(centers[:, 1] ** 2)
        # y-axis: x close to 0
        axis_penalty += axis_penalty * np.sum(centers[:, 0] ** 2)
        # Diagonals: x close to y or x close to -y
        axis_penalty += axis_penalty * np.sum((centers[:, 0] - centers[:, 1]) ** 2)
        axis_penalty += axis_penalty * np.sum((centers[:, 0] + centers[:, 1]) ** 2)
        
        obj += axis_penalty
        grad[:, 0] += 2.0 * axis_penalty * centers[:, 0] + 2.0 * axis_penalty * (centers[:, 0] - centers[:, 1]) + 2.0 * axis_penalty * (centers[:, 0] + centers[:, 1])
        grad[:, 1] += 2.0 * axis_penalty * centers[:, 1] + 2.0 * axis_penalty * (centers[:, 1] - centers[:, 0]) + 2.0 * axis_penalty * (centers[:, 1] + centers[:, 0])
    
    return obj, -grad


def _optimize_lbfgs(centers, beta, maxiter=500, gtol=1e-8, symmetry_penalty=0.0):
    """
    Optimize using L-BFGS-B.
    """
    try:
        res = minimize(
            _objective_and_gradient, 
            centers.flatten(), 
            args=(N, beta, symmetry_penalty),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'gtol': gtol, 'disp': False, 'maxfun': 5000}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return centers


def _get_hex_lattice(n, seed_offset=0):
    """Generate a hexagonal lattice initialization."""
    rng = np.random.default_rng(42 + seed_offset)
    s_est = np.sqrt(2 * np.pi / (n * np.sqrt(3)))
    
    points = []
    max_range = int(2.0 / s_est) + 3
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s_est + (j % 2) * s_est / 2.0
            y = j * s_est * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.95:
                points.append([x, y])
                
    if len(points) < n:
        points = []
        for _ in range(n * 10):
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
    
    points = np.array(points)
    
    if len(points) < n:
        while len(points) < n:
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points = np.vstack([points, [[x, y]]])
    
    # Add small random perturbation to break symmetry
    points = points + rng.normal(0, 0.01, points.shape)
    
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:n]]


def _get_ring_init(n, seed=42):
    """
    Generate a ring-based initialization that better matches expected optimal geometry.
    Concentric rings with points spaced evenly around each ring.
    """
    rng = np.random.default_rng(seed)
    
    rings = [1, 6, 12, 18, 13]
    while sum(rings) > n:
        rings[-1] -= 1
        if rings[-1] <= 0:
            rings.pop()
    while sum(rings) < n:
        rings.append(1)
    
    points = []
    ring_spacing = 0.25
    
    for ring_idx, count in enumerate(rings):
        if count == 0:
            continue
        if ring_idx == 0:
            points.append([0.0, 0.0])
        else:
            radius = ring_idx * ring_spacing
            angle_offset = rng.uniform(0, 2 * np.pi)
            for k in range(count):
                angle = angle_offset + 2 * np.pi * k / count
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                r_jitter = rng.normal(0, 0.01)
                x += r_jitter * np.cos(angle)
                y += r_jitter * np.sin(angle)
                points.append([x, y])
    
    points = np.array(points[:n])
    
    norms = np.linalg.norm(points, axis=1)
    mask = norms > 0.95
    if np.any(mask):
        points[mask] *= (0.95 / norms[mask])[:, np.newaxis]
    
    return points


def _get_random_init(n, seed):
    """Generate random initialization."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.85)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        centers.append([x, y])
    return np.array(centers)


def _polish(centers, start_time):
    """
    Progressively polish the solution with increasing beta.
    Uses adaptive symmetry-breaking penalty that decreases as beta increases.
    """
    current = centers.copy()
    best_score = _score(current)
    
    # Phase 1: Lower beta with stronger symmetry breaking
    betas_phase1 = [10, 50, 100, 500]
    for beta in betas_phase1:
        if time.time() > _DEADLINE - 10.0:
            break
        sym_pen = 0.01 * (100.0 / beta)
        new_centers = _optimize_lbfgs(current, beta, maxiter=300, gtol=1e-6, symmetry_penalty=sym_pen)
        r = np.hypot(new_centers[:, 0], new_centers[:, 1])
        mask = r > 1.0
        if np.any(mask):
            new_centers[mask] /= r[mask, None] * 1.0001
        new_score = _score(new_centers)
        if new_score > best_score:
            best_score = new_score
            current = new_centers.copy()
    
    # Phase 2: Higher beta with weaker symmetry breaking
    betas_phase2 = [1000, 3000, 5000, 10000]
    for beta in betas_phase2:
        if time.time() > _DEADLINE - 5.0:
            break
        sym_pen = 0.001 * (1000.0 / beta)
        new_centers = _optimize_lbfgs(current, beta, maxiter=500, gtol=1e-8, symmetry_penalty=sym_pen)
        r = np.hypot(new_centers[:, 0], new_centers[:, 1])
        mask = r > 1.0
        if np.any(mask):
            new_centers[mask] /= r[mask, None] * 1.0001
        new_score = _score(new_centers)
        if new_score > best_score:
            best_score = new_score
            current = new_centers.copy()
    
    return current, best_score


def _force_directed_refine(pts, r_current, max_iters=20):
    """
    Force-directed approach to refine positions.
    """
    pts = pts.copy().astype(np.float64)
    best_pts = pts.copy()
    best_score = _score(pts)
    best_r = r_current

    i_idx, j_idx = np.triu_indices(N, k=1)

    for iteration in range(max_iters):
        if time.time() > _DEADLINE - 5.0:
            break

        r_try = best_r * 1.001 + 0.0001

        dists = np.hypot(pts[:, 0], pts[:, 1])
        r_cont = 1.0 - np.max(dists)
        
        dx = pts[i_idx, 0] - pts[j_idx, 0]
        dy = pts[i_idx, 1] - pts[j_idx, 1]
        d_sq = dx * dx + dy * dy
        min_pair_sq = np.min(d_sq)
        r_pair = np.sqrt(min_pair_sq) / 2.0

        if min(r_cont, r_pair) >= r_try:
            best_r = r_try
            continue

        force = np.zeros_like(pts)

        d = np.sqrt(d_sq + 1e-12)
        for k in range(len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]
            if d[k] < 2 * r_try:
                overlap = 2 * r_try - d[k]
                if d[k] > 1e-10:
                    fx = (pts[i, 0] - pts[j, 0]) / d[k] * overlap * 0.5
                    fy = (pts[i, 1] - pts[j, 1]) / d[k] * overlap * 0.5
                    force[i] += [fx, fy]
                    force[j] -= [fx, fy]

        for i in range(N):
            if dists[i] + r_try > 1.0:
                if dists[i] > 1e-10:
                    pull = (dists[i] + r_try - 1.0) * 0.5
                    force[i, 0] -= pts[i, 0] / dists[i] * pull
                    force[i, 1] -= pts[i, 1] / dists[i] * pull

        step = 0.1
        new_pts = pts + step * force

        nr = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = nr > 1.0
        if np.any(mask):
            new_pts[mask] /= nr[mask, None] * 1.0001

        new_score = _score(new_pts)

        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            best_r = new_score
            pts = new_pts.copy()
        else:
            step *= 0.5
            if step < 0.001:
                break

    return best_pts, best_score


def construct():
    """Construct 50 circle centers in unit circle maximizing common radius."""
    best_pts = None
    best_score = 0.0

    # Multiple initialization strategies
    inits = [
        _get_hex_lattice(N, seed_offset=0),
        _get_hex_lattice(N, seed_offset=1),
        _get_ring_init(N, seed=42),
        _get_ring_init(N, seed=123),
        _get_random_init(N, seed=42),
        _get_random_init(N, seed=123),
        _get_hex_lattice(N, seed_offset=2),
        _get_ring_init(N, seed=456),
    ]

    for i, pts in enumerate(inits):
        if time.time() > _DEADLINE - 20.0:
            break

        # Project inside container
        r = np.hypot(pts[:, 0], pts[:, 1])
        mask = r > 1.0
        if np.any(mask):
            pts[mask] /= r[mask, None] * 1.001

        # Run adaptive polish with symmetry breaking
        optimized_pts, score = _polish(pts, _START_TIME)

        if score > best_score:
            best_score = score
            best_pts = optimized_pts.copy()

        # If we have time, try force-directed refinement
        if time.time() < _DEADLINE - 30.0 and i % 2 == 0:
            fd_pts, fd_score = _force_directed_refine(optimized_pts, score, max_iters=10)
            if fd_score > best_score:
                best_score = fd_score
                best_pts = fd_pts.copy()
                # Polish the force-directed result
                polished_pts, polished_score = _polish(fd_pts, _START_TIME)
                if polished_score > best_score:
                    best_score = polished_score
                    best_pts = polished_pts.copy()

    if best_pts is None:
        best_pts = _get_hex_lattice(N, seed_offset=0)
        best_score = _score(best_pts)

    # Final high-precision polish
    if time.time() < _DEADLINE - 3.0:
        polished_pts, polished_score = _polish(best_pts, _START_TIME)
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts

    # Final validation and cleanup
    r = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = r > 1.0
    if np.any(mask):
        best_pts[mask] /= r[mask, None] * 1.0001

    return best_pts
