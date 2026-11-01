"""Packing 50 equal circles in a unit circle - SDR-based QCQP with L-BFGS-B polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0


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
    v = np.array(values, dtype=np.float64)
    m = np.min(v)
    exp_terms = np.exp(-beta * (v - m))
    return m - (1.0 / beta) * np.log(np.sum(exp_terms))


def _objective_and_gradient(params, n, beta):
    """
    Objective to minimize: -soft_min(constraints)
    params: (n, 2) array of centers
    Returns (objective_value, gradient)
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
    
    # Pairwise part
    pair_weights = weights[:len(i_idx)]
    safe_dists = np.where(dists > 1e-8, dists, 1.0)
    unit_vecs = diff / safe_dists[:, np.newaxis]
    weighted_vecs = pair_weights[:, np.newaxis] * 0.5 * unit_vecs
    
    np.add.at(grad, i_idx, weighted_vecs)
    np.add.at(grad, j_idx, -weighted_vecs)
    
    # Boundary part
    bound_weights = weights[len(i_idx):]
    safe_norms = np.where(norms > 1e-8, norms, 1.0)
    grad -= bound_weights[:, np.newaxis] * centers / safe_norms[:, np.newaxis]
    
    return obj, -grad


def _optimize_lbfgs(centers, beta, maxiter=500, gtol=1e-8):
    """
    Optimize using L-BFGS-B.
    """
    try:
        res = minimize(
            _objective_and_gradient, 
            centers.flatten(), 
            args=(N, beta),
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
    """
    current = centers.copy()
    best_score = _score(current)
    
    betas = [50.0, 100.0, 200.0, 400.0, 800.0, 1500.0, 3000.0]
    
    for beta in betas:
        if time.time() > _DEADLINE - 10.0:
            break
        
        maxiter = 150
        gtol = 1e-8
        
        polished = _optimize_lbfgs(current, beta, maxiter, gtol)
        
        if np.all(np.isfinite(polished)):
            radii = np.hypot(polished[:, 0], polished[:, 1])
            mask = radii > 1.0
            if np.any(mask):
                polished[mask] *= (0.999 / radii[mask])[:, None]
            
            r = _score(polished)
            if r > best_score:
                best_score = r
                current = polished
    
    return current, best_score


def _high_beta_polish(centers, start_time):
    """
    Final high-precision polish with very high beta values.
    """
    current = centers.copy()
    best_score = _score(current)
    
    betas = [5000.0, 10000.0, 20000.0, 50000.0]
    
    for beta in betas:
        if time.time() > _DEADLINE - 5.0:
            break
        
        maxiter = 500
        gtol = 1e-12
        
        polished = _optimize_lbfgs(current, beta, maxiter, gtol)
        
        if np.all(np.isfinite(polished)):
            radii = np.hypot(polished[:, 0], polished[:, 1])
            mask = radii > 1.0
            if np.any(mask):
                polished[mask] *= (0.9999 / radii[mask])[:, None]
            
            r = _score(polished)
            if r > best_score:
                best_score = r
                current = polished
    
    return current, best_score


def _feasibility_search(r, initial_points, time_limit=5.0):
    """
    Try to find a feasible configuration for radius r starting from initial_points.
    Uses L-BFGS-B on violation function.
    Returns (success, points).
    """
    start_time = time.time()
    pts = initial_points.copy()
    
    def violation(params):
        c = params.reshape(N, 2)
        i_idx, j_idx = np.triu_indices(N, k=1)
        diff = c[i_idx, :] - c[j_idx, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=1))
        norms = np.linalg.norm(c, axis=1)
        
        # Violations: pair distances < 2r, boundary violations
        pair_viol = np.maximum(0, 2*r - dists)
        bound_viol = np.maximum(0, norms + r - 1.0)
        
        return np.sum(pair_viol ** 2) + np.sum(bound_viol ** 2)
    
    def violation_grad(params):
        c = params.reshape(N, 2)
        i_idx, j_idx = np.triu_indices(N, k=1)
        diff = c[i_idx, :] - c[j_idx, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=1) + 1e-12)
        norms = np.linalg.norm(c, axis=1) + 1e-12
        
        grad = np.zeros_like(c)
        
        # Pair violations
        for k in range(len(i_idx)):
            if dists[k] < 2*r:
                overlap = 2*r - dists[k]
                if dists[k] > 1e-10:
                    fx = (c[i_idx[k], 0] - c[j_idx[k], 0]) / dists[k] * overlap
                    fy = (c[i_idx[k], 1] - c[j_idx[k], 1]) / dists[k] * overlap
                    grad[i_idx[k], 0] += 2*fx
                    grad[i_idx[k], 1] += 2*fy
                    grad[j_idx[k], 0] -= 2*fx
                    grad[j_idx[k], 1] -= 2*fy
        
        # Boundary violations
        for i in range(N):
            if norms[i] + r > 1.0:
                overlap = norms[i] + r - 1.0
                if norms[i] > 1e-10:
                    grad[i, 0] += 2 * c[i, 0] / norms[i] * overlap
                    grad[i, 1] += 2 * c[i, 1] / norms[i] * overlap
        
        return grad.flatten()
    
    try:
        res = minimize(
            violation,
            pts.flatten(),
            jac=violation_grad,
            method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-10, 'disp': False}
        )
        
        final_pts = res.x.reshape(N, 2)
        if np.all(np.isfinite(final_pts)):
            final_viol = violation(final_pts.flatten())
            if final_viol < 1e-8:
                return True, final_pts
    except Exception:
        pass
    
    return False, pts


def _bisection_search(initial_points, r_low, r_high, time_limit=30.0):
    """
    Bisection search for the maximum feasible radius.
    """
    start_time = time.time()
    best_pts = initial_points.copy()
    best_r = r_low
    
    while r_high - r_low > 1e-7 and (time.time() - start_time) < time_limit:
        r_mid = (r_low + r_high) / 2.0
        
        success, pts = _feasibility_search(r_mid, initial_points, time_limit=min(5.0, time_limit - (time.time() - start_time)))
        
        if success:
            r_low = r_mid
            best_r = r_mid
            best_pts = pts.copy()
        else:
            r_high = r_mid
        
        # Update initial points for next iteration
        initial_points = best_pts.copy()
    
    return best_pts, best_r


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
        if time.time() > _DEADLINE - 25.0:
            break

        # Project inside container
        r = np.hypot(pts[:, 0], pts[:, 1])
        mask = r > 1.0
        if np.any(mask):
            pts[mask] /= r[mask, None] * 1.001

        # Run polish
        optimized_pts, score = _polish(pts, _START_TIME)

        if score > best_score:
            best_score = score
            best_pts = optimized_pts.copy()

        # Try bisection search with the polished result
        if time.time() < _DEADLINE - 40.0:
            r_low = score * 0.99
            r_high = min(score * 1.05, 0.14)
            bisect_pts, bisect_r = _bisection_search(optimized_pts, r_low, r_high, time_limit=20.0)
            
            if bisect_r > best_score:
                best_score = bisect_r
                best_pts = bisect_pts.copy()

    if best_pts is None:
        best_pts = _get_hex_lattice(N, seed_offset=0)
        best_score = _score(best_pts)

    # Final high-precision polish
    if time.time() < _DEADLINE - 10.0:
        polished_pts, polished_score = _high_beta_polish(best_pts, _START_TIME)
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts

    # Final bisection refinement
    if time.time() < _DEADLINE - 5.0:
        r_low = best_score * 0.999
        r_high = min(best_score * 1.002, 0.14)
        final_pts, final_r = _bisection_search(best_pts, r_low, r_high, time_limit=15.0)
        if final_r > best_score:
            best_score = final_r
            best_pts = final_pts

    # Final validation and cleanup
    r = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = r > 1.0
    if np.any(mask):
        best_pts[mask] /= r[mask, None] * 1.0001

    return best_pts
