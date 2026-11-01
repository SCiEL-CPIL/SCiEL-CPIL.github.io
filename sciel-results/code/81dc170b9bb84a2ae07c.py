"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0


def _score(positions):
    """
    Compute the valid packing radius.
    Returns 0 if invalid.
    """
    n = len(positions)
    if n != N:
        return 0.0
    if not np.all(np.isfinite(positions)):
        return 0.0
    
    radii = np.hypot(positions[:, 0], positions[:, 1])
    if np.any(radii > 1.0):
        return 0.0
        
    min_boundary = 1.0 - np.max(radii)
    if min_boundary < 0:
        return 0.0
        
    if n < 2:
        return min_boundary

    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, min_boundary)


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


def _optimize_lbfgs(centers, beta, maxiter, gtol):
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
            options={'maxiter': maxiter, 'gtol': gtol, 'disp': False}
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
    
    points = points + rng.normal(0, 0.01, points.shape)
    
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:n]]


def _get_ring_init(n, seed=42):
    """
    Generate a ring-based initialization.
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
        if time.time() - start_time > _DEADLINE - 10:
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
    
    # Use very high beta for near-exact min
    betas = [5000.0, 10000.0, 20000.0, 50000.0]
    
    for beta in betas:
        if time.time() - start_time > _DEADLINE - 5:
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
    Returns (success, points).
    """
    start_time = time.time()
    
    # L-BFGS-B on violation function
    def _violation_and_grad(x, r):
        points = x.reshape(N, 2)
        
        # Container violations
        dists_origin = np.linalg.norm(points, axis=1)
        container_violations = np.maximum(0, r + dists_origin - 1.0)
        
        # Pairwise violations
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        pair_dists = dists[mask]
        pair_violations = np.maximum(0, 2.0 * r - pair_dists)
        
        # Total objective
        obj = np.sum(container_violations**2) + np.sum(pair_violations**2)
        
        # Gradient
        grad = np.zeros_like(points)
        
        # Container gradient
        for i in range(N):
            if container_violations[i] > 0:
                r_i = dists_origin[i]
                if r_i > 1e-12:
                    grad[i] += 2.0 * container_violations[i] * (points[i] / r_i)
        
        # Pairwise gradient
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                if pair_violations[idx] > 0:
                    d_vec = points[i] - points[j]
                    d = dists[i, j]
                    if d > 1e-12:
                        factor = -2.0 * pair_violations[idx] / d
                        grad[i] += factor * d_vec
                        grad[j] -= factor * d_vec
                idx += 1
                
        return obj, grad.ravel()
    
    # Bounds: points must be in [-1, 1] x [-1, 1] (loose, constraint handles disk)
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    try:
        res = minimize(_violation_and_grad, initial_points.ravel(), args=(r,), method='L-BFGS-B', 
                       jac=True, bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
        
        # Check if feasible
        obj, _ = _violation_and_grad(res.x, r)
        if obj < 1e-8:
            return True, res.x.reshape(N, 2)
        else:
            return False, res.x.reshape(N, 2)
            
    except Exception:
        return False, initial_points


def construct():
    """
    Construct positions for N circles in the unit disk.
    Returns an array of shape (N, 2).
    """
    start_time = time.time()
    
    # 1. Generate multiple diverse initializations
    inits = []
    
    inits.append(_get_hex_lattice(N, seed_offset=0))
    inits.append(_get_hex_lattice(N, seed_offset=1))
    inits.append(_get_ring_init(N, seed=42))
    inits.append(_get_ring_init(N, seed=123))
    inits.append(_get_random_init(N, seed=7))
    inits.append(_get_random_init(N, seed=99))
    
    best_points = None
    best_score = 0.0
    
    # 2. Try each initialization with soft-min optimization
    for init in inits:
        if time.time() - start_time > _DEADLINE - 15:
            break
        
        # Polish with increasing beta
        polished, score = _polish(init, start_time)
        
        if score > best_score:
            best_score = score
            best_points = polished.copy()
    
    if best_points is None:
        best_points = inits[0]
        best_score = _score(best_points)
    
    # 3. Binary search to find the maximum feasible radius
    # Start with the current best score as a lower bound
    r_low = max(0.05, best_score * 0.95)
    r_high = 0.2
    
    # Do a few binary search steps
    n_binary_steps = 10
    for step in range(n_binary_steps):
        if time.time() - start_time > _DEADLINE - 10:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        success = False
        # Try each initialization for this r_mid
        for init in inits[:3]:  # Use only top 3 inits for speed
            if time.time() - start_time > _DEADLINE - 10:
                break
                
            is_feasible, points = _feasibility_search(r_mid, init)
            
            if is_feasible:
                success = True
                score = _score(points)
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
                r_low = r_mid
                break
                
        if not success:
            r_high = r_mid
    
    # 4. Final high-precision polish
    if time.time() - start_time < _DEADLINE - 5:
        best_points, best_score = _high_beta_polish(best_points, start_time)
    
    # 5. Final validation
    final_score = _score(best_points)
    if final_score > best_score:
        best_score = final_score
    
    # Ensure validity
    if not np.all(np.isfinite(best_points)):
        best_points = inits[0]
        best_score = _score(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
