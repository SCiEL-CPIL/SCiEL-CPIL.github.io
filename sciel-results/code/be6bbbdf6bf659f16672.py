"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 112.0


def _score(positions):
    """
    Compute the valid packing radius.
    Returns 0 if invalid.
    """
    n = len(positions)
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
    v = np.array(values)
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


def _force_directed_relax(centers, start_time, max_iters=500):
    """
    Force-directed relaxation: push apart overlapping circles, pull boundary circles inward.
    """
    current = centers.copy()
    n = len(current)
    
    for it in range(max_iters):
        if time.time() - start_time > _DEADLINE - 15:
            break
        
        forces = np.zeros_like(current)
        
        # Pairwise repulsion (for pairs closer than 2r, where r is estimated)
        i_idx, j_idx = np.triu_indices(n, k=1)
        diff = current[i_idx, :] - current[j_idx, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=1))
        
        # Estimate radius from current packing
        norms = np.linalg.norm(current, axis=1)
        boundary_dists = 1.0 - norms
        min_boundary = np.min(boundary_dists)
        min_pair = np.min(dists) / 2.0 if n > 1 else min_boundary
        r_est = min(min_boundary, min_pair)
        if r_est < 1e-6:
            r_est = 0.1
        
        # Repulsion force for pairs closer than 2*r_est * 1.1
        threshold = 2.0 * r_est * 1.1
        close_mask = dists < threshold
        if np.any(close_mask):
            close_diff = diff[close_mask]
            close_dists = dists[close_mask]
            safe_dists = np.where(close_dists > 1e-8, close_dists, 1e-8)
            # Force magnitude proportional to overlap
            overlap = (2.0 * r_est - close_dists) / (2.0 * r_est)
            overlap = np.clip(overlap, 0, 1)
            force_mag = overlap * 0.01
            unit_vecs = close_diff / safe_dists[:, np.newaxis]
            repulsive = force_mag[:, np.newaxis] * unit_vecs
            
            ci = i_idx[close_mask]
            cj = j_idx[close_mask]
            np.add.at(forces, ci, repulsive)
            np.add.at(forces, cj, -repulsive)
        
        # Boundary attraction: pull circles that are too close to boundary
        norms = np.linalg.norm(current, axis=1)
        boundary_dist = 1.0 - norms
        boundary_mask = boundary_dist < r_est * 0.5
        if np.any(boundary_mask):
            safe_norms = np.where(norms[boundary_mask] > 1e-8, norms[boundary_mask], 1e-8)
            pull_strength = (r_est * 0.5 - boundary_dist[boundary_mask]) * 0.02
            pull = -pull_strength[:, np.newaxis] * current[boundary_mask] / safe_norms[:, np.newaxis]
            forces[boundary_mask] += pull
        
        # Update positions with damping
        current += forces
        current *= 0.999  # slight damping
        
        # Keep inside container
        norms = np.linalg.norm(current, axis=1)
        mask = norms > 0.999
        if np.any(mask):
            current[mask] *= (0.999 / norms[mask])[:, None]
    
    return current


def _local_perturb_reoptimize(centers, start_time, n_perturb=3, perturb_scale=0.02):
    """
    Try local perturbations of some circles and re-optimize.
    """
    current = centers.copy()
    best_score = _score(current)
    n = len(current)
    rng = np.random.default_rng(999)
    
    for attempt in range(n_perturb):
        if time.time() - start_time > _DEADLINE - 20:
            break
        
        # Pick 3-5 random circles to perturb
        n_pert = rng.integers(3, 6)
        idx = rng.choice(n, size=n_pert, replace=False)
        
        perturbed = current.copy()
        for i in idx:
            angle = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(0, perturb_scale)
            perturbed[i, 0] += dist * np.cos(angle)
            perturbed[i, 1] += dist * np.sin(angle)
        
        # Quick re-optimize with moderate beta
        polished = _optimize_lbfgs(perturbed, 500.0, 200, 1e-9)
        
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


def _polish(centers, start_time):
    """
    Progressively polish the solution with increasing beta, including force-directed relaxation.
    """
    current = centers.copy()
    best_score = _score(current)
    
    # First do a quick force-directed relaxation
    if time.time() - start_time < _DEADLINE - 30:
        fd_result = _force_directed_relax(current, start_time, max_iters=200)
        r_fd = _score(fd_result)
        if r_fd > best_score:
            best_score = r_fd
            current = fd_result
    
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


def construct():
    """
    Construct positions for N circles in the unit disk.
    Returns an array of shape (N, 2).
    """
    start_time = time.time()
    
    # 1. Generate multiple diverse initializations
    inits = []
    
    inits.append(_get_hex_lattice(N, seed_offset=0))
    inits.append(_get_ring_init(N, seed=42))
    inits.append(_get_ring_init(N, seed=123))
    inits.append(_get_ring_init(N, seed=7))
    
    for seed in [1, 2, 3, 4, 5]:
        inits.append(_get_random_init(N, seed))
    
    # 2. Optimize each initialization
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Quick optimization on all inits
    for init in inits:
        if time.time() - start_time > _DEADLINE - 40:
            break
            
        optimized, score = _polish(init, start_time)
        
        if score > best_score:
            best_score = score
            best_pts = optimized
    
    if best_pts is None:
        best_pts = inits[0]
        best_score = _score(best_pts)
    
    # Phase 2: Deep optimization on the best candidate
    if time.time() - start_time < _DEADLINE - 25:
        current = best_pts.copy()
        
        betas = [200.0, 500.0, 1000.0, 2000.0, 5000.0]
        
        for beta in betas:
            if time.time() - start_time > _DEADLINE - 25:
                break
            
            maxiter = 300
            gtol = 1e-10
            
            optimized = _optimize_lbfgs(current, beta, maxiter, gtol)
            
            if np.all(np.isfinite(optimized)):
                radii = np.hypot(optimized[:, 0], optimized[:, 1])
                mask = radii > 1.0
                if np.any(mask):
                    optimized[mask] *= (0.999 / radii[mask])[:, None]
                
                score = _score(optimized)
                if score > best_score:
                    best_score = score
                    best_pts = optimized
                    current = optimized
    
    # Phase 2.5: Local perturbation + re-optimize to escape local minima
    if time.time() - start_time < _DEADLINE - 15:
        perturbed_pts, perturbed_score = _local_perturb_reoptimize(best_pts, start_time, n_perturb=5, perturb_scale=0.015)
        if perturbed_score > best_score:
            best_score = perturbed_score
            best_pts = perturbed_pts
    
    # Phase 3: High-precision final polish
    if time.time() - start_time < _DEADLINE - 5:
        final_pts, final_score = _high_beta_polish(best_pts, start_time)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts
    
    # Final validation
    radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
    max_allowed_radius = 1.0 - best_score - 1e-12
    
    mask = radii > max_allowed_radius
    if np.any(mask):
        best_pts[mask] *= (max_allowed_radius / radii[mask])[:, None]
    
    # Ensure no coincident centers
    i_idx, j_idx = np.triu_indices(N, k=1)
    diff = best_pts[i_idx, :] - best_pts[j_idx, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    min_dist = np.min(dists)
    if min_dist < 1e-6:
        # Separate coincident points slightly
        for k in range(len(i_idx)):
            if dists[k] < 1e-6:
                best_pts[i_idx[k], 0] += 1e-5
                best_pts[j_idx[k], 0] -= 1e-5
    
    return best_pts

# EVOLVE-BLOCK-END
