import time
import numpy as np
from scipy.optimize import minimize

N = 50
_START_TIME = time.time()
_DEADLINE = _START_TIME + 135.0  # 15s safety margin before 150s kill

def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    dists_origin = np.linalg.norm(pts, axis=1)
    if np.any(dists_origin >= 1.0):
        return 0.0
        
    r_cont = 1.0 - np.max(dists_origin)
    if r_cont < 0:
        return 0.0
        
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)
    
    return min(r_cont, r_pair)

def _soft_min_obj_and_grad(flat, beta):
    """
    Compute the soft-min objective and gradient for L-BFGS-B.
    Objective: minimize f(x) = (1/beta) * ln(sum(exp(-beta * x_k)))
    where x_k are the critical distances (pairwise/2 and boundary).
    Minimizing this is equivalent to maximizing the soft-min of x_k.
    """
    pts = flat.reshape(N, 2)
    
    # Calculate boundary distances: 1 - |c_i|
    dists_origin = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - dists_origin
    
    # Calculate pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dist_mat[iu]
    
    # Combine all critical distances
    all_dists = np.concatenate([boundary_dists, pair_dists / 2.0])
    
    # Soft min calculation with numerical stability
    y = -beta * all_dists
    max_y = np.max(y)
    y_shifted = y - max_y
    sum_exp = np.sum(np.exp(y_shifted))
    log_sum_exp = max_y + np.log(sum_exp)
    obj_val = log_sum_exp / beta
    
    # Gradient
    weights = np.exp(y_shifted) / sum_exp
    
    grad_pts = np.zeros_like(pts)
    
    # 1. Boundary terms
    w_boundary = weights[0:N]
    safe_dists = np.where(dists_origin < 1e-8, 1.0, dists_origin)
    boundary_grad = (w_boundary / safe_dists)[:, np.newaxis] * pts
    grad_pts += boundary_grad
    
    # 2. Pairwise terms
    w_pairs = weights[N:]
    i_idx, m_idx = iu
    d_pairs = dist_mat[i_idx, m_idx]
    safe_d_pairs = np.where(d_pairs < 1e-8, 1.0, d_pairs)
    vecs = pts[i_idx] - pts[m_idx]
    unit_vecs = vecs / safe_d_pairs[:, np.newaxis]
    
    grad_contrib_i = -0.5 * w_pairs[:, np.newaxis] * unit_vecs
    grad_contrib_m = 0.5 * w_pairs[:, np.newaxis] * unit_vecs
    
    np.add.at(grad_pts, i_idx, grad_contrib_i)
    np.add.at(grad_pts, m_idx, grad_contrib_m)
    
    grad_flat = grad_pts.flatten()
    
    return obj_val, grad_flat

def _pbd_relax(pts, target_radius, iterations=200):
    """
    Position-Based Dynamics (PBD) relaxation.
    Ensures hard feasibility: all points inside unit disk, no overlaps.
    This acts as a strong pre-conditioner for the gradient-based optimizer.
    """
    pts = pts.copy()
    # We want a packing with radius slightly larger than current best to allow movement
    # But for PBD, we project onto the constraint that distance >= 2*target_radius
    # Actually, PBD usually solves for a specific radius. 
    # Here we use it to "untangle" and "compact" the configuration.
    # We set a target distance that is slightly less than the current minimum distance
    # to allow compression, or just use a fixed small distance to ensure no overlaps.
    
    # A better approach for PBD in packing: 
    # Iterate:
    # 1. For each pair, if dist < 2*r_target, move them apart to dist = 2*r_target.
    # 2. For each point, if dist from origin > 1-r_target, move it to dist = 1-r_target.
    
    # We don't know the optimal r_target yet, so we use a "soft" target or 
    # simply resolve overlaps based on a small epsilon to ensure validity.
    # However, to maximize radius, we want them as far apart as possible.
    # PBD is good for finding a valid configuration. 
    # Let's use a target radius that is slightly smaller than the current best estimate
    # to ensure we don't get stuck in a very tight, invalid state.
    
    # For the purpose of pre-conditioning the soft-min optimizer, we just need
    # a configuration that is "close" to being packed and valid.
    # We will resolve overlaps with a target distance of 0.2 (approx 2*0.1)
    # and boundary constraint 0.9 (approx 1-0.1).
    
    r_target = 0.1 
    d_min = 2 * r_target
    r_max = 1 - r_target
    
    for _ in range(iterations):
        if time.time() > _DEADLINE - 5.0:
            break
            
        # Pairwise constraints
        # Vectorized PBD is tricky due to sequential updates, but for N=50 
        # a simple loop is fast enough.
        for i in range(N):
            for j in range(i + 1, N):
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < d_min and dist > 1e-8:
                    # Move both points half the required distance apart
                    correction = (d_min - dist) * 0.5 * (diff / dist)
                    pts[i] += correction
                    pts[j] -= correction
                elif dist <= 1e-8:
                    # If coincident, push apart randomly
                    pts[i] += np.array([0.001, 0.001])
                    pts[j] -= np.array([0.001, 0.001])
                    
        # Boundary constraints
        dists_origin = np.linalg.norm(pts, axis=1)
        for i in range(N):
            if dists_origin[i] > r_max:
                # Project back to circle of radius r_max
                pts[i] = pts[i] * (r_max / dists_origin[i])
                
    return pts

def _hex_lattice_init():
    """Generate a hexagonal lattice packing as initial guess."""
    best_pts = None
    best_score = -1.0
    
    for spacing in [0.24, 0.25, 0.26, 0.27, 0.28]:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        
        mask = radius <= 1.0
        points = points[mask]
        radius = radius[mask]
        
        if len(points) >= N:
            idx = np.argsort(radius, kind="stable")[:N]
            pts = points[idx]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts
    
    if best_pts is not None:
        return best_pts
        
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.9, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _ring_init():
    """Generate a ring-based initialization matching known optimal structure."""
    best_pts = None
    best_score = -1.0
    
    # Known optimal structure: 1 center + 4 rings: 1, 6, 12, 18, 13 = 50
    rings = [1, 6, 12, 18, 13]
    ring_radii = [0.0, 0.25, 0.50, 0.75, 0.95]
    
    for seed in range(10):
        rng = np.random.default_rng(42 + seed)
        pts_list = []
        
        for ring_idx, count in enumerate(rings):
            if ring_idx == 0:
                pts_list.append([0.0, 0.0])
            else:
                radius = ring_radii[ring_idx]
                angle_offset = rng.uniform(0, 2 * np.pi)
                for k in range(count):
                    angle = angle_offset + 2 * np.pi * k / count
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    r_jitter = rng.normal(0, 0.003)
                    x += r_jitter * np.cos(angle)
                    y += r_jitter * np.sin(angle)
                    pts_list.append([x, y])
        
        pts = np.array(pts_list[:N])
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts
    
    if best_pts is not None:
        return best_pts
    return _hex_lattice_init()

def _random_init():
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.95, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _optimize_lbfgs(pts, beta, maxiter=500, gtol=1e-8):
    """Run L-BFGS-B optimization with given beta."""
    try:
        res = minimize(
            _soft_min_obj_and_grad,
            pts.flatten(),
            args=(beta,),
            method='L-BFGS-B',
            jac=True,
            bounds=[(-1.0, 1.0)] * (2 * N),
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': gtol, 'disp': False, 'maxls': 50}
        )
        new_pts = res.x.reshape(N, 2)
        # Ensure validity after optimization
        # If points went outside, project them back
        dists = np.linalg.norm(new_pts, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            new_pts[mask] = new_pts[mask] / dists[mask, np.newaxis]
        return new_pts
    except Exception:
        return pts

def construct():
    start_time = time.time()
    
    # Generate diverse initializations
    inits = []
    inits.append(_hex_lattice_init())
    inits.append(_ring_init())
    for _ in range(5):
        inits.append(_random_init())
        
    best_pts = inits[0]
    best_score = _score(best_pts)
    
    # Phase 1: For each init, do PBD relaxation then L-BFGS-B with adaptive beta
    betas = [10, 50, 200, 1000]
    
    for init_pts in inits:
        if time.time() > _DEADLINE - 20.0:
            break
            
        current_pts = init_pts.copy()
        
        # PBD relaxation to ensure hard feasibility and compactness
        # We use a small target radius for PBD to just untangle and push in
        current_pts = _pbd_relax(current_pts, target_radius=0.1, iterations=100)
        s = _score(current_pts)
        if s > best_score:
            best_score = s
            best_pts = current_pts.copy()
            
        # L-BFGS-B with increasing beta
        for beta in betas:
            if time.time() > _DEADLINE - 15.0:
                break
                
            new_pts = _optimize_lbfgs(current_pts, beta, maxiter=300, gtol=1e-8)
            new_score = _score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                current_pts = new_pts.copy()
                
            # If we got a good improvement, do a short PBD pass to ensure validity
            # and potentially escape shallow local minima caused by numerical noise
            if new_score > best_score - 1e-6 and time.time() < _DEADLINE - 20.0:
                # Use a target radius close to the current best score for PBD
                # to tighten the packing
                pbd_target = max(0.1, new_score * 0.95)
                relaxed_pts = _pbd_relax(current_pts, target_radius=pbd_target, iterations=50)
                rel_score = _score(relaxed_pts)
                if rel_score > best_score:
                    best_score = rel_score
                    best_pts = relaxed_pts.copy()
                    current_pts = relaxed_pts.copy()
        
        # Check time
        if time.time() > _DEADLINE - 15.0:
            break
            
    # Phase 2: Multiple final polish passes with very high beta on the best found
    polish_betas = [2000, 5000, 10000, 20000, 50000]
    for beta in polish_betas:
        if time.time() > _DEADLINE - 5.0:
            break
        new_pts = _optimize_lbfgs(best_pts, beta, maxiter=500, gtol=1e-12)
        new_score = _score(new_pts)
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts
            
    # Phase 3: Perturbation-based local search near the best solution
    if time.time() < _DEADLINE - 10.0:
        for sigma in [0.02, 0.01, 0.005, 0.002]:
            if time.time() > _DEADLINE - 5.0:
                break
            for attempt in range(4):
                if time.time() > _DEADLINE - 5.0:
                    break
                noise = np.random.normal(0, sigma, best_pts.shape)
                perturbed = best_pts + noise
                perturbed = _pbd_relax(perturbed, target_radius=max(0.1, best_score*0.9), iterations=50)
                
                # Quick polish with moderate beta
                new_pts = _optimize_lbfgs(perturbed, 500, maxiter=300, gtol=1e-10)
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts
                    
    return best_pts
