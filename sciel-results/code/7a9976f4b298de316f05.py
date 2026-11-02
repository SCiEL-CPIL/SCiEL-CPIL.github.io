import time
import numpy as np
from scipy.optimize import minimize

N = 50
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # 5s safety margin before 150s kill

def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    dists_origin = np.linalg.norm(pts, axis=1)
    if np.any(dists_origin > 1.0):
        return 0.0
        
    r_cont = 1.0 - np.max(dists_origin)
    if r_cont < 0:
        return 0.0
        
    if N < 2:
        return r_cont
        
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)
    
    return min(r_cont, r_pair)

def _soft_min_obj_and_grad(flat, beta):
    """
    Compute the soft-min objective and gradient for L-BFGS-B.
    Objective: minimize f(x) = (1/beta) * ln(sum(exp(-beta * x_k)))
    where x_k are the critical distances (pairwise and boundary).
    Minimizing this is equivalent to maximizing the soft-min of x_k.
    """
    pts = flat.reshape(N, 2)
    
    # Calculate boundary distances: 1 - |c_i|
    dists_origin = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - dists_origin
    
    # Calculate pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    
    # We only need upper triangle for unique pairs
    iu = np.triu_indices(N, k=1)
    pair_dists = dist_mat[iu]
    
    # Combine all critical distances
    all_dists = np.concatenate([boundary_dists, pair_dists / 2.0])
    
    # Soft min calculation
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

def _deterministic_ring_init():
    """
    Generate a high-quality initial configuration based on the known optimal structure.
    Structure: 1 center, 6, 12, 18, 13 points in concentric rings.
    """
    pts_list = []
    
    # Ring 0: 1 point at center
    pts_list.append([0.0, 0.0])
    
    # Ring 1: 6 points
    r1 = 0.25
    for k in range(6):
        angle = 2 * np.pi * k / 6
        pts_list.append([r1 * np.cos(angle), r1 * np.sin(angle)])
        
    # Ring 2: 12 points
    r2 = 0.50
    offset2 = np.pi / 12 # Shift to interlock with previous ring
    for k in range(12):
        angle = offset2 + 2 * np.pi * k / 12
        pts_list.append([r2 * np.cos(angle), r2 * np.sin(angle)])
        
    # Ring 3: 18 points
    r3 = 0.75
    offset3 = 0.0
    for k in range(18):
        angle = offset3 + 2 * np.pi * k / 18
        pts_list.append([r3 * np.cos(angle), r3 * np.sin(angle)])
        
    # Ring 4: 13 points
    r4 = 0.95
    offset4 = np.pi / 13
    for k in range(13):
        angle = offset4 + 2 * np.pi * k / 13
        pts_list.append([r4 * np.cos(angle), r4 * np.sin(angle)])
        
    pts = np.array(pts_list[:N])
    
    # Ensure points are strictly inside
    dists = np.linalg.norm(pts, axis=1)
    mask = dists > 0.999
    pts[mask] = pts[mask] * (0.999 / dists[mask, np.newaxis])
    
    return pts

def _force_directed_relax(centers, max_steps=1000):
    """
    Force-directed relaxation using repulsive springs and boundary constraint.
    """
    current = centers.copy()
    n = len(current)
    best_score = _score(current)
    best_positions = current.copy()
    
    # Pre-compute indices for pairs
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    for step in range(max_steps):
        if time.time() > _DEADLINE - 10.0:
            break
        
        forces = np.zeros_like(current)
        
        # Repulsive forces between all pairs
        diff = current[i_idx, :] - current[j_idx, :]
        dists_sq = np.sum(diff ** 2, axis=1)
        dists = np.sqrt(dists_sq)
        safe_dists = np.where(dists > 1e-8, dists, 1.0)
        
        # 1/d^2 potential for strong short-range repulsion
        # We want to maximize min distance, so we push apart
        repulsion_strength = 1.0 / (safe_dists ** 2 + 1e-4)
        unit_vecs = diff / safe_dists[:, np.newaxis]
        weighted_forces = repulsion_strength[:, np.newaxis] * unit_vecs
        
        np.add.at(forces, i_idx, weighted_forces)
        np.add.at(forces, j_idx, -weighted_forces)
        
        # Boundary constraint: push inward if too close to edge
        norms = np.linalg.norm(current, axis=1)
        # Soft boundary: quadratic penalty outside radius 0.95
        # Force is proportional to (r - r_boundary)^2 * gradient of r
        boundary_force_mag = np.where(norms > 0.95, (norms - 0.95) ** 2 * 50.0, 0.0)
        safe_norms = np.where(norms > 1e-8, norms, 1.0)
        forces -= boundary_force_mag[:, np.newaxis] * current / safe_norms[:, np.newaxis]
        
        # Adaptive time step
        max_force = np.max(np.linalg.norm(forces, axis=1))
        dt = 0.001 / (1.0 + max_force * 2.0)
        
        current += dt * forces
        current = _project_to_disk(current, limit=0.999)
        
        # Check improvement
        s = _score(current)
        if s > best_score:
            best_score = s
            best_positions = current.copy()
            
    return best_positions

def _project_to_disk(pts, limit=0.999):
    """Project points back into the unit disk (with a safety margin)."""
    dists = np.linalg.norm(pts, axis=1)
    mask = dists > limit
    if np.any(mask):
        pts = pts.copy()
        pts[mask] = pts[mask] * (limit / dists[mask, np.newaxis])
    return pts

def _optimize_lbfgs(pts, beta, maxiter=1000, gtol=1e-10):
    """Run L-BFGS-B optimization with given beta."""
    try:
        res = minimize(
            _soft_min_obj_and_grad,
            pts.flatten(),
            args=(beta,),
            method='L-BFGS-B',
            jac=True,
            bounds=[(-1.0, 1.0)] * (2 * N),
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': gtol, 'disp': False}
        )
        new_pts = res.x.reshape(N, 2)
        new_pts = _project_to_disk(new_pts, limit=0.9999)
        return new_pts
    except Exception:
        return pts

def construct():
    start_time = time.time()
    
    # 1. Generate high-quality deterministic initialization
    init_pts = _deterministic_ring_init()
    best_pts = init_pts.copy()
    best_score = _score(best_pts)
    
    # 2. Force-directed relaxation to settle into a good local minimum
    relaxed_pts = _force_directed_relax(init_pts, max_steps=1000)
    relaxed_score = _score(relaxed_pts)
    if relaxed_score > best_score:
        best_score = relaxed_score
        best_pts = relaxed_pts.copy()
    
    # 3. L-BFGS-B optimization with increasing beta
    # Start with moderate beta to find the basin, then increase for precision
    betas = [50, 200, 1000, 5000, 20000]
    
    current_pts = best_pts.copy()
    
    for beta in betas:
        if time.time() > _DEADLINE - 15.0:
            break
            
        # Run L-BFGS-B
        new_pts = _optimize_lbfgs(current_pts, beta, maxiter=500, gtol=1e-10)
        new_score = _score(new_pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            current_pts = new_pts.copy()
        else:
            # If no improvement, try a small perturbation and re-relax
            if time.time() < _DEADLINE - 20.0:
                perturbed = current_pts + np.random.normal(0, 0.01, current_pts.shape)
                perturbed = _project_to_disk(perturbed, limit=0.999)
                perturbed = _force_directed_relax(perturbed, max_steps=500)
                p_score = _score(perturbed)
                if p_score > best_score:
                    best_score = p_score
                    best_pts = perturbed.copy()
                    current_pts = perturbed.copy()
    
    # 4. Final high-precision polish
    if time.time() < _DEADLINE - 5.0:
        final_pts = _optimize_lbfgs(best_pts, 50000, maxiter=2000, gtol=1e-12)
        final_score = _score(final_pts)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts

    return best_pts
