"""Combined program: hexagonal lattice init + force-directed relaxation + soft-min optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _hex_lattice_init(n):
    """
    Initialize points in a hexagonal lattice clipped to the unit circle.
    Try multiple spacings and pick the best initial score.
    """
    best_points = None
    best_score = -1.0
    
    # Try a range of spacings around the expected optimal
    # Optimal r ~ 0.053, so spacing ~ 2r ~ 0.106
    spacings = [0.10, 0.103, 0.106, 0.109, 0.112, 0.115, 0.12]
    
    for s in spacings:
        max_k = int(1.0 / (s * 0.5)) + 2
        candidates = []
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = s * (i + 0.5 * (j % 2))
                y = s * j * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    candidates.append([x, y])
        
        candidates = np.array(candidates)
        
        if len(candidates) < n:
            continue
        
        # Sort by distance to origin to pick the most central ones
        radii = np.linalg.norm(candidates, axis=1)
        indices = np.argsort(radii)[:n]
        points = candidates[indices].copy()
        
        # Add small random perturbation to break symmetry
        points += np.random.normal(0, 1e-4, points.shape)
        
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points
    
    if best_points is None:
        # Fallback: random points
        best_points = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(best_points, axis=1)
        mask = norms > 1.0
        best_points[mask] = best_points[mask] / norms[mask, np.newaxis] * 0.9
    
    return best_points

def _force_directed_relax(points, deadline, max_iters=3000, step_size=0.002):
    """
    Force-directed relaxation to push circles apart and improve the minimum radius.
    """
    current_points = points.copy()
    current_r = _compute_score(current_points)
    best_points = current_points.copy()
    best_r = current_r
    
    decay = 0.9995
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute pairwise distances
        diff = current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        radii = np.linalg.norm(current_points, axis=1)
        min_dist = np.min(dists)
        min_dist_boundary = np.min(1.0 - radii)
        r = min(min_dist / 2.0, min_dist_boundary)
        
        if r > best_r:
            best_r = r
            best_points = current_points.copy()
        
        # Target radius: slightly larger than current to drive improvement
        target_r = r * 1.001 + 0.00005
        
        forces = np.zeros_like(current_points)
        
        # Pairwise repulsion for pairs closer than 2*target_r
        mask_overlap = dists < 2.0 * target_r
        mask_overlap[np.diag_indices_from(mask_overlap)] = False
        
        safe_dists = np.where(dists < 1e-9, 1e-9, dists)
        overlap = np.where(mask_overlap, 2.0 * target_r - dists, 0.0)
        
        # Force: overlap * direction / dist
        force_matrix = overlap[:, :, np.newaxis] * diff / safe_dists[:, :, np.newaxis]
        forces += np.sum(force_matrix, axis=1)
        
        # Boundary constraint: pull points back if too close to edge
        mask_boundary = radii > 1.0 - target_r
        boundary_overlap = np.where(mask_boundary, radii - (1.0 - target_r), 0.0)
        safe_radii = np.where(radii < 1e-9, 1e-9, radii)
        boundary_forces = -current_points / safe_radii[:, np.newaxis] * boundary_overlap[:, np.newaxis]
        forces += boundary_forces
        
        # Normalize forces
        force_magnitudes = np.linalg.norm(forces, axis=1, keepdims=True)
        max_force = np.max(force_magnitudes)
        if max_force > 0:
            normalized_forces = forces / max_force
        else:
            normalized_forces = forces
        
        current_points += step_size * normalized_forces
        
        # Project back inside unit circle
        radii_new = np.linalg.norm(current_points, axis=1)
        out_of_bounds = radii_new > 1.0
        if np.any(out_of_bounds):
            current_points[out_of_bounds] *= (0.9999 / radii_new[out_of_bounds])[:, np.newaxis]
        
        # Decay step size
        step_size *= decay
        
        # Random jitter every 200 iterations to escape local minima
        if it % 200 == 0 and it > 0:
            current_points += np.random.randn(N, 2) * 0.0001
            radii_new = np.linalg.norm(current_points, axis=1)
            out_of_bounds = radii_new > 1.0
            if np.any(out_of_bounds):
                current_points[out_of_bounds] *= (0.9999 / radii_new[out_of_bounds])[:, np.newaxis]
    
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Initialization with best hex lattice
    points = _hex_lattice_init(N)
    best_points = points.copy()
    best_score = _compute_score(points)
    
    # 2. Force-directed relaxation phase
    if time.time() < deadline - 30:
        points = _force_directed_relax(points, deadline - 30, max_iters=5000, step_size=0.002)
        score = _compute_score(points)
        if score > best_score:
            best_points = points
            best_score = score
    
    # 3. Soft-min optimization with annealing
    def objective_and_grad(x, beta):
        p = x.reshape(-1, 2)
        
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu] / 2.0
        
        norms = np.linalg.norm(p, axis=1)
        bound_dists = 1.0 - norms
        
        z_pairs = -beta * pair_dists
        z_bounds = -beta * bound_dists
        
        max_z_pair = np.max(z_pairs)
        max_z_bound = np.max(z_bounds)
        max_z = max(max_z_pair, max_z_bound)
        
        exp_pairs = np.exp(z_pairs - max_z)
        exp_bounds = np.exp(z_bounds - max_z)
        
        sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
        
        log_sum_exp = max_z + np.log(sum_exp)
        
        val = log_sum_exp / beta
        
        w_pairs = exp_pairs / sum_exp
        w_bounds = exp_bounds / sum_exp
        
        grad = np.zeros_like(p)
        
        i_idx, j_idx = iu
        safe_pair_dists = np.where(pair_dists > 1e-12, pair_dists, 1.0)
        coeff = w_pairs * 0.25 / safe_pair_dists
        vecs = p[i_idx] - p[j_idx]
        
        np.add.at(grad, i_idx, coeff[:, np.newaxis] * vecs)
        np.add.at(grad, j_idx, -coeff[:, np.newaxis] * vecs)
        
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        bound_grad_vecs = -p / safe_norms[:, np.newaxis]
        grad += w_bounds[:, np.newaxis] * bound_grad_vecs
        
        return val, -grad
    
    # Annealing schedule
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
    
    current_points = best_points.flatten()
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
        
        try:
            res = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                current_points,
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 200,
                    'gtol': 1e-8,
                    'ftol': 1e-12,
                    'maxls': 25,
                    'disp': False
                }
            )
            
            if time.time() < deadline:
                candidate = res.x.reshape(-1, 2)
                candidate_score = _compute_score(candidate)
                
                if candidate_score > best_score:
                    best_points = candidate
                    best_score = candidate_score
                    current_points = candidate.flatten()
                else:
                    current_points = candidate.flatten()
                    
        except Exception:
            break
    
    # 4. Final high-precision polish
    if time.time() < deadline - 3:
        try:
            res_final = minimize(
                lambda x: objective_and_grad(x, 5000.0)[0],
                current_points,
                jac=lambda x: objective_and_grad(x, 5000.0)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 300,
                    'gtol': 1e-10,
                    'ftol': 1e-14,
                    'maxls': 30,
                    'disp': False
                }
            )
            
            if time.time() < deadline:
                candidate = res_final.x.reshape(-1, 2)
                candidate_score = _compute_score(candidate)
                if candidate_score > best_score:
                    best_points = candidate
                    best_score = candidate_score
                    
        except Exception:
            pass
    
    # Final validity check: ensure all points are inside the unit circle
    norms = np.linalg.norm(best_points, axis=1)
    outside = norms > 1.0
    if np.any(outside):
        best_points[outside] *= (0.9999 / norms[outside])[:, np.newaxis]
    
    return best_points

# EVOLVE-BLOCK-END
