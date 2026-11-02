"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START

def _get_packing_from_positions(p):
    """
    Given center positions p (N,2), compute maximum valid radii.
    r_i = min(dist_to_boundary_i, min_{j!=i} 0.5 * dist(i,j))
    """
    p = np.asarray(p, dtype=np.float64).reshape(N, 2).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    
    min_dist_to_others = np.min(dist, axis=1)
    dist_to_box = np.min(np.stack([p[:, 0], 1.0 - p[:, 0], p[:, 1], 1.0 - p[:, 1]], axis=1), axis=1)
    
    radii = np.minimum(dist_to_box, 0.5 * min_dist_to_others)
    radii = np.maximum(radii, 0.0)
    
    # Safety factor for strict non-overlap
    radii *= 0.99999
    
    # Ensure centers are within bounds considering radius
    p[:, 0] = np.clip(p[:, 0], radii, 1.0 - radii)
    p[:, 1] = np.clip(p[:, 1], radii, 1.0 - radii)
    
    return p, radii

def _soft_sum_radii(pos, beta=50.0):
    """Smooth surrogate for sum of radii using soft-min."""
    p = pos.reshape(N, 2)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-12)
    np.fill_diagonal(dist, np.inf)
    
    # Soft min of pairwise distances
    exp_dist = np.exp(-beta * dist)
    sum_exp_dist = np.sum(exp_dist, axis=1)
    D_min = -1.0 / beta * np.log(sum_exp_dist + 1e-300)
    
    # Soft min of box distances
    x = p[:, 0]
    y = p[:, 1]
    d_box_terms = np.stack([x, 1.0 - x, y, 1.0 - y], axis=1)
    exp_box = np.exp(-beta * d_box_terms)
    sum_exp_box = np.sum(exp_box, axis=1)
    D_box = -1.0 / beta * np.log(sum_exp_box + 1e-300)
    
    # Soft min of D_box and 0.5 * D_min
    a = D_box
    b = 0.5 * D_min
    exp_a = np.exp(-beta * a)
    exp_b = np.exp(-beta * b)
    sum_exp_R = exp_a + exp_b
    R = -1.0 / beta * np.log(sum_exp_R + 1e-300)
    
    return -np.sum(R)

def _soft_sum_radii_adaptive(pos, beta_schedule):
    """Wrapper that uses a scheduled beta value."""
    return _soft_sum_radii(pos, beta=beta_schedule[0])

def _soft_grad_sum_radii_adaptive(pos, beta_schedule):
    """Wrapper for gradient with scheduled beta."""
    return _soft_grad_sum_radii(pos, beta=beta_schedule[0])

def _soft_grad_sum_radii(pos, beta=50.0):
    """Gradient of the soft surrogate."""
    p = pos.reshape(N, 2)
    
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-12)
    np.fill_diagonal(dist, np.inf)
    
    # Weights for soft min of distances
    exp_dist = np.exp(-beta * dist)
    sum_exp_dist = np.sum(exp_dist, axis=1)
    
    # Gradient of D_min_i wrt p_i
    weights = exp_dist / sum_exp_dist[:, np.newaxis]
    valid_dist = np.where(dist > 1e-12, dist, 1.0)
    grad_vec = weights / valid_dist
    grad_vec = np.where(np.isfinite(grad_vec), grad_vec, 0.0)
    grad_D_min = np.sum(grad_vec[:, :, np.newaxis] * diff, axis=1)
    
    # Box part
    x = p[:, 0]
    y = p[:, 1]
    d_box_terms = np.stack([x, 1.0 - x, y, 1.0 - y], axis=1)
    exp_box = np.exp(-beta * d_box_terms)
    sum_exp_box = np.sum(exp_box, axis=1)
    
    w_box = exp_box / sum_exp_box[:, np.newaxis]
    grad_box_terms = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    grad_D_box = np.dot(w_box, grad_box_terms)
    
    # Combine for R_i
    a = -1.0 / beta * np.log(sum_exp_box + 1e-300)
    b = -0.5 / beta * np.log(sum_exp_dist + 1e-300)
    
    exp_a = np.exp(-beta * a)
    exp_b = np.exp(-beta * b)
    sum_exp_R = exp_a + exp_b
    
    w_a = exp_a / sum_exp_R
    w_b = exp_b / sum_exp_R
    
    grad_R = w_a[:, np.newaxis] * grad_D_box + w_b[:, np.newaxis] * (0.5 * grad_D_min)
    
    return -grad_R.flatten()

def _local_perturb_and_refine(pos, best_score, beta=100.0, n_perturb=10):
    """
    Try small random perturbations around current position to escape local minima.
    Returns improved position and score if found.
    """
    p = pos.reshape(N, 2).copy()
    current_score = best_score
    
    for _ in range(n_perturb):
        # Small random perturbation
        perturbation = np.random.normal(0, 0.01, size=(N, 2))
        p_perturbed = p + perturbation
        p_perturbed[:, 0] = np.clip(p_perturbed[:, 0], 0.0, 1.0)
        p_perturbed[:, 1] = np.clip(p_perturbed[:, 1], 0.0, 1.0)
        
        # Quick local optimization with higher beta
        try:
            res = minimize(_soft_sum_radii, p_perturbed.flatten(), 
                          jac=_soft_grad_sum_radii,
                          method='L-BFGS-B', 
                          bounds=[(0.0, 1.0)] * (2 * N),
                          args=(beta,),
                          options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10})
            
            p_new, radii_new = _get_packing_from_positions(res.x)
            score_new = np.sum(radii_new)
            
            if score_new > current_score:
                current_score = score_new
                p = p_new
        except Exception:
            continue
    
    return p, current_score

def _hex_initialization():
    """Generate a hexagonal grid initialization."""
    pts = []
    for i in range(6):
        for j in range(6):
            x = (i + 0.5) / 6.0
            y = (j + 0.5) / 6.0
            if i % 2 == 0:
                y += 0.5 / 6.0 * 0.8
            if 0.0 < x < 1.0 and 0.0 < y < 1.0:
                pts.append([x, y])
    
    if len(pts) < N:
        while len(pts) < N:
            pts.append([np.random.rand(), np.random.rand()])
            
    return np.array(pts[:N]).flatten()

def _grid_initialization(spacing=1.0/6.0):
    """Generate a regular grid initialization."""
    pts = []
    n_per_side = int(1.0 / spacing) + 1
    for i in range(n_per_side):
        for j in range(n_per_side):
            x = (i + 0.5) * spacing
            y = (j + 0.5) * spacing
            if 0.0 < x < 1.0 and 0.0 < y < 1.0:
                pts.append([x, y])
    
    # Shuffle and take N
    np.random.shuffle(pts)
    if len(pts) < N:
        while len(pts) < N:
            pts.append([np.random.rand(), np.random.rand()])
    return np.array(pts[:N]).flatten()

def _physics_preprocess(pos, n_iters=80):
    """
    Physics-based push-apart to spread out initial points.
    """
    p = pos.reshape(N, 2).copy()
    target_dist = 1.0 / np.sqrt(N) * 0.9
    for _ in range(n_iters):
        forces = np.zeros((N, 2), dtype=np.float64)
        for i in range(N):
            for j in range(i + 1, N):
                d = p[i] - p[j]
                dist = np.sqrt(np.sum(d ** 2))
                if dist < target_dist:
                    if dist > 1e-10:
                        f = 0.3 * (target_dist - dist) * d / dist
                    else:
                        f = np.array([0.01, 0.01])
                    forces[i] += f
                    forces[j] -= f
        p += 0.1 * forces
        p[:, 0] = np.clip(p[:, 0], 0.01, 0.99)
        p[:, 1] = np.clip(p[:, 1], 0.01, 0.99)
    return p.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = -np.inf
    best_result = None
    
    # Generate diverse initializations
    inits = []
    inits.append(_hex_initialization())
    inits.append(_grid_initialization(1.0/6.0))
    inits.append(_grid_initialization(1.0/5.0))
    for _ in range(4):
        inits.append(np.random.rand(2 * N))
    
    # Preprocess with physics
    for idx in range(len(inits)):
        if time.time() > deadline - 60.0:
            break
        inits[idx] = _physics_preprocess(inits[idx], n_iters=50)
    
    # Use soft-min surrogate for optimization
    bounds = [(0.0, 1.0)] * (2 * N)
    
    for init_pos in inits:
        if time.time() > deadline - 30.0:
            break
        
        # Adaptive beta: start soft, end sharp
        beta_schedule = [20.0, 50.0, 100.0]
        
        for beta in beta_schedule:
            if time.time() > deadline - 25.0:
                break
            
            # Try L-BFGS-B with smooth gradient
            try:
                res = minimize(_soft_sum_radii, init_pos, jac=_soft_grad_sum_radii,
                              method='L-BFGS-B', bounds=bounds,
                              args=(beta,),
                              options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
                
                p, radii = _get_packing_from_positions(res.x)
                score = np.sum(radii)
                
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack((p, radii))
                    init_pos = res.x  # Continue from improved position
                    
            except Exception:
                pass
    
    # Local perturbation to escape local minima
    if best_result is not None and time.time() < deadline - 15.0:
        try:
            p_perturbed, score_perturbed = _local_perturb_and_refine(
                best_result[:, :2].flatten(), best_score, beta=100.0, n_perturb=8)
            
            if score_perturbed > best_score:
                best_score = score_perturbed
                p_new, radii_new = _get_packing_from_positions(p_perturbed)
                best_result = np.column_stack((p_new, radii_new))
                
        except Exception:
            pass
    
    # Final high-precision polish with multiple passes
    if best_result is not None and time.time() < deadline - 5.0:
        for polish_pass in range(2):
            if time.time() > deadline - 5.0:
                break
            
            try:
                start_pos = best_result[:, :2].flatten()
                res_final = minimize(_soft_sum_radii, start_pos, jac=_soft_grad_sum_radii,
                                    method='L-BFGS-B', bounds=bounds,
                                    args=(200.0,),
                                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100})
                
                p_final, radii_final = _get_packing_from_positions(res_final.x)
                score_final = np.sum(radii_final)
                
                if score_final > best_score:
                    best_score = score_final
                    best_result = np.column_stack((p_final, radii_final))
                    
            except Exception:
                pass
    
    # Fallback
    if best_result is None:
        p, radii = _get_packing_from_positions(inits[0])
        best_result = np.column_stack((p, radii))
        
    return best_result

# EVOLVE-BLOCK-END
