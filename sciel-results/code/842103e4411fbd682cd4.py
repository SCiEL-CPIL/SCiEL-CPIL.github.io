"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _score(points):
    """Largest common radius: min of half pairwise distances and wall clearance."""
    d = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(d * d, axis=2))
    iu = np.triu_indices(N, k=1)
    min_pair = dists[iu].min()
    clearance = (0.5 - np.abs(points)).min()
    return min(min_pair * 0.5, clearance)

def _soft_min_radius(x, beta=100.0):
    """
    Soft-min surrogate for the maximum radius.
    x is a flat array of length 3N.
    Returns a smooth approximation of the minimum radius.
    """
    c = x.reshape(N, 3)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1e10)
    
    # Wall clearances
    wall_clearances = 0.5 - np.abs(c)  # (N, 3)
    
    # Collect all constraints: half pairwise distances and wall clearances
    iu = np.triu_indices(N, k=1)
    pair_vals = dists[iu] * 0.5  # (N*(N-1)/2,)
    wall_vals = wall_clearances.flatten()  # (3N,)
    
    all_vals = np.concatenate([pair_vals, wall_vals])
    
    # Soft minimum: -log(sum(exp(-beta * v))) / beta
    # Use log-sum-exp for numerical stability
    max_val = np.max(all_vals)
    log_sum = np.log(np.sum(np.exp(-beta * (all_vals - max_val))))
    soft_min = max_val - log_sum / beta
    
    return soft_min

def _soft_min_obj_grad(x, beta=100.0):
    """
    Objective and gradient for soft-min radius maximization.
    Returns (negative_soft_min, gradient).
    """
    c = x.reshape(N, 3)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1e10)
    
    wall_clearances = 0.5 - np.abs(c)
    
    iu = np.triu_indices(N, k=1)
    pair_vals = dists[iu] * 0.5
    wall_vals = wall_clearances.flatten()
    
    all_vals = np.concatenate([pair_vals, wall_vals])
    
    max_val = np.max(all_vals)
    exp_vals = np.exp(-beta * (all_vals - max_val))
    log_sum = np.log(np.sum(exp_vals))
    soft_min = max_val - log_sum / beta
    
    # Gradient of soft_min w.r.t. x
    # d(soft_min)/dx = -1/beta * d(log_sum)/dx
    # d(log_sum)/dx = -beta * sum(exp_i * dv_i/dx) / sum(exp_i)
    # So d(soft_min)/dx = sum(exp_i * dv_i/dx) / sum(exp_i)
    
    weights = exp_vals / np.sum(exp_vals)  # (n_constraints,)
    
    # Gradient of pairwise terms
    grad_pair = np.zeros((N, 3))
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    # For each pair (i,j), contribution to grad[i] is -0.5 * w_ij * unit_vec[i,j]
    # and to grad[j] is +0.5 * w_ij * unit_vec[i,j]
    i_idx, j_idx = iu
    w_pairs = weights[:len(i_idx)]  # weights for pairwise constraints
    
    # Accumulate gradients using scatter
    for i, j, w in zip(i_idx, j_idx, w_pairs):
        vec = unit_vecs[i, j]
        grad_pair[i] -= 0.5 * w * vec
        grad_pair[j] += 0.5 * w * vec
    
    # Gradient of wall terms
    grad_wall = np.zeros((N, 3))
    signs = np.sign(c)
    w_walls = weights[len(i_idx):]  # weights for wall constraints
    w_walls_reshaped = w_walls.reshape(N, 3)
    grad_wall = -w_walls_reshaped * signs
    
    total_grad = (grad_pair + grad_wall).flatten()
    
    return -soft_min, total_grad

def _pso_optimize(init_pts, time_limit=60.0, n_particles=10, n_iters=200, beta=100.0):
    """
    Particle Swarm Optimization for sphere packing.
    Each particle is a 300-dimensional configuration.
    """
    t_start = time.time()
    dim = N * 3
    
    # Initialize particles
    particles = np.zeros((n_particles, dim))
    velocities = np.zeros((n_particles, dim))
    
    # Initialize from init_pts with some spread
    for i in range(n_particles):
        if i == 0:
            particles[i] = init_pts.flatten()
        else:
            noise = np.random.randn(dim) * 0.05
            particles[i] = np.clip(init_pts.flatten() + noise, -0.48, 0.48)
    
    # Evaluate initial fitness
    best_personal = np.zeros((n_particles, dim))
    best_personal_scores = np.zeros(n_particles)
    global_best = np.zeros(dim)
    global_best_score = -np.inf
    
    for i in range(n_particles):
        score = _soft_min_radius(particles[i], beta)
        best_personal[i] = particles[i].copy()
        best_personal_scores[i] = score
        if score > global_best_score:
            global_best_score = score
            global_best = particles[i].copy()
    
    # PSO parameters
    w = 0.7  # inertia
    c1 = 1.5  # cognitive
    c2 = 1.5  # social
    max_velocity = 0.05
    
    for it in range(n_iters):
        if time.time() - t_start > time_limit:
            break
        
        for i in range(n_particles):
            if time.time() - t_start > time_limit:
                break
            
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            
            velocities[i] = (w * velocities[i] +
                           c1 * r1 * (best_personal[i] - particles[i]) +
                           c2 * r2 * (global_best - particles[i]))
            
            # Clip velocities
            velocities[i] = np.clip(velocities[i], -max_velocity, max_velocity)
            
            particles[i] = particles[i] + velocities[i]
            particles[i] = np.clip(particles[i], -0.49, 0.49)
            
            # Evaluate
            score = _soft_min_radius(particles[i], beta)
            
            if score > best_personal_scores[i]:
                best_personal_scores[i] = score
                best_personal[i] = particles[i].copy()
                
                if score > global_best_score:
                    global_best_score = score
                    global_best = particles[i].copy()
        
        # Adaptive beta: increase over time for sharper minimum
        beta *= 1.05
    
    return global_best.reshape(N, 3)

def _bfgs_polish(init_pts, target_r, max_iter=2000, gtol=1e-10, time_limit=10.0):
    """
    BFGS polish with hard constraints.
    """
    t_start = time.time()
    
    def energy_and_grad(x):
        if time.time() - t_start > time_limit:
            raise TimeoutError()
        c = x.reshape(N, 3)
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 1.0)
        
        active = dists < 2.0 * target_r
        active[np.diag_indices_from(active)] = False
        
        violations = np.where(active, 2.0 * target_r - dists, 0.0)
        energy_pair = 0.5 * np.sum(violations**2)
        
        safe_dists = np.where(dists > 1e-12, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        grad_pair = -np.sum(violations[:, :, np.newaxis] * unit_vecs, axis=1)
        
        wall_dists = 0.5 - np.abs(c)
        wall_violations = np.maximum(0.0, target_r - wall_dists)
        energy_wall = np.sum(wall_violations**2)
        
        signs = np.sign(c)
        grad_wall = 2.0 * wall_violations * signs
        
        total_energy = energy_pair + energy_wall
        total_grad = (grad_pair + grad_wall).flatten()
        
        return total_energy, total_grad
    
    try:
        result = minimize(
            energy_and_grad,
            init_pts.flatten(),
            method='BFGS',
            jac=True,
            options={'maxiter': max_iter, 'gtol': gtol, 'disp': False}
        )
        opt_pts = result.x.reshape(N, 3)
        return opt_pts
    except Exception:
        return init_pts

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # Safety margin under 150s

    # ---- Seed: FCC-like initialization ----
    # Generate FCC lattice points
    fcc_points = []
    n = 3
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            for k in range(-n, n+1):
                fcc_points.append([i, j, k])
                fcc_points.append([i+0.5, j+0.5, k])
                fcc_points.append([i+0.5, j, k+0.5])
                fcc_points.append([i, j+0.5, k+0.5])
    
    fcc_points = np.array(fcc_points)
    center = fcc_points.mean(axis=0)
    fcc_points = fcc_points - center
    
    # Scale to fit in cube
    max_extent = np.abs(fcc_points).max()
    fcc_points = fcc_points / max_extent * 0.45
    
    # Select N closest to origin
    dists_to_origin = np.linalg.norm(fcc_points, axis=1)
    idx = np.argsort(dists_to_origin)
    pts = fcc_points[idx[:N]].copy()
    
    best_pts = pts.copy()
    best_r = _score(best_pts)
    
    rng = np.random.RandomState(42)

    # Strategy 1: PSO with soft-min objective
    if time.time() - t0 < DEADLINE - 30.0:
        pso_time = min(50.0, DEADLINE - (time.time() - t0) - 25.0)
        pso_result = _pso_optimize(best_pts, time_limit=pso_time, 
                                    n_particles=12, n_iters=300, beta=80.0)
        
        if np.all(np.isfinite(pso_result)):
            pso_result = np.clip(pso_result, -0.5 + 1e-9, 0.5 - 1e-9)
            r_pso = _score(pso_result)
            
            if r_pso > best_r:
                best_r = r_pso
                best_pts = pso_result.copy()

    # Strategy 2: BFGS polish with increasing targets
    if time.time() - t0 < DEADLINE - 10.0:
        targets = [best_r + 0.001, best_r + 0.003, best_r + 0.005]
        
        for target_r in targets:
            if time.time() - t0 > DEADLINE - 5.0:
                break
            
            noise = rng.normal(0, 0.002, size=(N, 3))
            init_pts = np.clip(best_pts + noise, -0.49, 0.49)
            
            bfgs_time = min(15.0, DEADLINE - (time.time() - t0) - 3.0)
            opt_pts = _bfgs_polish(init_pts, target_r, 
                                    max_iter=3000, gtol=1e-11, 
                                    time_limit=bfgs_time)
            
            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                r_opt = _score(opt_pts)
                
                if r_opt > best_r:
                    best_r = r_opt
                    best_pts = opt_pts.copy()

    # Strategy 3: Second PSO pass with higher beta
    if time.time() - t0 < DEADLINE - 15.0:
        pso_time2 = min(25.0, DEADLINE - (time.time() - t0) - 10.0)
        pso_result2 = _pso_optimize(best_pts, time_limit=pso_time2,
                                     n_particles=8, n_iters=200, beta=200.0)
        
        if np.all(np.isfinite(pso_result2)):
            pso_result2 = np.clip(pso_result2, -0.5 + 1e-9, 0.5 - 1e-9)
            r_pso2 = _score(pso_result2)
            
            if r_pso2 > best_r:
                best_r = r_pso2
                best_pts = pso_result2.copy()

    # Final safety check
    best_pts = np.clip(best_pts, -0.5 + 1e-9, 0.5 - 1e-9)
    if not np.all(np.isfinite(best_pts)):
        # Fallback to grid
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts = grid[:N].copy()
        
    return best_pts
# EVOLVE-BLOCK-END
