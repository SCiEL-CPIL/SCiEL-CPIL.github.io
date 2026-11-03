import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _log_energy(points):
    """Compute log(1/d) energy using math.fsum for accuracy."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    log_dists = 0.5 * np.log(dists_sq[iu])
    return -math.fsum(log_dists.tolist())

def _fast_log_energy(points):
    """Compute log(1/d) energy using np.sum for speed."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    log_dists = 0.5 * np.log(dists_sq[iu])
    return -np.sum(log_dists)

def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _energy_and_grad(x):
    """Energy and gradient for L-BFGS-B. x is flat array of shape (N*3,)."""
    points = x.reshape(N, 3)
    points = _project_to_sphere(points)
    
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    
    # E = -sum_{i<j} log(d_ij)
    # dE/dx_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    weights = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    weights[mask] = 1.0 / dists_sq[mask]
    
    sum_weights = np.sum(weights, axis=1)
    weighted_sum = weights @ points
    
    grads = weighted_sum - points * sum_weights[:, np.newaxis]
    
    # Project gradient to tangent space
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points
    
    # Compute energy with math.fsum for consistency with final score
    iu = np.triu_indices(n, k=1)
    log_dists = 0.5 * np.log(dists_sq[iu])
    energy = -math.fsum(log_dists.tolist())
    
    return energy, grads_tangent.flatten()

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_uniform_sphere(n, seed=42):
    """Generate random uniform points on sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def _lbfgs_polish(points, max_iter=2000, ftol=1e-12, gtol=1e-10, time_limit=10.0):
    """Polish a point configuration using L-BFGS-B."""
    start = time.time()
    x0 = points.flatten()
    e0 = _log_energy(points)
    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'maxfun': max_iter * 10,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': 50
            }
        )
        if res.success or res.fun < e0:
            p_res = _project_to_sphere(res.x.reshape(N, 3))
            e_check = _log_energy(p_res)
            if e_check < e0 and time.time() - start < time_limit:
                return p_res
    except Exception:
        pass
    return points

def _tangent_vector(point, rng):
    """Generate a random unit tangent vector at point on sphere."""
    rand_vec = rng.randn(3)
    tangent = rand_vec - np.dot(rand_vec, point) * point
    norm = np.linalg.norm(tangent)
    if norm < 1e-15:
        rand_vec = rng.randn(3)
        tangent = rand_vec - np.dot(rand_vec, point) * point
        norm = np.linalg.norm(tangent)
    return tangent / norm

def _particle_swarm_optimization(points, max_iter=200, time_limit=60.0, rng=None, n_particles=8):
    """
    Particle Swarm Optimization on the sphere.
    Each particle is a full configuration of N points.
    Velocity is in tangent space (approximated by adding noise and re-projecting).
    """
    if rng is None:
        rng = np.random.RandomState(999)
    
    start = time.time()
    
    # Initialize swarm
    # Particle 0 is the current best
    particles = [points.copy() for _ in range(n_particles)]
    # Initialize others with small perturbations
    for i in range(1, n_particles):
        perturbed = particles[0].copy()
        for _ in range(rng.randint(5, 15)):
            idx = rng.randint(N)
            tangent = _tangent_vector(perturbed[idx], rng)
            step = 0.2
            new_pt = perturbed[idx] + step * tangent
            perturbed[idx] = new_pt / np.linalg.norm(new_pt)
        particles[i] = perturbed
        
    # Evaluate initial energies
    particle_energies = [_fast_log_energy(p) for p in particles]
    
    # Personal bests
    personal_bests = [p.copy() for p in particles]
    personal_best_energies = particle_energies.copy()
    
    # Global best
    global_best_idx = np.argmin(personal_best_energies)
    global_best = particles[global_best_idx].copy()
    global_best_energy = personal_best_energies[global_best_idx]
    
    # Best result overall (including input)
    best_points = points.copy()
    best_energy = _fast_log_energy(best_points)
    if global_best_energy < best_energy:
        best_energy = global_best_energy
        best_points = global_best.copy()
        
    # PSO Parameters
    # w: inertia, c1: cognitive, c2: social
    w = 0.5
    c1 = 1.5
    c2 = 1.5
    max_speed = 0.5  # Max step size in tangent space
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        
        for i in range(n_particles):
            if time.time() - start > time_limit:
                break
            
            p = particles[i]
            pb = personal_bests[i]
            gb = global_best
            
            # Velocity update (in flat space, then projected)
            # v = w*v + c1*r1*(pb - p) + c2*r2*(gb - p)
            # We don't store velocity explicitly, we compute a "step"
            # step = alpha*(pb - p) + beta*(gb - p)
            # This is a simplified PSO where we move towards bests.
            
            r1 = rng.rand()
            r2 = rng.rand()
            
            # Compute direction in flat space
            # Note: pb and gb are on sphere. p is on sphere.
            # Moving in flat space and re-projecting is a valid Riemannian PSO approximation.
            
            # Weighted average of directions
            # We want to move p towards pb and gb.
            # Let's use a simple convex combination of the target points.
            # New target = w_cog * pb + w_soc * gb
            # Then move p towards New Target.
            
            # Actually, standard PSO velocity update:
            # v_new = w * v_old + c1 * r1 * (p_best - p) + c2 * r2 * (g_best - p)
            # Since we don't have v_old, we can treat the "step" as the velocity.
            # But we need to limit the step size.
            
            # Let's compute a raw step vector
            step = 0.5 * (pb - p) + 0.5 * (gb - p)
            step = step * (c1 * r1 * 0.5 + c2 * r2 * 0.5) # Scale by coefficients
            
            # Limit step size
            step_norm = np.linalg.norm(step, axis=1)
            max_allowed = max_speed
            scale = np.where(step_norm > max_allowed, max_allowed / np.maximum(step_norm, 1e-15), 1.0)
            step = step * scale[:, np.newaxis]
            
            # Update position
            new_p = p + step
            new_p = _project_to_sphere(new_p)
            
            # Evaluate
            e_new = _fast_log_energy(new_p)
            
            if e_new < personal_best_energies[i]:
                personal_best_energies[i] = e_new
                personal_bests[i] = new_p.copy()
                
                if e_new < global_best_energy:
                    global_best_energy = e_new
                    global_best = new_p.copy()
                    
                    if e_new < best_energy:
                        best_energy = e_new
                        best_points = new_p.copy()
            
            particles[i] = new_p
            
            # Periodic polish of global best to stay in basin
            if it % 10 == 0 and time.time() - start < time_limit - 5.0:
                polished = _lbfgs_polish(global_best, max_iter=50, ftol=1e-8, gtol=1e-6, time_limit=2.0)
                e_polished = _fast_log_energy(polished)
                if e_polished < global_best_energy:
                    global_best_energy = e_polished
                    global_best = polished.copy()
                    if e_polished < best_energy:
                        best_energy = e_polished
                        best_points = polished.copy()
                # Update particle 0 to the polished global best?
                # Or just let it converge there.
                particles[0] = global_best.copy()
                personal_bests[0] = global_best.copy()
                personal_best_energies[0] = global_best_energy

    return best_points, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initializations
    inits = [
        _fibonacci_sphere(N),
        _random_uniform_sphere(N, seed=1),
        _random_uniform_sphere(N, seed=2),
        _random_uniform_sphere(N, seed=3),
        _random_uniform_sphere(N, seed=42),
    ]
    
    best_points = inits[0].copy()
    best_energy = _log_energy(best_points)
    
    # First pass: moderate optimization on all initializations
    for idx, init in enumerate(inits):
        if time.time() > deadline - 80.0:
            break
        
        polished = _lbfgs_polish(init, max_iter=1000, ftol=1e-10, gtol=1e-8, time_limit=10.0)
        e_polish = _log_energy(polished)
        
        if e_polish < best_energy:
            best_energy = e_polish
            best_points = polished.copy()
    
    # Second pass: Particle Swarm Optimization
    if time.time() < deadline - 50.0:
        pso_result, pso_energy = _particle_swarm_optimization(best_points, max_iter=200, time_limit=deadline - time.time() - 40.0)
        # Verify with precise energy
        e_pso_precise = _log_energy(pso_result)
        if e_pso_precise < best_energy:
            best_energy = e_pso_precise
            best_points = pso_result.copy()
    
    # Third pass: final high-precision polish
    if time.time() < deadline - 10.0:
        final_polished = _lbfgs_polish(best_points, max_iter=5000, ftol=1e-14, gtol=1e-12, time_limit=deadline - time.time() - 5.0)
        e_final = _log_energy(final_polished)
        if e_final < best_energy:
            best_points = final_polished.copy()
    
    best_points = _project_to_sphere(best_points)
    
    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)
    
    # Check for points too close
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    iu = np.triu_indices(N, k=1)
    if np.min(dists_sq[iu]) < 1e-18:
        return _fibonacci_sphere(N)
    
    return best_points
