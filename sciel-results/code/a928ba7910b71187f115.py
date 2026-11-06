"""Morse cluster global minima: gradient-driven particle swarm optimization with lattice seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 285.0  # 15s safety margin before 300s kill
_instance_call_count = [0]


def _time_left():
    return _HARD_DEADLINE - time.time()


def _morse_energy_positions(pos, rho):
    """Compute Morse energy for positions."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, 1e10)
    mask = dist > 0.5
    dist_masked = np.where(mask, dist, 1e10)
    x = rho * (1.0 - dist_masked)
    e = np.exp(x)
    pair_e = e * (e - 2.0)
    iu = np.triu_indices(n, k=1)
    total = np.sum(pair_e[iu[0], iu[1]])
    return total


def _morse_grad_positions(pos, rho):
    """Compute Morse gradient for positions (vectorized)."""
    n = len(pos)
    if n < 2:
        return np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, 1e10)
    np.fill_diagonal(dist_sq, 1e10)
    mask = dist > 0.5
    dist_masked = np.where(mask, dist, 1e10)
    x = rho * (1.0 - dist_masked)
    e = np.exp(x)
    de_dr = 2.0 * rho * e * (1.0 - e)
    # unit_vec[i,j] = diff[i,j] / dist[i,j]
    inv_dist = np.where(mask, 1.0 / dist_masked, 0.0)
    unit_vec = diff * inv_dist[:, :, np.newaxis]
    np.fill_diagonal(unit_vec, 0.0)
    # grad[i] = sum_j de_dr[i,j] * unit_vec[i,j]
    grad = np.sum(de_dr[:, :, np.newaxis] * unit_vec, axis=1)
    return grad


def _fcc_points(n, spacing=1.0):
    """Generate n points from an FCC lattice centered at origin."""
    a = spacing * 0.7071
    r_est = spacing * n ** (1.0 / 3.0) * 1.2
    points = []
    extent = int(np.ceil(r_est / a)) + 2
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                p = np.array([i, j, k]) * a * np.sqrt(2)
                if np.linalg.norm(p) <= r_est:
                    points.append(p)
                if i < extent and j < extent:
                    p2 = np.array([i + 0.5, j + 0.5, k]) * a * np.sqrt(2)
                    if np.linalg.norm(p2) <= r_est:
                        points.append(p2)
                if i < extent and k < extent:
                    p3 = np.array([i + 0.5, j, k + 0.5]) * a * np.sqrt(2)
                    if np.linalg.norm(p3) <= r_est:
                        points.append(p3)
                if j < extent and k < extent:
                    p4 = np.array([i, j + 0.5, k + 0.5]) * a * np.sqrt(2)
                    if np.linalg.norm(p4) <= r_est:
                        points.append(p4)
    if len(points) == 0:
        rng = np.random.default_rng(42)
        return rng.uniform(-r_est, r_est, (n, 3))
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx]
    if len(points) >= n:
        return points[:n]
    else:
        rng = np.random.default_rng(42)
        extra = rng.uniform(-r_est, r_est, (n - len(points), 3))
        return np.vstack([points, extra])


def _bcc_points(n, spacing=1.0):
    """Generate n points from a BCC lattice centered at origin."""
    a = 2.0 * spacing / np.sqrt(3.0)
    r_est = spacing * (n ** (1.0 / 3.0) + 2) * 0.8
    points = []
    extent = int(np.ceil(r_est / a)) + 2
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                p = np.array([i, j, k]) * a
                if np.linalg.norm(p) <= r_est:
                    points.append(p)
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5]) * a
                if np.linalg.norm(p2) <= r_est:
                    points.append(p2)
    if len(points) == 0:
        rng = np.random.default_rng(42)
        return rng.uniform(-r_est, r_est, (n, 3))
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx]
    if len(points) >= n:
        return points[:n]
    else:
        rng = np.random.default_rng(42)
        extra = rng.uniform(-r_est, r_est, (n - len(points), 3))
        return np.vstack([points, extra])


def _sc_points(n, spacing=1.0):
    """Generate n points from a simple cubic lattice centered at origin."""
    a = spacing
    r_est = spacing * (n ** (1.0 / 3.0) + 2) * 0.8
    points = []
    extent = int(np.ceil(r_est / a)) + 2
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                p = np.array([i, j, k]) * a
                if np.linalg.norm(p) <= r_est:
                    points.append(p)
    if len(points) == 0:
        rng = np.random.default_rng(42)
        return rng.uniform(-r_est, r_est, (n, 3))
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx]
    if len(points) >= n:
        return points[:n]
    else:
        rng = np.random.default_rng(42)
        extra = rng.uniform(-r_est, r_est, (n - len(points), 3))
        return np.vstack([points, extra])


def _random_compact(n, spacing=1.0, seed=0):
    """Generate n random points in a compact sphere with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    min_dist = 0.55 * spacing
    points = []
    attempts = 0
    max_attempts = n * 200
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    ok = False
                    break
            if ok:
                points.append(p)
        attempts += 1
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)


def _gradient_descent_step(pos, grad, lr=0.01, max_steps=50):
    """Perform a few gradient descent steps with adaptive learning rate."""
    current = pos.copy()
    current_grad = grad.copy()
    for step in range(max_steps):
        if _time_left() < 1.0:
            break
        step_size = lr / (1.0 + 0.1 * step)
        current = current - step_size * current_grad
        # Clamp to avoid extreme moves
        current = np.clip(current, -20.0, 20.0)
        current_grad = _morse_grad_positions(current, rho_global)
        gnorm = np.linalg.norm(current_grad)
        if gnorm < 1e-8:
            break
    return current


def _pso_optimize(init_positions, rho, n_particles, max_iter, time_budget):
    """
    Gradient-informed Particle Swarm Optimization.
    
    Each particle has a position (n,3), velocity (n,3), personal best, and the swarm
    maintains a global best. Velocity update combines:
      - inertia
      - cognitive term (toward personal best)
      - social term (toward global best)
      - gradient guidance (smooth direction from analytical gradient)
    
    This replaces discrete random kicks with continuous, gradient-informed motion.
    """
    global rho_global
    rho_global = rho
    
    n_atoms = init_positions.shape[0]
    rng = np.random.default_rng(12345)
    
    # Initialize particles: first from init, rest from perturbed versions
    particles = []
    for i in range(n_particles):
        if i == 0:
            p_pos = init_positions.copy()
        else:
            # Perturb the initial configuration with increasing noise
            noise_scale = 0.1 + 0.15 * i
            p_pos = init_positions + rng.uniform(-noise_scale, noise_scale, init_positions.shape)
        particles.append(p_pos)
    
    # Relax each particle to a local minimum
    particle_energies = []
    for i in range(n_particles):
        if _time_left() < 5.0:
            break
        e, relaxed = relax(particles[i], 'morse', rho)
        particles[i] = relaxed
        particle_energies.append(e)
    
    # Set up PSO state
    personal_best_pos = [p.copy() for p in particles]
    personal_best_energy = list(particle_energies)
    global_best_idx = np.argmin(personal_best_energy)
    global_best_pos = particles[global_best_idx].copy()
    global_best_energy = personal_best_energy[global_best_idx]
    
    # Velocities initialized to zero
    velocities = [np.zeros_like(p) for p in particles]
    
    # PSO parameters
    w = 0.9       # inertia
    c1 = 1.5      # cognitive coefficient
    c2 = 1.5      # social coefficient
    grad_weight = 0.05  # gradient guidance weight
    max_speed = 0.3     # velocity clamp
    
    for it in range(max_iter):
        if _time_left() < 3.0 or (time.time() - _START_TIME) > time_budget:
            break
        
        for i in range(n_particles):
            if _time_left() < 2.0:
                break
            
            # Compute gradient at current position for guidance
            grad = _morse_grad_positions(particles[i], rho)
            gnorm = np.linalg.norm(grad)
            if gnorm > 1e-10:
                grad_dir = -grad / gnorm  # direction of steepest descent
            else:
                grad_dir = np.zeros_like(particles[i])
            
            # Random coefficients for stochasticity
            r1 = rng.random(particles[i].shape)
            r2 = rng.random(particles[i].shape)
            
            # Velocity update
            vel = (w * velocities[i]
                   + c1 * r1 * (personal_best_pos[i] - particles[i])
                   + c2 * r2 * (global_best_pos - particles[i])
                   + grad_weight * grad_dir * np.ones(particles[i].shape))
            
            # Clamp velocity
            vnorm = np.linalg.norm(vel)
            if vnorm > max_speed:
                vel = vel * (max_speed / vnorm)
            
            velocities[i] = vel
            
            # Update position
            particles[i] = particles[i] + vel
            
            # Relax to local minimum
            e, particles[i] = relax(particles[i], 'morse', rho)
            
            # Update personal best
            if e < personal_best_energy[i]:
                personal_best_energy[i] = e
                personal_best_pos[i] = particles[i].copy()
            
            # Update global best
            if e < global_best_energy:
                global_best_energy = e
                global_best_pos = particles[i].copy()
                global_best_idx = i
        
        # Adaptive: reduce exploration over time
        w = max(0.5, w * 0.995)
        c1 = max(0.8, c1 * 0.998)
        c2 = min(2.5, c2 * 1.002)
    
    return global_best_energy, global_best_pos


def _enforce_min_dist(pos, min_d=0.5, iterations=20):
    """Push apart any atoms closer than min_d."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(iterations):
        moved = False
        for a in range(n):
            for b in range(a + 1, n):
                d = np.linalg.norm(pos[a] - pos[b])
                if d < min_d:
                    if d < 1e-10:
                        direction = np.array([1.0, 0.0, 0.0])
                    else:
                        direction = (pos[a] - pos[b]) / d
                    shift = (min_d - d) / 2.0 + 0.005
                    pos[a] += direction * shift
                    pos[b] -= direction * shift
                    moved = True
        if not moved:
            break
    return pos


def construct(instance):
    """Find low-energy Morse cluster configurations via gradient-informed PSO with diverse lattice seeds."""
    _instance_call_count[0] += 1
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Per-instance time budget
    per_instance_budget = 45.0
    if _instance_call_count[0] <= 2:
        per_instance_budget = 50.0
    elif _instance_call_count[0] >= 5:
        per_instance_budget = 40.0
    
    instance_deadline = time.time() + per_instance_budget
    
    # Generate multiple starting configurations from different lattices
    generators = [
        lambda: _fcc_points(size, 1.0),
        lambda: _bcc_points(size, 1.0),
        lambda: _sc_points(size, 1.0),
        lambda: _random_compact(size, 1.0, seed=42),
        lambda: _random_compact(size, 1.0, seed=123),
        lambda: _random_compact(size, 1.0, seed=456),
    ]
    
    best_energy = np.inf
    best_pos = None
    
    # PSO parameters: fewer particles but more iterations, gradient-guided
    n_particles = 5 if size > 50 else 7
    pso_max_iter = 25 if size > 50 else 35
    
    for i, gen in enumerate(generators):
        if time.time() > instance_deadline or _time_left() < 5.0:
            break
        
        try:
            init_pos = gen()
            init_pos = _enforce_min_dist(init_pos)
            
            # Relax to get a local minimum as starting point for PSO
            energy, pos = relax(init_pos, potential, parameter)
            
            # Run PSO from this local minimum
            pso_energy, pso_pos = _pso_optimize(
                pos, parameter,
                n_particles=n_particles,
                max_iter=pso_max_iter,
                time_budget=per_instance_budget
            )
            
            if pso_energy < best_energy:
                best_energy = pso_energy
                best_pos = pso_pos.copy()
                
        except Exception as e:
            continue
    
    # If we still don't have a result, do a final simple relaxation
    if best_pos is None:
        init_pos = _random_compact(size, 1.0, seed=0)
        best_pos = _enforce_min_dist(init_pos)
        best_energy, best_pos = relax(best_pos, potential, parameter)
    
    # Final validation: ensure minimum distance
    best_pos = _enforce_min_dist(best_pos)
    
    return best_pos
# EVOLVE-BLOCK-END
