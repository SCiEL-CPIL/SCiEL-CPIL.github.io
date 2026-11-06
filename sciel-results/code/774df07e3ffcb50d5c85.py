"""Morse cluster global minima: PSO with gradient-driven velocity and lattice seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 285.0  # 15s safety margin before 300s kill


def _time_left():
    return _HARD_DEADLINE - time.time()


def _morse_energy(pos, rho):
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
    exp_x = np.exp(x)
    pair_energy = exp_x * (exp_x - 2.0)
    # Zero out diagonal and invalid pairs
    pair_energy[~mask] = 0.0
    return float(np.sum(pair_energy) / 2.0)


def _morse_gradient(pos, rho):
    """Compute Morse energy gradient for positions."""
    n = len(pos)
    if n < 2:
        return np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, 1e10)
    mask = dist > 0.5
    dist_masked = np.where(mask, dist, 1e10)
    
    # dE/dr = rho * exp(2x) * (2*exp(x) - 1) where x = rho*(1-r)
    # Actually, E = exp(x)(exp(x)-2) = exp(2x) - 2exp(x)
    # dE/dx = 2exp(2x) - 2exp(x) = 2exp(x)(exp(x)-1)
    # dx/dr = -rho
    # dE/dr = -rho * 2exp(x)(exp(x)-1)
    
    x = rho * (1.0 - dist_masked)
    exp_x = np.exp(x)
    dE_dr = -rho * 2.0 * exp_x * (exp_x - 1.0)
    
    # Gradient: dE/dpos_i = sum_j dE_dr_ij * (pos_i - pos_j) / r_ij
    # Use broadcasting
    inv_dist = np.where(mask, 1.0 / dist_masked, 0.0)
    # dE_dr shape (n,n), inv_dist shape (n,n)
    # grad[i] = sum_j dE_dr[i,j] * inv_dist[i,j] * (pos[i] - pos[j])
    # diff[i,j] = pos[i] - pos[j]
    factor = dE_dr * inv_dist  # shape (n,n)
    factor[~mask] = 0.0
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    return grad


def _fcc_points(n):
    """Generate FCC-like compact cluster points."""
    # Generate points in a cubic grid with FCC offset, then take n closest to center
    s = int(np.ceil((2 * n) ** (1.0 / 3.0)))
    coords = []
    for i in range(-s, s + 1):
        for j in range(-s, s + 1):
            for k in range(-s, s + 1):
                # FCC: simple cubic + face-centered
                coords.append([i, j, k])
                coords.append([i + 0.5, j + 0.5, k])
                coords.append([i + 0.5, j, k + 0.5])
                coords.append([i, j + 0.5, k + 0.5])
    coords = np.array(coords)
    # Center
    coords -= coords.mean(axis=0)
    # Sort by distance to center
    dists = np.linalg.norm(coords, axis=1)
    idx = np.argsort(dists)[:n]
    return coords[idx]


def _bcc_points(n):
    """Generate BCC-like compact cluster points."""
    s = int(np.ceil((2 * n) ** (1.0 / 3.0)))
    coords = []
    for i in range(-s, s + 1):
        for j in range(-s, s + 1):
            for k in range(-s, s + 1):
                coords.append([i, j, k])
                coords.append([i + 0.5, j + 0.5, k + 0.5])
    coords = np.array(coords)
    coords -= coords.mean(axis=0)
    dists = np.linalg.norm(coords, axis=1)
    idx = np.argsort(dists)[:n]
    return coords[idx]


def _icosahedral_points(n):
    """Generate icosahedral shell points."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    verts /= np.linalg.norm(verts[0])
    
    points = [np.zeros(3)]  # center
    # First shell
    for v in verts:
        points.append(v.copy())
    # Second shell (midpoints of edges)
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            mid = (verts[i] + verts[j]) / 2
            mid = 2.0 * mid / np.linalg.norm(mid)
            points.append(mid)
    
    points = np.array(points)
    if len(points) >= n:
        # Sort by radius, take n closest
        radii = np.linalg.norm(points, axis=1)
        idx = np.argsort(radii)[:n]
        return points[idx]
    else:
        # Pad with random
        rng = np.random.default_rng(42)
        while len(points) < n:
            p = rng.normal(0, 1.5, 3)
            points = np.vstack([points, [p]])
        return points[:n]


def _random_compact(n, rng):
    """Generate random compact cluster."""
    radius = n ** (1.0 / 3.0) * 0.8
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.55 for q in points):
            points.append(p)
    return np.array(points)


def _pso_optimize(initial_positions, rho, n_particles=6, max_iter=50, time_budget=None):
    """Particle swarm optimization with gradient-driven velocity."""
    n_atoms = len(initial_positions)
    
    # Initialize particles
    particles = np.zeros((n_particles, n_atoms, 3))
    velocities = np.zeros((n_particles, n_atoms, 3))
    energies = np.zeros(n_particles)
    best_positions = np.zeros((n_particles, n_atoms, 3))
    best_energies = np.full(n_particles, np.inf)
    
    # First particle is the initial position
    particles[0] = initial_positions
    # Other particles are perturbed versions
    for i in range(1, n_particles):
        particles[i] = initial_positions + np.random.normal(0, 0.3, (n_atoms, 3))
    
    # Evaluate initial energies
    for i in range(n_particles):
        energies[i] = _morse_energy(particles[i], rho)
        best_positions[i] = particles[i].copy()
        best_energies[i] = energies[i]
    
    global_best_idx = np.argmin(best_energies)
    global_best_energy = best_energies[global_best_idx]
    global_best_pos = best_positions[global_best_idx].copy()
    
    inertia_w = 0.7
    cognitive_c1 = 1.5
    social_c2 = 1.5
    max_vel = 0.5
    
    for it in range(max_iter):
        if time_budget is not None and _time_left() < time_budget:
            break
        
        for i in range(n_particles):
            # Update velocity using PSO formula + gradient guidance
            r1 = np.random.rand(n_atoms, 3)
            r2 = np.random.rand(n_atoms, 3)
            
            vel_update = (inertia_w * velocities[i] +
                         cognitive_c1 * r1 * (best_positions[i] - particles[i]) +
                         social_c2 * r2 * (global_best_pos - particles[i]))
            
            # Add gradient-based nudge
            grad = _morse_gradient(particles[i], rho)
            vel_update -= 0.1 * grad
            
            velocities[i] = np.clip(vel_update, -max_vel, max_vel)
            
            # Update position
            particles[i] += velocities[i]
            
            # Compute new energy
            new_energy = _morse_energy(particles[i], rho)
            
            if new_energy < best_energies[i]:
                best_energies[i] = new_energy
                best_positions[i] = particles[i].copy()
            
            if new_energy < global_best_energy:
                global_best_energy = new_energy
                global_best_pos = particles[i].copy()
        
        # Adaptive inertia
        inertia_w = max(0.4, 0.9 - 0.5 * it / max_iter)
    
    return global_best_pos, global_best_energy


def construct(instance):
    """PSO-based optimization with diverse lattice seeds."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(size)
    
    # Generate diverse initial seeds
    seeds = []
    
    # FCC seed
    try:
        fcc = _fcc_points(size)
        # Scale to reasonable size
        fcc *= 1.0
        seeds.append(fcc)
    except:
        pass
    
    # BCC seed
    try:
        bcc = _bcc_points(size)
        bcc *= 1.0
        seeds.append(bcc)
    except:
        pass
    
    # Icosahedral seed
    try:
        ico = _icosahedral_points(size)
        seeds.append(ico)
    except:
        pass
    
    # Random compact seeds
    for _ in range(3):
        try:
            rand = _random_compact(size, rng)
            seeds.append(rand)
        except:
            pass
    
    # Filter valid seeds (all pairs >= 0.5 apart)
    valid_seeds = []
    for seed in seeds:
        if len(seed) != size:
            continue
        # Check minimum distance
        diff = seed[:, np.newaxis, :] - seed[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        if np.min(dist) >= 0.5:
            valid_seeds.append(seed)
    
    if not valid_seeds:
        # Fallback: simple random
        radius = size ** (1.0 / 3.0) * 0.8
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
        valid_seeds.append(np.array(points))
    
    best_positions = None
    best_energy = np.inf
    
    # For each seed, run PSO
    for seed in valid_seeds:
        if _time_left() < 10.0:
            break
        
        # First relax locally
        try:
            relaxed_energy, relaxed_pos = relax(seed.copy(), potential, parameter)
            if relaxed_energy < best_energy:
                best_energy = relaxed_energy
                best_positions = relaxed_pos.copy()
        except:
            pass
        
        # Then run PSO from the relaxed position
        if _time_left() < 5.0:
            break
        
        try:
            pso_pos, pso_energy = _pso_optimize(
                relaxed_pos if 'relaxed_pos' in dir() else seed,
                parameter,
                n_particles=4,
                max_iter=30,
                time_budget=5.0
            )
            if pso_energy < best_energy:
                best_energy = pso_energy
                best_positions = pso_pos.copy()
        except:
            pass
    
    if best_positions is None:
        # Final fallback
        best_positions = valid_seeds[0]
    
    return best_positions
# EVOLVE-BLOCK-END
