"""Physics-driven self-assembly for Dzugutov cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient


def _dzugutov_energy_and_gradient(positions, A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16):
    """Compute total energy and gradient for Dzugutov potential."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    energy = 0.0
    grad = np.zeros_like(positions)
    
    for i in range(n):
        for j in range(i + 1, n):
            r = dist[i, j]
            if r < 1e-10:
                continue
            
            # Energy contribution
            if r < a:
                v = A * (r ** (-m) - B) * np.exp(c / (r - a))
                # dv/dr
                dv_dr = A * (-m * r ** (-m - 1)) * np.exp(c / (r - a)) + A * (r ** (-m) - B) * np.exp(c / (r - a)) * (-c / (r - a) ** 2)
            elif r < b:
                v = B * np.exp(d / (r - b))
                dv_dr = B * np.exp(d / (r - b)) * (-d / (r - b) ** 2)
            else:
                v = 0.0
                dv_dr = 0.0
            
            energy += v
            
            # Gradient: force on i is -dv/dr * (r_i - r_j)/r, on j is dv/dr * (r_i - r_j)/r
            if r > 1e-10 and r < b:
                r_vec = positions[i] - positions[j]
                f = -dv_dr / r * r_vec
                grad[i] += f
                grad[j] -= f
    
    return energy, grad


def _langevin_step(positions, velocities, dt, temperature, damping, rng):
    """One Langevin dynamics step."""
    n = len(positions)
    energy, grad = _dzugutov_energy_and_gradient(positions)
    
    # F = -grad (force)
    forces = -grad
    
    # Langevin: dv = (F - gamma*v)*dt + sigma*sqrt(dt)*noise
    sigma = np.sqrt(2 * temperature * damping)
    noise = rng.normal(0, 1, size=positions.shape)
    
    new_velocities = velocities + (forces - damping * velocities) * dt + sigma * np.sqrt(dt) * noise
    new_positions = positions + new_velocities * dt
    
    # Enforce minimum distance constraint by pushing apart if too close
    for i in range(n):
        for j in range(i + 1, n):
            d = new_positions[i] - new_positions[j]
            dist = np.linalg.norm(d)
            if dist < 0.5 and dist > 1e-10:
                push = (0.5 - dist) / dist * d
                new_positions[i] += 0.5 * push
                new_positions[j] -= 0.5 * push
    
    return new_positions, new_velocities


def _self_assemble_cluster(n, seed, max_steps=5000, temperature_start=1.0, temperature_end=0.05, damping=1.0):
    """Build cluster by sequential addition with Langevin relaxation."""
    rng = np.random.default_rng(seed)
    
    # Start with 2 atoms
    positions = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
    velocities = np.zeros((2, 3))
    
    for target_n in range(3, n + 1):
        # Add a new atom at a random position near the cluster
        if target_n == 3:
            # Place third atom to form a triangle
            angle = rng.uniform(0, 2 * np.pi)
            new_atom = np.array([1.1 * np.cos(angle), 1.1 * np.sin(angle), 0.0])
        else:
            # Place near existing cluster
            center = np.mean(positions, axis=0)
            radius = 1.1 * target_n ** (1.0 / 3.0) * 0.8
            new_atom = center + rng.uniform(-radius, radius, 3)
            # Ensure minimum distance
            for _ in range(100):
                if all(np.linalg.norm(new_atom - p) >= 0.5 for p in positions):
                    break
                new_atom = center + rng.uniform(-radius, radius, 3)
        
        positions = np.vstack([positions, new_atom])
        velocities = np.vstack([velocities, np.zeros(3)])
        
        # Relax with Langevin dynamics, cooling temperature
        progress = (target_n - 2) / max(1, n - 2)
        temp = temperature_start * (1 - progress) + temperature_end * progress
        
        for step in range(max_steps // max(1, n - 2)):
            if step % 10 == 0:
                # Adaptive temperature based on energy
                e, _ = _dzugutov_energy_and_gradient(positions)
                if e > 0:
                    temp *= 0.99
                else:
                    temp = max(temperature_end, temp * 1.01)
            
            positions, velocities = _langevin_step(positions, velocities, 0.01, temp, damping, rng)
        
        # Final local relaxation
        energy, pos = relax(positions, 'dzugutov', None)
        positions = pos
    
    return positions


def _multi_start_langevin(n, seed, n_starts=6, max_steps=8000, temperature_start=0.8, temperature_end=0.02, damping=0.8):
    """Multi-start Langevin dynamics for full cluster."""
    rng = np.random.default_rng(seed)
    best_energy = np.inf
    best_positions = None
    
    for start in range(n_starts):
        # Random compact initial configuration
        radius = 1.1 * n ** (1.0 / 3.0) * 0.7
        positions = rng.uniform(-radius, radius, size=(n, 3))
        
        # Enforce minimum distance
        for i in range(n):
            for j in range(i + 1, n):
                for _ in range(50):
                    d = positions[i] - positions[j]
                    dist = np.linalg.norm(d)
                    if dist >= 0.5:
                        break
                    positions[i] += (0.5 - dist) / (dist + 1e-10) * d * 0.5
                    positions[j] -= (0.5 - dist) / (dist + 1e-10) * d * 0.5
        
        velocities = rng.normal(0, 0.1, size=(n, 3))
        
        # Langevin dynamics with cooling
        n_total_steps = max_steps // max(1, n_starts)
        for step in range(n_total_steps):
            progress = step / n_total_steps
            temp = temperature_start * (1 - progress) ** 2 + temperature_end * progress
            
            if step % 20 == 0:
                e, _ = _dzugutov_energy_and_gradient(positions)
                if e > 0.5 * (best_energy if best_energy < np.inf else 0):
                    temp *= 0.95
            
            positions, velocities = _langevin_step(positions, velocities, 0.008, temp, damping, rng)
        
        # Final local relaxation
        energy, pos = relax(positions, 'dzugutov', None)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = pos
    
    return best_positions


def _combined_search(n, seed, time_budget=45):
    """Combine self-assembly and multi-start Langevin."""
    import time
    start_time = time.time()
    
    best_energy = np.inf
    best_positions = None
    
    # Strategy 1: Self-assembly
    if time.time() - start_time < time_budget * 0.4:
        pos = _self_assemble_cluster(n, seed, max_steps=3000)
        energy, pos = relax(pos, 'dzugutov', None)
        if energy < best_energy:
            best_energy = energy
            best_positions = pos
    
    # Strategy 2: Multi-start Langevin
    remaining_time = time_budget - (time.time() - start_time)
    if remaining_time > 5:
        n_starts = max(2, int(remaining_time / 8))
        pos = _multi_start_langevin(n, seed + 1000, n_starts=n_starts, 
                                     max_steps=int(remaining_time * 150),
                                     temperature_start=0.6, temperature_end=0.01)
        if pos is not None:
            energy, pos = relax(pos, 'dzugutov', None)
            if energy < best_energy:
                best_energy = energy
                best_positions = pos
    
    # Final polish with additional local relaxation attempts
    if best_positions is not None and time.time() - start_time < time_budget - 2:
        for _ in range(3):
            pos = best_positions + np.random.default_rng(seed + 9999).normal(0, 0.05, size=best_positions.shape)
            # Enforce min distance
            for i in range(len(pos)):
                for j in range(i + 1, len(pos)):
                    d = pos[i] - pos[j]
                    dist = np.linalg.norm(d)
                    if dist < 0.5 and dist > 1e-10:
                        push = (0.5 - dist) / dist * d
                        pos[i] += 0.5 * push
                        pos[j] -= 0.5 * push
            energy, pos = relax(pos, 'dzugutov', None)
            if energy < best_energy:
                best_energy = energy
                best_positions = pos
    
    return best_positions


def construct(instance):
    """Physics-driven self-assembly for Dzugutov cluster global minima."""
    size = instance["n"]
    potential = instance["potential"]
    
    if potential != 'dzugutov':
        # Fallback for other potentials
        spacing = {"morse": 1.0, "sutton_chen": 0.72}.get(potential, 1.0)
        rng = np.random.default_rng(size)
        radius = spacing * size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                points.append(p)
        energy, positions = relax(np.array(points), potential, instance.get("rho", instance.get("kind")))
        return positions
    
    # For Dzugutov, use physics-driven self-assembly
    seed = size * 7919  # Deterministic seed per cluster size
    positions = _combined_search(size, seed, time_budget=45)
    
    if positions is None:
        # Fallback
        rng = np.random.default_rng(size)
        radius = 1.1 * size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        energy, positions = relax(np.array(points), 'dzugutov', None)
    
    return positions
# EVOLVE-BLOCK-END
