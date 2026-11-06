"""Morse cluster global minima: simulated annealing with Langevin dynamics."""
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
    # Only compute for pairs with distance > 0 (avoid diagonal)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dist[mask]
    # Morse potential: E = exp(rho*(1-r))*(exp(rho*(1-r))-2)
    x = rho * (1.0 - d)
    e = np.exp(x)
    energy = np.sum(e * (e - 2.0))
    return float(energy)


def _morse_gradient(pos, rho):
    """Compute Morse gradient for positions."""
    n = len(pos)
    if n < 2:
        return np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, 1e10)
    
    # Gradient of Morse potential w.r.t. positions
    # dE/dr = rho * exp(rho*(1-r)) * (2*exp(rho*(1-r)) - 1)
    x = rho * (1.0 - dist)
    e = np.exp(x)
    dEdr = rho * e * (2.0 * e - 1.0)
    
    # Gradient w.r.t. each coordinate: dE/dx_i = sum_j dEdr_ij * (x_i - x_j)/r_ij
    # For each pair i<j, force on i is +dEdr_ij * (r_i - r_j)/r_ij
    # and force on j is -dEdr_ij * (r_i - r_j)/r_ij
    grad = np.zeros_like(pos)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    for i in range(n):
        for j in range(i + 1, n):
            r = dist[i, j]
            if r < 1e-10:
                continue
            force = dEdr[i, j] / r
            grad[i] += force * diff[i, j]
            grad[j] -= force * diff[i, j]
    return grad


def _generate_lattice_seed(n, rng):
    """Generate a compact lattice-like seed."""
    # Try to pack atoms in a roughly cubic arrangement
    side = int(np.ceil(n ** (1.0 / 3.0)))
    points = []
    for x in range(side):
        for y in range(side):
            for z in range(side):
                if len(points) >= n:
                    break
                points.append([x * 1.0, y * 1.0, z * 1.0])
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    
    # If we didn't get enough, add random points
    while len(points) < n:
        p = rng.uniform(-side * 0.5, side * 0.5, 3)
        points.append(p)
    
    points = np.array(points[:n], dtype=float)
    # Center the cluster
    points -= np.mean(points, axis=0)
    # Add small random perturbation to break symmetry
    points += rng.normal(0, 0.05, points.shape)
    return points


def _simulated_annealing(pos, rho, n, max_time):
    """Run simulated annealing with Langevin dynamics."""
    rng = np.random.default_rng(int(time.time() * 1000) % 2**32)
    
    current_energy = _morse_energy(pos, rho)
    best_pos = pos.copy()
    best_energy = current_energy
    
    # Annealing schedule
    T_start = 5.0
    T_end = 0.001
    n_steps = int(max_time * 500)  # ~500 steps per second
    n_steps = max(n_steps, 100)
    
    # Linear cooling schedule
    temperatures = np.linspace(T_start, T_end, n_steps)
    
    # Langevin parameters
    dt = 0.01
    mass = 1.0
    
    start_time = time.time()
    for step in range(n_steps):
        if time.time() - start_time > max_time - 5:
            break
        
        T = temperatures[min(step, n_steps - 1)]
        
        # Compute gradient
        grad = _morse_gradient(pos, rho)
        
        # Langevin update: x_new = x + (dt/m)*F + sqrt(2*T*dt/m)*noise
        force = -grad
        noise = rng.normal(0, 1, pos.shape)
        pos_new = pos + (dt / mass) * force + np.sqrt(2 * T * dt / mass) * noise
        
        # Compute new energy
        new_energy = _morse_energy(pos_new, rho)
        
        # Metropolis acceptance
        if new_energy < current_energy:
            pos = pos_new
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = pos_new.copy()
        else:
            delta_e = new_energy - current_energy
            if delta_e < 1e-10 or rng.random() < np.exp(-delta_e / max(T, 1e-10)):
                pos = pos_new
                current_energy = new_energy
    
    # Final local relaxation
    energy, positions = relax(best_pos, 'morse', rho)
    return energy, positions


def construct(instance):
    """Find low-energy Morse cluster configurations using simulated annealing."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    if potential != 'morse':
        # Fallback for non-Morse potentials
        rng = np.random.default_rng(size)
        radius = 1.0 * size ** (1.0 / 3.0) * 0.75
        best = None
        for _ in range(4):
            points = []
            while len(points) < size:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 for q in points):
                    points.append(p)
            energy, positions = relax(np.array(points), potential, parameter)
            if best is None or energy < best[0]:
                best = (energy, positions)
        return best[1]
    
    # Time budget for this instance
    time_budget = min(40.0, max(5.0, 280.0 / 6.0))  # ~40s per instance, at least 5s
    
    best_energy = np.inf
    best_pos = None
    
    # Multiple restarts with different seeds
    n_restarts = max(1, int(time_budget / 10.0))
    
    for restart in range(n_restarts):
        if _time_left() < 10:
            break
        
        rng = np.random.default_rng(size * 1000 + restart)
        
        # Generate seed
        if restart == 0:
            # Lattice seed
            pos = _generate_lattice_seed(size, rng)
        elif restart == 1:
            # Random compact
            radius = 1.0 * size ** (1.0 / 3.0) * 0.75
            points = []
            while len(points) < size:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 for q in points):
                    points.append(p)
            pos = np.array(points)
        else:
            # Perturbed lattice
            pos = _generate_lattice_seed(size, rng)
            pos += rng.normal(0, 0.2, pos.shape)
        
        # Run simulated annealing
        sa_time = time_budget / (n_restarts - restart) - 2
        sa_time = max(2.0, min(sa_time, 30.0))
        
        energy, positions = _simulated_annealing(pos, parameter, size, sa_time)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = positions
    
    # Final check: ensure minimum distance constraint
    if best_pos is None:
        # Fallback: return relaxed lattice
        rng = np.random.default_rng(size)
        pos = _generate_lattice_seed(size, rng)
        _, best_pos = relax(pos, 'morse', parameter)
    
    # Verify and fix any close pairs
    n = len(best_pos)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(best_pos[i] - best_pos[j])
            if d < 0.5:
                # Push apart
                if d < 1e-10:
                    best_pos[j] += 0.51 * np.array([1, 0, 0])
                else:
                    direction = (best_pos[i] - best_pos[j]) / d
                    best_pos[i] += direction * (0.51 - d) / 2
                    best_pos[j] -= direction * (0.51 - d) / 2
    
    return best_pos
# EVOLVE-BLOCK-END
