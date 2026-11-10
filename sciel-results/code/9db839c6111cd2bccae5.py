"""Morse cluster global minima via stochastic collision dynamics (event-driven MD)."""
import numpy as np
from scipy.spatial import cKDTree
import time

def _morse_pair_energies_and_forces(pos, rho):
    """Compute total Morse energy and forces for all pairs.
    
    E_ij = exp(rho*(1-r_ij)) * (exp(rho*(1-r_ij)) - 2)
    dE/dr = rho * exp(rho*(1-r_ij)) * (exp(rho*(1-r_ij)) - 1)  (force on i is -dE/dr * r_hat)
    """
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    
    tree = cKDTree(pos)
    # Use a cutoff: for large rho, interactions decay fast
    # exp(rho*(1-r)) decays as r increases. For r > 1 + log(2)/rho + some margin, energy ~ 0
    # Let's use a generous cutoff
    cutoff = 1.0 + np.log(1000.0) / rho + 0.5  # beyond this, energy < ~0.001 per pair
    
    pairs = tree.query_pairs(cutoff, output_type='ndarray')
    
    total_energy = 0.0
    forces = np.zeros_like(pos)
    
    if len(pairs) == 0:
        return 0.0, forces
    
    for i, j in pairs:
        rij = pos[i] - pos[j]
        r = np.linalg.norm(rij)
        if r < 1e-10:
            continue
        exp_term = np.exp(rho * (1.0 - r))
        e_ij = exp_term * (exp_term - 2.0)
        total_energy += e_ij
        # Force magnitude: dE/dr = rho * exp_term * (exp_term - 1)
        # Force on i is -dE/dr * (rij / r)
        f_mag = rho * exp_term * (exp_term - 1.0)
        f_vec = f_mag * rij / r
        forces[i] += f_vec
        forces[j] -= f_vec
    
    return total_energy, forces


def _event_driven_md(pos, rho, n_steps=2000, dt=0.01, cooling_rate=0.999, min_dist=0.5):
    """Run event-driven MD with velocity Verlet and thermostatting.
    
    Starts from given positions with random velocities, cools down to find low-energy states.
    """
    n = len(pos)
    pos = pos.copy().astype(float)
    
    # Initialize with small random velocities
    rng = np.random.default_rng(hash(tuple(pos.ravel())) % 2**32)
    vel = rng.normal(0.0, 0.1, size=(n, 3))
    
    # Compute initial energy and forces
    energy, forces = _morse_pair_energies_and_forces(pos, rho)
    
    best_energy = energy
    best_pos = pos.copy()
    
    for step in range(n_steps):
        # Velocity Verlet: half-step velocity
        vel_half = vel + 0.5 * dt * forces
        pos_new = pos + dt * vel_half
        
        # Check minimum distance constraint; if violated, push apart
        for i in range(n):
            for j in range(i + 1, n):
                d = pos_new[i] - pos_new[j]
                dist = np.linalg.norm(d)
                if dist < min_dist:
                    if dist < 1e-10:
                        # Randomly separate
                        d = rng.normal(0, 1, 3)
                        dist = np.linalg.norm(d)
                    # Push apart to maintain min_dist
                    correction = (min_dist - dist) * d / dist * 0.5
                    pos_new[i] += correction
                    pos_new[j] -= correction
        
        # Compute new forces
        energy_new, forces_new = _morse_pair_energies_and_forces(pos_new, rho)
        
        # Second half-step velocity
        vel_new = vel_half + 0.5 * dt * forces_new
        
        # Thermostat: scale velocities to maintain target temperature
        # Temperature decreases over time
        target_temp = 0.5 * cooling_rate ** step
        kinetic = 0.5 * np.sum(vel_new ** 2)
        if kinetic > 1e-10:
            # Scale to target kinetic energy (3/2 * n * T)
            target_kinetic = 1.5 * n * target_temp
            scale = np.sqrt(target_kinetic / kinetic) if kinetic > 0 else 1.0
            vel_new *= scale
        
        pos = pos_new
        vel = vel_new
        energy = energy_new
        forces = forces_new
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
    
    return best_pos, best_energy


def _random_compact_init(n, spacing=1.0, seed=42):
    """Generate a compact random initial configuration."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 * spacing for q in points):
                points.append(p)
        attempts += 1
    
    # If we didn't get enough points, fill in randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])


def construct(instance):
    """Find low-energy Morse cluster configuration using stochastic MD."""
    n = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 6))
    
    if potential != "morse":
        # Fallback for non-Morse (shouldn't happen per task spec)
        return _random_compact_init(n, seed=n)
    
    start_time = time.time()
    time_budget = 300.0 / 6  # ~50s per instance
    
    best_energy = np.inf
    best_pos = None
    
    # Multiple restarts with different initial conditions
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() - start_time > time_budget * 0.8:
            break
        
        # Generate initial configuration
        if restart == 0:
            # Start from a somewhat ordered structure (hexagonal-ish layers)
            pos = _hexagonal_init(n, seed=restart)
        else:
            pos = _random_compact_init(n, spacing=1.0, seed=restart * 1000 + 7)
        
        # Run MD with different parameters for diversity
        if restart < 3:
            # Longer, slower cooling
            pos_md, energy_md = _event_driven_md(
                pos, rho, n_steps=3000, dt=0.008, cooling_rate=0.9995
            )
        elif restart < 6:
            # Medium
            pos_md, energy_md = _event_driven_md(
                pos, rho, n_steps=2000, dt=0.012, cooling_rate=0.999
            )
        else:
            # Shorter, faster cooling for diversity
            pos_md, energy_md = _event_driven_md(
                pos, rho, n_steps=1500, dt=0.015, cooling_rate=0.9985
            )
        
        if energy_md < best_energy:
            best_energy = energy_md
            best_pos = pos_md.copy()
    
    if best_pos is None:
        best_pos = _random_compact_init(n, seed=n)
    
    return best_pos


def _hexagonal_init(n, seed=0):
    """Initialize atoms in a hexagonal close-packed arrangement."""
    rng = np.random.default_rng(seed)
    points = []
    a = 1.0  # lattice parameter
    dz = a * np.sqrt(2.0 / 3.0)
    
    layer = 0
    while len(points) < n:
        # Hexagonal layer
        r_max = int(np.ceil(np.sqrt(n) / 2)) + 1
        for i in range(-r_max, r_max + 1):
            for j in range(-r_max, r_max + 1):
                if len(points) >= n:
                    break
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0) / 2.0 * j
                z = layer * dz
                p = np.array([x, y, z])
                # Small random perturbation
                p += rng.normal(0, 0.01, 3)
                points.append(p)
            if len(points) >= n:
                break
        layer += 1
        if layer > 20:
            break
    
    # If still not enough, add random points
    while len(points) < n:
        p = rng.uniform(-3, 3, 3)
        points.append(p)
    
    return np.array(points[:n])
