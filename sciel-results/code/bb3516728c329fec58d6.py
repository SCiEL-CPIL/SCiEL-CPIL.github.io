"""Physics-driven self-assembly for Dzugutov cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy for given positions."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-10:
                continue
            if r < a:
                energy += A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                energy += B * np.exp(d / (r - b))
    return energy

def _dzugutov_gradient(positions):
    """Compute Dzugutov potential gradient for given positions."""
    n = len(positions)
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            r = np.linalg.norm(rij)
            if r < 1e-10:
                continue
            if r < a:
                # V(r) = A*(r^-m - B)*exp(c/(r-a))
                # dV/dr = A*[(-m*r^(-m-1))*exp(c/(r-a)) + (r^-m - B)*exp(c/(r-a))*(-c/(r-a)^2)]
                dVdr = A * (-m * r ** (-m - 1)) * np.exp(c / (r - a)) + \
                       A * (r ** (-m) - B) * np.exp(c / (r - a)) * (-c / (r - a) ** 2)
            elif r < b:
                # V(r) = B*exp(d/(r-b))
                # dV/dr = B*exp(d/(r-b))*(-d/(r-b)^2)
                dVdr = B * np.exp(d / (r - b)) * (-d / (r - b) ** 2)
            else:
                dVdr = 0.0
            fij = dVdr * rij / r
            grad[i] += fij
            grad[j] -= fij
    return grad

def _langevin_dynamics(positions, dt=0.002, kT=0.3, n_steps=500, rng=None):
    """Langevin dynamics to relax the cluster."""
    if rng is None:
        rng = np.random.default_rng()
    n = len(positions)
    pos = positions.copy().astype(np.float64)
    vel = np.zeros_like(pos)
    
    for step in range(n_steps):
        grad = _dzugutov_gradient(pos)
        # Langevin: m*dv/dt = -grad - gamma*v + sqrt(2*gamma*kT)*noise
        gamma = 1.0
        noise = rng.normal(0, np.sqrt(2 * gamma * kT / dt), pos.shape)
        vel = vel * (1 - gamma * dt) - grad * dt + noise
        pos = pos + vel * dt
        # Keep atoms at least 0.5 apart by projection if too close
        for i in range(n):
            for j in range(i + 1, n):
                rij = pos[i] - pos[j]
                r = np.linalg.norm(rij)
                if r < 0.5:
                    if r < 1e-10:
                        rij = rng.normal(0, 0.1, 3)
                        r = np.linalg.norm(rij)
                    correction = (0.5 - r) * 0.5
                    pos[i] += correction * rij / r
                    pos[j] -= correction * rij / r
    
    # Final relaxation using the provided relax function
    energy, final_pos = relax(pos, 'dzugutov', None)
    return energy, final_pos

def _self_assemble(n_atoms, rng, deadline):
    """Grow cluster by adding atoms one at a time with relaxation."""
    # Start with a small stable core (e.g., 13-atom icosahedron)
    phi = (1 + np.sqrt(5)) / 2
    core = np.array([
        [0, 0, 0],
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=np.float64)
    # Scale to appropriate spacing (~1.1 for Dzugutov)
    scale = 1.1 / (2 * phi)
    core = core * scale
    
    if n_atoms <= 13:
        # Just relax the core and trim or pad
        if n_atoms < 13:
            core = core[:n_atoms]
        energy, pos = relax(core, 'dzugutov', None)
        return energy, pos
    
    pos = core.copy()
    current_n = 13
    
    while current_n < n_atoms and time.time() < deadline:
        # Add a new atom at a random position near the surface
        # Find the center of mass and typical radius
        com = np.mean(pos, axis=0)
        radii = np.linalg.norm(pos - com, axis=1)
        r_typical = np.mean(radii)
        
        # Try to add atom at random direction at radius slightly larger than typical
        for attempt in range(20):
            direction = rng.normal(0, 1, 3)
            direction /= np.linalg.norm(direction)
            r_new = r_typical * (1.0 + 0.2 * rng.uniform(0, 1))
            new_atom = com + direction * r_new
            
            # Check minimum distance
            dists = np.linalg.norm(pos - new_atom, axis=1)
            if np.min(dists) >= 0.5:
                break
        
        pos = np.vstack([pos, new_atom.reshape(1, 3)])
        current_n += 1
        
        # Relax the current cluster
        energy, pos = relax(pos, 'dzugutov', None)
        
        # Check if we're running out of time
        if time.time() > deadline - 10:
            break
    
    # If we didn't reach n_atoms, pad with atoms placed carefully
    while len(pos) < n_atoms:
        com = np.mean(pos, axis=0)
        radii = np.linalg.norm(pos - com, axis=1)
        r_typical = np.mean(radii)
        for _ in range(50):
            direction = rng.normal(0, 1, 3)
            direction /= np.linalg.norm(direction)
            r_new = r_typical * (1.0 + 0.3 * rng.uniform(0, 1))
            new_atom = com + direction * r_new
            dists = np.linalg.norm(pos - new_atom, axis=1)
            if np.min(dists) >= 0.5:
                break
        pos = np.vstack([pos, new_atom.reshape(1, 3)])
    
    # Final relaxation
    energy, pos = relax(pos, 'dzugutov', None)
    return energy, pos

def _basin_hopping(positions, rng, deadline, n_trials=50):
    """Basin hopping to find lower energy structures."""
    best_energy, best_pos = _dzugutov_energy(positions), positions.copy()
    current_energy, current_pos = best_energy, best_pos
    
    for trial in range(n_trials):
        if time.time() > deadline - 5:
            break
        
        # Perturb the structure
        scale = 0.3 * rng.uniform(0.5, 2.0)
        perturbed = current_pos + rng.normal(0, scale, current_pos.shape)
        
        # Relax the perturbed structure
        energy, relaxed = relax(perturbed, 'dzugutov', None)
        
        # Metropolis acceptance
        if energy < current_energy:
            current_energy, current_pos = energy, relaxed
            if energy < best_energy:
                best_energy, best_pos = energy, relaxed
        else:
            # Accept with some probability to escape local minima
            delta_E = energy - current_energy
            kT = 0.5
            if rng.uniform(0, 1) < np.exp(-delta_E / kT):
                current_energy, current_pos = energy, relaxed
    
    return best_energy, best_pos

def construct(instance):
    """Find low-energy Dzugutov cluster structures using physics-driven self-assembly."""
    n_atoms = instance["n"]
    start_time = time.time()
    total_deadline = start_time + 290  # Safety margin for 300s limit
    
    # Multiple independent self-assembly runs
    best_energy = np.inf
    best_positions = None
    
    n_runs = 6  # Number of independent attempts
    
    for run in range(n_runs):
        if time.time() > total_deadline - 30:
            break
        
        rng = np.random.default_rng(run * 1000 + n_atoms)
        
        # Self-assembly with Langevin
        energy, positions = _self_assemble(n_atoms, rng, total_deadline - 20)
        
        # Basin hopping from the self-assembled structure
        bh_deadline = min(total_deadline - 10, time.time() + 30)
        energy, positions = _basin_hopping(positions, rng, bh_deadline, n_trials=30)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = positions
    
    # Final high-precision relaxation
    if best_positions is not None:
        final_energy, final_positions = relax(best_positions, 'dzugutov', None)
        return final_positions
    
    # Fallback: random compact cluster
    rng = np.random.default_rng(n_atoms)
    radius = 1.1 * n_atoms ** (1.0 / 3.0) * 0.75
    points = []
    while len(points) < n_atoms:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
    energy, positions = relax(np.array(points), 'dzugutov', None)
    return positions
# EVOLVE-BLOCK-END
