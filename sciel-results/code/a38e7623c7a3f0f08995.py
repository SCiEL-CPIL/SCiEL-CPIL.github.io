"""Stochastic many-body collision dynamics for Morse cluster global minima."""
import numpy as np
from scipy.optimize import minimize
import time

def _morse_energy(positions, rho):
    """Compute total Morse potential energy."""
    n = len(positions)
    E = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-10:
                continue
            x = rho * (1.0 - r)
            E += np.exp(x) * (np.exp(x) - 2.0)
    return E

def _morse_energy_gradient(positions, rho):
    """Compute gradient of total Morse potential energy."""
    n = len(positions)
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            r = np.linalg.norm(rij)
            if r < 1e-10:
                continue
            x = rho * (1.0 - r)
            exp_x = np.exp(x)
            # dE/dr = 2*rho*exp(2x) - 2*rho*exp(x) = 2*rho*exp_x*(exp_x - 1)
            dE_dr = 2.0 * rho * exp_x * (exp_x - 1.0)
            # grad_i = -dE/dr * rij/r, grad_j = +dE/dr * rij/r
            factor = dE_dr / r
            grad[i] -= factor * rij
            grad[j] += factor * rij
    return grad

def _velocity_verlet_step(positions, velocities, dt, rho, n_steps=1):
    """Perform Velocity Verlet integration steps."""
    for _ in range(n_steps):
        grad = _morse_energy_gradient(positions, rho)
        velocities += 0.5 * dt * grad
        positions += dt * velocities
        grad = _morse_energy_gradient(positions, rho)
        velocities += 0.5 * dt * grad
    return positions, velocities

def _cooling_schedule(initial_T, final_T, n_steps, gamma=0.98):
    """Geometric cooling schedule."""
    T = initial_T
    schedule = []
    for i in range(n_steps):
        schedule.append(T)
        T *= gamma
    return schedule

def _simulate_md(positions, rho, n_atoms, time_budget=45.0):
    """
    Run stochastic MD with thermostatting to find low-energy configurations.
    Uses velocity Verlet with Langevin-like thermostatting.
    """
    start_time = time.time()
    
    n = n_atoms
    # Initialize with small random velocities
    rng = np.random.default_rng(hash((rho, n)) % (2**32))
    velocities = rng.normal(0, 0.1, size=(n, 3))
    
    # Parameters
    dt = 0.005
    initial_T = 2.0
    final_T = 0.01
    total_steps = 50000
    
    # Cooling: geometric
    gamma = (final_T / initial_T) ** (1.0 / total_steps)
    
    best_positions = positions.copy()
    best_energy = _morse_energy(positions, rho)
    
    current_T = initial_T
    
    for step in range(total_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Velocity Verlet step
        positions, velocities = _velocity_verlet_step(positions, velocities, dt, rho)
        
        # Thermostat: scale velocities to maintain target temperature
        # KE = 0.5 * sum(m*v^2), m=1
        KE = 0.5 * np.sum(velocities**2)
        target_KE = 1.5 * n * current_T
        if KE > 1e-10:
            scale = np.sqrt(target_KE / KE)
            velocities *= scale
        
        # Add small random kicks for stochasticity
        velocities += rng.normal(0, 0.01 * np.sqrt(current_T), size=(n, 3))
        
        # Cool
        current_T *= gamma
        
        # Check energy
        energy = _morse_energy(positions, rho)
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()
    
    return best_positions, best_energy

def _local_refine(positions, rho, max_iter=2000):
    """Local refinement using BFGS from scipy."""
    def objective(x):
        pos = x.reshape(-1, 3)
        return _morse_energy(pos, rho)
    
    def gradient(x):
        pos = x.reshape(-1, 3)
        return _morse_energy_gradient(pos, rho).flatten()
    
    x0 = positions.flatten()
    result = minimize(objective, x0, jac=gradient, method='BFGS', 
                      options={'maxiter': max_iter, 'gtol': 1e-8})
    return result.x.reshape(-1, 3), result.fun

def construct(instance):
    """
    Find low-energy arrangement of n atoms bound by Morse potentials.
    Uses stochastic MD self-assembly followed by local refinement.
    """
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    if potential != "morse":
        # Fallback for other potentials (shouldn't happen per contract)
        from autoresearch_suite.chem_clusters import relax
        rng = np.random.default_rng(size)
        radius = size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 for q in points):
                points.append(p)
        energy, positions = relax(np.array(points), potential, rho)
        return positions
    
    # Time budget per instance: ~48s (6 instances * 48s = 288s < 300s)
    time_per_instance = 48.0
    
    # Generate multiple initial configurations and run MD on each
    rng = np.random.default_rng(hash((rho, size)) % (2**32))
    
    # Strategy: run 2-3 independent MD simulations with different initial conditions
    # and take the best result after local refinement
    
    n_simulations = 3
    time_per_sim = time_per_instance / (n_simulations + 1)  # reserve time for refinement
    
    best_positions = None
    best_energy = float('inf')
    
    for sim_idx in range(n_simulations):
        # Generate initial gas-like configuration
        radius = size ** (1.0 / 3.0) * 1.2
        positions = rng.uniform(-radius, radius, size=(size, 3))
        
        # Ensure minimum distance of 0.5
        for _ in range(100):  # Try to resolve overlaps
            violated = False
            for i in range(size):
                for j in range(i + 1, size):
                    d = np.linalg.norm(positions[i] - positions[j])
                    if d < 0.5:
                        violated = True
                        # Push apart
                        if d < 1e-10:
                            positions[j] += rng.normal(0, 0.1, size=3)
                        else:
                            direction = (positions[j] - positions[i]) / d
                            positions[i] -= 0.25 * direction
                            positions[j] += 0.25 * direction
            if not violated:
                break
        
        # Run MD simulation
        md_positions, md_energy = _simulate_md(positions, rho, size, time_budget=time_per_sim)
        
        # Local refinement
        refined_positions, refined_energy = _local_refine(md_positions, rho, max_iter=1000)
        
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_positions = refined_positions
    
    # Final check: ensure minimum distance constraint
    if best_positions is not None:
        # If any atoms are too close, nudge them
        for i in range(size):
            for j in range(i + 1, size):
                d = np.linalg.norm(best_positions[i] - best_positions[j])
                if d < 0.5:
                    if d < 1e-10:
                        best_positions[j] += np.array([0.5, 0, 0])
                    else:
                        direction = (best_positions[j] - best_positions[i]) / d
                        best_positions[i] -= 0.26 * direction
                        best_positions[j] += 0.26 * direction
    
    return best_positions
