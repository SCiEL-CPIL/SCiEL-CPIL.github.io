"""
Strategy: Stochastic Annealing via Velocity Verlet MD with Custom Thermostat.
This approach fundamentally differs from the current deterministic multi-start L-BFGS-B method.
Instead of relaxing fixed initial configurations with a gradient-based solver, we simulate the
dynamics of the atoms using Velocity Verlet integration with a Langevin thermostat (or simple
velocity rescaling). The system starts at a high temperature (disordered gas) and is slowly
cooled (annealed) to a very low temperature. This allows the cluster to explore configuration
space, overcome energy barriers, and settle into deep local minima, potentially finding the
global minimum through structural reorganization. We run multiple such annealing trajectories
in parallel (sequentially in code, but conceptually independent) to increase the chance of
finding the global minimum.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 295.0  # 300s limit, keep 5s safety margin

def _morse_energy_and_grad(positions, rho):
    """Compute exact Morse energy and gradient."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Pairwise differences
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)  # Avoid division by zero
    
    # Morse potential: V(r) = exp(rho(1-r)) * (exp(rho(1-r)) - 2)
    # Let x = rho(1-r)
    # V = e^x (e^x - 2) = e^(2x) - 2e^x
    # dV/dr = (2e^(2x) - 2e^x) * dx/dr = 2e^x(e^x - 1) * (-rho)
    # dV/dri = -dV/dr * (ri - rj) / r
    
    x = rho * (1.0 - r)
    ex = np.exp(x)
    energy = 0.5 * float(np.sum(ex * (ex - 2.0)))
    
    # Gradient
    # dV/dr = -rho * 2 * ex * (ex - 1)
    dV_dr = -rho * 2.0 * ex * (ex - 1.0)
    
    # Vectorized gradient calculation
    inv_r = np.zeros_like(r)
    mask = r > 1e-12
    inv_r[mask] = 1.0 / r[mask]
    
    # factor[i,j] = dV_dr[i,j] * inv_r[i,j]
    factor = dV_dr * inv_r
    
    # sum_j factor[i,j] -> row_sum (n,)
    row_sum = np.sum(factor, axis=1)
    
    # sum_j factor[i,j] * positions[j] -> factor @ positions (n, 3)
    weighted_pos = factor.dot(positions)
    
    grad = positions * row_sum[:, np.newaxis] - weighted_pos
    
    return energy, grad

def _is_valid(positions, min_dist=0.5):
    """Check if all pairwise distances are >= min_dist."""
    if len(positions) < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    return np.all(r >= min_dist)

def _generate_gas(n, rng, radius_scale=1.5):
    """Generate n atoms randomly in a compact region (gas-like)."""
    radius = radius_scale * n ** (1.0 / 3.0)
    points = np.zeros((n, 3))
    for i in range(n):
        while True:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                # Check minimum distance to existing atoms to avoid initial singularity
                if i == 0:
                    points[i] = p
                    break
                dists = np.linalg.norm(points[:i] - p, axis=1)
                if np.min(dists) > 0.6: # Slightly larger than min constraint to be safe
                    points[i] = p
                    break
    return points

def _velocity_verlet_anneal(positions, rho, T_start=5.0, T_end=0.01, n_steps=2000, dt=0.01, gamma=0.1, rng=None):
    """
    Perform Velocity Verlet MD with Langevin thermostatting (simplified).
    Returns the final positions and energy.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    n = len(positions)
    x = positions.copy()
    v = rng.normal(0.0, np.sqrt(T_start / n), (n, 3)) # Initial velocities scaled by T
    
    # We use a simple velocity rescaling thermostat to maintain temperature
    # This is less robust than full Langevin but faster and simpler.
    # Alternatively, we can use a Nosé-Hoover-like approach, but rescaling is often sufficient for annealing.
    
    for step in range(n_steps):
        # Linear cooling schedule
        T_current = T_start * (T_end / T_start) ** (step / n_steps)
        
        # 1. Update velocities half step
        x, grad = _morse_energy_and_grad(x, rho)
        v += 0.5 * dt * grad / n # Mass = 1
        
        # 2. Update positions
        x += dt * v
        
        # 3. Compute new forces
        x, grad_new = _morse_energy_and_grad(x, rho)
        
        # 4. Update velocities half step
        v += 0.5 * dt * grad_new / n
        
        # 5. Thermostat: Rescale velocities to target temperature
        # Kinetic energy KE = 0.5 * sum(v^2). Target KE = (3/2) * n * T
        v2 = np.sum(v**2)
        if v2 > 1e-10:
            current_T = v2 / (3.0 * n)
            # Smooth rescaling to avoid violent shocks
            # v_new = v * sqrt(T_target / T_current)
            # To be gentle, we can mix
            target_v2 = 3.0 * n * T_current
            scale_factor = np.sqrt(target_v2 / v2)
            # Damping factor to make it more like Langevin
            # v = (1-gamma)*v + gamma*v_target_dist
            # Here we just rescale for simplicity and speed
            v *= scale_factor
            
        # Check time limit
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
            
    final_energy, _ = _morse_energy_and_grad(x, rho)
    return x, final_energy

def _local_minimize(positions, rho):
    """Local minimization using L-BFGS-B as a final polish."""
    def fun(x):
        x = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(x, rho)
        g = g.flatten()
        return e, g
    
    try:
        res = minimize(fun, positions.flatten(), method='L-BFGS-B', jac=True, 
                       options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6})
        final_pos = res.x.reshape(-1, 3)
        final_energy = res.fun
        
        if _is_valid(final_pos):
            return final_pos, final_energy
    except Exception:
        pass
    
    return positions, _morse_energy_and_grad(positions, rho)[0]

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Fixed seed for reproducibility of the random parts, but we use multiple runs
    base_seed = 42
    
    best_energy = float('inf')
    best_positions = None
    
    # We will run multiple independent annealing trajectories
    # Each trajectory starts from a different random gas configuration
    n_runs = 8 # Number of independent annealing runs
    
    # Estimate time per run to determine how many we can afford
    # Roughly 2000 steps * (energy calc cost). For n=80, energy calc is ~O(n^2) = 6400 ops.
    # 2000 * 6400 * 8 runs = ~100M ops. Should be fast in numpy.
    
    for run in range(n_runs):
        if time.time() - _START_TIME > _TIME_LIMIT - 20:
            break
            
        rng = np.random.default_rng(base_seed + run)
        
        # Generate initial gas configuration
        pos0 = _generate_gas(size, rng, radius_scale=1.2)
        
        # Anneal
        try:
            pos_annealed, e_annealed = _velocity_verlet_anneal(pos0, parameter, T_start=3.0, T_end=0.05, n_steps=1500, dt=0.005, rng=rng)
            
            if _is_valid(pos_annealed):
                # Polish with L-BFGS-B to ensure we are at a local minimum
                pos_final, e_final = _local_minimize(pos_annealed, parameter)
                
                if e_final < best_energy:
                    best_energy = e_final
                    best_positions = pos_final
        except Exception:
            # Fallback to simple relax if MD fails
            try:
                energy, positions = relax(pos0, potential, parameter)
                if _is_valid(positions) and energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass

    # If no valid position found, return a valid random one
    if best_positions is None:
        radius = 1.0 * size ** (1.0 / 3.0) * 1.5
        points = []
        rng = np.random.default_rng(0)
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        best_positions = np.array(points)
    
    return best_positions
# EVOLVE-BLOCK-END
