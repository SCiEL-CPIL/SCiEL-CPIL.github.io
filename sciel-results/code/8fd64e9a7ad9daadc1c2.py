"""
Strategy: Dissipative Molecular Dynamics (Fire Algorithm) for Global Minimization.
Instead of gradient-based local minimization (L-BFGS-B) combined with basin hopping,
we use a physics-based simulation: Velocity Verlet integration with strong damping
(friction) and a "Fire" (Fast Inertial Relaxation Engine) algorithm. 
This allows the cluster to self-assemble from a high-temperature gas state. 
The Fire algorithm adapts the time step and cooling rate dynamically to accelerate
convergence while maintaining stability, effectively performing a global search 
through kinetic energy fluctuations before settling into low-energy basins.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 290.0  # 300s limit, keep 10s safety margin

def _morse_energy_and_grad(positions, rho):
    """Compute exact Morse energy and gradient."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Vectorized distance calculation
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    
    x = rho * (1.0 - r)
    ex = np.exp(x)
    energy = 0.5 * float(np.sum(ex * (ex - 2.0)))
    
    # Gradient calculation
    # dV/dr = -rho * 2 * exp(x) * (exp(x) - 1)
    dV_dr = -rho * 2.0 * ex * (ex - 1.0)
    
    inv_r = np.zeros_like(r)
    mask = r > 1e-12
    inv_r[mask] = 1.0 / r[mask]
    
    # Force on i from j: F_ij = -dV/dr * (r_i - r_j) / r
    # Sum over j for each i
    # factor[i, j] = -dV_dr[i, j] / r[i, j]
    factor = -dV_dr * inv_r
    
    # grad_i = sum_j factor[i, j] * (r_i - r_j)
    # grad_i = r_i * sum_j factor[i, j] - sum_j factor[i, j] * r_j
    row_sum = np.sum(factor, axis=1)
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

def _generate_gas(n, temperature, rng):
    """Generate a random gas configuration with velocities."""
    # Positions in a cube
    side = n ** (1/3) * 2.0
    pos = rng.uniform(-side/2, side/2, (n, 3))
    
    # Velocities according to Maxwell-Boltzmann
    # sigma_v = sqrt(temperature)
    vel = rng.normal(0, np.sqrt(temperature), (n, 3))
    return pos, vel

def _fire_optimize(positions, rho, t_max=10.0, dt_init=0.01, dt_min=0.001, dt_max=0.1,
                   f_min=0.9, f_max=1.1, n_steps_max=2000, rng=None):
    """
    Fire algorithm (Fast Inertial Relaxation Engine) for minimizing energy.
    Combines dynamics with adaptive time steps and cooling.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    pos = positions.copy()
    n = len(pos)
    
    # Initial velocity
    vel = rng.normal(0, 0.1, pos.shape)
    
    dt = dt_init
    t = 0.0
    
    # Fire parameters
    alpha = 0.01 # Damping coefficient (friction)
    
    # State
    energy, grad = _morse_energy_and_grad(pos, rho)
    best_energy = energy
    best_pos = pos.copy()
    
    # Track previous velocity direction for Fire logic
    v_prev = vel.copy()
    
    for step in range(n_steps_max):
        # Check time limit
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
            
        # Calculate forces
        force = -grad
        
        # Update velocity (Velocity Verlet half step + damping)
        # v_new = (v + a*dt) * exp(-alpha*dt) + ... 
        # Simplified Fire update:
        # 1. Compute kinetic energy
        ke = 0.5 * np.sum(vel**2)
        
        # 2. Check if power P = v . F is positive
        power = np.sum(vel * force)
        
        if power > 0:
            # Accelerate: increase dt
            dt = dt * f_max
            # No change to velocity direction logic explicitly in simple Fire, 
            # but we can dampen less or increase step
        else:
            # Decelerate: decrease dt
            dt = dt * f_min
            
        # Clamp dt
        dt = np.clip(dt, dt_min, dt_max)
        
        # Integrate
        # v_new = (v + F*dt) * exp(-alpha*dt)
        # This is a simplified damping. For better stability, use:
        # v = (v + F*dt) * (1 - alpha*dt)  (for small alpha*dt)
        
        v_new = (vel + force * dt) * (1.0 - alpha * dt)
        
        # Check for stability: if energy increases too much or dt was too large
        # We can also check if the step caused an instability (e.g. atoms colliding too hard)
        
        pos_new = pos + v_new * dt
        
        # Check validity (min distance)
        # If invalid, we might need to reduce dt and retry or just accept penalty?
        # For now, if invalid, we revert and reduce dt significantly.
        if not _is_valid(pos_new, 0.5):
            dt = dt * 0.5
            continue
            
        energy_new, grad_new = _morse_energy_and_grad(pos_new, rho)
        
        # If energy explodes, revert
        if energy_new > energy + 10.0:
            dt = dt * 0.5
            continue
            
        # Update state
        pos = pos_new
        vel = v_new
        energy = energy_new
        grad = grad_new
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
            
        # Cooling: reduce velocity magnitude over time to settle
        # This mimics annealing.
        vel *= 0.995
        
        # Stop if converged
        if np.linalg.norm(grad) < 1e-4 and ke < 1e-4:
            break
            
        t += dt
        if t > t_max:
            break
            
    return best_pos, best_energy

def _generate_crystalline_starts(n, rng):
    """Generate diverse crystalline and random starts."""
    starts = []
    
    # 1. FCC
    a = np.sqrt(2.0)
    points = []
    max_coord = int(np.ceil(n ** (1/3) * 1.5)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                base = np.array([i, j, k]) * a
                for dx, dy, dz in [(0,0,0), (0, 0.5*a, 0.5*a), (0.5*a, 0, 0.5*a), (0.5*a, 0.5*a, 0)]:
                    p = base + np.array([dx, dy, dz])
                    points.append(p)
    points = np.array(points)
    points -= np.mean(points, axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    starts.append(points[idx[:n]])
    
    # 2. BCC
    a = 2.0 / np.sqrt(3.0)
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                base = np.array([i, j, k]) * a
                for dx, dy, dz in [(0,0,0), (0.5*a, 0.5*a, 0.5*a)]:
                    p = base + np.array([dx, dy, dz])
                    points.append(p)
    points = np.array(points)
    points -= np.mean(points, axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    starts.append(points[idx[:n]])
    
    # 3. Random compact
    radius = 1.0 * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.6 for q in points):
            points.append(p)
        attempts += 1
    if len(points) == n:
        starts.append(np.array(points))
        
    return starts

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(42)
    
    best_energy = float('inf')
    best_positions = None
    
    # Time budget per start
    # We have ~290s total. 6 instances. ~48s per instance.
    # We will try 3-4 starts per instance with Fire optimization.
    
    starts = _generate_crystalline_starts(size, rng)
    
    # Add a high-temperature gas start
    gas_pos, _ = _generate_gas(size, temperature=5.0, rng=rng)
    starts.append(gas_pos)
    
    for i, pos in enumerate(starts):
        # Check time
        if time.time() - _START_TIME > _TIME_LIMIT - 30:
            break
            
        # Run Fire optimization
        # Adjust t_max based on remaining time and number of starts
        # Rough estimate: Fire step is fast, but many steps.
        # Let's allocate ~10s per start
        fire_pos, fire_energy = _fire_optimize(pos, parameter, t_max=5.0, dt_init=0.01, 
                                               n_steps_max=5000, rng=rng)
        
        if _is_valid(fire_pos):
            if fire_energy < best_energy:
                best_energy = fire_energy
                best_positions = fire_pos.copy()
                
    # If we have time left, do a final polish on the best position using L-BFGS-B
    # This combines the global search of Fire with the precision of gradient descent
    if best_positions is not None and time.time() - _START_TIME < _TIME_LIMIT - 10:
        try:
            from scipy.optimize import minimize
            
            def fun(x):
                x = x.reshape(-1, 3)
                e, g = _morse_energy_and_grad(x, parameter)
                return e, g.flatten()
                
            res = minimize(fun, best_positions.flatten(), method='L-BFGS-B', jac=True,
                           options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-7})
            polished_pos = res.x.reshape(-1, 3)
            polished_energy = res.fun
            
            if _is_valid(polished_pos) and polished_energy < best_energy:
                best_energy = polished_energy
                best_positions = polished_pos
        except Exception:
            pass # Ignore if polishing fails
            
    # Fallback
    if best_positions is None:
        radius = 1.0 * size ** (1.0 / 3.0) * 1.5
        points = []
        attempts = 0
        while len(points) < size and attempts < size * 200:
            p = rng.uniform(-radius, radius, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
            attempts += 1
        best_positions = np.array(points)
        
    return best_positions
# EVOLVE-BLOCK-END
