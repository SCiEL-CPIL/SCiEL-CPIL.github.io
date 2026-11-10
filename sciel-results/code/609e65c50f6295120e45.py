"""Morse cluster global minima via stochastic collision dynamics (event-driven MD with thermostat)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Try to import chem_clusters for faster energy/gradient if available
try:
    import chem_clusters
    _HAS_CHEM = True
except ImportError:
    _HAS_CHEM = False

def _fast_energy(positions, rho):
    """Compute Morse energy using chem_clusters if available, else vectorized numpy."""
    if _HAS_CHEM:
        try:
            return chem_clusters.morse_energy(positions, rho)
        except Exception:
            pass
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(np.maximum(dist_sq, 1e-20))
    x = rho * (1.0 - dist)
    # Clip x to avoid overflow
    x = np.clip(x, -50, 50)
    exp_x = np.exp(x)
    pair_energy = exp_x * (exp_x - 2.0)
    total = 0.5 * np.sum(pair_energy)
    return total

def _fast_gradient(positions, rho):
    """Compute Morse energy gradient using chem_clusters if available, else vectorized numpy."""
    if _HAS_CHEM:
        try:
            return chem_clusters.morse_energy_gradient(positions, rho)
        except Exception:
            pass
    n = len(positions)
    if n < 2:
        return np.zeros_like(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1e-20)
    dist = np.sqrt(dist_sq)
    x = rho * (1.0 - dist)
    x = np.clip(x, -50, 50)
    exp_x = np.exp(x)
    # dE/dr = -2*rho*exp(rho*(1-r))*(exp(rho*(1-r))-1)
    dE_dr = -2.0 * rho * exp_x * (exp_x - 1.0)
    # Set diagonal to 0
    np.fill_diagonal(dE_dr, 0.0)
    # force_ij = dE_dr_ij * diff_ij / dist_ij
    # diff[i,j] = pos[i] - pos[j]
    # grad[i] = sum_j dE_dr[i,j] * (pos[i]-pos[j]) / dist[i,j]
    # = sum_j dE_dr[i,j] * diff[i,j] / dist[i,j]
    # Vectorized: multiply dE_dr by diff/dist element-wise, then sum over j axis
    inv_dist = 1.0 / np.maximum(dist, 1e-10)
    np.fill_diagonal(inv_dist, 0.0)
    # factor[i,j] = dE_dr[i,j] / dist[i,j]
    factor = dE_dr * inv_dist
    # grad[i] = sum_j factor[i,j] * diff[i,j]
    grad = np.einsum('ij,ijk->ik', factor, diff)
    return grad

def _md_stage(positions, velocities, rho, n_steps, dt, temperature, rng):
    """
    Velocity Verlet MD with Langevin thermostat for cluster assembly.
    Returns final positions and energy.
    """
    n = len(positions)
    pos = positions.copy()
    vel = velocities.copy()
    
    gamma = 0.5  # friction coefficient
    noise_std = np.sqrt(2.0 * gamma * temperature * dt)
    
    for step in range(n_steps):
        # Compute forces (negative gradient)
        grad = _fast_gradient(pos, rho)
        force = -grad
        
        # Velocity Verlet: half step velocity
        vel_half = vel + 0.5 * dt * force
        
        # Full step position
        pos_new = pos + dt * vel_half
        
        # Compute new forces
        grad_new = _fast_gradient(pos_new, rho)
        force_new = -grad_new
        
        # Full step velocity
        vel_new = vel_half + 0.5 * dt * force_new
        
        # Langevin thermostat
        vel_new = vel_new * (1.0 - gamma * dt)
        vel_new += noise_std * rng.standard_normal((n, 3))
        
        pos = pos_new
        vel = vel_new
    
    return pos, _fast_energy(pos, rho)

def _local_relax(positions, rho, max_iter=200):
    """Local minimization using L-BFGS-B."""
    def objective(x):
        pos = x.reshape(-1, 3)
        return _fast_energy(pos, rho)
    
    def gradient(x):
        pos = x.reshape(-1, 3)
        return _fast_gradient(pos, rho).flatten()
    
    x0 = positions.flatten()
    try:
        result = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'gtol': 1e-5, 'ftol': 1e-8})
        return result.x.reshape(-1, 3), result.fun
    except Exception:
        return positions, _fast_energy(positions, rho)

def _check_min_distance(positions, min_d=0.5):
    """Check if all pairs are at least min_d apart."""
    n = len(positions)
    if n < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, np.inf)
    min_d_sq = min_d * min_d
    return np.all(dist_sq >= min_d_sq)

def _init_gas(n, radius, rng):
    """Initialize n atoms as a gas in a sphere of given radius with min distance 0.51."""
    positions = []
    attempts = 0
    max_attempts = n * 200
    min_d = 0.51
    while len(positions) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - np.array(q)) >= min_d for q in positions):
                positions.append(p)
        attempts += 1
    
    if len(positions) < n:
        # Fallback: Fibonacci sphere
        positions = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            z = 1 - 2 * (i + 0.5) / n
            r = np.sqrt(max(0, 1 - z*z))
            theta = 2 * np.pi * i / golden_ratio
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * radius
            positions.append(p)
    
    return np.array(positions)

def _init_lattice(n, rng):
    """Initialize n atoms on a BCC-like lattice for better starting structure."""
    # Create a cubic lattice and take first n points
    side = int(np.ceil(n ** (1.0/3.0))) + 2
    points = []
    for x in range(side):
        for y in range(side):
            for z in range(side):
                points.append([x, y, z])
                # Add BCC center
                points.append([x + 0.5, y + 0.5, z + 0.5])
        if len(points) >= n:
            break
        if len(points) >= n:
            break
    if len(points) < n:
        side += 2
        points = []
        for x in range(side):
            for y in range(side):
                for z in range(side):
                    points.append([x, y, z])
                    points.append([x + 0.5, y + 0.5, z + 0.5])
    
    # Shuffle and take first n
    idx = rng.permutation(len(points))[:n]
    positions = np.array([points[i] for i in idx], dtype=float)
    
    # Center the structure
    positions -= positions.mean(axis=0)
    
    # Scale to reasonable size
    max_r = np.max(np.linalg.norm(positions, axis=1))
    if max_r > 0:
        target_r = 0.5 * n ** (1.0/3.0) + 1.0
        positions *= target_r / max_r
    
    # Add small random perturbation
    positions += rng.normal(0, 0.05, positions.shape)
    
    return positions

def construct(instance):
    """
    Construct low-energy Morse cluster using stochastic collision dynamics.
    
    Strategy:
    1. Multiple restarts with diverse initializations (gas, lattice)
    2. Run MD with gradual cooling (Langevin thermostat)
    3. Local relax with L-BFGS-B
    4. Keep best valid configuration
    """
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]
    
    # Time budget: ~45s per instance (6 instances * 45s = 270s < 300s)
    time_budget = 45.0
    start_time = time.time()
    
    best_energy = np.inf
    best_positions = None
    
    # Number of restarts: adapt based on n
    # Smaller clusters can afford more restarts
    if n <= 40:
        n_restarts = 12
    elif n <= 60:
        n_restarts = 10
    else:
        n_restarts = 8
    
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > time_budget * 0.85:
            break
        
        rng = np.random.default_rng(restart * 7919 + n * 104729 + 42)
        
        # Alternate between gas and lattice initialization
        if restart % 2 == 0:
            radius = 0.5 * n ** (1.0/3.0) + 2.0
            positions = _init_gas(n, radius, rng)
        else:
            positions = _init_lattice(n, rng)
        
        # Small random perturbation to break symmetry
        positions += rng.normal(0, 0.02, positions.shape)
        
        velocities = rng.uniform(-0.5, 0.5, (n, 3))
        
        # Multi-stage cooling schedule
        # Start hot to allow structural reorganization, then cool
        stages = [
            (300, 0.005, 8.0),    # Hot: break initial structure
            (300, 0.005, 3.0),    # Warm: begin assembly
            (300, 0.005, 1.0),    # Medium: refine structure
            (300, 0.003, 0.3),    # Cool: fine-tune
            (200, 0.003, 0.05),   # Very cool: settle
            (100, 0.002, 0.01),   # Near zero: final settle
        ]
        
        for n_steps, dt, T in stages:
            elapsed = time.time() - start_time
            if elapsed > time_budget * 0.92:
                break
            positions, E = _md_stage(positions, velocities, rho, n_steps, dt, T, rng)
            # Re-thermalize slightly between stages
            velocities = rng.normal(0, 0.05, (n, 3))
        
        # Local relaxation
        positions_relaxed, E_relaxed = _local_relax(positions, rho, max_iter=300)
        
        # Check constraints: atoms at least 0.5 apart
        if _check_min_distance(positions_relaxed, 0.5):
            if E_relaxed < best_energy:
                best_energy = E_relaxed
                best_positions = positions_relaxed
        
        # If invalid, try to fix by nudging close pairs apart
        else:
            # Simple fix: iteratively push apart close pairs
            pos_fixed = positions_relaxed.copy()
            for _ in range(50):
                moved = False
                for i in range(n):
                    for j in range(i+1, n):
                        d = pos_fixed[i] - pos_fixed[j]
                        dist = np.linalg.norm(d)
                        if dist < 0.5 and dist > 1e-10:
                            push = (0.5 - dist) / dist * d
                            pos_fixed[i] += 0.5 * push
                            pos_fixed[j] -= 0.5 * push
                            moved = True
                if not moved:
                    break
            if _check_min_distance(pos_fixed, 0.5):
                E_fixed = _fast_energy(pos_fixed, rho)
                if E_fixed < best_energy:
                    best_energy = E_fixed
                    best_positions = pos_fixed
    
    if best_positions is None:
        # Fallback: use a simple spherical arrangement
        positions = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        radius = 0.5 * n ** (1.0/3.0) + 1.0
        for i in range(n):
            z = 1 - 2 * (i + 0.5) / n
            r = np.sqrt(max(0, 1 - z*z))
            theta = 2 * np.pi * i / golden_ratio
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * radius
            positions.append(p)
        best_positions = np.array(positions)
    
    return best_positions
# EVOLVE-BLOCK-END
