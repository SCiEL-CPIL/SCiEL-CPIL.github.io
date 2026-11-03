"""50 equal circles in a unit circle - dynamics-based optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
_DEADLINE = time.time() + 145.0  # safety margin before 150s kill


def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    dist_origin = np.hypot(centres[:, 0], centres[:, 1])
    r_contain = 1.0 - np.max(dist_origin)
    
    # Min pairwise distance / 2
    min_pair = np.inf
    for i in range(N):
        d = np.hypot(centres[i+1:, 0] - centres[i, 0], centres[i+1:, 1] - centres[i, 1])
        if d.size > 0:
            m = np.min(d)
            if m < min_pair:
                min_pair = m
    r_pair = min_pair / 2.0
    return min(r_contain, r_pair)


def _lj_energy(centres, sigma=1.0, epsilon=1.0, r_target=0.125):
    """Compute Lennard-Jones-like energy with confinement.
    Uses a soft repulsive potential to push circles apart and a harmonic well to keep them inside.
    """
    n = len(centres)
    energy = 0.0
    
    # Pairwise Lennard-Jones-like repulsion
    for i in range(n):
        for j in range(i+1, n):
            d = np.hypot(centres[i, 0] - centres[j, 0], centres[i, 1] - centres[j, 1])
            if d < 1e-10:
                d = 1e-10
            # Soft repulsion: strong when d < 2*r_target, weak otherwise
            if d < 2.5 * r_target:
                # Use a smooth repulsive potential
                energy += epsilon * (sigma/d)**12 - epsilon * (sigma/d)**6
            else:
                energy += 0.0
    
    # Confinement: harmonic potential pushing circles toward center
    for i in range(n):
        r = np.hypot(centres[i, 0], centres[i, 1])
        if r > 0.9:
            # Quadratic penalty for being outside 0.9 radius
            energy += 0.5 * 100.0 * (r - 0.9)**2
        elif r > 0.85:
            # Gentle push
            energy += 0.5 * 10.0 * (r - 0.85)**2
    
    return energy


def _lj_forces(centres, sigma=1.0, epsilon=1.0, r_target=0.125):
    """Compute forces from Lennard-Jones-like potential and confinement."""
    n = len(centres)
    forces = np.zeros_like(centres)
    
    # Pairwise forces
    for i in range(n):
        for j in range(i+1, n):
            dx = centres[i, 0] - centres[j, 0]
            dy = centres[i, 1] - centres[j, 1]
            d = np.hypot(dx, dy)
            if d < 1e-10:
                # Random small perturbation to break symmetry
                dx = (np.random.rand() - 0.5) * 1e-6
                dy = (np.random.rand() - 0.5) * 1e-6
                d = np.hypot(dx, dy)
            
            if d < 2.5 * r_target:
                # Lennard-Jones force: F = 12*eps*sigma^12/d^13 - 6*eps*sigma^6/d^7
                f_mag = epsilon * (12.0 * (sigma/d)**12 - 6.0 * (sigma/d)**6) / d
                fx = f_mag * dx
                fy = f_mag * dy
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy
    
    # Confinement forces
    for i in range(n):
        r = np.hypot(centres[i, 0], centres[i, 1])
        if r > 1e-10:
            if r > 0.9:
                f_mag = -100.0 * (r - 0.9) / r
            elif r > 0.85:
                f_mag = -10.0 * (r - 0.85) / r
            else:
                f_mag = 0.0
            forces[i, 0] += f_mag * centres[i, 0]
            forces[i, 1] += f_mag * centres[i, 1]
    
    return forces


def _velocity_verlet_step(centres, velocities, dt, sigma=1.0, epsilon=1.0, r_target=0.125):
    """One velocity-Verlet integration step with adaptive damping."""
    n = len(centres)
    forces1 = _lj_forces(centres, sigma, epsilon, r_target)
    
    # Update positions
    new_centres = centres + velocities * dt + 0.5 * forces1 * dt**2
    
    # Compute new forces
    forces2 = _lj_forces(new_centres, sigma, epsilon, r_target)
    
    # Update velocities with damping
    new_velocities = (velocities + 0.5 * (forces1 + forces2) * dt) * 0.95  # damping factor
    
    return new_centres, new_velocities


def _anneal_dynamics(centres_init, max_steps=5000, dt=0.005, sigma=0.25, epsilon=1.0, r_target=0.125,
                     temperature_start=0.1, temperature_end=0.001):
    """Run velocity-Verlet dynamics with annealing."""
    n = len(centres_init)
    centres = centres_init.copy()
    velocities = np.zeros_like(centres)
    
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    for step in range(max_steps):
        # Check deadline
        if time.time() > _DEADLINE - 10.0:
            break
        
        # Adaptive dt based on temperature
        t_frac = step / max_steps
        temperature = temperature_start * (temperature_end / temperature_start) ** t_frac
        dt_current = dt * (1.0 + temperature * 10.0)
        
        # Velocity-Verlet step
        centres, velocities = _velocity_verlet_step(centres, velocities, dt_current, sigma, epsilon, r_target)
        
        # Add small random perturbation for exploration
        if step % 100 == 0:
            centres += (np.random.randn(n, 2) - 0.5) * temperature * 0.01
        
        # Keep circles inside the unit disk
        dists = np.hypot(centres[:, 0], centres[:, 1])
        for i in range(n):
            if dists[i] > 0.98:
                scale = 0.98 / dists[i]
                centres[i] *= scale
                velocities[i] *= 0.5
        
        # Track best
        current_radius = _compute_radius(centres)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centres = centres.copy()
        
        # Adaptive step size reduction if energy increases too much
        if step % 500 == 0 and step > 0:
            # Check if we're making progress
            if best_radius < 0.12:
                dt *= 0.8
    
    return best_centres, best_radius


def _local_optimize(centres):
    """Local optimization using Nelder-Mead on individual coordinates."""
    def objective(params):
        pts = params.reshape(-1, 2)
        r = _compute_radius(pts)
        return -r
    
    x0 = centres.flatten()
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
    
    if result.fun < -_compute_radius(centres):
        return result.x.reshape(-1, 2)
    return centres


def _hexagonal_init(n=50):
    """Initialize with hexagonal lattice pattern."""
    points = []
    spacing = 0.22
    
    for j in range(-10, 11):
        for i in range(-10, 11):
            x = i * spacing + 0.5 * spacing * (j % 2)
            y = j * spacing * np.sqrt(3) / 2
            if np.hypot(x, y) < 0.9:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) >= n:
        # Sort by distance from origin, take closest n
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists)[:n]
        return points[idx]
    
    # Fallback: random
    return np.random.randn(n, 2) * 0.5


def construct():
    global _DEADLINE
    t_start = time.time()
    
    # Multiple restarts with different initializations
    best_centres = None
    best_radius = 0.0
    
    # Try several different initializations and dynamics parameters
    configs = [
        {'sigma': 0.25, 'epsilon': 1.0, 'r_target': 0.125, 'dt': 0.005, 'max_steps': 3000},
        {'sigma': 0.22, 'epsilon': 1.5, 'r_target': 0.126, 'dt': 0.004, 'max_steps': 3000},
        {'sigma': 0.28, 'epsilon': 0.8, 'r_target': 0.124, 'dt': 0.006, 'max_steps': 2500},
    ]
    
    for cfg_idx, cfg in enumerate(configs):
        if time.time() > _DEADLINE - 20.0:
            break
        
        # Different initializations for each config
        if cfg_idx == 0:
            init_centres = _hexagonal_init(N)
        elif cfg_idx == 1:
            init_centres = _hexagonal_init(N) + (np.random.randn(N, 2) - 0.5) * 0.05
        else:
            init_centres = _hexagonal_init(N) + (np.random.randn(N, 2) - 0.5) * 0.1
        
        # Run dynamics
        centres, radius = _anneal_dynamics(
            init_centres,
            max_steps=cfg['max_steps'],
            dt=cfg['dt'],
            sigma=cfg['sigma'],
            epsilon=cfg['epsilon'],
            r_target=cfg['r_target']
        )
        
        if radius > best_radius:
            best_radius = radius
            best_centres = centres.copy()
        
        # Local polish
        if time.time() < _DEADLINE - 15.0:
            polished = _local_optimize(centres)
            polished_radius = _compute_radius(polished)
            if polished_radius > best_radius:
                best_radius = polished_radius
                best_centres = polished
    
    # Final verification and safety
    if best_centres is None:
        # Fallback to simple hexagonal
        best_centres = _hexagonal_init(N)
        best_radius = _compute_radius(best_centres)
    
    # Ensure all circles are strictly inside
    dist_origin = np.hypot(best_centres[:, 0], best_centres[:, 1])
    max_dist = np.max(dist_origin)
    if max_dist > 0.999:
        scale = 0.999 / max_dist
        best_centres = best_centres * scale
    
    # Final radius check
    final_radius = _compute_radius(best_centres)
    if final_radius <= 0:
        # Emergency fallback
        return _fallback_hex()
    
    return best_centres


def _fallback_hex():
    """Simple hexagonal packing as fallback."""
    spacing = 0.24
    points = []
    for j in range(-8, 9):
        for i in range(-8, 9):
            x = i * spacing + 0.5 * spacing * (j % 2)
            y = j * spacing * np.sqrt(3) / 2
            if np.hypot(x, y) < 1.0 - spacing / 2:
                points.append([x, y])
    points = np.array(points)
    if len(points) >= N:
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists)[:N]
        return points[idx]
    return np.zeros((N, 2))
