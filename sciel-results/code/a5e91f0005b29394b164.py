"""
Strategy: Deterministic Multi-Start Annealing (DMSA) with Structured Crystalline Initialization.
Instead of pure gradient descent (L-BFGS-B), we use a combination of:
1. Diverse structured initializations (FCC, BCC, Icosahedral, Spherical shells)
2. Simulated Annealing as the global optimizer (deterministic temperature schedule)
3. L-BFGS-B for local refinement after each annealing step
This approach systematically explores the configuration space while maintaining
reproducibility and efficient local optimization.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 290.0  # 300s limit, keep 10s safety margin


def _morse_energy_and_grad(positions, rho):
    """Compute exact Morse energy and gradient."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    
    x = rho * (1.0 - r)
    ex = np.exp(x)
    energy = 0.5 * float(np.sum(ex * (ex - 2.0)))
    
    dV_dr = -rho * 2.0 * ex * (ex - 1.0)
    
    inv_r = np.zeros_like(r)
    mask = r > 1e-12
    inv_r[mask] = 1.0 / r[mask]
    
    factor = -dV_dr * inv_r
    row_sum = np.sum(factor, axis=1)
    weighted_pos = factor.dot(positions)
    grad = positions * row_sum[:, np.newaxis] - weighted_pos
    
    return energy, grad


def _local_minimize(positions, rho, maxiter=200):
    """Perform local minimization using L-BFGS-B."""
    def fun(x):
        x = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(x, rho)
        return e, g.flatten()
    
    try:
        res = minimize(fun, positions.flatten(), method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-7})
        return res.x.reshape(-1, 3), res.fun
    except Exception:
        return positions, _morse_energy_and_grad(positions, rho)[0]


def _generate_fcc(n, rng):
    """Generate n atoms on an FCC lattice, cropped to a sphere."""
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
    points = points[idx[:n]]
    return points


def _generate_bcc(n, rng):
    """Generate n atoms on a BCC lattice, cropped to a sphere."""
    a = 2.0 / np.sqrt(3.0)
    points = []
    max_coord = int(np.ceil(n ** (1/3) * 1.5)) + 2
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
    points = points[idx[:n]]
    return points


def _generate_icosahedron(n, rng):
    """Generate n atoms in an icosahedral-like structure with shells."""
    if n <= 1:
        return np.zeros((n, 3))
    
    points = [np.zeros(3)]
    
    # Icosahedron vertices (12 vertices)
    phi = (1 + np.sqrt(5)) / 2
    ico_vertices = np.array([
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico_vertices = ico_vertices / np.linalg.norm(ico_vertices, axis=1, keepdims=True)
    
    # Add shells
    shell_radii = [1.0, 1.8, 2.6, 3.4, 4.2]
    for radius in shell_radii:
        if len(points) >= n:
            break
        
        # Add points on shell
        for v in ico_vertices:
            if len(points) >= n:
                break
            p = v * radius
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        
        # Add more points if needed
        while len(points) < n:
            theta = rng.uniform(0, 2 * np.pi)
            cos_theta = rng.uniform(-1, 1)
            sin_theta = np.sqrt(1 - cos_theta**2)
            p = np.array([
                radius * sin_theta * np.cos(theta),
                radius * sin_theta * np.sin(theta),
                radius * cos_theta
            ])
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
            else:
                break
    
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points


def _generate_shell(n, rng):
    """Generate n atoms on concentric spherical shells."""
    if n <= 1:
        return np.zeros((n, 3))
    
    points = [np.zeros(3)]
    
    # Calculate shell radii
    shell_count = int(np.ceil(n ** (1/3) / 2))
    shell_radii = np.linspace(1.0, n ** (1/3) * 1.2, shell_count)
    
    for radius in shell_radii:
        if len(points) >= n:
            break
        
        # Fibonacci sphere for uniform distribution
        num_points_on_shell = min(int(4 * np.pi * radius**2 / 0.5**2), n - len(points))
        for i in range(num_points_on_shell):
            if len(points) >= n:
                break
            
            # Fibonacci sphere
            golden_ratio = (1 + np.sqrt(5)) / 2
            theta = 2 * np.pi * i / golden_ratio
            cos_theta = 1 - 2 * (i + 0.5) / num_points_on_shell
            sin_theta = np.sqrt(1 - cos_theta**2)
            
            p = np.array([
                radius * sin_theta * np.cos(theta),
                radius * sin_theta * np.sin(theta),
                radius * cos_theta
            ])
            
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
    
    # Fill remaining with random points
    while len(points) < n:
        radius = rng.uniform(0.5, n ** (1/3) * 1.5)
        theta = rng.uniform(0, 2 * np.pi)
        cos_theta = rng.uniform(-1, 1)
        sin_theta = np.sqrt(1 - cos_theta**2)
        p = np.array([
            radius * sin_theta * np.cos(theta),
            radius * sin_theta * np.sin(theta),
            radius * cos_theta
        ])
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
        else:
            break
    
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points


def _simulated_annealing(positions, rho, n_iterations=100, T_initial=1.0, T_final=0.01, rng=None):
    """
    Simulated Annealing: deterministic temperature schedule with Metropolis acceptance.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    pos, energy = _local_minimize(positions, rho)
    best_pos = pos.copy()
    best_energy = energy
    
    for it in range(n_iterations):
        # Check time
        if time.time() - _START_TIME > _TIME_LIMIT - 20:
            break
        
        # Temperature schedule (exponential cooling)
        T = T_initial * (T_final / T_initial) ** (it / n_iterations)
        
        # Perturb
        perturbation = 0.3 * np.sqrt(T / T_initial)
        new_pos = pos + rng.normal(0, perturbation, pos.shape)
        
        # Local minimize from perturbed position
        new_pos, new_energy = _local_minimize(new_pos, rho)
        
        # Metropolis acceptance
        if new_energy < energy:
            pos, energy = new_pos, new_energy
        else:
            delta = new_energy - energy
            if rng.random() < np.exp(-delta / T):
                pos, energy = new_pos, new_energy
        
        if energy < best_energy:
            best_pos = pos.copy()
            best_energy = energy
    
    return best_pos, best_energy


def _is_valid(positions, min_dist=0.5):
    """Check if all pairwise distances are >= min_dist."""
    if len(positions) < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    return np.all(r >= min_dist)


def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial configurations
    configs = []
    
    fcc = _generate_fcc(size, rng)
    configs.append(fcc)
    
    bcc = _generate_bcc(size, rng)
    configs.append(bcc)
    
    ico = _generate_icosahedron(size, rng)
    configs.append(ico)
    
    shell = _generate_shell(size, rng)
    configs.append(shell)
    
    best_energy = float('inf')
    best_positions = None
    
    # Allocate time budget: ~40s per instance, 6 instances = 240s
    # Use simulated annealing from each initial config
    time_per_config = 30.0  # seconds per initial config
    
    for i, pos in enumerate(configs):
        if time.time() - _START_TIME > _TIME_LIMIT - 30:
            break
        
        # Do simulated annealing from this initial config
        sa_pos, sa_energy = _simulated_annealing(pos, parameter, n_iterations=50, 
                                                  T_initial=1.0, T_final=0.01, rng=rng)
        
        if _is_valid(sa_pos):
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_positions = sa_pos
    
    # If we have time left, do more annealing from the best
    if best_positions is not None and time.time() - _START_TIME < _TIME_LIMIT - 40:
        # Additional annealing from best position
        extra_pos, extra_energy = _simulated_annealing(best_positions, parameter, 
                                                        n_iterations=30, T_initial=0.5, 
                                                        T_final=0.005, rng=rng)
        if _is_valid(extra_pos) and extra_energy < best_energy:
            best_energy = extra_energy
            best_positions = extra_pos
    
    # Fallback if no valid position found
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
