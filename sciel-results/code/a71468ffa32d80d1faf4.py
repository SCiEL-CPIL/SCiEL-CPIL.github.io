"""
Strategy: Two-Phase Basin Hopping with Crystalline and Random Initialization.
Phase 1: Broad exploration from diverse seeds (FCC, BCC, Icosahedral, Random).
Phase 2: Focused polishing from the best candidate found in Phase 1.
Final Step: High-precision L-BFGS polish on the best candidate.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 325.0  # 330s limit, keep 5s safety margin


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


def _local_minimize(positions, rho, maxiter=500, ftol=1e-12, gtol=1e-9):
    """Perform local minimization using L-BFGS-B."""
    def fun(x):
        x = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(x, rho)
        return e, g.flatten()
    
    try:
        res = minimize(fun, positions.flatten(), method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
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


def _generate_random(n, rng):
    """Generate n atoms randomly in a sphere, ensuring min distance."""
    radius = 1.0 * n ** (1.0 / 3.0) * 1.5
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 500:
        p = rng.uniform(-radius, radius, 3)
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
        attempts += 1
    
    # If we didn't get enough, fill with a lattice
    while len(points) < n:
        p = np.array([rng.uniform(-radius, radius), rng.uniform(-radius, radius), rng.uniform(-radius, radius)])
        points.append(p)
        
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points


def _is_valid(positions, min_dist=0.5):
    """Check if all pairwise distances are >= min_dist."""
    if len(positions) < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    return np.all(r >= min_dist)


def _basin_hopping(positions, rho, n_iterations, rng, T_start=1.0, T_end=0.1, perturbation_scale=1.0):
    """
    Basin Hopping with simulated annealing acceptance.
    """
    pos, energy = _local_minimize(positions, rho)
    best_pos = pos.copy()
    best_energy = energy
    no_improve_count = 0
    
    for it in range(n_iterations):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        
        # Temperature schedule
        T = T_start * (T_end / T_start) ** (it / max(1, n_iterations - 1))
        
        # Perturbation
        scale = perturbation_scale * (0.5 + 0.5 * (T / T_start))
        new_pos = pos + rng.normal(0, scale, pos.shape)
        
        # Local minimize
        new_pos, new_energy = _local_minimize(new_pos, rho)
        
        # Metropolis acceptance
        if new_energy < energy:
            pos, energy = new_pos, new_energy
            no_improve_count = 0
        else:
            delta = new_energy - energy
            if rng.random() < np.exp(-delta / T):
                pos, energy = new_pos, new_energy
                no_improve_count += 1
            else:
                no_improve_count += 1
        
        # Track best
        if energy < best_energy:
            best_pos = pos.copy()
            best_energy = energy
        
        # Reset strategy: if stuck for 20 steps, reset to a random config
        if no_improve_count > 20:
            # Reset to a random perturbation of the best so far
            pos = best_pos + rng.normal(0, 0.5, best_pos.shape)
            pos, energy = _local_minimize(pos, rho)
            no_improve_count = 0
    
    return best_pos, best_energy


def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial configurations
    configs = []
    configs.append(_generate_fcc(size, rng))
    configs.append(_generate_bcc(size, rng))
    configs.append(_generate_icosahedron(size, rng))
    configs.append(_generate_random(size, rng))
    
    best_energy = float('inf')
    best_positions = None
    
    # Phase 1: Broad Exploration
    # Time budget: ~40s per instance
    # Allocate ~25s for Phase 1, ~15s for Phase 2
    phase1_time_limit = 25.0
    phase2_time_limit = 15.0
    
    # Determine iterations based on time
    # Estimate time per iteration: ~0.05s for n=80, ~0.01s for n=38
    est_iter_time = 0.01 * (size / 38.0)
    n_iter_phase1 = int(phase1_time_limit / est_iter_time)
    n_iter_phase1 = max(50, min(500, n_iter_phase1))
    
    # Run BH from each config
    for i, pos in enumerate(configs):
        if time.time() - _START_TIME > _TIME_LIMIT - (phase2_time_limit + 10):
            break
        
        # Use a subset of iterations for each config to ensure diversity
        iters_for_config = max(10, n_iter_phase1 // len(configs))
        
        bh_pos, bh_energy = _basin_hopping(pos, parameter, iters_for_config, rng, 
                                            T_start=1.0, T_end=0.5, perturbation_scale=1.0)
        
        if _is_valid(bh_pos):
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_pos
    
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
        best_energy = _morse_energy_and_grad(best_positions, parameter)[0]
    
    # Phase 2: Focused Polishing
    if time.time() - _START_TIME < _TIME_LIMIT - 5:
        # Run BH from the best position with smaller perturbations
        n_iter_phase2 = max(50, min(300, int(phase2_time_limit / est_iter_time)))
        
        bh_pos, bh_energy = _basin_hopping(best_positions, parameter, n_iter_phase2, rng, 
                                            T_start=0.5, T_end=0.1, perturbation_scale=0.5)
        
        if _is_valid(bh_pos) and bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_pos
    
    # Final Polish: High-precision L-BFGS
    if time.time() - _START_TIME < _TIME_LIMIT - 1:
        final_pos, final_energy = _local_minimize(best_positions, parameter, maxiter=1000, ftol=1e-15, gtol=1e-12)
        if _is_valid(final_pos) and final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_pos
    
    return best_positions
# EVOLVE-BLOCK-END
