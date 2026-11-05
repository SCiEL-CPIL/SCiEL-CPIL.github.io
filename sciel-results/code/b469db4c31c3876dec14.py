# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 315.0  # safety margin before 330s kill

def _get_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return _DEADLINE - (time.time() - _GLOBAL_START_TIME)

def _morse_energy(x, rho):
    """Compute Morse energy for a set of positions (vectorized)."""
    n = x.shape[0]
    if n < 2:
        return 0.0
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    er = rho * (1.0 - dist)
    er = np.clip(er, -500, 500)
    exp_er = np.exp(er)
    energy = np.sum(exp_er * (exp_er - 2.0)) / 2.0
    return float(energy)

def _min_distance(x):
    """Compute minimum distance between any pair of atoms (vectorized)."""
    n = x.shape[0]
    if n < 2:
        return np.inf
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    return float(np.min(dist))

def _is_valid(x):
    """Check if the configuration is valid (atoms at least 0.5 apart)."""
    return _min_distance(x) >= 0.5

def _icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron, normalized to unit length."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    return verts

def _generate_density_cluster(n, rho, seed=42):
    """
    Generate a cluster by sampling from a density field.
    For low rho: Uniform sphere.
    For high rho: Icosahedral density.
    """
    rng = np.random.default_rng(seed)
    
    # Estimate radius based on n
    # Volume ~ n * (1.0)^3 (approx)
    # R ~ (3n/4pi)^(1/3)
    R = 1.5 * (n / (4.0/3.0 * np.pi)) ** (1.0/3.0)
    
    if rho < 5.0:
        # Low rho: Spherical
        # Sample points in a sphere of radius R
        points = []
        while len(points) < n:
            # Sample from normal distribution and reject if outside R
            p = rng.normal(0, R/2, 3)
            if np.linalg.norm(p) <= R:
                points.append(p)
        points = np.array(points)
    else:
        # High rho: Icosahedral
        # Create a density field that is high near icosahedral shells
        verts = _icosahedron_vertices()
        
        # Target radii for shells: 1, 1.75, 2.5, ...
        # We will sample points and weight them by proximity to these shells
        
        points = []
        attempts = 0
        max_attempts = n * 100
        while len(points) < n and attempts < max_attempts:
            p = rng.uniform(-R, R, 3)
            if np.linalg.norm(p) > R:
                attempts += 1
                continue
            
            # Calculate distance to nearest icosahedral shell vertex
            # This is a simplified density model
            dist_to_center = np.linalg.norm(p)
            
            # We want peaks at radii corresponding to icosahedral shells
            # Shell 1: r=1, Shell 2: r~1.7, Shell 3: r~2.4
            # Density ~ exp(-alpha * (r - r_shell)^2)
            
            best_density = 0.0
            for shell_r in [1.0, 1.7, 2.4, 3.1]:
                if shell_r > R: continue
                d = abs(dist_to_center - shell_r)
                density = math.exp(-10.0 * d*d)
                if density > best_density:
                    best_density = density
            
            # Also add a small uniform component to allow filling gaps
            best_density += 0.05
            
            # Accept with probability proportional to density
            if rng.random() < best_density:
                points.append(p)
            attempts += 1
        
        if len(points) < n:
            # Fill remaining with random points in sphere
            while len(points) < n:
                p = rng.uniform(-R, R, 3)
                if np.linalg.norm(p) <= R:
                    points.append(p)
        points = np.array(points)
    
    return points

def _generate_fcc_cluster(n, seed=42):
    """Generate an FCC-like cluster."""
    rng = np.random.default_rng(seed)
    
    radius = 1.2 * (n ** (1.0/3.0)) + 1.0
    points = []
    
    a = 1.0 / np.sqrt(2.0)
    basis = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    
    max_coord = int(radius / a) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in basis:
                    p = a * np.array([i + dx, j + dy, k + dz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
    
    points.sort(key=lambda p: np.linalg.norm(p))
    
    if len(points) >= n:
        return np.array(points[:n])
    else:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                points.append(p)
            else:
                points.append(p * 0.5)
        return np.array(points)

def _generate_icosahedral_cluster(n, seed=42):
    """
    Generate an icosahedral cluster with concentric shells.
    """
    rng = np.random.default_rng(seed)
    positions = [np.array([0.0, 0.0, 0.0])]
    remaining = n - 1
    
    verts = _icosahedron_vertices()
    
    # Shell 1: 12 atoms at radius 1.0
    if remaining >= 12:
        positions.extend(verts * 1.0)
        remaining -= 12
    
    # Shell 2: 42 atoms at radius 1.75
    if remaining >= 1:
        n_shell2 = min(remaining, 42)
        # Approximate placement on second shell
        # Use golden spiral on sphere of radius 1.75
        extra = []
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for k in range(n_shell2):
            z = 1.0 - 2.0 * (k + 0.5) / n_shell2
            r = np.sqrt(1.0 - z*z)
            theta = golden_angle * k
            extra.append(np.array([r*np.cos(theta), r*np.sin(theta), z]) * 1.75)
        positions.extend(extra)
        remaining -= n_shell2
        
    # Additional shells
    shell_idx = 3
    while remaining > 0:
        shell_radius = 1.0 + (shell_idx - 1) * 0.75
        max_in_shell = min(remaining, 4 * shell_idx * (shell_idx + 1) + 2)
        extra = []
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for k in range(max_in_shell):
            z = 1.0 - 2.0 * (k + 0.5) / max_in_shell
            r = np.sqrt(1.0 - z*z)
            theta = golden_angle * k
            extra.append(np.array([r*np.cos(theta), r*np.sin(theta), z]) * shell_radius)
        positions.extend(extra)
        remaining -= max_in_shell
        shell_idx += 1
    
    return np.array(positions[:n])

def _relax_and_validate(positions, potential, rho):
    """Relax positions and check validity."""
    n = positions.shape[0]
    try:
        energy, relaxed = relax(positions, potential, rho)
        if _min_distance(relaxed) >= 0.5:
            return float(energy), relaxed
        else:
            return float('inf'), None
    except Exception:
        return float('inf'), None

def _basin_hop(start_positions, potential, rho, max_iterations=100, initial_step=0.5, initial_temp=3.0):
    """
    Perform basin hopping optimization.
    Returns (best_energy, best_positions).
    """
    n = start_positions.shape[0]
    current_energy, current_positions = _relax_and_validate(start_positions, potential, rho)
    
    if current_positions is None:
        # Try to fix initial invalidity
        perturbed = start_positions + np.random.normal(0, 0.1, start_positions.shape)
        current_energy, current_positions = _relax_and_validate(perturbed, potential, rho)
        if current_positions is None:
            return float('inf'), None
    
    best_energy = current_energy
    best_positions = current_positions.copy()
    
    step_size = initial_step
    temp = initial_temp
    
    for it in range(max_iterations):
        if _get_remaining_time() < 3.0:
            break
        
        noise = np.random.normal(0, step_size, current_positions.shape)
        new_positions = current_positions + noise
        
        new_energy, new_relaxed = _relax_and_validate(new_positions, potential, rho)
        
        if new_relaxed is None:
            # Retry with smaller step
            noise = np.random.normal(0, step_size * 0.5, current_positions.shape)
            new_positions = current_positions + noise
            new_energy, new_relaxed = _relax_and_validate(new_positions, potential, rho)
        
        if new_relaxed is None:
            continue
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or (temp > 1e-8 and np.random.rand() < math.exp(-delta_e / temp)):
            current_energy = new_energy
            current_positions = new_relaxed
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_positions.copy()
        
        temp *= 0.95
        step_size *= 0.99
        
        if it > 0 and it % 50 == 0:
            step_size *= 1.5
            temp *= 1.5

    return best_energy, best_positions

def construct(instance):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 6))
    potential = instance["potential"]
    
    if rho is None:
        rho = 6
    
    best_energy = float('inf')
    best_positions = None
    
    initial_guesses = []
    
    # 1. Density-based guesses (Structural Change)
    # Generate diverse starts based on density fields
    for seed in [42, 1, 2, 3, 4, 5]:
        try:
            initial_guesses.append(_generate_density_cluster(n, rho, seed=seed))
        except Exception:
            pass
            
    # 2. Lattice-based guesses (Fallback/Diversity)
    if rho < 5.0:
        for seed in [10, 11, 12]:
            try:
                initial_guesses.append(_generate_fcc_cluster(n, seed=seed))
            except Exception:
                pass
    else:
        for seed in [20, 21, 22]:
            try:
                initial_guesses.append(_generate_icosahedral_cluster(n, seed=seed))
            except Exception:
                pass
                
    # Relax all initial guesses
    relaxed_results = []
    for guess in initial_guesses:
        if _get_remaining_time() < 10.0:
            break
        
        energy, positions = _relax_and_validate(guess, potential, rho)
        if positions is not None and energy < float('inf'):
            relaxed_results.append((energy, positions))
    
    relaxed_results.sort(key=lambda x: x[0])
    
    if relaxed_results:
        best_energy = relaxed_results[0][0]
        best_positions = relaxed_results[0][1].copy()
    
    # Basin hopping from multiple starting points
    # Prioritize the best relaxed results
    num_bh_starts = min(len(relaxed_results), 6)
    
    if n <= 40:
        max_iter = 150
    elif n <= 60:
        max_iter = 100
    else:
        max_iter = 60
        
    if rho < 5.0:
        initial_step = 0.6
        initial_temp = 5.0
    else:
        initial_step = 0.4
        initial_temp = 3.0
        
    for idx in range(num_bh_starts):
        if _get_remaining_time() < 15.0:
            break
        
        start_energy, start_pos = relaxed_results[idx]
        
        bh_energy, bh_positions = _basin_hop(
            start_pos, potential, rho,
            max_iterations=max_iter,
            initial_step=initial_step,
            initial_temp=initial_temp
        )
        
        if bh_positions is not None and bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_positions.copy()
            
    # Safety: if no valid positions found, generate a fallback
    if best_positions is None or not _is_valid(best_positions):
        best_positions = _generate_density_cluster(n, rho, seed=99)
        if not _is_valid(best_positions):
            best_positions = _generate_fcc_cluster(n, seed=0)
        if not _is_valid(best_positions):
            # Last resort: random
            best_positions = np.random.uniform(-2, 2, (n, 3))
    
    return best_positions
# EVOLVE-BLOCK-END
