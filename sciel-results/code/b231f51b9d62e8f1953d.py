"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
TIME_LIMIT = 285.0  # Safety margin before 300s limit


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def _generate_icosahedron(n, scale=1.0):
    """Generate icosahedral structure with n atoms for Dzugutov potential."""
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of regular icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms * scale
    
    if n <= 12:
        return verts[:n].copy()
    
    # Add inner shells for larger n
    points = [verts]
    current_count = 12
    
    # Add inner icosahedron (scaled down)
    if n > 12:
        inner_scale = scale * 0.55
        inner_verts = verts.copy()
        inner_verts = inner_verts / norms * inner_scale
        points.append(inner_verts)
        current_count += 12
    
    # Add center atom
    if n > 24:
        points.append(np.array([[0.0, 0.0, 0.0]]))
        current_count += 1
    
    # Add more shells if needed
    shell_scales = [0.75, 0.45, 0.30, 0.15]
    for s in shell_scales:
        if current_count >= n:
            break
        shell_verts = verts.copy()
        shell_verts = shell_verts / norms * (scale * s)
        points.append(shell_verts)
        current_count += 12
    
    all_points = np.vstack(points)
    
    # If we have more than n, trim; if less, add random points
    if len(all_points) >= n:
        return all_points[:n].copy()
    
    # Add random points in shell
    rng = np.random.default_rng(42)
    while len(all_points) < n:
        theta = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-1, 1)
        r = np.sqrt(1 - z**2)
        p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * scale * 0.9
        all_points = np.vstack([all_points, p])
    
    return all_points[:n].copy()


def _generate_bcc_cluster(n, scale=1.0):
    """Generate BCC-like cluster structure."""
    rng = np.random.default_rng(123)
    points = []
    
    # Center
    points.append([0.0, 0.0, 0.0])
    
    # Add atoms in shells
    shell_radii = [scale * 0.5, scale * 0.8, scale * 1.0, scale * 1.2, scale * 1.4]
    for radius in shell_radii:
        if len(points) >= n:
            break
        # Add atoms on sphere surface
        theta = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-1, 1)
        r = np.sqrt(1 - z**2)
        p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * radius
        points.append(p)
    
    # Fill remaining with random points
    while len(points) < n:
        p = rng.uniform(-scale, scale, 3)
        if np.linalg.norm(p) <= scale * 1.1:
            points.append(p)
    
    return np.array(points[:n])


def _random_compact(n, seed, spacing=1.1, radius_factor=0.75):
    """Generate a random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * radius_factor
    points = []
    min_dist = 0.8 * spacing
    attempts = 0
    max_attempts = n * 500
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    
    # If we couldn't place all atoms with strict constraint, relax it
    if len(points) < n:
        min_dist = 0.5
        attempts = 0
        while len(points) < n and attempts < max_attempts:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= min_dist for q in points):
                    points.append(p)
            attempts += 1
    
    # Last resort: just add random points
    while len(points) < n:
        p = rng.uniform(-radius * 0.5, radius * 0.5, 3)
        points.append(p)
    
    return np.array(points[:n])


def construct(instance):
    """Construct low-energy Dzugutov cluster using structured initializations and relaxation."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    # Strategy 1: Icosahedral structure (primary for Dzugutov)
    ico_scale = 1.0 * size ** (1.0 / 3.0) * 0.4
    ico_points = _generate_icosahedron(size, scale=ico_scale)
    energy, positions = relax(ico_points, potential, parameter)
    if energy < best_energy:
        best_energy = energy
        best_positions = positions
    
    if _time_remaining() < 40:
        return best_positions
    
    # Strategy 2: Scaled icosahedron with different scale
    ico_scale2 = 1.0 * size ** (1.0 / 3.0) * 0.35
    ico_points2 = _generate_icosahedron(size, scale=ico_scale2)
    energy, positions = relax(ico_points2, potential, parameter)
    if energy < best_energy:
        best_energy = energy
        best_positions = positions
    
    if _time_remaining() < 40:
        return best_positions
    
    # Strategy 3: BCC-like structure
    bcc_scale = 1.0 * size ** (1.0 / 3.0) * 0.45
    bcc_points = _generate_bcc_cluster(size, scale=bcc_scale)
    energy, positions = relax(bcc_points, potential, parameter)
    if energy < best_energy:
        best_energy = energy
        best_positions = positions
    
    if _time_remaining() < 40:
        return best_positions
    
    # Strategy 4: Random compact with minimum distance (inspiration approach)
    rc_points = _random_compact(size, seed=size, spacing=1.1, radius_factor=0.75)
    energy, positions = relax(rc_points, potential, parameter)
    if energy < best_energy:
        best_energy = energy
        best_positions = positions
    
    if _time_remaining() < 40:
        return best_positions
    
    # Strategy 5: Perturbed icosahedron
    ico_points3 = _generate_icosahedron(size, scale=ico_scale)
    rng2 = np.random.default_rng(size + 1)
    perturbation = rng2.normal(0, 0.1, ico_points3.shape)
    ico_points3 = ico_points3 + perturbation
    energy, positions = relax(ico_points3, potential, parameter)
    if energy < best_energy:
        best_energy = energy
        best_positions = positions
    
    if _time_remaining() < 40:
        return best_positions
    
    # Strategy 6: Second random compact with different seed
    rc_points2 = _random_compact(size, seed=size + 100, spacing=1.1, radius_factor=0.75)
    energy, positions = relax(rc_points2, potential, parameter)
    if energy < best_energy:
        best_energy = energy
        best_positions = positions
    
    if _time_remaining() < 40:
        return best_positions
    
    # Strategy 7: Third random compact with different parameters
    rc_points3 = _random_compact(size, seed=size + 200, spacing=1.0, radius_factor=0.8)
    energy, positions = relax(rc_points3, potential, parameter)
    if energy < best_energy:
        best_energy = energy
        best_positions = positions
    
    if _time_remaining() < 40:
        return best_positions
    
    # Strategy 8: Icosahedron with larger scale
    ico_scale3 = 1.0 * size ** (1.0 / 3.0) * 0.45
    ico_points4 = _generate_icosahedron(size, scale=ico_scale3)
    energy, positions = relax(ico_points4, potential, parameter)
    if energy < best_energy:
        best_energy = energy
        best_positions = positions
    
    return best_positions
# EVOLVE-BLOCK-END
