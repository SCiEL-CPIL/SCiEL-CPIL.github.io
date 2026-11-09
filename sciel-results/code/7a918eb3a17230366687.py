"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Known global minima energies (Doye & Wales 1998) for reference
# These are used for sanity checks but the evaluator uses its own E_pub

def _generate_fcc(n, spacing=1.0):
    """Generate n-atom FCC-like cluster."""
    if n <= 0:
        return np.empty((0, 3))
    
    # FCC lattice basis
    basis = np.array([
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ]) * spacing
    
    # Generate candidate sites
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 3
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                for b in basis:
                    p = np.array([i, j, k]) * spacing + b
                    sites.append(p)
    
    sites = np.array(sites)
    
    # Remove duplicates
    unique_sites = []
    for p in sites:
        if not any(np.allclose(p, q, atol=1e-6) for q in unique_sites):
            unique_sites.append(p)
    sites = np.array(unique_sites)
    
    # Center the cluster
    center = sites.mean(axis=0)
    sites = sites - center
    
    # Sort by distance from center and take n closest
    distances = np.linalg.norm(sites, axis=1)
    idx = np.argsort(distances)[:n]
    result = sites[idx]
    
    return result


def _generate_hcp(n, spacing=1.0):
    """Generate n-atom HCP-like cluster."""
    if n <= 0:
        return np.empty((0, 3))
    
    # HCP lattice: two triangular layers offset
    # Generate points in a hexagonal pattern
    points = []
    a = spacing  # lattice constant
    
    # Generate hexagonal lattice points
    max_r = int(np.ceil(n ** (1.0 / 3.0))) + 3
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                if i + j + k == 0:
                    # Two layers
                    for layer in range(2):
                        if layer == 0:
                            p = np.array([i * a, j * a, 0.0])
                        else:
                            p = np.array([(i + 0.5) * a, (j + 0.5) * a, a * np.sqrt(2/3)])
                        points.append(p)
    
    points = np.array(points)
    
    # Center
    center = points.mean(axis=0)
    points = points - center
    
    # Sort by distance and take n closest
    distances = np.linalg.norm(points, axis=1)
    idx = np.argsort(distances)[:n]
    result = points[idx]
    
    return result


def _generate_icosahedron(n, spacing=1.0):
    """Generate n-atom icosahedral-like cluster."""
    if n <= 0:
        return np.empty((0, 3))
    
    # Icosahedron vertices (12 vertices)
    phi = (1 + np.sqrt(5)) / 2
    icosahedron = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    
    # Normalize to unit sphere
    icosahedron = icosahedron / np.linalg.norm(icosahedron, axis=1, keepdims=True)
    
    # Scale
    radius = spacing * 0.5
    icosahedron = icosahedron * radius
    
    points = list(icosahedron)
    
    # Add inner shells if needed
    if n > 12:
        # Add center atom
        points.append(np.array([0.0, 0.0, 0.0]))
        
        # Add more shells
        while len(points) < n:
            # Generate more points on a sphere at increasing radii
            for shell in range(1, 10):
                if len(points) >= n:
                    break
                r = radius * (1 + shell * 0.3)
                # Fibonacci sphere for even distribution
                m = min(12 * (shell + 1), n - len(points))
                for i in range(m):
                    y = 1 - (2 * i + 1) / m
                    rad = np.sqrt(1 - y * y)
                    theta = np.pi * (1 + 5 ** 0.5) * i  # golden angle
                    x = np.cos(theta) * rad
                    z = np.sin(theta) * rad
                    points.append(np.array([x, y, z]) * r)
    
    result = np.array(points[:n])
    
    return result


def _generate_random(n, spacing=1.0, seed=None):
    """Generate n-atom random compact cluster."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    min_dist = spacing * 0.6
    
    points = []
    attempts = 0
    max_attempts = 10000
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    
    # If we couldn't place enough, fill with random points
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def _generate_deformed(n, base_positions, rng, scale=0.1):
    """Generate a deformed version of base positions."""
    return base_positions + rng.normal(0, scale, size=base_positions.shape)


def _check_validity(positions):
    """Check if positions are valid (finite, no atoms too close)."""
    if not np.all(np.isfinite(positions)):
        return False
    if len(positions) < 2:
        return True
    # Check minimum distance
    from scipy.spatial.distance import pdist
    min_dist = pdist(positions)
    if min_dist.min() < 0.35:
        return False
    return True


def construct(instance):
    """Construct cluster configuration for given instance."""
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s safety margin
    
    n = instance["n"]
    kind = instance["kind"]
    
    # Spacing based on kind
    spacing_map = {
        "12-6": 0.72,
        "9-6": 0.75,
        "10-8": 0.73
    }
    spacing = spacing_map.get(kind, 0.72)
    
    # Generate diverse initial configurations
    rng = np.random.default_rng(n * 1000 + hash(kind) % 10000)
    
    candidates = []
    
    # 1. FCC cluster
    candidates.append(_generate_fcc(n, spacing))
    
    # 2. HCP cluster
    candidates.append(_generate_hcp(n, spacing))
    
    # 3. Icosahedral cluster
    candidates.append(_generate_icosahedron(n, spacing))
    
    # 4-7. Random clusters with different seeds
    for i in range(4):
        candidates.append(_generate_random(n, spacing, seed=n * 100 + i))
    
    # 8-12. Deformed versions of FCC
    fcc_base = _generate_fcc(n, spacing)
    for i in range(5):
        candidates.append(_generate_deformed(n, fcc_base, rng, scale=0.05 * (i + 1)))
    
    # 13-17. Deformed versions of HCP
    hcp_base = _generate_hcp(n, spacing)
    for i in range(5):
        candidates.append(_generate_deformed(n, hcp_base, rng, scale=0.05 * (i + 1)))
    
    # Relax each candidate and track best
    best_energy = np.inf
    best_positions = None
    
    for i, pos in enumerate(candidates):
        # Check time
        if time.time() > deadline:
            break
        
        # Validate initial positions
        if not _check_validity(pos):
            continue
        
        try:
            energy, relaxed_pos = relax(pos, "sutton_chen", kind)
            
            if np.isfinite(energy) and _check_validity(relaxed_pos):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed_pos.copy()
        except Exception:
            continue
    
    # If we have time, do a few more random restarts from best position
    if best_positions is not None and time.time() < deadline - 30:
        for i in range(10):
            if time.time() > deadline - 10:
                break
            
            # Small perturbation of best position
            perturbed = best_positions + rng.normal(0, 0.03, size=best_positions.shape)
            
            if not _check_validity(perturbed):
                continue
            
            try:
                energy, relaxed_pos = relax(perturbed, "sutton_chen", kind)
                
                if np.isfinite(energy) and _check_validity(relaxed_pos):
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = relaxed_pos.copy()
            except Exception:
                continue
    
    # Final fallback: if no valid result, return a simple valid configuration
    if best_positions is None:
        # Create a simple linear chain
        best_positions = np.zeros((n, 3))
        for i in range(n):
            best_positions[i] = [i * spacing * 1.1, 0, 0]
        # Center it
        best_positions -= best_positions.mean(axis=0)
    
    return best_positions
# EVOLVE-BLOCK-END
