"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient


def _icosahedron_vertices(radius=1.0):
    """Generate icosahedron vertices."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts * (radius / norms[:, np.newaxis])
    return verts


def _octahedron_vertices(radius=1.0):
    """Generate octahedron vertices."""
    verts = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts * (radius / norms[:, np.newaxis])
    return verts


def _tetrahedron_vertices(radius=1.0):
    """Generate tetrahedron vertices."""
    verts = np.array([
        [1, 1, 1], [1, -1, -1],
        [-1, 1, -1], [-1, -1, 1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts * (radius / norms[:, np.newaxis])
    return verts


def _build_polytetrahedral_cluster(n, rng):
    """Build a cluster with polytetrahedral character using icosahedral-like packing."""
    # Start with an icosahedral core, then add atoms in shells
    core_radius = 1.1
    ico = _icosahedron_vertices(core_radius)
    
    points = [np.zeros(3)]  # center atom
    points.extend(list(ico))  # 12 icosahedron vertices = 13 total
    
    # Add more atoms in shells if needed
    if n > 13:
        # Second shell: octahedral positions
        octa = _octahedron_vertices(core_radius * 1.5)
        points.extend(list(octa))  # 6 more = 19 total
        
        # Third shell: more icosahedral-like
        if n > 19:
            ico2 = _icosahedron_vertices(core_radius * 2.0)
            points.extend(list(ico2))  # 12 more = 31 total
        
        # Continue adding in roughly spherical shells
        while len(points) < n:
            # Add atoms on a sphere with random perturbations
            shell_radius = core_radius * (1.0 + 0.5 * (len(points) - 13) / max(1, n - 13))
            for _ in range(10):
                if len(points) >= n:
                    break
                theta = rng.uniform(0, 2 * np.pi)
                phi = rng.uniform(0, np.pi)
                r = shell_radius * rng.uniform(0.95, 1.05)
                p = np.array([
                    r * np.sin(phi) * np.cos(theta),
                    r * np.sin(phi) * np.sin(theta),
                    r * np.cos(phi)
                ])
                # Check minimum distance
                if all(np.linalg.norm(p - q) >= 0.9 for q in points):
                    points.append(p)
            if len(points) < n:
                # If we're stuck, add with smaller minimum distance
                for _ in range(50):
                    if len(points) >= n:
                        break
                    theta = rng.uniform(0, 2 * np.pi)
                    phi = rng.uniform(0, np.pi)
                    r = shell_radius * rng.uniform(0.9, 1.1)
                    p = np.array([
                        r * np.sin(phi) * np.cos(theta),
                        r * np.sin(phi) * np.sin(theta),
                        r * np.cos(phi)
                    ])
                    if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                        points.append(p)
    
    points = points[:n]
    return np.array(points)


def _build_random_cluster(n, rng, spacing=1.1):
    """Build a random compact cluster."""
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                points.append(p)
    
    # If we didn't get enough points, relax the constraint
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
    
    return np.array(points[:n])


def _build_tetrahedral_cluster(n, rng):
    """Build a cluster with tetrahedral character."""
    tet = _tetrahedron_vertices(1.5)
    points = [np.zeros(3)]
    points.extend(list(tet))
    
    while len(points) < n:
        # Add atoms near existing tetrahedral vertices
        base = points[rng.integers(0, len(points))]
        p = base + rng.normal(0, 0.5, 3)
        if all(np.linalg.norm(p - q) >= 0.8 for q in points):
            points.append(p)
        if len(points) >= n:
            break
    
    # Fill remaining with random
    while len(points) < n:
        p = rng.uniform(-3, 3, 3)
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
    
    return np.array(points[:n])


def construct(instance):
    """Find low-energy Dzugutov cluster using polytetrahedral-inspired starts."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    time_limit = 290.0  # Safety margin before 300s limit
    
    rng = np.random.default_rng(size)
    
    # Generate diverse starting configurations
    candidates = []
    
    # 1. Polytetrahedral-inspired (icosahedral core)
    candidates.append(_build_polytetrahedral_cluster(size, rng))
    
    # 2. Tetrahedral-inspired
    candidates.append(_build_tetrahedral_cluster(size, rng))
    
    # 3-6. Random compact clusters with different seeds
    for seed_offset in range(4):
        rng_local = np.random.default_rng(size + 1000 + seed_offset)
        candidates.append(_build_random_cluster(size, rng_local))
    
    best_energy = np.inf
    best_positions = None
    
    for i, pos in enumerate(candidates):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Relax locally
        try:
            energy, positions = relax(pos, potential, parameter)
        except Exception:
            continue
        
        if energy < best_energy:
            best_energy = energy
            best_positions = positions
    
    # If we have time, do a few more random restarts
    if best_positions is not None and time.time() - start_time < time_limit - 20:
        for restart in range(5):
            if time.time() - start_time > time_limit - 10:
                break
            
            rng_restart = np.random.default_rng(size + 5000 + restart)
            pos = _build_random_cluster(size, rng_restart)
            
            try:
                energy, positions = relax(pos, potential, parameter)
            except Exception:
                continue
            
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
    
    if best_positions is None:
        # Fallback: return a simple linear arrangement
        spacing = 0.6
        positions = np.array([[i * spacing, 0, 0] for i in range(size)])
        return positions
    
    return best_positions
# EVOLVE-BLOCK-END
