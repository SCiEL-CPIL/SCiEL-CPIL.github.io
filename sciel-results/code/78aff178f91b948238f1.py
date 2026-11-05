"""New program using structured seeds + diverse perturbations for Morse cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(positions, rho):
    """Compute exact Morse energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    # Only compute upper triangle
    iu = np.triu_indices(n, 1)
    r = dist[iu]
    val = rho * (1.0 - r)
    exp_val = np.exp(val)
    energy = np.sum(exp_val * (exp_val - 2.0))
    return float(energy)


def _icosahedron_vertices():
    """Return vertices of a regular icosahedron centered at origin, edge length 1."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    verts /= 2.0
    return verts


def _make_icosahedral_cluster(n):
    """Create an icosahedral-like cluster of n atoms."""
    # Start with icosahedron core (12 atoms)
    verts = _icosahedron_vertices()
    points = list(verts)
    
    # Add layers around
    while len(points) < n:
        # Add atoms on the surface of a sphere with increasing radius
        shell_radius = 1.0 + 0.3 * (len(points) // 12)
        theta = np.random.uniform(0, 2 * np.pi)
        phi_angle = np.random.uniform(0, np.pi)
        p = shell_radius * np.array([
            np.sin(phi_angle) * np.cos(theta),
            np.sin(phi_angle) * np.sin(theta),
            np.cos(phi_angle)
        ])
        # Check minimum distance
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
    
    return np.array(points[:n])


def _make_decahedral_cluster(n):
    """Create a decahedral-like cluster (FCC-based)."""
    # FCC lattice points
    points = []
    # Generate FCC points in a cube
    a = 1.0  # lattice parameter
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                # FCC: corners and face centers
                # Corner
                p = a * np.array([i, j, k])
                points.append(p)
                # Face centers
                if i < 3 and j < 3:
                    points.append(a * np.array([i + 0.5, j + 0.5, k]))
                if i < 3 and k < 3:
                    points.append(a * np.array([i + 0.5, j, k + 0.5]))
                if j < 3 and k < 3:
                    points.append(a * np.array([i, j + 0.5, k + 0.5]))
    
    # Remove duplicates and keep closest to origin
    points = np.array(points)
    # Remove duplicates
    unique_points = []
    for p in points:
        if all(np.linalg.norm(p - q) > 0.1 for q in unique_points):
            unique_points.append(p)
    unique_points = np.array(unique_points)
    
    # Sort by distance from origin
    dists = np.linalg.norm(unique_points, axis=1)
    idx = np.argsort(dists)
    unique_points = unique_points[idx]
    
    # Scale to get approximately n atoms with reasonable spacing
    if len(unique_points) >= n:
        return unique_points[:n]
    else:
        # Pad with random points
        while len(unique_points) < n:
            p = np.random.uniform(-3, 3, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in unique_points):
                unique_points = np.vstack([unique_points, [p]])
        return unique_points[:n]


def _make_fcc_fragment(n):
    """Create an FCC fragment of n atoms."""
    points = []
    a = 1.0
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                # FCC lattice
                p = a * np.array([i, j, k])
                points.append(p)
                if i < 4 and j < 4:
                    points.append(a * np.array([i + 0.5, j + 0.5, k]))
                if i < 4 and k < 4:
                    points.append(a * np.array([i + 0.5, j, k + 0.5]))
                if j < 4 and k < 4:
                    points.append(a * np.array([i, j + 0.5, k + 0.5]))
    
    points = np.array(points)
    # Remove duplicates
    unique_points = []
    for p in points:
        if all(np.linalg.norm(p - q) > 0.1 for q in unique_points):
            unique_points.append(p)
    unique_points = np.array(unique_points)
    
    # Sort by distance from origin
    dists = np.linalg.norm(unique_points, axis=1)
    idx = np.argsort(dists)
    unique_points = unique_points[idx]
    
    if len(unique_points) >= n:
        return unique_points[:n]
    else:
        while len(unique_points) < n:
            p = np.random.uniform(-4, 4, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in unique_points):
                unique_points = np.vstack([unique_points, [p]])
        return unique_points[:n]


def _random_cluster(n, seed=None):
    """Create a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = n ** (1.0 / 3.0) * 0.7
    points = []
    for _ in range(n):
        for _ in range(100):
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
                break
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _perturb_cluster(positions, magnitude=0.1, seed=None):
    """Perturb a cluster slightly."""
    rng = np.random.default_rng(seed)
    perturbed = positions + rng.uniform(-magnitude, magnitude, positions.shape)
    return perturbed


def construct(instance):
    """Find the lowest-energy arrangement using diverse seeds and local relaxation."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # Generate diverse initial clusters
    seeds = []
    
    # Icosahedral
    try:
        seeds.append(_make_icosahedral_cluster(size))
    except:
        pass
    
    # Decahedral
    try:
        seeds.append(_make_decahedral_cluster(size))
    except:
        pass
    
    # FCC
    try:
        seeds.append(_make_fcc_fragment(size))
    except:
        pass
    
    # Random clusters with different seeds
    for seed in [0, 1, 2, 3, 4, 5]:
        try:
            seeds.append(_random_cluster(size, seed=seed))
        except:
            pass
    
    # Perturbed versions of good seeds
    if len(seeds) > 0:
        for i in range(min(3, len(seeds))):
            for perturb_seed in [100, 200, 300]:
                try:
                    seeds.append(_perturb_cluster(seeds[i], magnitude=0.15, seed=perturb_seed))
                except:
                    pass
    
    # Relax each seed and keep the best
    best_energy = np.inf
    best_positions = None
    
    for idx, seed in enumerate(seeds):
        if time.time() > deadline:
            break
        
        try:
            energy, positions = relax(seed, potential, rho)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except:
            continue
    
    # If we have time, do more random restarts
    if best_positions is not None and time.time() < deadline - 30:
        rng = np.random.default_rng(42)
        for restart in range(20):
            if time.time() > deadline:
                break
            try:
                seed = _random_cluster(size, seed=1000 + restart)
                energy, positions = relax(seed, potential, rho)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            except:
                continue
    
    if best_positions is None:
        # Fallback: return a valid random cluster
        best_positions = _random_cluster(size, seed=0)
    
    return best_positions
# EVOLVE-BLOCK-END
