"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _icosahedron_vertices():
    """Return normalized icosahedron vertices (12 points)."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ], dtype=float)
    return verts / np.linalg.norm(verts, axis=1, keepdims=True)


def _icosahedron_edges():
    """Return edge directions (20 faces * 3 edges, normalized)."""
    verts = _icosahedron_vertices()
    # Find all pairs that are edges (shortest distance)
    dists = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            dists.append((d, i, j))
    dists.sort()
    # The 30 edges are the shortest 30 distances
    edge_pairs = [(dists[k][1], dists[k][2]) for k in range(30)]
    # For each edge, compute the midpoint direction from center
    edge_dirs = []
    for i, j in edge_pairs:
        mid = (verts[i] + verts[j]) / 2.0
        edge_dirs.append(mid / np.linalg.norm(mid))
    return np.array(edge_dirs)


def _icosahedron_face_centers():
    """Return 20 face center directions (normalized)."""
    verts = _icosahedron_vertices()
    # 20 faces of icosahedron - use known face normals
    # Each face is a triangle; the face center direction is the normalized sum of 3 vertex normals
    # Standard icosahedron faces (indices into the 12 vertices):
    faces = [
        [0, 1, 4], [0, 4, 8], [0, 8, 10], [0, 10, 6], [0, 6, 1],
        [1, 6, 5], [1, 5, 7], [1, 7, 2], [1, 2, 0],
        [2, 7, 5], [2, 5, 9], [2, 9, 3], [2, 3, 1],
        [3, 9, 4], [3, 4, 8], [3, 8, 11], [3, 11, 7], [3, 7, 2],
        [4, 9, 11], [4, 11, 8],
    ]
    centers = []
    for f in faces:
        c = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        centers.append(c / np.linalg.norm(c))
    return np.array(centers)


def _build_icosahedral_cluster(n, seed=0):
    """Build a cluster with icosahedral core + surface atoms at high-symmetry sites."""
    rng = np.random.default_rng(seed)
    positions = []
    
    # Core icosahedron: 12 atoms at radius ~0.72
    core_radius = 0.72
    core = _icosahedron_vertices() * core_radius
    positions.extend(core)
    n_remaining = n - 12
    
    if n_remaining <= 0:
        return np.array(positions[:n])
    
    # Second shell: 30 edge-midpoint positions at radius ~1.05
    edge_dirs = _icosahedron_edges()
    shell2_radius = 1.05
    shell2 = edge_dirs * shell2_radius
    
    # Add as many edge positions as needed
    n_add2 = min(n_remaining, 30)
    positions.extend(shell2[:n_add2])
    n_remaining -= n_add2
    
    if n_remaining <= 0:
        return np.array(positions[:n])
    
    # Third shell: 20 face centers at radius ~1.35
    face_dirs = _icosahedron_face_centers()
    shell3_radius = 1.35
    shell3 = face_dirs * shell3_radius
    
    n_add3 = min(n_remaining, 20)
    positions.extend(shell3[:n_add3])
    n_remaining -= n_add3
    
    if n_remaining <= 0:
        return np.array(positions[:n])
    
    # Fourth shell: additional surface positions
    # Use a denser spherical distribution
    if n_remaining > 0:
        # Fibonacci sphere for remaining atoms
        indices = np.arange(n_remaining)
        phi = np.arccos(1 - 2 * (indices + 0.5) / n_remaining)
        theta = np.pi * (1 + np.sqrt(5)) * indices
        shell4_radius = 1.55
        shell4 = np.column_stack([
            shell4_radius * np.sin(phi) * np.cos(theta),
            shell4_radius * np.sin(phi) * np.sin(theta),
            shell4_radius * np.cos(phi)
        ])
        positions.extend(shell4)
    
    return np.array(positions[:n])


def _build_deformed_icosahedral(n, seed=0):
    """Build icosahedral cluster with random deformations for diversity."""
    rng = np.random.default_rng(seed)
    positions = _build_icosahedral_cluster(n, seed=seed)
    
    # Apply small random perturbations
    perturbation_scale = 0.08
    positions += rng.uniform(-perturbation_scale, perturbation_scale, positions.shape)
    
    return positions


def _build_fcc_fragment(n, seed=0):
    """Build FCC lattice fragment."""
    rng = np.random.default_rng(seed)
    a = 0.72
    basis = [
        [0, 0, 0],
        [0, a/2, a/2],
        [a/2, 0, a/2],
        [a/2, a/2, 0]
    ]
    
    side = int(np.ceil((n / 4) ** (1/3))) + 2
    candidates = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    p = np.array([i * a + b[0], j * a + b[1], k * a + b[2]])
                    candidates.append(p)
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    indices = np.argsort(dists)[:n]
    positions = candidates[indices]
    positions -= np.mean(positions, axis=0)
    return positions


def _build_random_cluster(n, seed=0):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = 0.72 * n ** (1.0 / 3.0) * 0.8
    
    points = []
    attempts = 0
    while len(points) < n and attempts < 5000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def _validate(positions, min_dist=0.35):
    """Check if positions are valid (finite, no overlaps)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    # Vectorized check
    for i in range(n):
        dists = np.linalg.norm(positions[i+1:] - positions[i], axis=1)
        if np.any(dists < min_dist):
            return False
    return True


def _basin_hopping(positions, potential, kind, n_iterations=3, perturbation=0.3, rng=None):
    """Basin hopping: relax, perturb, relax, keep best."""
    if rng is None:
        rng = np.random.default_rng()
    
    energy, pos = relax(positions, potential, kind)
    
    for _ in range(n_iterations):
        # Perturb
        perturbed = pos + rng.uniform(-perturbation, perturbation, pos.shape)
        
        if not _validate(perturbed):
            continue
        
        try:
            e2, pos2 = relax(perturbed, potential, kind)
        except Exception:
            continue
        
        if _validate(pos2) and e2 < energy:
            energy = e2
            pos = pos2
    
    return energy, pos


def construct(instance):
    """Structure prediction with basin hopping for Sutton-Chen clusters."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind")
    
    start_time = time.time()
    deadline = start_time + 270  # Leave 60s margin for safety
    
    best_energy = np.inf
    best_positions = None
    
    # Strategy 1: Icosahedral core + basin hopping (primary)
    # Multiple icosahedral starts with different deformations
    for seed in range(6):
        if time.time() > deadline:
            break
        
        positions = _build_icosahedral_cluster(size, seed=seed)
        if not _validate(positions):
            positions = _build_deformed_icosahedral(size, seed=seed)
            if not _validate(positions):
                continue
        
        rng = np.random.default_rng(seed * 1000 + 42)
        try:
            energy, relaxed = _basin_hopping(positions, potential, kind, 
                                             n_iterations=4, perturbation=0.25, rng=rng)
        except Exception:
            continue
        
        if _validate(relaxed) and energy < best_energy:
            best_energy = energy
            best_positions = relaxed
    
    # Strategy 2: Deformed icosahedral + basin hopping
    for seed in range(4):
        if time.time() > deadline:
            break
        
        positions = _build_deformed_icosahedral(size, seed=seed + 100)
        if not _validate(positions):
            continue
        
        rng = np.random.default_rng(seed * 2000 + 77)
        try:
            energy, relaxed = _basin_hopping(positions, potential, kind,
                                             n_iterations=3, perturbation=0.3, rng=rng)
        except Exception:
            continue
        
        if _validate(relaxed) and energy < best_energy:
            best_energy = energy
            best_positions = relaxed
    
    # Strategy 3: FCC fragment + basin hopping
    for seed in range(3):
        if time.time() > deadline:
            break
        
        positions = _build_fcc_fragment(size, seed=seed)
        if not _validate(positions):
            continue
        
        rng = np.random.default_rng(seed * 3000 + 13)
        try:
            energy, relaxed = _basin_hopping(positions, potential, kind,
                                             n_iterations=3, perturbation=0.3, rng=rng)
        except Exception:
            continue
        
        if _validate(relaxed) and energy < best_energy:
            best_energy = energy
            best_positions = relaxed
    
    # Strategy 4: Random + basin hopping
    for seed in range(3):
        if time.time() > deadline:
            break
        
        positions = _build_random_cluster(size, seed=seed)
        if not _validate(positions):
            continue
        
        rng = np.random.default_rng(seed * 4000 + 99)
        try:
            energy, relaxed = _basin_hopping(positions, potential, kind,
                                             n_iterations=2, perturbation=0.35, rng=rng)
        except Exception:
            continue
        
        if _validate(relaxed) and energy < best_energy:
            best_energy = energy
            best_positions = relaxed
    
    # If no valid result, fall back
    if best_positions is None:
        positions = _build_icosahedral_cluster(size, seed=0)
        if not _validate(positions):
            positions = _build_random_cluster(size, seed=42)
        energy, best_positions = relax(positions, potential, kind)
    
    return best_positions
# EVOLVE-BLOCK-END
