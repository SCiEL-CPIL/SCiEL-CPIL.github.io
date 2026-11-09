"""Sutton-Chen transition-metal cluster global minima: multi-start structured + random seeds with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Per-kind parameters: (n_exp, m_exp, c)
_KIND_PARAMS = {
    '12-6': (12, 6, 144.41),
    '9-6': (9, 6, 39.432),
    '10-8': (10, 8, 34.408),
}

# Spacing per kind for seed generation
_SPACING = {
    '12-6': 0.72,
    '9-6': 0.72,
    '10-8': 0.72,
}


def _validate_positions(pos):
    """Vectorized validity check: finite coords and min pair distance >= 0.35."""
    if not np.all(np.isfinite(pos)):
        return False
    n = pos.shape[0]
    if n < 2:
        return True
    # Compute all pairwise distances via broadcasting (n x n x 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.35


def _fcc_cluster(n, spacing=0.72):
    """Generate n-atom FCC-like cluster by filling nearest FCC lattice sites to origin."""
    if n <= 0:
        return np.empty((0, 3))
    basis = np.array([[0, 0, 0],
                      [0, 0.5, 0.5],
                      [0.5, 0, 0.5],
                      [0.5, 0.5, 0]], dtype=float)
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                for b in basis:
                    p = np.array([i, j, k], dtype=float) + b
                    sites.append(p)
    sites = np.array(sites)
    sites = sites * spacing
    # Sort by distance to origin, take nearest n
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n]]


def _hcp_cluster(n, spacing=0.72):
    """Generate n-atom HCP-like cluster by filling nearest HCP lattice sites to origin."""
    if n <= 0:
        return np.empty((0, 3))
    # HCP: two interpenetrating hexagonal lattices offset in z
    a = spacing
    c = spacing * np.sqrt(8.0 / 3.0)
    # Hexagonal basis vectors
    v1 = np.array([a, 0, 0])
    v2 = np.array([-a / 2, a * np.sqrt(3) / 2, 0])
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                # Layer k
                base = i * v1 + j * v2
                z = k * c
                # Two atoms per unit cell in HCP
                p1 = base + np.array([0, 0, z])
                p2 = base + np.array([a / 2, a * np.sqrt(3) / 6, z + c / 2])
                sites.append(p1)
                sites.append(p2)
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n]]


def _icosahedral_cluster(n, spacing=0.72):
    """Generate n-atom icosahedral-like cluster: nested icosahedra."""
    if n <= 0:
        return np.empty((0, 3))
    # Icosahedron vertices (12)
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)

    # Nested shells: 1 (center) + 12 (1st) + 30 (2nd) + 32 (3rd) + ...
    # For simplicity, use icosahedral shells with increasing radius
    # Shell radii: r, r*sqrt(3), r*sqrt(5), ...
    # Points per shell: 1, 12, 30, 32, 48, 60, ...
    # We'll generate points on icosahedral vertices at multiple radii

    points = [np.zeros(3)]  # center
    if n <= 1:
        return np.array(points[:n])

    # 1st shell: 12 vertices at radius r1
    r1 = spacing
    for v in ico_verts:
        points.append(v * r1)
    if len(points) >= n:
        return np.array(points[:n])

    # 2nd shell: 30 points (midpoints of edges of icosahedron) at radius r2
    r2 = spacing * np.sqrt(3.0)
    # Edges of icosahedron: pairs of vertices at distance 2 (in the unit ico)
    ico_norms = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    # Compute pairwise distances to find edges
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(ico_norms[i] - ico_norms[j])
            if 1.0 < d < 2.0:  # edge
                mid = (ico_norms[i] + ico_norms[j]) / 2.0
                mid = mid / np.linalg.norm(mid)
                points.append(mid * r2)
    if len(points) >= n:
        return np.array(points[:n])

    # 3rd shell: 32 points (face centers) at radius r3
    r3 = spacing * np.sqrt(5.0)
    # Faces of icosahedron: triangles of 3 mutually adjacent vertices
    # For simplicity, use face centers by finding triplets
    # This is complex; instead, use a denser set of points on a sphere
    # Use Fibonacci sphere for remaining points
    remaining = n - len(points)
    if remaining > 0:
        # Fibonacci sphere
        for k in range(remaining):
            t = (k + 0.5) / remaining
            inc = np.pi * (3 - np.sqrt(5))
            z = 1 - 2 * t
            r = np.sqrt(1 - z * z)
            theta = inc * k
            p = np.array([r * np.cos(theta), r * np.sin(theta), z])
            points.append(p * r3)

    return np.array(points[:n])


def _random_compact(n, spacing=0.72, seed=42):
    """Generate a random compact cluster with minimum separation."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    max_attempts = n * 500
    attempt = 0
    while len(points) < n and attempt < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.7 * spacing for q in points):
                points.append(p)
        attempt += 1
    # If we couldn't place all, fill remaining with small random offsets
    while len(points) < n:
        base = points[len(points) % max(1, len(points))] if points else np.zeros(3)
        p = base + rng.normal(0, spacing * 0.3, 3)
        points.append(p)
    return np.array(points[:n])


def construct(instance):
    """Multi-start structured + random seeds, relaxed with L-BFGS-B, with deadline guard."""
    start_time = time.time()
    deadline = start_time + 290.0  # 40s safety margin before 330s kill

    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    spacing = _SPACING.get(kind, 0.72)

    # Generate diverse starting configurations
    seeds = []

    # Structured seeds at multiple scales
    for scale in [0.85, 1.0, 1.15]:
        s = spacing * scale
        seeds.append(_fcc_cluster(size, s))
        seeds.append(_hcp_cluster(size, s))
        seeds.append(_icosahedral_cluster(size, s))

    # Random compact seeds with different seeds
    for rs in [size, size + 1, size + 7, size + 13]:
        seeds.append(_random_compact(size, spacing, seed=rs))

    # Remove duplicates (by rounding) and ensure validity
    unique_seeds = []
    seen = set()
    for s in seeds:
        if s.shape[0] != size:
            continue
        key = tuple(np.round(s, 3).flatten())
        if key not in seen:
            seen.add(key)
            if _validate_positions(s):
                unique_seeds.append(s)

    best_energy = np.inf
    best_positions = None

    # Relax each seed
    for i, seed in enumerate(unique_seeds):
        if time.time() > deadline:
            break
        try:
            energy, positions = relax(seed, potential, kind)
            if np.isfinite(energy) and _validate_positions(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            continue

    # If no valid result yet, try a simple fallback
    if best_positions is None:
        # Last resort: linear chain
        positions = np.zeros((size, 3))
        positions[:, 0] = np.arange(size) * spacing
        best_positions = positions
        best_energy = np.inf

    # Polish pass: re-relax the best result with a fresh start
    if best_positions is not None and time.time() < deadline - 5.0:
        try:
            # Small perturbation to escape any local minimum
            rng = np.random.default_rng(0)
            perturbed = best_positions + rng.normal(0, 0.05 * spacing, best_positions.shape)
            energy2, positions2 = relax(perturbed, potential, kind)
            if np.isfinite(energy2) and _validate_positions(positions2):
                if energy2 < best_energy:
                    best_energy = energy2
                    best_positions = positions2

            # Also try relaxing the original best again
            energy3, positions3 = relax(best_positions, potential, kind)
            if np.isfinite(energy3) and _validate_positions(positions3):
                if energy3 < best_energy:
                    best_energy = energy3
                    best_positions = positions3
        except Exception:
            pass

    # Final validation
    if not _validate_positions(best_positions):
        # Fallback: ensure validity
        best_positions = np.zeros((size, 3))
        best_positions[:, 0] = np.arange(size) * spacing

    return best_positions
# EVOLVE-BLOCK-END
