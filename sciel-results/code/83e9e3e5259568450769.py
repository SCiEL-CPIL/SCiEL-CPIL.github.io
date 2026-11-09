"""Sutton-Chen transition-metal cluster global minima: multi-start with structured seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _fcc_cluster(n, spacing=0.72):
    """Generate an n-atom fcc-like cluster by filling fcc lattice sites nearest to origin."""
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
                    pos = np.array([i, j, k]) * spacing + b * spacing
                    sites.append(pos)
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)[:n]
    return sites[idx]


def _hcp_cluster(n, spacing=0.72):
    """Generate an n-atom hcp-like cluster by filling hcp lattice sites nearest to origin."""
    if n <= 0:
        return np.empty((0, 3))
    a = spacing
    c = np.sqrt(8.0 / 3.0) * a
    # hcp basis in hexagonal coords: (0,0,0) and (1/3, 2/3, 1/2) in fractional
    # Convert to cartesian: a1 = a*(1,0,0), a2 = a*(-1/2, sqrt(3)/2, 0), a3 = (0,0,c)
    a1 = np.array([a, 0, 0])
    a2 = np.array([-a/2, a*np.sqrt(3)/2, 0])
    a3 = np.array([0, 0, c])
    basis_frac = [np.array([0, 0, 0]), np.array([1/3, 2/3, 0.5])]
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                for bf in basis_frac:
                    pos = i * a1 + j * a2 + k * a3 + bf[0]*a1 + bf[1]*a2 + bf[2]*a3
                    sites.append(pos)
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)[:n]
    return sites[idx]


def _icosahedral_cluster(n, spacing=0.72):
    """Generate an n-atom icosahedral-like cluster (Mackay icosahedra)."""
    if n <= 0:
        return np.empty((0, 3))
    # Mackay icosahedra shells: 1, 12, 42, 92, 140, ...
    # Generate points on icosahedral shells
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices (12)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    
    # Generate shells
    all_points = [np.array([0.0, 0.0, 0.0])]  # center
    # Shell 1: 12 vertices
    shell1 = verts * spacing
    all_points.append(shell1)
    # Shell 2: 42 points (edges and face centers)
    # For simplicity, use a denser sampling
    if n > 13:
        # Generate more points by subdividing
        shell2 = []
        # Edge midpoints (30 edges)
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if 0.9 < d < 1.1:  # approximate edge length check
                    edges.append((i, j))
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2
            mid /= np.linalg.norm(mid)
            shell2.append(mid * 2 * spacing)
        # Face centers (20 faces)
        # Approximate: add points at 2*spacing in various directions
        for v in verts:
            shell2.append(v * 2 * spacing)
        # Add more to reach 42
        for v in verts:
            for w in verts:
                if np.linalg.norm(v - w) > 1.5:
                    p = (v + w) / 2
                    p /= np.linalg.norm(p)
                    shell2.append(p * 2 * spacing)
        shell2 = np.array(shell2)
        # Remove duplicates approximately
        unique = [shell2[0]]
        for p in shell2[1:]:
            if all(np.linalg.norm(p - q) > 0.1 for q in unique):
                unique.append(p)
        all_points.append(np.array(unique))
    
    # Combine and take nearest n
    all_pts = np.vstack(all_points)
    dists = np.linalg.norm(all_pts, axis=1)
    idx = np.argsort(dists)[:n]
    return all_pts[idx]


def _random_compact(n, spacing=0.72, seed=42):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
                points.append(p)
        attempts += 1
    # If we couldn't place all, just add random points
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _validate(positions):
    """Check if positions are valid (finite, min distance > 0.35)."""
    if not np.all(np.isfinite(positions)):
        return False
    if len(positions) < 2:
        return True
    # Vectorized distance check
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) > 0.35


def construct(instance):
    """Multi-start optimization with structured seeds and random starts."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    deadline = time.time() + 280  # Safety margin before 330s kill
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starting structures
    starts = []
    
    # Structured seeds
    try:
        starts.append(_fcc_cluster(size))
    except Exception:
        pass
    try:
        starts.append(_hcp_cluster(size))
    except Exception:
        pass
    try:
        starts.append(_icosahedral_cluster(size))
    except Exception:
        pass
    
    # Random compact starts with different seeds
    for seed in [42, 123, 789, 2024, 31415, 27182]:
        try:
            starts.append(_random_compact(size, seed=seed))
        except Exception:
            pass
    
    # Try each start
    for i, start in enumerate(starts):
        if time.time() > deadline - 30:  # Need at least 30s for final validation
            break
        try:
            energy, positions = relax(start, potential, kind)
            if _validate(positions) and energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue
    
    # If no valid result, try more random starts
    if best_positions is None:
        for seed in range(100, 200):
            if time.time() > deadline - 30:
                break
            try:
                start = _random_compact(size, seed=seed)
                energy, positions = relax(start, potential, kind)
                if _validate(positions) and energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            except Exception:
                continue
    
    # Final polish: try a few more relaxations from best position with small perturbations
    if best_positions is not None and time.time() < deadline - 20:
        rng = np.random.default_rng(999)
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            perturbed = best_positions + rng.normal(0, 0.05, size=(size, 3))
            try:
                energy, positions = relax(perturbed, potential, kind)
                if _validate(positions) and energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            except Exception:
                continue
    
    if best_positions is None:
        # Fallback: return a simple linear chain
        best_positions = np.array([[i * 0.5, 0, 0] for i in range(size)], dtype=float)
    
    return best_positions
# EVOLVE-BLOCK-END
