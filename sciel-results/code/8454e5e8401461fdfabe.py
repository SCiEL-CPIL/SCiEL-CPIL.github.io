"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _generate_icosahedral(n, seed=0):
    """Generate positions based on icosahedral shells."""
    rng = np.random.default_rng(seed)
    
    # Icosahedron vertices (12 vertices)
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico_vertices = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ])
    ico_vertices = ico_vertices / np.linalg.norm(ico_vertices, axis=1, keepdims=True)
    
    # Fibonacci sphere for outer shells
    def fib_sphere(n_pts, radius, rng):
        if n_pts <= 0:
            return np.array([]).reshape(0, 3)
        if n_pts == 1:
            return np.array([[0, 0, radius]])
        
        # Fibonacci lattice
        golden = (1 + np.sqrt(5)) / 2
        indices = np.arange(n_pts)
        phi = np.arccos(1 - 2 * (indices + 0.5) / n_pts)
        theta = np.pi * (1 + np.sqrt(5)) * indices
        
        points = np.column_stack([
            radius * np.sin(phi) * np.cos(theta),
            radius * np.sin(phi) * np.sin(theta),
            radius * np.cos(phi)
        ])
        
        # Add small random perturbation
        points += rng.uniform(-0.05, 0.05, points.shape)
        return points
    
    positions = []
    
    # Inner icosahedron (12 atoms)
    if n >= 12:
        inner_radius = 0.72
        inner = ico_vertices * inner_radius
        positions.extend(inner)
        n -= 12
    
    # Outer shell(s)
    if n > 0:
        outer_radius = 1.2
        outer = fib_sphere(n, outer_radius, rng)
        positions.extend(outer)
    
    return np.array(positions)


def _generate_fcc(n, seed=0):
    """Generate positions from FCC lattice fragments."""
    rng = np.random.default_rng(seed)
    
    # FCC basis vectors
    a = 0.72  # typical spacing
    basis = [
        [0, 0, 0],
        [0, a/2, a/2],
        [a/2, 0, a/2],
        [a/2, a/2, 0]
    ]
    
    # Generate FCC lattice points
    side = int(np.ceil((n / 4) ** (1/3))) + 2
    candidates = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    p = np.array([i * a + b[0], j * a + b[1], k * a + b[2]])
                    candidates.append(p)
    
    candidates = np.array(candidates)
    
    # Select n points closest to center
    dists = np.linalg.norm(candidates, axis=1)
    indices = np.argsort(dists)[:n]
    positions = candidates[indices]
    
    # Center the cluster
    positions -= np.mean(positions, axis=0)
    
    return positions


def _generate_hcp(n, seed=0):
    """Generate positions from HCP lattice fragments."""
    rng = np.random.default_rng(seed)
    
    a = 0.72  # in-plane spacing
    c = a * np.sqrt(8/3)  # ideal HCP c/a ratio
    
    positions = []
    # Generate layers
    layer_spacing = c / 2
    n_layers = int(np.ceil(np.sqrt(n / 3)))
    
    for layer in range(n_layers):
        z = layer * layer_spacing
        # AB stacking
        offset = np.array([0, 0, 0]) if layer % 2 == 0 else np.array([a/2, a/(2*np.sqrt(3)), 0])
        
        # Hexagonal arrangement in this layer
        max_r = int(np.ceil(np.sqrt(n / (3 * np.pi / (a**2 * 0.866)))))
        for i in range(-max_r, max_r + 1):
            for j in range(-max_r, max_r + 1):
                x = a * (i + j / 2) + offset[0]
                y = a * np.sqrt(3) / 2 * j + offset[1]
                p = np.array([x, y, z])
                if len(positions) < n:
                    positions.append(p)
                else:
                    break
            if len(positions) >= n:
                break
        if len(positions) >= n:
            break
    
    positions = np.array(positions[:n])
    positions -= np.mean(positions, axis=0)
    
    return positions


def _generate_random(n, spacing=0.72, seed=0):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
                points.append(p)
        attempts += 1
    
    # If we couldn't place all points, add random ones
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
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(positions[i] - positions[j]) < min_dist:
                return False
    return True


def construct(instance):
    """Best of multiple structured starting clusters, each relaxed locally."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind")
    
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s margin
    
    # Generate diverse initial configurations
    generators = [
        lambda s: _generate_icosahedral(size, seed=s),
        lambda s: _generate_fcc(size, seed=s),
        lambda s: _generate_hcp(size, seed=s),
        lambda s: _generate_random(size, seed=s),
    ]
    
    best_energy = np.inf
    best_positions = None
    
    n_starts = 8  # Number of different starting configurations to try
    
    for i in range(n_starts):
        if time.time() > deadline:
            break
        
        # Cycle through generators
        gen_idx = i % len(generators)
        seed = i * 12345 + 7
        positions = generators[gen_idx](seed)
        
        # Validate
        if not _validate(positions):
            # Try to fix by small perturbation
            positions += np.random.default_rng(seed).uniform(-0.1, 0.1, positions.shape)
            if not _validate(positions):
                continue
        
        # Relax
        try:
            energy, relaxed = relax(positions, potential, kind)
        except Exception:
            continue
        
        if not _validate(relaxed):
            continue
        
        if energy < best_energy:
            best_energy = energy
            best_positions = relaxed
    
    # If no valid result, fall back to a simple random start
    if best_positions is None:
        positions = _generate_random(size, seed=42)
        energy, best_positions = relax(positions, potential, kind)
    
    return best_positions
# EVOLVE-BLOCK-END
