"""Sutton-Chen cluster global minima search using structured templates and multi-start relaxation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _fcc_positions(n, a=0.72):
    """Generate n positions from an FCC lattice fragment."""
    points = []
    basis = [
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0],
    ]
    side = int(np.ceil((n / 4) ** (1.0 / 3.0))) + 2
    candidates = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    p = np.array([i * a + b[0] * a, j * a + b[1] * a, k * a + b[2] * a])
                    candidates.append(p)
    # Sort by distance from center
    candidates = np.array(candidates)
    center = candidates.mean(axis=0)
    dists = np.linalg.norm(candidates - center, axis=1)
    order = np.argsort(dists)
    points = candidates[order[:n]]
    # Center the cluster
    points -= points.mean(axis=0)
    return points


def _hcp_positions(n, a=0.72):
    """Generate n positions from an HCP lattice fragment."""
    points = []
    basis = [
        [0, 0, 0],
        [0.5, 0.288675, 0],  # 1/sqrt(3)
        [0.25, 0.866025/2, 0.5 * a * 0.577350],  # approx
    ]
    # Simpler: use a hexagonal close-packed arrangement
    # Layer spacing
    layer_spacing = a * 0.577350
    # Generate layers
    max_layer = int(np.ceil((n / 6) ** (1.0 / 3.0))) + 2
    for layer in range(-max_layer, max_layer + 1):
        z = layer * layer_spacing
        # In each layer, place atoms in a hexagonal pattern
        for i in range(-max_layer, max_layer + 1):
            for j in range(-max_layer, max_layer + 1):
                x = i * a + (j % 2) * 0.5 * a
                y = j * a * 0.866025
                points.append([x, y, z])
    points = np.array(points)
    center = points.mean(axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    points -= points.mean(axis=0)
    return points


def _icosahedron_positions(n, a=0.72):
    """Generate positions based on icosahedral symmetry."""
    # Start with a central atom and add shells
    points = [[0, 0, 0]]
    # Icosahedron vertices (unit sphere)
    phi = (1 + np.sqrt(5)) / 2
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    ico_vertices /= np.linalg.norm(ico_vertices, axis=1, keepdims=True)
    r1 = a * 1.0
    for v in ico_vertices:
        if len(points) < n:
            points.append(v * r1)
    # Second shell
    if len(points) < n:
        r2 = a * 1.6
        # Add more points on a sphere
        theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        for t in theta:
            if len(points) < n:
                points.append([r2 * np.cos(t), r2 * np.sin(t), 0])
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points


def _random_compact(n, spacing=0.72, seed=42):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    points = []
    for _ in range(10000):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - np.array(q)) >= 0.5 * spacing for q in points):
                points.append(p)
    if len(points) < n:
        # Fallback: just random points
        points = [list(rng.uniform(-radius, radius, 3)) for _ in range(n)]
    return np.array(points[:n])


def construct(instance):
    """Multi-start optimization using structured templates and random starts."""
    size = instance["n"]
    kind = instance["kind"]
    n_exp = instance["n_exp"]
    m_exp = instance["m_exp"]
    c = instance["c"]
    
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # FCC
    try:
        initial_configs.append(_fcc_positions(size))
    except:
        pass
    
    # HCP
    try:
        initial_configs.append(_hcp_positions(size))
    except:
        pass
    
    # Icosahedron
    try:
        initial_configs.append(_icosahedron_positions(size))
    except:
        pass
    
    # Random compact with different seeds
    for seed in [42, 123, 456, 789, 1011, 1314]:
        try:
            initial_configs.append(_random_compact(size, seed=seed))
        except:
            pass
    
    best_energy = np.inf
    best_positions = None
    
    for i, config in enumerate(initial_configs):
        if time.time() > deadline:
            break
        try:
            energy, positions = relax(config, 'sutton_chen', kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    if best_positions is None:
        # Fallback: return random positions
        rng = np.random.default_rng(0)
        best_positions = rng.uniform(-1, 1, (size, 3))
    
    return best_positions
# EVOLVE-BLOCK-END
