"""Dzugutov cluster global minima via polytetrahedral geometric initialization with local relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_DEADLINE = time.time() + 280.0  # safety margin before 300s kill


def _icosahedron_vertices(scale=1.0):
    """Return 12 vertices of a regular icosahedron scaled so edge length ~ scale."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    # Edge length of this icosahedron
    d = np.linalg.norm(verts[0] - verts[4])
    verts = verts * (scale / d)
    return verts


def _face_centers(vertices):
    """Return centers of all triangular faces of an icosahedron."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # Icosahedron face indices
    faces = [
        (0, 1, 4), (0, 1, 5), (0, 4, 8), (0, 8, 5), (0, 2, 6),
        (0, 2, 7), (0, 6, 9), (0, 9, 7), (1, 4, 10), (1, 10, 5),
        (1, 5, 10), (2, 6, 11), (2, 11, 7), (2, 7, 11), (4, 8, 3),
        (4, 3, 10), (5, 8, 3), (5, 3, 10), (6, 9, 11), (6, 11, 3),
        (7, 9, 11), (7, 11, 3), (9, 11, 3), (10, 3, 8)
    ]
    centers = []
    for f in faces:
        c = (vertices[f[0]] + vertices[f[1]] + vertices[f[2]]) / 3.0
        centers.append(c)
    return np.array(centers)


def _second_shell_points(icosahedron_verts, scale=1.0):
    """Return 30 points on second shell (face centers) of icosahedron."""
    return _face_centers(icosahedron_verts)


def _make_partial_icosahedron(n, base_spacing=1.1):
    """Construct a partial icosahedral cluster with n atoms."""
    ico = _icosahedron_vertices(base_spacing)
    # First shell: 12 vertices
    points = [ico]
    
    if n <= 12:
        # Just take first n vertices
        result = ico[:n].copy()
        return result
    
    # Second shell: 30 face centers
    face_centers = _face_centers(ico)
    points.append(face_centers)
    
    all_points = np.vstack(points)
    if len(all_points) >= n:
        return all_points[:n].copy()
    
    # For n > 42, add third shell (edges of icosahedron, 30 edges)
    # Or just repeat with slight perturbation
    extra_needed = n - len(all_points)
    # Add points on edges of the icosahedron
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    edges = [
        (0, 1), (0, 4), (0, 5), (0, 2), (0, 6), (0, 7),
        (1, 4), (1, 5), (1, 10), (2, 6), (2, 7), (2, 11),
        (3, 8), (3, 9), (3, 10), (3, 11), (4, 8), (4, 10),
        (5, 8), (5, 10), (6, 9), (6, 11), (7, 9), (7, 11),
        (8, 9), (8, 10), (9, 11), (10, 11)
    ]
    edge_points = []
    for e in edges:
        mid = (ico[e[0]] + ico[e[1]]) / 2.0
        edge_points.append(mid)
    edge_points = np.array(edge_points)
    all_points = np.vstack([all_points, edge_points])
    
    if len(all_points) >= n:
        return all_points[:n].copy()
    
    # For very large n, add a third icosahedral shell
    outer_ico = _icosahedron_vertices(base_spacing * 1.6)
    all_points = np.vstack([all_points, outer_ico])
    
    if len(all_points) >= n:
        return all_points[:n].copy()
    
    # Fallback: random compact
    radius = base_spacing * n ** (1.0/3.0) * 0.75
    rng = np.random.default_rng(n)
    while len(all_points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            all_points = np.vstack([all_points, p])
    
    return all_points[:n].copy()


def _relax_with_gradient(positions, potential, parameter, max_steps=500, tol=1e-8):
    """Local relaxation using gradient descent with adaptive step size."""
    n = len(positions)
    if n < 2:
        return positions
    
    # Try the built-in relax first
    try:
        energy, pos = relax(positions.copy(), potential, parameter)
        return pos
    except Exception:
        pass
    
    # Fallback: simple gradient descent
    pos = positions.copy()
    lr = 0.01
    for step in range(max_steps):
        if time.time() > _DEADLINE - 5:
            break
        try:
            grad = dzugutov_energy_gradient(pos)
        except Exception:
            break
        norm = np.linalg.norm(grad)
        if norm < tol:
            break
        pos = pos - lr * grad / (norm + 1e-12) * 0.1
        # Adaptive learning rate
        lr = max(0.001, lr * 0.99)
    
    return pos


def construct(instance):
    """Find low-energy Dzugutov cluster via polytetrahedral geometric initialization."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Try multiple geometric seeds with local relaxation
    candidates = []
    
    # Seed 1: Standard partial icosahedron
    try:
        seed1 = _make_partial_icosahedron(size, base_spacing=1.1)
        candidates.append(seed1)
    except Exception:
        pass
    
    # Seed 2: Slightly different spacing
    try:
        seed2 = _make_partial_icosahedron(size, base_spacing=1.15)
        candidates.append(seed2)
    except Exception:
        pass
    
    # Seed 3: Random compact (fallback)
    rng = np.random.default_rng(size)
    radius = 1.1 * size ** (1.0/3.0) * 0.75
    points = []
    while len(points) < size:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * 1.1 for q in points):
            points.append(p)
    candidates.append(np.array(points))
    
    # Relax each candidate and keep the best
    best_energy = np.inf
    best_positions = None
    
    for seed in candidates:
        if time.time() > _DEADLINE - 10:
            break
        try:
            energy, positions = relax(seed.copy(), potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    # If no relaxation succeeded, return the best seed
    if best_positions is None:
        return candidates[0]
    
    # Additional refinement: try a few more perturbations around the best
    if time.time() < _DEADLINE - 20:
        rng2 = np.random.default_rng(42)
        for _ in range(5):
            perturbed = best_positions + rng2.normal(0, 0.05, best_positions.shape)
            try:
                energy, positions = relax(perturbed.copy(), potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                continue
    
    return best_positions
# EVOLVE-BLOCK-END
