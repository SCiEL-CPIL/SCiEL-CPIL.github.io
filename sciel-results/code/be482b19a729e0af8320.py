"""Sutton-Chen transition-metal cluster global minima - structured multi-start search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _fcc_cluster(n, spacing=0.72):
    """Generate an n-atom fcc-like cluster by filling fcc lattice sites by distance from center."""
    if n <= 0:
        return np.empty((0, 3))
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 3
    candidates = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                for b in basis:
                    p = np.array([i, j, k]) * spacing + b * spacing
                    r = np.linalg.norm(p)
                    candidates.append((r, p))
    candidates.sort(key=lambda x: x[0])
    # Deduplicate and pick n closest
    seen = set()
    points = []
    for r, p in candidates:
        key = tuple(np.round(p, 6))
        if key not in seen:
            seen.add(key)
            points.append(p)
            if len(points) == n:
                break
    if len(points) < n:
        # Fill remaining randomly
        rng = np.random.default_rng(42)
        while len(points) < n:
            p = rng.uniform(-spacing, spacing, 3)
            if all(np.linalg.norm(p - q) >= 0.3 * spacing for q in points):
                points.append(p)
    return np.array(points)


def _hcp_cluster(n, spacing=0.72):
    """Generate an n-atom hcp-like cluster."""
    if n <= 0:
        return np.empty((0, 3))
    # HCP: close-packed layers with AB stacking
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    candidates = []
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 3
    for layer in range(-max_shell, max_shell + 1):
        z = layer * c / 2.0
        # AB stacking offset
        if layer % 2 == 0:
            offset = np.array([0, 0, 0])
        else:
            offset = np.array([a / 2, a / (2 * np.sqrt(3)), 0])
        for i in range(-max_shell, max_shell + 1):
            for j in range(-max_shell, max_shell + 1):
                x = i * a + (j % 2) * a / 2 + offset[0]
                y = j * a * np.sqrt(3) / 2 + offset[1]
                p = np.array([x, y, z])
                r = np.linalg.norm(p)
                candidates.append((r, p))
    candidates.sort(key=lambda x: x[0])
    seen = set()
    points = []
    for r, p in candidates:
        key = tuple(np.round(p, 6))
        if key not in seen:
            seen.add(key)
            points.append(p)
            if len(points) == n:
                break
    if len(points) < n:
        rng = np.random.default_rng(43)
        while len(points) < n:
            p = rng.uniform(-spacing, spacing, 3)
            if all(np.linalg.norm(p - q) >= 0.3 * spacing for q in points):
                points.append(p)
    return np.array(points)


def _icosahedral_cluster(n, spacing=0.72):
    """Generate an n-atom icosahedral-like cluster (Mackay icosahedron)."""
    if n <= 0:
        return np.empty((0, 3))
    
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    verts /= np.linalg.norm(verts[0]) * 0.5  # normalize to unit radius
    
    # Generate points on concentric icosahedral shells
    candidates = [(0.0, np.zeros(3))]  # center
    shells = [1, 2, 3, 4, 5]
    for shell in shells:
        for v in verts:
            p = v * shell * spacing * 0.5
            candidates.append((np.linalg.norm(p), p))
        # Add midpoints of edges for denser packing
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                d = np.linalg.norm(verts[i] - verts[j])
                if 0.8 < d < 1.2:  # adjacent vertices
                    p = (verts[i] + verts[j]) / 2 * shell * spacing * 0.5
                    candidates.append((np.linalg.norm(p), p))
    
    candidates.sort(key=lambda x: x[0])
    seen = set()
    points = []
    for r, p in candidates:
        key = tuple(np.round(p, 6))
        if key not in seen:
            seen.add(key)
            points.append(p)
            if len(points) == n:
                break
    if len(points) < n:
        rng = np.random.default_rng(44)
        while len(points) < n:
            p = rng.uniform(-spacing, spacing, 3)
            if all(np.linalg.norm(p - q) >= 0.3 * spacing for q in points):
                points.append(p)
    return np.array(points)


def _random_cluster(n, spacing=0.72, seed=0):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 50:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
            points.append(p)
        attempts += 1
    # Fill remaining if needed
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _perturb(positions, spacing=0.72, seed=0, magnitude=0.1):
    """Add small random perturbation to a cluster."""
    rng = np.random.default_rng(seed)
    perturbed = positions + rng.uniform(-magnitude, magnitude, size=positions.shape)
    return perturbed


def construct(instance):
    """Multi-start search with structured and random initial geometries."""
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s limit
    
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho", "12-6"))
    spacing = 0.72
    
    # Generate diverse initial configurations
    generators = [
        ("fcc", _fcc_cluster),
        ("hcp", _hcp_cluster),
        ("ico", _icosahedral_cluster),
    ]
    
    initial_configs = []
    
    # Add structured clusters
    for name, gen in generators:
        try:
            config = gen(size, spacing)
            if config.shape == (size, 3) and np.all(np.isfinite(config)):
                initial_configs.append(config)
        except Exception:
            pass
    
    # Add random clusters with different seeds
    for seed in range(8):
        try:
            config = _random_cluster(size, spacing, seed=seed)
            if config.shape == (size, 3) and np.all(np.isfinite(config)):
                initial_configs.append(config)
        except Exception:
            pass
    
    # Add perturbed versions of structured clusters
    for i, config in enumerate(initial_configs[:3]):
        for seed in range(3):
            try:
                perturbed = _perturb(config, spacing, seed=100 + i * 10 + seed, magnitude=0.08)
                if np.all(np.isfinite(perturbed)):
                    initial_configs.append(perturbed)
            except Exception:
                pass
    
    # Relax each configuration and keep the best
    best_energy = np.inf
    best_positions = initial_configs[0] if initial_configs else np.zeros((size, 3))
    
    for idx, config in enumerate(initial_configs):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        try:
            energy, positions = relax(config, potential, kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    # If we have time left, do additional random starts
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        extra_seed = 1000
        while time.time() - start_time < deadline - 10:
            try:
                config = _random_cluster(size, spacing, seed=extra_seed)
                energy, positions = relax(config, potential, kind)
                if np.isfinite(energy) and energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass
            extra_seed += 1
    
    # Final validity check
    if not np.all(np.isfinite(best_positions)):
        best_positions = initial_configs[0] if initial_configs else np.zeros((size, 3))
    
    return best_positions
# EVOLVE-BLOCK-END
