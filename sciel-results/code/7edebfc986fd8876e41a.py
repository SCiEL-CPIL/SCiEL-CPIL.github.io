"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Time management
_TOTAL_TIME_LIMIT = 285.0  # safety margin before 300s kill
_START_TIME = None


def _time_left():
    if _START_TIME is None:
        return _TOTAL_TIME_LIMIT
    return _TOTAL_TIME_LIMIT - (time.time() - _START_TIME)


def _validate(positions):
    """Check if positions are valid (finite, min distance >= 0.35)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    # Compute all pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.35


def _fcc_positions(n):
    """Generate FCC-like positions for n atoms."""
    positions = []
    # FCC lattice points
    a = 0.72  # typical spacing
    # Generate a cubic chunk of FCC lattice
    max_coord = int(np.ceil(n ** (1/3) / 2)) + 2
    fcc_points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC: corners and face centers
                fcc_points.append([i * a, j * a, k * a])
                fcc_points.append([(i + 0.5) * a, (j + 0.5) * a, k * a])
                fcc_points.append([(i + 0.5) * a, j * a, (k + 0.5) * a])
                fcc_points.append([i * a, (j + 0.5) * a, (k + 0.5) * a])
    
    fcc_points = np.array(fcc_points)
    # Center the points
    fcc_points -= np.mean(fcc_points, axis=0)
    
    # Sort by distance from center and take first n
    dists = np.linalg.norm(fcc_points, axis=1)
    sorted_idx = np.argsort(dists)
    positions = fcc_points[sorted_idx[:n]]
    return positions


def _hcp_positions(n):
    """Generate HCP-like positions for n atoms."""
    positions = []
    a = 0.72
    c_over_a = np.sqrt(8.0 / 3.0)  # ideal HCP c/a ratio
    c = c_over_a * a
    
    # Generate HCP layers
    max_layer = int(np.ceil(n ** (1/3))) + 2
    for layer in range(-max_layer, max_layer + 1):
        z = layer * c
        if layer % 2 == 0:
            # A layer
            for i in range(-max_layer, max_layer + 1):
                for j in range(-max_layer, max_layer + 1):
                    x = (i + 0.5 * (j % 2)) * a
                    y = j * a * np.sqrt(3) / 2
                    positions.append([x, y, z])
        else:
            # B layer
            for i in range(-max_layer, max_layer + 1):
                for j in range(-max_layer, max_layer + 1):
                    x = (i + 0.5 * (j % 2)) * a + a / 2
                    y = j * a * np.sqrt(3) / 2 + a * np.sqrt(3) / 6
                    positions.append([x, y, z])
    
    positions = np.array(positions)
    positions -= np.mean(positions, axis=0)
    
    dists = np.linalg.norm(positions, axis=1)
    sorted_idx = np.argsort(dists)
    positions = positions[sorted_idx[:n]]
    return positions


def _bcc_positions(n):
    """Generate BCC-like positions for n atoms."""
    positions = []
    a = 0.72
    max_coord = int(np.ceil(n ** (1/3) / 2)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                positions.append([i * a, j * a, k * a])
                positions.append([(i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a])
    
    positions = np.array(positions)
    positions -= np.mean(positions, axis=0)
    
    dists = np.linalg.norm(positions, axis=1)
    sorted_idx = np.argsort(dists)
    positions = positions[sorted_idx[:n]]
    return positions


def _random_compact(n, seed=None):
    """Generate random compact cluster positions."""
    rng = np.random.default_rng(seed)
    radius = 0.72 * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.6 * 0.72 for q in points):
                points.append(p)
    if len(points) < n:
        # Fill remaining with random positions
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)


def _icosahedron_positions(n):
    """Generate icosahedral-like positions."""
    # Start with icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    a = 0.72
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]) * a / np.sqrt(1 + phi**2)
    
    if n <= 13:
        if n == 13:
            return np.vstack([vertices, [[0, 0, 0]]])
        else:
            # Take subset
            return vertices[:n]
    
    # For larger n, add more layers
    positions = list(vertices)
    # Add atoms in shells
    shell_radii = [a * 0.72, a * 1.44, a * 2.16]
    for r in shell_radii:
        if len(positions) >= n:
            break
        # Add points on sphere of radius r
        num_to_add = min(n - len(positions), 20)
        for _ in range(num_to_add):
            # Random point on sphere
            theta = np.random.uniform(0, 2 * np.pi)
            phi_angle = np.random.uniform(0, np.pi)
            p = r * np.array([
                np.sin(phi_angle) * np.cos(theta),
                np.sin(phi_angle) * np.sin(theta),
                np.cos(phi_angle)
            ])
            if all(np.linalg.norm(p - q) >= 0.5 * a for q in positions):
                positions.append(p)
    
    positions = np.array(positions[:n])
    return positions


def _small_cluster_template(n, kind):
    """Generate a template based on known small cluster structures."""
    # For small clusters, use known optimal or near-optimal structures
    if n <= 13:
        return _icosahedron_positions(n)
    elif n <= 55:
        # Mackay icosahedron for n up to 55
        return _icosahedron_positions(n)
    else:
        # For larger clusters, use FCC-like
        return _fcc_positions(n)


def _ensure_valid(positions, n):
    """Ensure positions are valid and have exactly n atoms."""
    if not _validate(positions):
        # Try to fix by spreading out atoms
        rng = np.random.default_rng(42)
        positions = positions.copy()
        for i in range(n):
            for _ in range(100):
                # Check if atom i is too close to others
                too_close = False
                for j in range(n):
                    if i != j and np.linalg.norm(positions[i] - positions[j]) < 0.35:
                        too_close = True
                        break
                if not too_close:
                    break
                # Move atom i randomly
                positions[i] += rng.uniform(-0.5, 0.5, 3)
        if not _validate(positions):
            # Last resort: regenerate
            positions = _random_compact(n, seed=123)
    return positions[:n]


def construct(instance):
    """Find global minimum energy structure for Sutton-Chen clusters."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance["kind"]
    
    # Generate multiple starting configurations
    start_configs = []
    
    # FCC-based
    try:
        fcc = _fcc_positions(n)
        if _validate(fcc):
            start_configs.append(fcc)
    except:
        pass
    
    # HCP-based
    try:
        hcp = _hcp_positions(n)
        if _validate(hcp):
            start_configs.append(hcp)
    except:
        pass
    
    # BCC-based
    try:
        bcc = _bcc_positions(n)
        if _validate(bcc):
            start_configs.append(bcc)
    except:
        pass
    
    # Icosahedron-based
    try:
        ico = _icosahedron_positions(n)
        if _validate(ico):
            start_configs.append(ico)
    except:
        pass
    
    # Random compact configurations
    for seed in range(8):
        try:
            rand = _random_compact(n, seed=seed)
            if _validate(rand):
                start_configs.append(rand)
        except:
            pass
    
    # Small cluster template
    try:
        template = _small_cluster_template(n, kind)
        if _validate(template):
            start_configs.append(template)
    except:
        pass
    
    if not start_configs:
        # Fallback
        start_configs = [_random_compact(n, seed=0)]
    
    # Relax each configuration and keep the best
    best_energy = np.inf
    best_positions = None
    
    for config in start_configs:
        if _time_left() < 10:
            break
        
        config = _ensure_valid(config, n)
        
        try:
            energy, positions = relax(config, 'sutton_chen', kind)
            if _validate(positions) and energy < best_energy:
                best_energy = energy
                best_positions = positions
        except:
            continue
    
    # If we have time, try more random restarts
    while _time_left() > 30:
        seed = int(time.time() * 1000) % 100000
        try:
            rand = _random_compact(n, seed=seed)
            if _validate(rand):
                rand = _ensure_valid(rand, n)
                energy, positions = relax(rand, 'sutton_chen', kind)
                if _validate(positions) and energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except:
            continue
    
    if best_positions is None:
        # Last resort
        best_positions = _random_compact(n, seed=0)
        best_positions = _ensure_valid(best_positions, n)
    
    return best_positions
# EVOLVE-BLOCK-END
