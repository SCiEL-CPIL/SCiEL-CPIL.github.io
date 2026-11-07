"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Time management
_TOTAL_TIME_LIMIT = 290.0  # safety margin before 300s kill
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


def _fcc_positions(n, a=0.72):
    """Generate FCC-like positions for n atoms."""
    positions = []
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


def _hcp_positions(n, a=0.72):
    """Generate HCP-like positions for n atoms."""
    positions = []
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


def _bcc_positions(n, a=0.72):
    """Generate BCC-like positions for n atoms."""
    positions = []
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


def _fibonacci_sphere(n, radius, seed=None):
    """Generate points on a sphere using Fibonacci distribution."""
    rng = np.random.default_rng(seed)
    points = []
    if n == 1:
        return np.array([[0, 0, radius]])
    
    # Fibonacci sphere algorithm
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
        radius_at_y = np.sqrt(1 - y * y)
        theta = golden_angle * i
        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y
        points.append([x * radius, y * radius, z * radius])
    
    return np.array(points)


def _nested_shell_positions(n, seed=None):
    """Generate positions using nested spherical shells (Fibonacci distribution)."""
    rng = np.random.default_rng(seed)
    
    # Determine shell structure
    # For clusters, typical shell sizes are: 1, 12, 42, 92, ...
    # We'll use a simple approach: distribute atoms across shells
    
    positions = []
    
    # First atom at center
    if n >= 1:
        positions.append([0, 0, 0])
    
    # Determine how many shells and how many atoms per shell
    # Use a heuristic based on n
    shell_configs = {
        13: [1, 12],
        55: [1, 12, 42],
        147: [1, 12, 42, 92],
    }
    
    # Find the closest shell configuration
    best_config = None
    best_diff = float('inf')
    for config_n, config in shell_configs.items():
        diff = abs(config_n - n)
        if diff < best_diff:
            best_diff = diff
            best_config = config
    
    if best_config is None:
        # Create a custom configuration
        # Start with 1 atom at center
        remaining = n - 1
        # Add shells of increasing size
        shell_sizes = [12]
        while remaining > 0:
            if remaining <= 12:
                shell_sizes.append(remaining)
                remaining = 0
            else:
                shell_sizes.append(12)
                remaining -= 12
        best_config = [1] + shell_sizes
    
    # Generate positions for each shell
    current = 0
    for shell_idx, shell_size in enumerate(best_config):
        if current >= n:
            break
        
        if shell_idx == 0:
            # Center atom
            if current < n:
                positions.append([0, 0, 0])
                current += 1
        else:
            # Shell at radius r
            # Radius increases with shell index
            base_radius = 0.72
            radius = base_radius * np.sqrt(shell_idx)
            
            # Generate Fibonacci sphere points for this shell
            shell_points = _fibonacci_sphere(shell_size, radius, seed=seed + shell_idx)
            
            for p in shell_points:
                if current < n:
                    positions.append(p)
                    current += 1
    
    # If we have more than n positions, truncate
    positions = np.array(positions[:n])
    
    return positions


def _deformed_lattice(n, kind, seed=None):
    """Generate deformed lattice positions."""
    rng = np.random.default_rng(seed)
    
    # Start with FCC positions
    positions = _fcc_positions(n)
    
    # Add small random perturbations
    noise_scale = 0.05
    positions += rng.uniform(-noise_scale, noise_scale, positions.shape)
    
    return positions


def _growth_based_positions(n, seed=None):
    """Generate positions by growing from a small core."""
    rng = np.random.default_rng(seed)
    
    # Start with a small icosahedron core (13 atoms)
    if n <= 13:
        return _icosahedron_positions(n)
    
    core = _icosahedron_positions(13)
    positions = list(core)
    
    # Add atoms one by one at optimal positions
    while len(positions) < n:
        # Try to add atom at a position that minimizes energy
        # Sample candidate positions on a sphere around the cluster
        center = np.mean(positions, axis=0)
        max_dist = np.max([np.linalg.norm(p - center) for p in positions])
        
        best_pos = None
        best_energy = float('inf')
        
        # Sample candidate positions
        for _ in range(20):
            # Random direction
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(0, np.pi)
            r = max_dist * rng.uniform(0.8, 1.2)
            
            candidate = center + r * np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])
            
            # Check minimum distance
            min_dist = min([np.linalg.norm(candidate - p) for p in positions])
            if min_dist < 0.5:
                continue
            
            # Estimate energy (simplified)
            energy = 0
            for p in positions:
                d = np.linalg.norm(candidate - p)
                if d > 0.35:
                    energy += d ** (-12) - 144.41 * np.sqrt(d ** (-6))
            
            if energy < best_energy:
                best_energy = energy
                best_pos = candidate
        
        if best_pos is not None:
            positions.append(best_pos)
        else:
            # Fallback: add at random position
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(0, np.pi)
            r = max_dist * 1.1
            positions.append(center + r * np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ]))
    
    return np.array(positions)


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
    
    # Nested shell positions
    try:
        shell = _nested_shell_positions(n, seed=42)
        if _validate(shell):
            start_configs.append(shell)
    except:
        pass
    
    # Deformed lattice
    try:
        deformed = _deformed_lattice(n, kind, seed=42)
        if _validate(deformed):
            start_configs.append(deformed)
    except:
        pass
    
    # Growth-based positions
    try:
        growth = _growth_based_positions(n, seed=42)
        if _validate(growth):
            start_configs.append(growth)
    except:
        pass
    
    # Random compact configurations
    for seed in range(6):
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
