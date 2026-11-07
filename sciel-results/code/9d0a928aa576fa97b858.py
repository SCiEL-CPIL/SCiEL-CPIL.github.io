"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Time management
_TOTAL_TIME_LIMIT = 295.0  # safety margin before 300s kill
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
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.35


def _fcc_positions(n, spacing=1.0):
    """Generate n positions in an FCC-like compact arrangement."""
    if n == 0:
        return np.zeros((0, 3))
    
    # FCC lattice vectors
    a = spacing
    # Generate FCC points in a cube
    points = []
    # Use a cubic shell approach for FCC
    # FCC: simple cubic + face centers
    max_shell = int(np.ceil((n * 4 / np.pi) ** (1/3))) + 2
    
    # Generate candidate FCC positions
    candidates = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                # Simple cubic lattice points
                candidates.append(np.array([i, j, k]) * a)
                # Face-centered positions
                if i >= 0 and j >= 0:
                    candidates.append((np.array([i, j, k]) + np.array([0.5, 0.5, 0])) * a)
                if i >= 0 and k >= 0:
                    candidates.append((np.array([i, j, k]) + np.array([0.5, 0, 0.5])) * a)
                if j >= 0 and k >= 0:
                    candidates.append((np.array([i, j, k]) + np.array([0, 0.5, 0.5])) * a)
    
    candidates = np.array(candidates)
    
    # Sort by distance from center
    dists = np.linalg.norm(candidates, axis=1)
    indices = np.argsort(dists)
    
    # Select n closest points, ensuring minimum distance
    selected = []
    for idx in indices:
        p = candidates[idx]
        if all(np.linalg.norm(p - q) >= 0.45 * a for q in selected):
            selected.append(p)
        if len(selected) == n:
            break
    
    if len(selected) < n:
        # Fill remaining with random points
        rng = np.random.default_rng(42)
        while len(selected) < n:
            p = rng.uniform(-max_shell * a, max_shell * a, 3)
            if all(np.linalg.norm(p - q) >= 0.45 * a for q in selected):
                selected.append(p)
    
    return np.array(selected[:n])


def _hcp_positions(n, spacing=1.0):
    """Generate n positions in an HCP-like compact arrangement."""
    if n == 0:
        return np.zeros((0, 3))
    
    a = spacing
    c = a * np.sqrt(8/3)  # ideal HCP c/a ratio
    
    positions = []
    # Build HCP layer by layer
    # Each layer is a hexagonal close-packed arrangement
    layer_sizes = [1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
    
    layer_idx = 0
    while len(positions) < n:
        # Determine how many atoms in this layer
        if layer_idx == 0:
            layer_n = 1
        else:
            # Hexagonal number: 1 + 3*k*(k+1) for k layers
            k = layer_idx
            layer_n = 3 * k * (k + 1)
        
        # Cap layer size
        remaining = n - len(positions)
        layer_n = min(layer_n, remaining)
        
        # Generate hexagonal arrangement for this layer
        layer_points = []
        if layer_idx == 0:
            layer_points.append(np.array([0, 0, layer_idx * c]))
        else:
            # Hexagonal packing
            for ring in range(layer_idx + 1):
                if ring == 0:
                    pts = [np.array([0, 0])]
                else:
                    pts = []
                    for i in range(6 * ring):
                        angle = i * np.pi / (3 * ring)
                        r = ring * a * np.sqrt(3) / 2
                        pts.append(np.array([r * np.cos(angle), r * np.sin(angle)]))
                for pt in pts:
                    layer_points.append(np.append(pt, layer_idx * c))
        
        # Alternate AB stacking
        offset = np.array([0, 0]) if layer_idx % 2 == 0 else np.array([a * np.sqrt(3)/2, a/2])
        layer_points = [p + np.array([offset[0], offset[1], 0]) for p in layer_points]
        
        positions.extend(layer_points)
        layer_idx += 1
    
    return np.array(positions[:n])


def _bcc_positions(n, spacing=1.0):
    """Generate n positions in a BCC-like compact arrangement."""
    if n == 0:
        return np.zeros((0, 3))
    
    a = spacing
    positions = []
    
    # BCC: simple cubic + body center
    max_shell = int(np.ceil((n / 2) ** (1/3))) + 2
    
    candidates = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                candidates.append(np.array([i, j, k]) * a)
                # Body center
                candidates.append((np.array([i, j, k]) + np.array([0.5, 0.5, 0.5])) * a)
    
    candidates = np.array(candidates)
    
    # Sort by distance from center
    dists = np.linalg.norm(candidates, axis=1)
    indices = np.argsort(dists)
    
    # Select n closest points, ensuring minimum distance
    selected = []
    for idx in indices:
        p = candidates[idx]
        if all(np.linalg.norm(p - q) >= 0.45 * a for q in selected):
            selected.append(p)
        if len(selected) == n:
            break
    
    if len(selected) < n:
        rng = np.random.default_rng(42)
        while len(selected) < n:
            p = rng.uniform(-max_shell * a, max_shell * a, 3)
            if all(np.linalg.norm(p - q) >= 0.45 * a for q in selected):
                selected.append(p)
    
    return np.array(selected[:n])


def _icosahedron_positions(n, spacing=1.0):
    """Generate n positions in an icosahedral-like arrangement."""
    if n == 0:
        return np.zeros((0, 3))
    
    # Start with icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    a = spacing / (2 * phi)
    
    ico_vertices = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    ico_vertices = np.array(ico_vertices) * a
    ico_vertices = ico_vertices / np.linalg.norm(ico_vertices, axis=1, keepdims=True) * spacing
    
    positions = [np.array([0.0, 0.0, 0.0])]  # Center
    positions.extend(ico_vertices.tolist())  # 12 vertices = 13 total
    
    # Add more shells
    if n > 13:
        # Add a second shell at ~2x radius
        rng = np.random.default_rng(42)
        radius2 = spacing * 1.8
        while len(positions) < n:
            p = rng.uniform(-radius2, radius2, 3)
            if np.linalg.norm(p) <= radius2:
                if all(np.linalg.norm(p - q) >= 0.45 * spacing for q in positions):
                    positions.append(p)
    
    return np.array(positions[:n])


def _random_compact(n, spacing=0.72, seed=None):
    """Generate n positions in a random compact arrangement."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
        attempts += 1
    
    # If we couldn't place enough, relax the constraint
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])


def _ensure_valid(positions, spacing=0.72):
    """Ensure positions are valid; if not, try to fix them."""
    if not _validate(positions):
        # Perturb and try again
        rng = np.random.default_rng(12345)
        for _ in range(10):
            perturbed = positions + rng.normal(0, 0.05, positions.shape)
            if _validate(perturbed):
                return perturbed
        # Last resort: spread out
        scale = 1.1
        scaled = positions * scale
        if _validate(scaled):
            return scaled
    return positions


def construct(instance):
    """Find low-energy configuration for Sutton-Chen cluster."""
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    potential = instance["potential"]
    
    # Generate diverse initial configurations
    initializations = []
    
    # 1. FCC
    try:
        init = _fcc_positions(n, spacing=0.72)
        if _validate(init):
            initializations.append(('fcc', init))
    except:
        pass
    
    # 2. HCP
    try:
        init = _hcp_positions(n, spacing=0.72)
        if _validate(init):
            initializations.append(('hcp', init))
    except:
        pass
    
    # 3. BCC
    try:
        init = _bcc_positions(n, spacing=0.72)
        if _validate(init):
            initializations.append(('bcc', init))
    except:
        pass
    
    # 4. Icosahedral
    try:
        init = _icosahedron_positions(n, spacing=0.72)
        if _validate(init):
            initializations.append(('ico', init))
    except:
        pass
    
    # 5-8. Random compact with different seeds
    for seed in [42, 123, 456, 789]:
        try:
            init = _random_compact(n, spacing=0.72, seed=seed)
            if _validate(init):
                initializations.append((f'rand{seed}', init))
        except:
            pass
    
    # Relax each initialization and keep the best
    best_energy = np.inf
    best_positions = None
    
    for name, init in initializations:
        if _time_left() < 30:
            break
        
        try:
            energy, positions = relax(init, potential, kind)
            if np.isfinite(energy) and _validate(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except:
            pass
    
    # If we still have time, try more random restarts
    if best_positions is not None and _time_left() > 60:
        rng = np.random.default_rng(999)
        for i in range(20):
            if _time_left() < 30:
                break
            try:
                init = _random_compact(n, spacing=0.72, seed=999 + i)
                if _validate(init):
                    energy, positions = relax(init, potential, kind)
                    if np.isfinite(energy) and _validate(positions):
                        if energy < best_energy:
                            best_energy = energy
                            best_positions = positions
            except:
                pass
    
    if best_positions is None:
        # Fallback: return a simple valid configuration
        best_positions = _random_compact(n, spacing=0.72, seed=0)
        best_positions = _ensure_valid(best_positions)
    
    return best_positions
# EVOLVE-BLOCK-END
