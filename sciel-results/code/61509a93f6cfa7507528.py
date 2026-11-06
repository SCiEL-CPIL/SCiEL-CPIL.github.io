# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
_TIME_LIMIT = 290.0  # safety margin before 300s hard kill


def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron (unit circumradius)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0,  1,  phi], [0, -1,  phi], [0,  1, -phi], [0, -1, -phi],
        [ 1,  phi, 0], [-1,  phi, 0], [ 1, -phi, 0], [-1, -phi, 0],
        [ phi, 0,  1], [ phi, 0, -1], [-phi, 0,  1], [-phi, 0, -1],
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms


def _golden_spiral(n, radius):
    """Distribute n points on a sphere of given radius using golden spiral."""
    if n == 0:
        return np.empty((0, 3))
    if n == 1:
        return np.array([[0.0, 0.0, radius]])
    idx = np.arange(n)
    # golden angle
    ga = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - (2.0 * idx + 1.0) / n
    z = np.clip(z, -1.0, 1.0)
    r = np.sqrt(np.maximum(0.0, 1.0 - z**2))
    theta = ga * idx
    x = np.cos(theta) * r
    y = np.sin(theta) * r
    return radius * np.stack([x, y, z], axis=1)


def _build_icosahedral_cluster(n, shell_radii=None):
    """
    Build a cluster by placing atoms on concentric icosahedral shells.
    Returns positions array of shape (n, 3).
    """
    if shell_radii is None:
        # Default: inner core + outer shell
        # For 38-60 atoms, use 1+12+30 or 1+12+32 etc.
        # Let's use a scheme: try to fit into icosahedral shells
        # Shell 0: 1 atom at center (optional)
        # Shell 1: 12 atoms (icosahedron)
        # Shell 2: up to 30+ atoms (cuboctahedron / truncated icosahedron vertices)
        # For simplicity, use golden spiral on spheres
        
        # Strategy: place first 12 on inner icosahedron, rest on outer sphere
        inner_n = min(12, n)
        outer_n = n - inner_n
        inner_r = 1.0
        outer_r = 1.8  # ratio ~ sqrt(3) for icosahedral shells
        
        positions = []
        if inner_n > 0:
            verts = _icosahedron_vertices()
            # Take first inner_n vertices
            positions.append(verts[:inner_n] * inner_r)
        if outer_n > 0:
            positions.append(_golden_spiral(outer_n, outer_r))
        
        if not positions:
            return np.empty((0, 3))
        return np.vstack(positions)
    
    # Custom shell radii: list of (count, radius)
    positions = []
    for count, radius in shell_radii:
        if count <= 0:
            continue
        if count == 1 and radius == 0:
            positions.append(np.array([[0.0, 0.0, 0.0]]))
        elif count <= 12:
            verts = _icosahedron_vertices()
            positions.append(verts[:count] * radius)
        else:
            positions.append(_golden_spiral(count, radius))
    if not positions:
        return np.empty((0, 3))
    return np.vstack(positions)


def _try_relax(positions, potential, parameter, max_time=15.0):
    """Relax positions, with a time guard."""
    try:
        energy, relaxed = relax(positions, potential, parameter)
        return energy, relaxed
    except Exception:
        return np.inf, positions


def _random_perturbation(positions, rng, sigma=0.1):
    """Add small random noise to break symmetry."""
    return positions + rng.normal(0, sigma, positions.shape)


def construct(instance):
    """
    Find low-energy Dzugutov cluster by starting from icosahedral-like
    structures and relaxing.
    """
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(42)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate several candidate starting structures
    candidates = []
    
    # 1. Basic icosahedral shell
    candidates.append(_build_icosahedral_cluster(n))
    
    # 2. Different shell ratios
    if n <= 40:
        # 1 + 12 + (n-13)
        inner_outer = [(1, 0.0), (12, 1.0), (n - 13, 1.8)]
        candidates.append(_build_icosahedral_cluster(n, inner_outer))
    elif n <= 52:
        # 1 + 12 + 30 + (n-43)
        inner_outer = [(1, 0.0), (12, 1.0), (30, 1.8), (max(0, n - 43), 2.5)]
        candidates.append(_build_icosahedral_cluster(n, inner_outer))
    else:
        # 1 + 12 + 30 + 32 + (n-75) but we only go to 60
        # 1 + 12 + 30 + (n-43)
        inner_outer = [(1, 0.0), (12, 1.0), (30, 1.8), (max(0, n - 43), 2.5)]
        candidates.append(_build_icosahedral_cluster(n, inner_outer))
    
    # 3. Pure golden spiral (no icosahedral bias)
    candidates.append(_golden_spiral(n, 2.0))
    
    # 4. Two-shell with different radii
    half = n // 2
    candidates.append(np.vstack([
        _golden_spiral(half, 1.2),
        _golden_spiral(n - half, 2.2)
    ]))
    
    # 5. Icosahedral with perturbation
    base = _build_icosahedral_cluster(n)
    candidates.append(_random_perturbation(base, rng, 0.05))
    
    # 6. Another shell configuration
    if n >= 45:
        candidates.append(_build_icosahedral_cluster(n, [(12, 1.0), (n - 12, 1.9)]))
    
    # Relax each candidate, keeping the best
    for i, pos in enumerate(candidates):
        if _time_remaining() < 5.0:
            break
        
        # Ensure minimum distance constraint
        # Check pairwise distances
        valid = True
        for a in range(len(pos)):
            for b in range(a + 1, len(pos)):
                if np.linalg.norm(pos[a] - pos[b]) < 0.5:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Add small perturbation to fix overlaps
            pos = _random_perturbation(pos, rng, 0.1)
        
        energy, relaxed = _try_relax(pos, potential, parameter)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = relaxed
    
    # If we have time, try more random restarts
    if best_positions is None:
        # Fallback: random compact cluster
        radius = 1.1 * n ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(
                np.linalg.norm(p - q) >= 0.55 for q in points
            ):
                points.append(p)
        best_positions = np.array(points)
        best_energy, best_positions = _try_relax(best_positions, potential, parameter)
    
    # Additional random restarts if time permits
    attempts = 0
    while _time_remaining() > 10.0 and attempts < 20:
        attempts += 1
        # Random compact cluster
        radius = 1.1 * n ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(
                np.linalg.norm(p - q) >= 0.55 for q in points
            ):
                points.append(p)
        if len(points) < n:
            continue
        pos = np.array(points)
        energy, relaxed = _try_relax(pos, potential, parameter)
        if energy < best_energy:
            best_energy = energy
            best_positions = relaxed
    
    # Final: ensure minimum distance
    if best_positions is not None:
        # Check and fix any close pairs
        for a in range(len(best_positions)):
            for b in range(a + 1, len(best_positions)):
                d = np.linalg.norm(best_positions[a] - best_positions[b])
                if d < 0.5:
                    # Push apart
                    mid = (best_positions[a] + best_positions[b]) / 2.0
                    diff = best_positions[b] - best_positions[a]
                    if np.linalg.norm(diff) > 1e-10:
                        diff = diff / np.linalg.norm(diff)
                    else:
                        diff = np.array([1.0, 0, 0])
                    shift = diff * (0.5 - d + 0.01) / 2.0
                    best_positions[a] -= shift
                    best_positions[b] += shift
    
    return best_positions
# EVOLVE-BLOCK-END
