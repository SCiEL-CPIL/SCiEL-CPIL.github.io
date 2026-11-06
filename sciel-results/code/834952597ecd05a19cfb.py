"""Improved Dzugutov cluster search with Mackay icosahedra, basin hopping, and kick-and-relax."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
TIME_LIMIT = 290.0  # Safety margin before 300s limit


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def _mckay_icosahedron(n, scale=1.0):
    """Generate Mackay icosahedral cluster with up to n atoms.
    
    Mackay icosahedra have shells: center (1), inner icosahedron (12),
    outer icosahedron (42), next (92), etc.
    For Dzugutov potential, icosahedral/polytetrahedral order is favored.
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of regular icosahedron (unit)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # 30 edge midpoints of icosahedron (for 42-atom shell)
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            # Check if i,j are connected (distance = 2 in original coords before normalization)
            d = np.linalg.norm(verts[i] - verts[j])
            if abs(d - 2.0) < 0.01:
                edges.append((i, j))
    
    edge_mids = np.array([(verts_unit[i] + verts_unit[j]) / 2.0 for i, j in edges])
    edge_mids = edge_mids / np.linalg.norm(edge_mids, axis=1, keepdims=True)
    
    # 20 face centers (for additional points if needed)
    # Faces of icosahedron
    faces = [
        [0, 1, 8], [0, 8, 4], [0, 4, 2], [0, 2, 6], [0, 6, 1],
        [1, 6, 9], [1, 9, 5], [1, 5, 3], [1, 3, 8],
        [2, 4, 10], [2, 10, 7], [2, 7, 6],
        [3, 5, 11], [3, 11, 10], [3, 10, 4], [3, 4, 8],
        [5, 9, 11], [5, 11, 7], [5, 7, 3],
        [7, 11, 10], [7, 10, 2], [7, 2, 6], [7, 6, 9], [7, 9, 5]
    ]
    # Fix: icosahedron has 20 faces
    face_centers = []
    for f in faces:
        fc = (verts_unit[f[0]] + verts_unit[f[1]] + verts_unit[f[2]]) / 3.0
        fc = fc / np.linalg.norm(fc)
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    
    points = []
    
    # Shell 0: center
    if n >= 1:
        points.append([0.0, 0.0, 0.0])
    
    # Shell 1: 12 vertices at radius r1
    if n >= 13:
        r1 = scale * 0.55
        points.extend((verts_unit * r1).tolist())
    
    # Shell 2: 30 edge midpoints at radius r2
    if n >= 43:
        r2 = scale * 0.85
        points.extend((edge_mids * r2).tolist())
    
    # Shell 3: 20 face centers at radius r3 (if needed)
    if n >= 63:
        r3 = scale * 0.75
        points.extend((face_centers * r3).tolist())
    
    # Shell 4: Another icosahedron at larger radius
    if n >= 93:
        r4 = scale * 1.1
        points.extend((verts_unit * r4).tolist())
    
    # Convert to array
    points = np.array(points)
    
    # If we have more than n, trim
    if len(points) >= n:
        return points[:n].copy()
    
    # If we have fewer, add more points on spheres
    rng = np.random.default_rng(42)
    while len(points) < n:
        theta = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-1, 1)
        r = np.sqrt(1 - z**2)
        radius = scale * (0.6 + 0.3 * rng.random())
        p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * radius
        points = np.vstack([points, p])
    
    return points[:n].copy()


def _icosahedral_shells(n, scale=1.0):
    """Alternative icosahedral shell construction for Dzugutov."""
    phi = (1 + np.sqrt(5)) / 2
    
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    points = []
    
    # Try different shell configurations
    # Configuration A: center + 12 + 30 + 20 + ...
    configs = [
        # (radius_factor, count, type)
        [(0.0, 1, 'center'), (0.5, 12, 'verts'), (0.8, 30, 'edges')],
        [(0.0, 1, 'center'), (0.55, 12, 'verts'), (0.9, 12, 'verts'), (1.2, 12, 'verts')],
        [(0.0, 1, 'center'), (0.45, 12, 'verts'), (0.75, 12, 'verts'), (1.05, 12, 'verts'), (1.35, 12, 'verts')],
    ]
    
    for config in configs:
        pts = []
        for radius_f, count, ptype in config:
            if len(pts) >= n:
                break
            if ptype == 'center':
                pts.append([0.0, 0.0, 0.0])
            elif ptype == 'verts':
                r = scale * radius_f
                shell = (verts_unit * r)
                pts.extend(shell.tolist())
            elif ptype == 'edges':
                # 30 edge midpoints
                edges = []
                for i in range(12):
                    for j in range(i + 1, 12):
                        d = np.linalg.norm(verts[i] - verts[j])
                        if abs(d - 2.0) < 0.01:
                            edges.append((i, j))
                edge_mids = np.array([(verts_unit[i] + verts_unit[j]) / 2.0 for i, j in edges])
                edge_mids = edge_mids / np.linalg.norm(edge_mids, axis=1, keepdims=True)
                r = scale * radius_f
                shell = (edge_mids * r)
                pts.extend(shell.tolist())
        
        if len(pts) >= n:
            pts = np.array(pts[:n])
            # Check minimum distance
            valid = True
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    if np.linalg.norm(pts[i] - pts[j]) < 0.45:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return pts
    
    # Fallback: Mackay icosahedron
    return _mckay_icosahedron(n, scale)


def _random_compact(n, seed, spacing=1.1, radius_factor=0.75):
    """Generate a random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * radius_factor
    points = []
    min_dist = 0.8 * spacing
    attempts = 0
    max_attempts = n * 300
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    
    # If we couldn't place all atoms with strict constraint, relax it
    if len(points) < n:
        min_dist = 0.5
        attempts = 0
        while len(points) < n and attempts < max_attempts:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= min_dist for q in points):
                    points.append(p)
            attempts += 1
    
    # Last resort: just add random points
    while len(points) < n:
        p = rng.uniform(-radius * 0.5, radius * 0.5, 3)
        points.append(p)
    
    return np.array(points[:n])


def _kick_and_relax(positions, potential, parameter, kick_sigma, seed):
    """Perturb positions and relax."""
    rng = np.random.default_rng(seed)
    perturbed = positions + rng.normal(0, kick_sigma, positions.shape)
    energy, relaxed = relax(perturbed, potential, parameter)
    return energy, relaxed


def _basin_hopping(positions, potential, parameter, n_kicks=5, kick_sigma=0.15, seed=0):
    """Simple basin hopping around a starting position."""
    best_energy, best_pos = positions.copy(), positions.copy()
    # Get initial energy
    e0, _ = relax(positions, potential, parameter)
    best_energy = e0
    
    for i in range(n_kicks):
        if _time_remaining() < 15:
            break
        energy, new_pos = _kick_and_relax(best_pos, potential, parameter, kick_sigma, seed + i)
        if energy < best_energy:
            best_energy = energy
            best_pos = new_pos
    
    return best_energy, best_pos


def construct(instance):
    """Construct low-energy Dzugutov cluster using Mackay icosahedra, basin hopping, and kicks."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    # Scale factor for the cluster
    base_scale = size ** (1.0 / 3.0) * 0.42
    
    # Strategy 1: Mackay icosahedron (primary for Dzugutov)
    ico_points = _mckay_icosahedron(size, scale=base_scale)
    energy, positions = relax(ico_points, potential, parameter)
    if energy < best_energy:
        best_energy = energy
        best_positions = positions.copy()
    
    if _time_remaining() < 60:
        return best_positions
    
    # Strategy 2: Basin hopping around Mackay icosahedron
    bh_energy, bh_pos = _basin_hopping(
        best_positions, potential, parameter,
        n_kicks=8, kick_sigma=0.12, seed=size
    )
    if bh_energy < best_energy:
        best_energy = bh_energy
        best_positions = bh_pos.copy()
    
    if _time_remaining() < 55:
        return best_positions
    
    # Strategy 3: Alternative icosahedral shells
    ico2_points = _icosahedral_shells(size, scale=base_scale * 1.05)
    energy, positions = relax(ico2_points, potential, parameter)
    if energy < best_energy:
        best_energy = energy
        best_positions = positions.copy()
    
    if _time_remaining() < 50:
        return best_positions
    
    # Strategy 4: Basin hopping around alternative icosahedron
    bh_energy2, bh_pos2 = _basin_hopping(
        best_positions, potential, parameter,
        n_kicks=6, kick_sigma=0.15, seed=size + 100
    )
    if bh_energy2 < best_energy:
        best_energy = bh_energy2
        best_positions = bh_pos2.copy()
    
    if _time_remaining() < 45:
        return best_positions
    
    # Strategy 5: Random compact initialization
    rc_points = _random_compact(size, seed=size, spacing=1.1, radius_factor=0.75)
    energy, positions = relax(rc_points, potential, parameter)
    if energy < best_energy:
        best_energy = energy
        best_positions = positions.copy()
    
    if _time_remaining() < 40:
        return best_positions
    
    # Strategy 6: Basin hopping around random compact
    bh_energy3, bh_pos3 = _basin_hopping(
        best_positions, potential, parameter,
        n_kicks=5, kick_sigma=0.18, seed=size + 200
    )
    if bh_energy3 < best_energy:
        best_energy = bh_energy3
        best_positions = bh_pos3.copy()
    
    if _time_remaining() < 35:
        return best_positions
    
    # Strategy 7: Different scale icosahedron
    ico_scale2 = base_scale * 0.9
    ico3_points = _mckay_icosahedron(size, scale=ico_scale2)
    energy, positions = relax(ico3_points, potential, parameter)
    if energy < best_energy:
        best_energy = energy
        best_positions = positions.copy()
    
    if _time_remaining() < 30:
        return best_positions
    
    # Strategy 8: Basin hopping with larger kicks
    bh_energy4, bh_pos4 = _basin_hopping(
        best_positions, potential, parameter,
        n_kicks=6, kick_sigma=0.25, seed=size + 300
    )
    if bh_energy4 < best_energy:
        best_energy = bh_energy4
        best_positions = bh_pos4.copy()
    
    if _time_remaining() < 25:
        return best_positions
    
    # Strategy 9: Second random compact
    rc_points2 = _random_compact(size, seed=size + 500, spacing=1.0, radius_factor=0.8)
    energy, positions = relax(rc_points2, potential, parameter)
    if energy < best_energy:
        best_energy = energy
        best_positions = positions.copy()
    
    if _time_remaining() < 20:
        return best_positions
    
    # Strategy 10: Final basin hopping on best
    bh_energy5, bh_pos5 = _basin_hopping(
        best_positions, potential, parameter,
        n_kicks=8, kick_sigma=0.10, seed=size + 400
    )
    if bh_energy5 < best_energy:
        best_energy = bh_energy5
        best_positions = bh_pos5.copy()
    
    if _time_remaining() < 15:
        return best_positions
    
    # Strategy 11: Larger scale icosahedron
    ico_scale3 = base_scale * 1.1
    ico4_points = _mckay_icosahedron(size, scale=ico_scale3)
    energy, positions = relax(ico4_points, potential, parameter)
    if energy < best_energy:
        best_energy = energy
        best_positions = positions.copy()
    
    if _time_remaining() < 10:
        return best_positions
    
    # Strategy 12: Final basin hopping
    bh_energy6, bh_pos6 = _basin_hopping(
        best_positions, potential, parameter,
        n_kicks=5, kick_sigma=0.20, seed=size + 500
    )
    if bh_energy6 < best_energy:
        best_energy = bh_energy6
        best_positions = bh_pos6.copy()
    
    return best_positions
# EVOLVE-BLOCK-END
