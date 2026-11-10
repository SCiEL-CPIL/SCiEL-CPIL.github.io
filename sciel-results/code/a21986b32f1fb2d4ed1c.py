"""Dzugutov cluster global minima: portfolio of structural heuristics + parallel relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 320.0  # safety margin before 330s kill


def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)


def _generate_mackay_icosahedron(n, rng):
    """Generate a Mackay icosahedral cluster of n atoms."""
    # Icosahedral vertices and edges for shell generation
    # Use golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0])
    
    # Center atom
    points = [np.zeros(3)]
    
    # First shell: 12 atoms at icosahedral vertices
    for v in verts:
        points.append(v * 1.5)
    
    # Second shell: 30 atoms (edges of icosahedron, 30 edges)
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 2.0:  # edges of icosahedron
                edges.append((i, j))
    
    for i, j in edges[:30]:
        mid = (verts[i] + verts[j]) / 2
        mid = mid / np.linalg.norm(mid)
        points.append(mid * 2.5)
    
    # Third shell: 32 atoms (faces of icosahedron)
    faces = [
        (0, 1, 8), (0, 1, 10), (0, 2, 8), (0, 2, 11),
        (1, 3, 8), (1, 3, 10), (1, 4, 8), (1, 4, 5),
        (2, 3, 11), (2, 3, 10), (2, 6, 11), (2, 6, 7),
        (3, 7, 11), (3, 7, 10), (3, 4, 5), (3, 4, 8),
        (4, 5, 8), (4, 5, 9), (5, 9, 8), (5, 9, 10),
        (6, 7, 11), (6, 7, 12), (7, 12, 11), (7, 12, 10),
        (8, 9, 13), (9, 13, 10), (9, 13, 14), (13, 14, 10),
        (13, 14, 15), (14, 15, 10), (15, 10, 14)
    ]
    
    # For a proper Mackay icosahedron, we need more careful construction
    # Let's use a simpler approach: generate points in icosahedral shells
    points = [np.zeros(3)]
    
    # Shell 1: 12 atoms
    for v in verts:
        points.append(v * 1.4)
    
    # Shell 2: 30 atoms (midpoints of edges)
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 2.0:
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid)
                points.append(mid * 2.2)
    
    # Shell 3: 32 atoms (face centers)
    # Find faces by checking triples of vertices that form triangles
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[k] - verts[i])
                if d1 < 2.0 and d2 < 2.0 and d3 < 2.0:
                    center = (verts[i] + verts[j] + verts[k]) / 3
                    center = center / np.linalg.norm(center)
                    points.append(center * 2.8)
    
    # Trim or pad to n atoms
    if len(points) > n:
        points = points[:n]
    elif len(points) < n:
        while len(points) < n:
            p = rng.uniform(-3.5, 3.5, 3)
            if all(np.linalg.norm(p - q) >= 0.6 for q in points):
                points.append(p)
    
    return np.array(points[:n])


def _generate_fcc(n, rng):
    """Generate an FCC cluster of n atoms."""
    points = []
    a = 1.4  # lattice spacing
    
    # FCC basis
    basis = [
        [0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]
    ]
    
    # Generate FCC lattice points
    max_idx = 3
    for i in range(-max_idx, max_idx+1):
        for j in range(-max_idx, max_idx+1):
            for k in range(-max_idx, max_idx+1):
                for b in basis:
                    p = np.array([i, j, k]) * a + np.array(b) * a
                    if len(points) < n:
                        if all(np.linalg.norm(p - q) >= 0.6 for q in points):
                            points.append(p)
    
    # Center the cluster
    if points:
        center = np.mean(points, axis=0)
        points = [p - center for p in points]
    
    # Pad with random points if needed
    while len(points) < n:
        p = rng.uniform(-2.5, 2.5, 3)
        if all(np.linalg.norm(p - q) >= 0.6 for q in points):
            points.append(p)
    
    return np.array(points[:n])


def _generate_random_cluster(n, rng):
    """Generate a random compact cluster."""
    points = []
    radius = 1.1 * n ** (1.0/3.0) * 0.75
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.7 for q in points):
            points.append(p)
    
    return np.array(points)


def construct(instance):
    """Best of multiple structural heuristics, each relaxed locally."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(size + 12345)
    
    best = None
    time_left = _get_time_remaining()
    
    # Generate diverse starting structures
    candidates = []
    
    # 1. Random compact
    candidates.append(_generate_random_cluster(size, rng))
    
    # 2. Mackay icosahedron
    try:
        candidates.append(_generate_mackay_icosahedron(size, rng))
    except:
        pass
    
    # 3. FCC
    try:
        candidates.append(_generate_fcc(size, rng))
    except:
        pass
    
    # 4-7: More random starts with different parameters
    for i in range(4):
        candidates.append(_generate_random_cluster(size, rng))
    
    # Relax all candidates in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for i, cand in enumerate(candidates):
            if _get_time_remaining() < 10:
                break
            fut = executor.submit(relax, cand, potential, parameter)
            futures[fut] = i
        
        for fut in concurrent.futures.as_completed(futures):
            if _get_time_remaining() < 5:
                break
            try:
                energy, positions = fut.result()
                if best is None or energy < best[0]:
                    best = (energy, positions)
            except:
                pass
    
    if best is None:
        # Fallback: return the first candidate
        return candidates[0]
    
    return best[1]
# EVOLVE-BLOCK-END
