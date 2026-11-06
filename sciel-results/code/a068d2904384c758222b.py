import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import time

def _get_icosahedron_vertices(radius=1.0):
    """
    Generate the 12 vertices of a regular icosahedron centered at origin.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi],
        [0, -1, phi],
        [0, 1, -phi],
        [0, -1, -phi],
        [1, phi, 0],
        [-1, phi, 0],
        [1, -phi, 0],
        [-1, -phi, 0],
        [phi, 0, 1],
        [phi, 0, -1],
        [-phi, 0, 1],
        [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    return verts * radius

def _get_icosahedron_edges(verts):
    """
    Get the 30 edges of the icosahedron defined by the 12 vertices.
    Returns list of (i, j) index pairs.
    """
    min_dist = np.inf
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < min_dist:
                min_dist = d
    
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, min_dist, rtol=1e-6):
                edges.append((i, j))
    return edges

def _get_icosahedron_faces(verts, edges):
    """
    Get the 20 faces of the icosahedron.
    Returns list of (i, j, k) index triples.
    """
    adj = {i: set() for i in range(12)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    
    faces = []
    for i in range(12):
        for j in adj[i]:
            if j > i:
                for k in adj[i] & adj[j]:
                    if k > j:
                        faces.append((i, j, k))
    
    return faces

def _build_mackay_55(r1=1.0, r2=1.87):
    """
    Build the perfect 55-atom Mackay icosahedron.
    Returns array of shape (55, 3).
    """
    verts = _get_icosahedron_vertices(1.0)
    edges = _get_icosahedron_edges(verts)
    
    points = [np.array([0.0, 0.0, 0.0])]
    
    for v in verts:
        points.append(v * r1)
    
    shell2 = []
    for v in verts:
        shell2.append(v)
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        shell2.append(mid)
    
    shell2 = np.array(shell2)
    norms = np.linalg.norm(shell2, axis=1)
    shell2 = shell2 / norms[:, np.newaxis] * r2
    
    for v in shell2:
        points.append(v)
    
    return np.array(points[:55])

def _build_icosahedral_cluster(n, r1=1.0, r2=1.87, r3=2.55, seed=0):
    """
    Build an icosahedral cluster for n atoms using shell growth.
    Shell 0: 1 atom (center)
    Shell 1: 12 atoms (icosahedron vertices) at r1
    Shell 2: 42 atoms (12 vertices + 30 edge midpoints) at r2
    Shell 3: 92 atoms (20 face centers + 30 edge midpoints scaled + ...) at r3
    """
    verts = _get_icosahedron_vertices(1.0)
    edges = _get_icosahedron_edges(verts)
    faces = _get_icosahedron_faces(verts, edges)
    
    rng = np.random.default_rng(seed)
    
    # Shell 0: center
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 1: 12 vertices
    shell1 = [v * r1 for v in verts]
    
    # Shell 2: 42 atoms (12 vertices + 30 edge midpoints)
    shell2 = []
    for v in verts:
        shell2.append(v)
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        shell2.append(mid)
    shell2 = np.array(shell2)
    norms = np.linalg.norm(shell2, axis=1)
    shell2 = shell2 / norms[:, np.newaxis] * r2
    
    # Shell 3: 92 atoms (20 face centers + 60 additional)
    # Face centers: 20
    face_centers = []
    for i, j, k in faces:
        fc = (verts[i] + verts[j] + verts[k]) / 3.0
        fc = fc / np.linalg.norm(fc)
        face_centers.append(fc)
    
    # Additional positions for shell 3: use a denser icosahedral pattern
    # Generate more points on the sphere using subdivision
    shell3_points = list(face_centers)
    
    # Add edge midpoints of the subdivided icosahedron
    # For each face, add points at 1/3 and 2/3 along edges
    for i, j, k in faces:
        v1, v2, v3 = verts[i], verts[j], verts[k]
        # Points at 1/3 and 2/3 along each edge
        for (p, q) in [(v1, v2), (v2, v3), (v3, v1)]:
            pt1 = (2*p + q) / 3.0
            pt2 = (p + 2*q) / 3.0
            shell3_points.append(pt1 / np.linalg.norm(pt1))
            shell3_points.append(pt2 / np.linalg.norm(pt2))
    
    # Remove duplicates approximately
    shell3_unique = []
    for p in shell3_points:
        is_dup = False
        for q in shell3_unique:
            if np.linalg.norm(p - q) < 0.05:
                is_dup = True
                break
        if not is_dup:
            shell3_unique.append(p)
    
    shell3 = np.array(shell3_unique)
    norms = np.linalg.norm(shell3, axis=1)
    shell3 = shell3 / norms[:, np.newaxis] * r3
    
    # Now assemble the cluster
    if n <= 1:
        return np.array(points[:n])
    
    if n <= 13:
        points.extend(shell1[:n-1])
        return np.array(points[:n])
    
    if n <= 55:
        points.extend(shell1)
        needed_from_shell2 = n - 13
        # Shuffle shell2 for diversity
        idx = np.arange(len(shell2))
        rng.shuffle(idx)
        points.extend(shell2[idx[:needed_from_shell2]])
        return np.array(points[:n])
    
    # n > 55: use full shell1 and shell2, add from shell3
    points.extend(shell1)
    points.extend(shell2)
    needed_from_shell3 = n - 55
    idx = np.arange(len(shell3))
    rng.shuffle(idx)
    points.extend(shell3[idx[:needed_from_shell3]])
    
    return np.array(points[:n])

def _make_compact_random_seed(n, seed=0):
    """
    Generate a compact random seed by placing atoms in a small volume
    with minimum distance constraint.
    """
    rng = np.random.default_rng(seed)
    points = []
    min_dist = 0.5
    
    # Start with a few random points
    for _ in range(n):
        for attempt in range(200):
            p = rng.uniform(-2.5, 2.5, 3)
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            if valid:
                points.append(p)
                break
        else:
            # If we can't place, try closer to existing points
            if points:
                base = points[rng.integers(len(points))]
                p = base + rng.normal(0, 0.3, 3)
                valid = True
                for q in points:
                    if np.linalg.norm(p - q) < min_dist:
                        valid = False
                        break
                if valid:
                    points.append(p)
                    break
    
    return np.array(points)

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + 280  # 20s margin before 300s limit
    
    best_energy = float('inf')
    best_positions = None
    
    def try_relax(init_positions, label=""):
        nonlocal best_energy, best_positions
        if time.time() > deadline:
            return False
        if init_positions is None or len(init_positions) != size:
            return False
        try:
            e, pos = relax(init_positions, potential, None)
            if e < best_energy and np.all(np.isfinite(pos)):
                best_energy = e
                best_positions = pos.copy()
                return True
        except Exception:
            pass
        return False

    # Generate diverse seeds
    seeds = []
    
    # 1. Icosahedral seeds with different parameters and seeds
    ico_params = [
        (1.0, 1.87, 2.55),
        (1.0, 1.90, 2.60),
        (0.95, 1.85, 2.50),
        (1.05, 1.92, 2.65),
        (1.0, 1.87, 2.70),
    ]
    
    for pi, (r1, r2, r3) in enumerate(ico_params):
        for s in range(3):
            try:
                seed_pos = _build_icosahedral_cluster(size, r1, r2, r3, seed=pi*10+s)
                if len(seed_pos) == size:
                    seeds.append(seed_pos)
            except Exception:
                pass
    
    # 2. Compact random seeds
    for s in range(5):
        try:
            seed_pos = _make_compact_random_seed(size, seed=100+s)
            if len(seed_pos) == size:
                seeds.append(seed_pos)
        except Exception:
            pass
    
    # Relax all seeds, tracking time
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 30:
            break
        try_relax(seed, label=f"seed_{i}")
    
    # Basin hopping on the best found
    if best_positions is not None:
        # Adaptive perturbation basin hopping
        base_amp = 0.15
        for attempt in range(15):
            if time.time() > deadline - 10:
                break
            try:
                rng = np.random.default_rng(2000 + attempt)
                amp = base_amp * (1 + 0.2 * attempt)
                perturbed = best_positions + rng.normal(0, amp, best_positions.shape)
                try_relax(perturbed, label=f"bh_{attempt}")
            except Exception:
                pass
    
    # Second round: more aggressive basin hopping from multiple starting points
    if best_positions is not None and time.time() < deadline - 20:
        # Create perturbed versions of the best and try more
        for i in range(10):
            if time.time() > deadline - 5:
                break
            try:
                rng = np.random.default_rng(3000 + i)
                amp = 0.2 + 0.05 * i
                perturbed = best_positions + rng.normal(0, amp, best_positions.shape)
                try_relax(perturbed, label=f"bh2_{i}")
            except Exception:
                pass
    
    # Final fallback
    if best_positions is None:
        try:
            best_positions = _build_icosahedral_cluster(size, 1.0, 1.87, 2.55, seed=0)
        except Exception:
            best_positions = _make_compact_random_seed(size, seed=0)
    
    return best_positions
