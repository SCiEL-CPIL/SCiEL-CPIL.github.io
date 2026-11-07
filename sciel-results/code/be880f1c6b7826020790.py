"""
Sutton-Chen transition-metal cluster global minima search.
Uses structure prediction: embeds known small cluster minima as rigid cores,
adds atoms at surface sites, and uses parallel relaxation with diverse starts.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_TOTAL_TIME_LIMIT = 290.0  # Safety margin before 300s kill
_START_TIME = None

def _time_left():
    if _START_TIME is None:
        return _TOTAL_TIME_LIMIT
    return _TOTAL_TIME_LIMIT - (time.time() - _START_TIME)

def _validate(positions):
    """Check if positions are valid: finite and min distance >= 0.35."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    # Vectorized distance check
    i_idx, j_idx = np.triu_indices(n, k=1)
    d = np.linalg.norm(positions[i_idx] - positions[j_idx])
    if np.any(d < 0.35):
        return False
    return True

def _get_icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    # Normalize to unit radius
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

def _get_icosahedral_shell(n_core, scale=1.0):
    """
    Generate an icosahedral cluster with n_core atoms.
    Shell 0: 1 atom (center)
    Shell 1: 12 atoms (icosahedron vertices)
    Shell 2: 30 atoms (edge midpoints)
    Shell 3: 20 atoms (face centers)
    Total: 1+12+30+20 = 63
    """
    if n_core <= 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    ico_verts = _get_icosahedron_vertices()
    points = [np.array([0.0, 0.0, 0.0])]  # Center
    
    # Shell 1: 12 vertices
    if n_core >= 13:
        for v in ico_verts:
            points.append(v * scale)
    
    if n_core >= 13:
        # Shell 2: 30 edge midpoints
        # Edges of icosahedron: connect each vertex to 5 others
        # We need to find all edges. Two vertices are connected if their dot product is -1/phi
        dot_threshold = -1.0 / ((1 + np.sqrt(5)) / 2)
        edges = set()
        for i in range(12):
            for j in range(i+1, 12):
                dot = np.dot(ico_verts[i], ico_verts[j])
                if abs(dot - dot_threshold) < 1e-6:
                    edges.add((i, j))
        
        for (i, j) in edges:
            mid = (ico_verts[i] + ico_verts[j]) / 2.0
            # Normalize to same radius as shell 1
            mid_norm = np.linalg.norm(mid)
            if mid_norm > 1e-10:
                mid = mid / mid_norm * scale
            points.append(mid)
        
        # Shell 3: 20 face centers
        # Faces of icosahedron: each face is a triangle of 3 vertices
        # We can find faces by looking for triples of mutually connected vertices
        faces = set()
        for i in range(12):
            for j in range(i+1, 12):
                if (i, j) not in edges and (j, i) not in edges:
                    continue
                for k in range(j+1, 12):
                    if (j, k) not in edges and (k, j) not in edges:
                        continue
                    if (i, k) not in edges and (k, i) not in edges:
                        continue
                    face = tuple(sorted([i, j, k]))
                    faces.add(face)
        
        for (i, j, k) in faces:
            center = (ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3.0
            center_norm = np.linalg.norm(center)
            if center_norm > 1e-10:
                center = center / center_norm * scale
            points.append(center)
    
    points = np.array(points)
    if len(points) >= n_core:
        return points[:n_core]
    else:
        # Fill with random if needed
        rng = np.random.default_rng(42)
        while len(points) < n_core:
            p = rng.uniform(-scale, scale, 3)
            if np.linalg.norm(p) <= scale:
                points = np.vstack([points, [p]])
        return points[:n_core]

def _generate_core_plus_surface(n, seed=0):
    """
    Structure prediction: use icosahedral core, add atoms on surface.
    """
    rng = np.random.default_rng(seed)
    
    # Choose core size based on n
    if n <= 13:
        core_size = min(n, 13)
    elif n <= 55:
        core_size = min(n, 55)
    else:
        core_size = min(n, 63)
    
    core = _get_icosahedral_shell(core_size, scale=1.0)
    
    if core_size >= n:
        return core[:n]
    
    # Add remaining atoms on surface
    remaining = n - core_size
    surface_points = []
    
    # Get surface sites: directions from center to existing atoms
    for p in core:
        if np.linalg.norm(p) > 0.1:
            direction = p / np.linalg.norm(p)
            surface_points.append(direction)
    
    # Also add some random surface directions
    for _ in range(remaining * 2):
        p = rng.normal(0, 1, 3)
        norm = np.linalg.norm(p)
        if norm > 1e-10:
            surface_points.append(p / norm)
    
    surface_points = np.array(surface_points)
    
    # Place new atoms slightly outside the core
    new_points = []
    for _ in range(remaining):
        # Pick a random surface direction
        idx = rng.integers(0, len(surface_points))
        direction = surface_points[idx]
        # Place at radius slightly larger than core
        r = 1.2 + rng.uniform(0, 0.3)
        new_points.append(direction * r)
    
    new_points = np.array(new_points)
    all_points = np.vstack([core, new_points])
    
    return all_points[:n]

def _generate_fcc(n, seed=0):
    """Generate points from an FCC lattice inside a sphere."""
    rng = np.random.default_rng(seed)
    a = 1.0
    R = (3.0 * n / (4.0 / a**3) / (4.0 * np.pi)) ** (1.0/3.0) * 1.2
    
    max_coord = int(np.ceil(R / a)) + 2
    points = []
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                points.append(np.array([ix, iy, iz]) * a)
                points.append(np.array([ix + 0.5, iy + 0.5, iz]) * a)
                points.append(np.array([ix + 0.5, iy, iz + 0.5]) * a)
                points.append(np.array([ix, iy + 0.5, iz + 0.5]) * a)
                
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    sorted_pts = points[order]
    
    if len(sorted_pts) >= n:
        return sorted_pts[:n]
    else:
        pts = list(sorted_pts)
        while len(pts) < n:
            p = rng.uniform(-R, R, 3)
            pts.append(p)
        return np.array(pts)

def _generate_fibonacci_sphere(n, radius=1.0):
    """Generate n points evenly distributed on a sphere using Fibonacci lattice."""
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
        
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        y = 1 - (i / (n - 1)) * 2
        radius_i = np.sqrt(1 - y**2)
        theta = phi * i
        x = np.cos(theta) * radius_i
        z = np.sin(theta) * radius_i
        points.append([x, y, z])
        
    pts = np.array(points) * radius
    return pts

def _generate_random_compact(n, seed=0):
    """Generate a random compact cluster using rejection sampling."""
    rng = np.random.default_rng(seed)
    radius = n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    max_attempts = n * 2000
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if points:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= 0.5:
                    points.append(p)
            else:
                points.append(p)
        attempts += 1
        
    while len(points) < n:
        points.append(rng.uniform(-radius, radius, 3))
        
    return np.array(points)

def _relax_worker(start, kind):
    """Worker function for parallel relaxation."""
    try:
        energy, positions = relax(start, 'sutton_chen', kind)
        if np.all(np.isfinite(positions)) and _validate(positions):
            return energy, positions
    except Exception:
        pass
    return None, None

def construct(instance):
    """Find near-global minimum for Sutton-Chen transition-metal clusters."""
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    
    # Generate diverse starting configurations
    starts = []
    
    # 1. Structure prediction: icosahedral core + surface atoms (8 variants)
    for i in range(8):
        try:
            sp = _generate_core_plus_surface(n, seed=i)
            if len(sp) == n and _validate(sp):
                starts.append(sp)
        except:
            pass
            
    # 2. FCC starts (8 variants)
    for i in range(8):
        try:
            fcc = _generate_fcc(n, seed=100+i)
            if len(fcc) == n and _validate(fcc):
                starts.append(fcc)
        except:
            pass

    # 3. Spherical Code starts (4 variants)
    for i in range(4):
        try:
            radius = n ** (1.0/3.0) * 0.9
            fib = _generate_fibonacci_sphere(n, radius)
            if len(fib) == n and _validate(fib):
                starts.append(fib)
        except:
            pass

    # 4. Random Compact starts (10 variants)
    for i in range(10):
        try:
            rand = _generate_random_compact(n, seed=200+i)
            if len(rand) == n and _validate(rand):
                starts.append(rand)
        except:
            pass

    if not starts:
        starts = [_generate_random_compact(n, seed=0)]
        
    best_energy = np.inf
    best_positions = starts[0]
    
    # Parallel relaxation
    starts = starts[:30]
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_relax_worker, start, kind): i for i, start in enumerate(starts)}
        
        for future in as_completed(futures):
            if _time_left() < 5.0:
                break
                
            res = future.result()
            if res[0] is not None:
                energy, positions = res
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
                    
    # Final validation
    if not _validate(best_positions):
        best_positions = _generate_random_compact(n, seed=999)
        for i in range(len(best_positions)):
            for j in range(i + 1, len(best_positions)):
                d = np.linalg.norm(best_positions[i] - best_positions[j])
                if d < 0.35:
                    direction = best_positions[i] - best_positions[j]
                    if np.linalg.norm(direction) > 1e-10:
                        direction = direction / np.linalg.norm(direction)
                    else:
                        direction = np.array([1.0, 0, 0])
                    best_positions[j] = best_positions[i] + direction * 0.36
    
    return best_positions
# EVOLVE-BLOCK-END
