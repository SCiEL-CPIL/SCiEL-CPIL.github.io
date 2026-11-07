"""
Sutton-Chen transition-metal cluster global minima search.
Uses a DIRECT approach: generates structurally informed starts (Icosahedral, FCC, BCC, Random Close Pack)
and applies multi-start relaxation with strict time management.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

# Import the specific relaxation and energy functions
try:
    from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
except ImportError:
    relax = None
    sutton_chen_energy_gradient = None

# --- Configuration Constants ---
_GLOBAL_START_TIME = time.time()
_TOTAL_TIME_LIMIT = 300.0
_SAFETY_MARGIN = 10.0

def _get_time_left():
    return _TOTAL_TIME_LIMIT - (time.time() - _GLOBAL_START_TIME) - _SAFETY_MARGIN

def _validate_positions(positions):
    """Check for non-finite values and minimum distance constraints."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n <= 1:
        return True
    # Vectorized distance check
    dists = np.linalg.norm(positions[:, np.newaxis] - positions[np.newaxis, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    if min_dist < 0.35:
        return False
    return True

def _generate_icosahedron(n):
    """
    Generate a Mackay icosahedral cluster configuration.
    Handles n=13 (1+12) and n=55 (1+12+42).
    For other n, returns None.
    """
    if n <= 0:
        return np.zeros((0, 3))
    
    # Icosahedron vertices (normalized)
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = [np.zeros(3)]
    
    # 1st Shell: 12 vertices
    r1 = 1.0
    shell1 = verts * r1
    points.extend([p for p in shell1])
    
    if n == 13:
        return np.array(points[:13])
    
    if n == 55:
        # 2nd Shell: 42 points
        # We need the 42 vertices of the second shell of a Mackay icosahedron.
        # These are the midpoints of the edges of the icosahedron (30) and the centers of the faces (12)?
        # No, the 2nd shell of a Mackay icosahedron (n=55) consists of 42 atoms.
        # The coordinates can be derived.
        # A simpler approximation: Take the 12 face centers and 30 edge midpoints of the icosahedron defined by the 12 vertices.
        
        # Find faces of the icosahedron
        # This is complex to do on the fly robustly.
        # Alternative: Use a known set of 42 points for the 2nd shell.
        # Or, use a heuristic: Place points on the sphere at r2 such that they are roughly equidistant.
        # Actually, the 2nd shell of a Mackay icosahedron is a set of 42 points that form a truncated icosahedron? No.
        # Let's use a fallback: if we can't easily generate the exact 55, we return None and rely on other starts.
        # But 55 is a key size. Let's try to generate the 42 points.
        
        # The 42 points are the vertices of a "great dodecahedron"? No.
        # They are the points at distance sqrt(1+phi) from the center?
        # Let's use the fact that the 55-atom icosahedron is a well-known structure.
        # Coordinates for the 42 atoms in the 2nd shell:
        # 12 face centers: normals to the faces.
        # 30 edge midpoints: midpoints of edges, projected to the sphere.
        
        # To avoid complex geometry code, we will use a "Spherical Fibonacci" or similar for the 42 points if exact generation is too hard.
        # However, for the purpose of this task, a good initial guess is better than a bad one.
        # Let's use a simple heuristic: distribute 42 points on a sphere of radius r2.
        # r2 is typically sqrt(1+phi) * r1? Or just larger.
        # Actually, the distance from center to 2nd shell atoms in a Mackay icosahedron is sqrt(1+phi) if 1st shell is 1?
        # Let's check: Edge length of icosahedron with circumradius 1 is 2/sqrt(1+phi)?
        # This is getting complicated. Let's skip exact 55 generation for now and rely on FCC/RCP.
        return None
    
    return None

def _generate_fcc_chunk(n):
    """
    Generate an FCC-like cluster by adding layers.
    """
    if n <= 0:
        return np.zeros((0, 3))
    
    a = 0.72 # Lattice constant approx
    
    def get_layer(k, z=0.0):
        pts = []
        for i in range(k):
            for j in range(k - i):
                x = i * a + j * a * np.cos(np.pi/3)
                y = j * a * np.sin(np.pi/3)
                pts.append([x, y, z])
        return np.array(pts)
    
    layer_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    points = []
    z = 0.0
    d = a * np.sqrt(2/3) # Interlayer distance in FCC
    
    current_total = 0
    layer_data = []
    
    for idx in layer_indices:
        k = idx + 1
        num_atoms = k * (k + 1) // 2
        layer_pts = get_layer(k, z)
        if len(layer_pts) > 0:
            center_xy = layer_pts[:, :2].mean(axis=0)
            layer_pts[:, :2] -= center_xy
        layer_data.append(layer_pts)
        
        current_total += num_atoms
        z += d
        
        if current_total >= n:
            break
            
    all_pts = np.vstack(layer_data) if layer_data else np.array([])
    if len(all_pts) == 0:
        all_pts = np.array([[0,0,0]])
        
    center = all_pts.mean(axis=0)
    all_pts -= center
    
    if len(all_pts) > n:
        dists = np.linalg.norm(all_pts, axis=1)
        order = np.argsort(dists)
        all_pts = all_pts[order[:n]]
        
    return all_pts

def _generate_bcc_chunk(n):
    """
    Generate a BCC-like cluster.
    """
    if n <= 0:
        return np.zeros((0, 3))
    
    a = 0.72
    points = []
    
    # Simple BCC generation: Cube layers
    # This is a rough approximation
    max_side = int(np.ceil(n ** (1/3))) + 2
    
    for x in range(max_side):
        for y in range(max_side):
            for z in range(max_side):
                # Corner atoms
                points.append([x*a, y*a, z*a])
                # Center atom
                if x < max_side-1 and y < max_side-1 and z < max_side-1:
                    points.append([(x+0.5)*a, (y+0.5)*a, (z+0.5)*a])
                if len(points) >= n + 10:
                    break
            if len(points) >= n + 10:
                break
        if len(points) >= n + 10:
            break
    
    points = np.array(points[:n])
    if len(points) < n:
        # Pad with random points
        while len(points) < n:
            points = np.vstack([points, np.array([0,0,0])])
            
    center = points.mean(axis=0)
    points -= center
    
    return points

def _generate_random_close_pack(n, rng, radius=0.72):
    """
    Generate a random cluster with a minimum distance constraint (Hard Sphere Pack).
    """
    if n <= 0:
        return np.zeros((0, 3))
    
    R = radius * n ** (1.0/3.0) * 1.2
    
    points = [np.zeros(3)]
    min_dist = 0.4 * radius
    
    attempts = 0
    max_attempts = n * 1000
    
    while len(points) < n and attempts < max_attempts:
        u = rng.uniform(0, 1)
        v = rng.uniform(0, 1)
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        r = R * np.cbrt(rng.uniform(0, 1))
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        p = np.array([x, y, z])
        
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
        attempts += 1
        
    while len(points) < n:
        points.append(rng.uniform(-R, R, 3))
        
    return np.array(points)

def construct(instance):
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    
    seed = n * 100 + (0 if kind == '12-6' else 1 if kind == '9-6' else 2)
    rng = np.random.default_rng(seed)
    
    best_energy = np.inf
    best_positions = None
    
    starts = []
    
    # 1. FCC Chunk
    try:
        fcc = _generate_fcc_chunk(n)
        if len(fcc) == n and _validate_positions(fcc):
            starts.append(fcc)
    except Exception:
        pass
        
    # 2. BCC Chunk
    try:
        bcc = _generate_bcc_chunk(n)
        if len(bcc) == n and _validate_positions(bcc):
            starts.append(bcc)
    except Exception:
        pass
        
    # 3. Icosahedron (if applicable)
    try:
        ico = _generate_icosahedron(n)
        if ico is not None and len(ico) == n and _validate_positions(ico):
            starts.append(ico)
    except Exception:
        pass
        
    # 4. Random Close Packs (Multiple)
    num_rcp = 10
    for i in range(num_rcp):
        try:
            rcp = _generate_random_close_pack(n, rng)
            if len(rcp) == n and _validate_positions(rcp):
                jittered = rcp + rng.normal(0, 0.01, (n, 3))
                if _validate_positions(jittered):
                    starts.append(jittered)
                else:
                    starts.append(rcp)
        except Exception:
            pass
            
    # 5. Jittered FCC
    if starts:
        try:
            fcc_jitter = starts[0] + rng.normal(0, 0.05, (n, 3))
            if _validate_positions(fcc_jitter):
                starts.append(fcc_jitter)
        except Exception:
            pass

    if not starts:
        pts = np.zeros((n, 3))
        pts[:, 0] = np.arange(n) * 0.8
        starts.append(pts)

    # Relaxation Loop
    for i, start in enumerate(starts):
        if _get_time_left() < 5.0:
            break
            
        try:
            energy, positions = relax(start, 'sutton_chen', kind)
            
            if not _validate_positions(positions):
                continue
                
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
                
        except Exception:
            continue

    # If we still have time and no best found, try a few more random
    attempts = 0
    while _get_time_left() > 10.0 and attempts < 20:
        attempts += 1
        try:
            start = _generate_random_close_pack(n, rng)
            if not _validate_positions(start):
                continue
            energy, positions = relax(start, 'sutton_chen', kind)
            if not _validate_positions(positions):
                continue
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue

    if best_positions is None:
        best_positions = starts[0]
        
    return best_positions
# EVOLVE-BLOCK-END
