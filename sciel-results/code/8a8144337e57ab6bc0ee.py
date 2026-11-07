import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Constants for time management
DEADLINE = 330.0  # Seconds
START_TIME = time.time()

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def _fcc_positions(n, spacing=1.0):
    """
    Generate positions for a truncated FCC cluster.
    We generate points in an FCC lattice and keep the n closest to the origin.
    """
    # FCC lattice vectors
    a = spacing
    # Generate a grid of points in a box large enough to contain n atoms
    # Density of FCC is 4 atoms per unit cell volume a^3.
    # Volume ~ 4/3 * pi * R^3. n ~ 4/a^3 * 4/3 * pi * R^3 => R ~ (3*n*a^3/(16*pi))^(1/3)
    R_est = (3 * n * a**3 / (16 * np.pi)) ** (1/3) * 1.5  # Add margin
    
    # Grid range
    max_idx = int(R_est / a) + 2
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # FCC lattice points: (i,j,k) where i+j+k is even? No, FCC is:
                # (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in units of a?
                # Standard cubic cell:
                # 0,0,0
                # 0, 0.5, 0.5
                # 0.5, 0, 0.5
                # 0.5, 0.5, 0
                
                base = np.array([i, j, k]) * a
                offsets = [
                    np.array([0, 0, 0]),
                    np.array([0, 0.5, 0.5]) * a,
                    np.array([0.5, 0, 0.5]) * a,
                    np.array([0.5, 0.5, 0]) * a
                ]
                for off in offsets:
                    p = base + off
                    if np.linalg.norm(p) <= R_est:
                        points.append(p)
    
    if len(points) < n:
        return None
    
    points = np.array(points)
    # Sort by distance to origin
    dists = np.linalg.norm(points, axis=1)
    idxs = np.argsort(dists)[:n]
    return points[idxs]

def _mackay_positions(n, spacing=1.0):
    """
    Generate positions for a Mackay icosahedral cluster.
    Mackay numbers: 1, 12, 42, 120, 220, 320...
    We generate layers and pick the closest n to the center.
    """
    # Icosahedron vertices (radius 1)
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    # Normalize to radius 1
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = [np.zeros(3)] # Center
    
    # Layer 1: 12 vertices
    # Distance from center to vertex in Mackay is a = 1 (if edge length is sqrt(2)?)
    # In standard Mackay, layer 1 is at distance 1.
    # Let's scale by spacing.
    # The nearest neighbor distance in Mackay is usually taken as 1.
    # The distance from center to layer 1 is 1.
    layer_dist = 1.0 * spacing
    points.extend([v * layer_dist for v in verts])
    
    # Layer 2: 30 points (midpoints of edges) + 12 points (vertices of larger icosahedron?)
    # Actually, Mackay layer 2 has 30 points at distance sqrt(2) * a? No.
    # Distance from center to layer 2 is sqrt(2) * a if a is NN distance?
    # Let's just generate a dense set of points on the icosahedral symmetry and pick closest.
    
    # Simpler approach: Generate points on the icosahedral surface.
    # We need up to n points.
    # If n is small, just use the first layers.
    
    # For robustness, if n > 42, we might need layer 2.
    # Layer 2 points are midpoints of edges of the icosahedron, scaled by sqrt(2)?
    # Let's generate all edge midpoints.
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            v1, v2 = verts[i], verts[j]
            if np.linalg.norm(v1 - v2) < 1.1: # Close to edge length
                mid = (v1 + v2) / 2
                mid = mid / np.linalg.norm(mid)
                # Distance from center to midpoint is 1/sqrt(2) * 2? No.
                # Norm of mid is 1/sqrt(2) approx?
                # Let's just add them and scale.
                points.append(mid * (np.sqrt(2) * spacing)) # Approx layer 2 radius
    
    if len(points) < n:
        # Add more layers or just fill with random on surface?
        # For now, if we don't have enough, return None or pad with random
        # Let's just return what we have if it's at least n? No, we need exactly n.
        # If we don't have enough, we can't form a valid Mackay cluster of size n easily without more logic.
        # We'll rely on FCC for larger N.
        return None

    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idxs = np.argsort(dists)[:n]
    return points[idxs]

def _hcp_positions(n, spacing=1.0):
    """
    Generate positions for a truncated HCP cluster.
    """
    a = spacing
    # HCP layers: A B A B ...
    # Layer A: Hexagonal lattice
    # Layer B: Hexagonal lattice shifted
    
    points = []
    # Center atom
    points.append(np.zeros(3))
    
    # Layer 1 (z=0)
    r = a
    # 6 atoms around center
    for k in range(6):
        theta = 2 * np.pi * k / 6
        points.append([r * np.cos(theta), r * np.sin(theta), 0])
    
    # Layer 2 (z=h)
    # h = sqrt(2/3) * a
    h = np.sqrt(2/3) * a
    # 3 atoms above center? No, 3 atoms in the "holes"
    # HCP stacking:
    # Layer 1: 7 atoms (center + 6)
    # Layer 2: 3 atoms (above triangles) + 6 atoms (outer ring)?
    # This is getting complex. Let's use a simpler heuristic:
    # Generate HCP lattice points and keep closest n.
    
    max_r = (n ** (1/3)) * a * 1.5
    max_z = max_r
    max_idx = int(max_r / a) + 2
    
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # HCP lattice points
                # a1 = a (1, 0, 0)
                # a2 = a (-1/2, sqrt(3)/2, 0)
                # a3 = a (0, 0, sqrt(2/3))
                
                # Basis:
                # (0,0,0)
                # (0,0, h/2) ? No, standard HCP basis is (0,0,0) and (1/3, 2/3, 1/2) in fractional coords?
                # Let's use Cartesian.
                
                x = i * a + j * (-0.5 * a)
                y = j * (np.sqrt(3)/2 * a)
                z = k * (np.sqrt(2/3) * a)
                
                # Point 1
                p1 = np.array([x, y, z])
                if np.linalg.norm(p1) <= max_r:
                    points.append(p1)
                
                # Point 2 (shifted)
                # Shift is (1/3, 2/3, 1/2) in lattice vectors?
                # a1 = (1,0,0), a2 = (-1/2, sqrt(3)/2, 0), a3 = (0,0, sqrt(2/3))
                # Shift vector = 1/3*a1 + 2/3*a2 + 1/2*a3
                # x_shift = 1/3*a + 2/3*(-1/2*a) = 0
                # y_shift = 2/3*(sqrt(3)/2*a) = sqrt(3)/3 * a
                # z_shift = 1/2 * sqrt(2/3) * a
                shift = np.array([0, np.sqrt(3)/3 * a, np.sqrt(2/3)/2 * a])
                p2 = p1 + shift
                if np.linalg.norm(p2) <= max_r:
                    points.append(p2)
    
    if len(points) < n:
        return None
        
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idxs = np.argsort(dists)[:n]
    return points[idxs]

def _random_positions(n, spacing=1.0, seed=None):
    """
    Generate random positions in a sphere, ensuring min distance.
    """
    rng = np.random.default_rng(seed)
    # Estimate radius
    R = spacing * n ** (1/3) * 1.2
    points = []
    attempts = 0
    max_attempts = 5000
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-R, R, 3)
        if np.linalg.norm(p) > R:
            continue
        # Check min distance
        if points:
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) < 0.8 * spacing:
                attempts += 1
                continue
        points.append(p)
        attempts += 1
    
    if len(points) < n:
        return None
    return np.array(points)

def construct(instance):
    """
    Construct a low-energy cluster configuration for the given instance.
    """
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    
    # Parameters for the potential (for reference, relax handles them)
    # kind is used by relax
    
    best_energy = np.inf
    best_pos = None
    
    # Generate diverse starting structures
    starts = []
    
    # 1. FCC
    pos_fcc = _fcc_positions(n, spacing=1.0)
    if pos_fcc is not None:
        starts.append(pos_fcc)
        
    # 2. HCP
    pos_hcp = _hcp_positions(n, spacing=1.0)
    if pos_hcp is not None:
        starts.append(pos_hcp)
        
    # 3. Mackay
    pos_mackay = _mackay_positions(n, spacing=1.0)
    if pos_mackay is not None:
        starts.append(pos_mackay)
        
    # 4. Random (2 seeds)
    for i in range(2):
        pos_rand = _random_positions(n, spacing=1.0, seed=n*100+i)
        if pos_rand is not None:
            starts.append(pos_rand)
            
    # If we have no starts (unlikely), fall back to a simple random
    if not starts:
        starts.append(_random_positions(n, spacing=1.0, seed=0))
        
    # Relax each start
    for i, pos in enumerate(starts):
        if _time_left() < 5.0:
            break
            
        try:
            energy, relaxed_pos = relax(pos, potential, kind)
            # Validate
            if np.isfinite(energy):
                # Check min distance
                if n > 1:
                    # Simple check: if any distance is too small, it might be invalid
                    # relax should handle this, but let's be safe
                    pass 
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos
        except Exception:
            continue
            
    if best_pos is None:
        # Emergency fallback: return a valid random configuration
        return _random_positions(n, spacing=1.0, seed=1)
        
    return best_pos
