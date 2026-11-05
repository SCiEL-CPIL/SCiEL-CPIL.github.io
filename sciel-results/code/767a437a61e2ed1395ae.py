"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

def _mackay_icosahedron_vertices(shells=3):
    """
    Generate vertices for a Mackay icosahedron.
    Shell 0: 1 atom (center)
    Shell 1: 12 atoms (icosahedron vertices)
    Shell 2: 42 atoms
    Shell 3: 92 atoms
    Total for 3 shells: 147 atoms.
    """
    phi = (1 + np.sqrt(5)) / 2
    # Basic icosahedron vertices (radius 1)
    v0 = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to radius 1
    norms = np.linalg.norm(v0, axis=1)
    v0 = v0 / norms[:, np.newaxis]
    
    points = [np.zeros(3)]  # Center
    
    if shells >= 1:
        points.extend([v for v in v0])
        
    if shells >= 2:
        # Shell 2 consists of:
        # - 6 vertices of an octahedron (scaled)
        # - 12 midpoints of icosahedron edges (scaled)
        # - 24 midpoints of ... actually, Mackay shell 2 is 42 points.
        # Standard Mackay construction:
        # The 12 vertices of the icosahedron are at distance R1.
        # The next shell (R2) has 42 points.
        # These are formed by projecting the vertices of a larger icosahedron? 
        # No, standard Mackay icosahedron is formed by stacking tetrahedra.
        # Let's use a simpler robust method: generate points on the surface of an icosahedron 
        # by subdividing faces.
        
        # Alternative robust method:
        # Generate a dense set of points on a sphere with icosahedral symmetry.
        # For the purpose of seeding, we can just use a scaled up icosahedron 
        # and add points in the "holes".
        
        # Let's stick to a simpler heuristic that works well for Dzugutov:
        # Dzugutov potential likes 12-coordination.
        # A common low energy structure is a "bi-icosahedron" or similar.
        # However, for n=38-60, a single icosahedral core with a partial outer shell is often best.
        
        # Let's generate the 42 points of the second shell explicitly.
        # The 42 points are:
        # 6 points at the octahedral positions (0,1,0) etc, scaled.
        # 12 points at the midpoints of the 12 edges of the icosahedron? No, that's 30 edges.
        # Actually, the 42 points are the vertices of a "truncated icosahedron" like shape?
        
        # Let's use a known coordinate set for Mackay icosahedron shell 2.
        # Or simpler: just use a dense random sampling on the icosahedral surface 
        # and relax. The relaxation will find the local minimum.
        # To be more efficient, I will generate a "super-icosahedron" by taking 
        # the 12 vertices and adding points at the centers of the 20 faces, 
        # and midpoints of edges, etc.
        
        # Face centers of icosahedron (20 faces):
        # We can compute these by averaging the 3 vertices of each face.
        # Edges of icosahedron (30 edges):
        # Midpoints of edges.
        # Vertices (12).
        # Total: 12 + 30 + 20 = 62 points. This is more than 42, but we can select.
        
        # Let's compute face centers and edge midpoints of the base icosahedron.
        # Faces of icosahedron (indices of vertices):
        faces = [
            [0, 1, 4], [0, 1, 10], [0, 2, 4], [0, 2, 6], [0, 3, 6],
            [0, 3, 10], [1, 4, 7], [1, 7, 11], [1, 11, 10], [1, 10, 4],
            [2, 4, 8], [2, 8, 5], [2, 5, 6], [2, 6, 4], [3, 6, 5],
            [3, 5, 9], [3, 9, 11], [3, 11, 10], [4, 7, 8], [8, 7, 5]
        ]
        
        face_centers = []
        for f in faces:
            center = (v0[f[0]] + v0[f[1]] + v0[f[2]]) / 3.0
            # Normalize to same radius as vertices? No, face centers are closer to center.
            # In Mackay, the second shell is further out.
            # Let's just generate a set of candidate points on a sphere of radius R2
            # with icosahedral symmetry.
            pass
            
        # Simpler Strategy for Shells 2 and 3:
        # Generate a large number of points on a sphere with icosahedral symmetry
        # by using the vertices of a subdivided icosahedron.
        # Subdivide each face into 4 smaller triangles (midpoints of edges).
        # This adds 30 edge midpoints and 20 face centers? No, 1 subdivision adds:
        # 30 new points (edge midpoints) + 20 new points (face centers) = 50 new points?
        # Actually, 1 subdivision of 20 faces:
        # Each face becomes 4. New vertices are the 3 edge midpoints.
        # Total new vertices = 30 (each edge shared by 2 faces, so 30 unique midpoints).
        # So Shell 2 could be the 30 edge midpoints + 12 vertices = 42? 
        # Yes! The 42 points of the second Mackay shell are the 12 vertices + 30 edge midpoints of the icosahedron, 
        # all scaled to the same radius.
        
        edges = set()
        for f in faces:
            edges.add(tuple(sorted((f[0], f[1]))))
            edges.add(tuple(sorted((f[1], f[2]))))
            edges.add(tuple(sorted((f[0], f[2]))))
            
        edge_midpoints = []
        for e in edges:
            mid = (v0[e[0]] + v0[e[1]]) / 2.0
            # Normalize to unit radius to be on the sphere
            norm = np.linalg.norm(mid)
            if norm > 1e-6:
                mid = mid / norm
            edge_midpoints.append(mid)
            
        # Combine vertices and edge midpoints, all on unit sphere
        shell2_points = list(v0) + edge_midpoints
        # Scale to desired radius for shell 2
        # In Mackay, R2/R1 is approx 1.414 (sqrt(2)) for close packing? 
        # For Dzugutov, optimal bond length is around 1.0-1.1.
        # If R1 = 1.0, R2 should be approx 2 * sin(pi/5) / sin(2pi/5) * R1? 
        # Let's just use R2 = 1.6 * R1 as a reasonable guess for second shell.
        r2 = 1.6
        points.extend([p * r2 for p in shell2_points])
        
    if shells >= 3:
        # Shell 3 has 92 points.
        # This is getting complex to generate analytically.
        # For n up to 60, Shell 1 (13 atoms) + part of Shell 2 (42 atoms) is sufficient.
        # Max n = 60. 13 + 42 = 55.
        # So for n=60, we need 5 more atoms from a third shell or just perturb.
        # I will generate a few random points on a sphere of radius R3=2.4 to fill up to 60.
        
        # Let's add 10 random points on a sphere of radius 2.4
        # to ensure we have enough candidates for n=60.
        rng = np.random.default_rng(42)
        for _ in range(10):
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(0, np.pi)
            p = np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])
            points.append(p * 2.4)
            
    return np.array(points)

# Pre-compute the large set of vertices
_MACKAY_VERTS = _mackay_icosahedron_vertices(shells=2)
# Add some more points to cover n=60 comfortably
_rng_fill = np.random.default_rng(123)
_extra_points = []
for _ in range(20):
    theta = _rng_fill.uniform(0, 2 * np.pi)
    phi = _rng_fill.uniform(0, np.pi)
    p = np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    _extra_points.append(p * 2.4)
_MACKAY_VERTS = np.vstack([_MACKAY_VERTS, np.array(_extra_points)])

def _build_icosahedral_seed(n, rng):
    """Select n points from the pre-computed Mackay icosahedron set."""
    # Shuffle indices to get different subsets? No, we want the "core" first.
    # The points are ordered: Center, Shell1, Shell2, Extras.
    # We just take the first n points.
    if n <= len(_MACKAY_VERTS):
        return _MACKAY_VERTS[:n].copy()
    else:
        # Fallback
        return _build_random_cluster(n, rng)

def _build_random_cluster(n, rng, spacing=1.1):
    """Build a random compact cluster."""
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 5000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check distance to existing points
            if len(points) == 0:
                points.append(p)
            else:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= 0.8 * spacing:
                    points.append(p)
    
    # If we didn't get enough, relax constraint
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if len(points) == 0 or np.min(np.linalg.norm(np.array(points) - p, axis=1)) >= 0.5:
            points.append(p)
    
    return np.array(points[:n])

def construct(instance):
    """Find low-energy Dzugutov cluster using icosahedral-inspired starts."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    time_limit = 290.0  # Safety margin
    
    rng = np.random.default_rng(size)
    
    candidates = []
    
    # 1. Icosahedral seed (Mackay)
    candidates.append(_build_icosahedral_seed(size, rng))
    
    # 2. Perturbed Icosahedral seed
    ico_seed = _build_icosahedral_seed(size, rng)
    perturbed = ico_seed + rng.normal(0, 0.1, size)
    candidates.append(perturbed)
    
    # 3-6. Random compact clusters
    for i in range(4):
        rng_local = np.random.default_rng(size + 1000 + i)
        candidates.append(_build_random_cluster(size, rng_local))
        
    best_energy = np.inf
    best_positions = None
    
    for i, pos in enumerate(candidates):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Ensure no atoms are too close (relax might fail or be slow)
        # Simple check: if min dist < 0.5, add noise
        if len(pos) > 1:
            # Efficient min distance check for small n
            # Using cdist is fast
            from scipy.spatial.distance import cdist
            dists = cdist(pos, pos)
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
            if min_dist < 0.5:
                # Add small random perturbation to break symmetry/overlap
                pos = pos + rng.normal(0, 0.01, size)
                
        try:
            energy, positions = relax(pos, potential, parameter)
            # Verify energy is finite
            if np.isfinite(energy):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            continue
    
    # If we have time, do a few more random restarts
    if best_positions is not None and time.time() - start_time < time_limit - 20:
        for restart in range(5):
            if time.time() - start_time > time_limit - 10:
                break
            
            rng_restart = np.random.default_rng(size + 5000 + restart)
            pos = _build_random_cluster(size, rng_restart)
            
            try:
                energy, positions = relax(pos, potential, parameter)
                if np.isfinite(energy) and energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                continue
    
    if best_positions is None:
        # Fallback: return the icosahedral seed if relaxation failed
        if len(candidates) > 0:
            return candidates[0]
        else:
            spacing = 0.6
            return np.array([[i * spacing, 0, 0] for i in range(size)])
    
    return best_positions
# EVOLVE-BLOCK-END
