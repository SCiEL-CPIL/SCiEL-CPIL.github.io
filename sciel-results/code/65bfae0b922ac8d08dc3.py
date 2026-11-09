"""Dzugutov cluster global minima via Mackay icosahedral initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

_DEADLINE = time.time() + 280.0  # Safety margin before 300s kill

def _icosahedron_vertices(scale=1.0):
    """Return 12 vertices of a regular icosahedron scaled so edge length ~ scale."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit sphere first, then scale
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    # The edge length of a regular icosahedron inscribed in a unit sphere is 1.05146...
    # We want the edge length to be ~ 1.0 (typical bond length in Dzugutov is ~1.0-1.1)
    # Let's just normalize to unit sphere and let relax adjust the scale.
    return verts

def _generate_mackay_55():
    """Generate coordinates for the 55-atom Mackay icosahedron."""
    # 1. 12 vertices of a regular icosahedron
    v = _icosahedron_vertices()
    
    # 2. 30 edge midpoints
    # Identify edges: pairs of vertices with distance ~ 1.051
    # Instead of explicit edge finding, we can generate all pairs and filter
    # Or simpler: The 55-atom cluster is symmetric. 
    # Let's construct the 42 outer atoms as:
    # - 12 atoms at the "face centers" of the 20 faces? No, 20 faces.
    # - 30 atoms at the "edge midpoints" of the 30 edges.
    
    # Let's find the 30 edges by distance
    d = np.linalg.norm(v[:, np.newaxis] - v[np.newaxis, :], axis=2)
    # The minimum non-zero distance is the edge length
    min_d = np.min(d[np.triu_indices(12, k=1)])
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(d[i,j] - min_d) < 1e-5:
                edges.append((i,j))
    
    edge_mids = []
    for i, j in edges:
        mid = (v[i] + v[j]) / 2.0
        # Normalize to unit sphere? No, in Mackay icosahedron, 
        # the edge midpoints are at a different radial distance than vertices.
        # However, for initialization, placing them at the geometric midpoint 
        # of the chord on the unit sphere is a good start.
        # Actually, the midpoint of the chord is inside the sphere.
        # Let's project it to the sphere? No, that would be wrong.
        # Let's just use the vector sum and normalize to a specific radius.
        # In the Mackay icosahedron, the ratio of radii is fixed.
        # r_vertex = R
        # r_edge_mid = R * cos(theta/2) ?
        # Let's just place them at the normalized midpoint direction, scaled by 0.95 * R?
        # Better: Just use the raw midpoint coordinates. Relax will fix it.
        edge_mids.append(mid)
    edge_mids = np.array(edge_mids)
    
    # 3. 20 face centers
    # Identify faces: triples of vertices forming an equilateral triangle
    faces = []
    # We can find faces by looking for triples (i,j,k) where d(i,j)=d(j,k)=d(k,i)=min_d
    # This is O(N^3) which is fine for N=12
    for i in range(12):
        for j in range(i+1, 12):
            if abs(d[i,j] - min_d) > 1e-5: continue
            for k in range(j+1, 12):
                if abs(d[j,k] - min_d) > 1e-5: continue
                if abs(d[i,k] - min_d) > 1e-5: continue
                # Found a face
                face_center = (v[i] + v[j] + v[k]) / 3.0
                faces.append(face_center)
    
    # There are 20 faces
    face_centers = np.array(faces)
    
    # Combine all atoms
    # Center
    center = np.array([[0,0,0]])
    
    # Scale the whole structure
    # Typical Dzugutov bond length is ~1.0
    # The edge length of our unit-sphere icosahedron is ~1.05
    # Let's scale by 1.0 / 1.05
    scale_factor = 1.0 / min_d
    
    verts_scaled = v * scale_factor
    edge_mids_scaled = edge_mids * scale_factor
    face_centers_scaled = face_centers * scale_factor
    
    all_atoms = np.vstack([center, verts_scaled, edge_mids_scaled, face_centers_scaled])
    return all_atoms

def _select_subset(positions, n):
    """Select n atoms from the 55-atom structure, keeping the core."""
    # positions[0] is center
    # positions[1:13] are vertices
    # positions[13:43] are edge mids
    # positions[43:55] are face centers
    
    if n == 55:
        return positions
    
    # Always keep center (1) and vertices (12) -> 13 atoms
    if n >= 13:
        core = positions[:13]
        outer = positions[13:]
        need = n - 13
        if need == 0:
            return core
        if need > len(outer):
            # Should not happen for n <= 55
            return positions[:n]
        
        # Select 'need' atoms from outer
        # Prioritize symmetry? Just random for now, or sorted by distance from center?
        # Random selection with a fixed seed per n for reproducibility
        rng = np.random.default_rng(n)
        indices = rng.choice(len(outer), need, replace=False)
        selected_outer = outer[indices]
        return np.vstack([core, selected_outer])
    else:
        # n < 13, select from vertices
        rng = np.random.default_rng(n)
        indices = rng.choice(12, n-1, replace=False)
        return np.vstack([positions[:1], positions[1:13][indices]])

def _add_atoms(positions, n):
    """Add atoms to a 55-atom structure to reach n."""
    if n == 55:
        return positions
    if n < 55:
        return _select_subset(positions, n)
    
    # n > 55
    # Add n-55 atoms at random positions near the surface
    rng = np.random.default_rng(n)
    new_atoms = []
    # Surface radius is approx 1.0
    for _ in range(n - 55):
        # Random direction
        dir = rng.normal(size=3)
        dir = dir / np.linalg.norm(dir)
        # Radius slightly larger than surface
        r = 1.2 + rng.uniform(0, 0.2)
        new_atoms.append(dir * r)
    
    new_atoms = np.array(new_atoms)
    return np.vstack([positions, new_atoms])

def construct(instance):
    """Find the lowest-energy structure for the given instance."""
    n = instance["n"]
    potential = instance["potential"]
    
    # Generate the 55-atom Mackay icosahedron
    base_positions = _generate_mackay_55()
    
    best_energy = np.inf
    best_positions = None
    
    # Number of restarts
    # We have ~50s per instance. Relax is fast.
    # Let's do 20 restarts.
    n_restarts = 20
    
    for i in range(n_restarts):
        if time.time() > _DEADLINE:
            break
        
        # Prepare initial guess
        if n <= 55:
            # Select subset
            # Add slight perturbation to break symmetry and avoid flat minima?
            # Actually, exact symmetry is good for Mackay.
            # But for non-Mackay N, we need to break symmetry.
            # Let's add small noise to the selected atoms.
            pos = _select_subset(base_positions, n)
            # Perturb
            noise = np.random.default_rng(i).normal(0, 0.01, pos.shape)
            pos = pos + noise
        else:
            pos = _add_atoms(base_positions, n)
            # Perturb
            noise = np.random.default_rng(i).normal(0, 0.01, pos.shape)
            pos = pos + noise
        
        # Relax
        try:
            energy, relaxed_pos = relax(pos, potential)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos
        except Exception as e:
            pass
            
    return best_positions
# EVOLVE-BLOCK-END
