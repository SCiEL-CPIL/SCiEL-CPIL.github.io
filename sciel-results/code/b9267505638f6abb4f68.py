import numpy as np
from scipy.optimize import minimize
import time

N = 55

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _get_icosahedron_vectors():
    """
    Returns the 12 unit vectors to the vertices of a regular icosahedron
    and the 30 unit vectors to the midpoints of the edges.
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of a regular icosahedron
    # (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1)
    v1 = np.array([0, 1, phi])
    v2 = np.array([1, phi, 0])
    v3 = np.array([phi, 0, 1])
    
    vertices = []
    for v in [v1, v2, v3]:
        for sx in [1, -1]:
            for sy in [1, -1]:
                # Permutations are handled by the specific arrays above, 
                # but we need to generate all 12.
                pass
    
    # Explicit list of 12 vertices
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts_unit = verts / norms[:, np.newaxis]
    
    # 30 edge midpoints
    # An edge exists between any two vertices that are not opposite and not identical.
    # In a regular icosahedron, each vertex is connected to 5 others.
    # Total edges = 12 * 5 / 2 = 30.
    
    midpoints = []
    for i in range(12):
        for j in range(i + 1, 12):
            # Check if they form an edge.
            # The distance between connected vertices is constant.
            # We can just compute the midpoint and normalize.
            # However, we need to distinguish edges from non-edges (diagonals).
            # The shortest distance between any two distinct vertices in an icosahedron is the edge length.
            # Let's compute all pairwise distances and pick the smallest 30 unique pairs.
            pass
            
    # Simpler way to get midpoints:
    # The icosahedron has 20 faces. Each face is a triangle.
    # The midpoints of the edges are shared by 2 faces.
    # Alternatively, the 30 midpoints are the vertices of a regular icosidodecahedron.
    # Coordinates of icosidodecahedron vertices:
    # All permutations of (0, ±1, ±phi)
    # (±1, ±phi, 0)
    # (±phi, 0, ±1)
    # Wait, the icosidodecahedron has 30 vertices.
    # Its vertices are the edge midpoints of the icosahedron.
    # The coordinates are all permutations of (0, 1, phi).
    
    ico_mid = []
    # Permutations of (0, 1, phi) with signs
    base = [0, 1, phi]
    perms = [
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, 0, phi], [1, 0, -phi], [1, phi, 0], [1, -phi, 0],
        [1, phi, 0], [1, -phi, 0], # Duplicates? No, let's be careful.
    ]
    
    # Let's just generate all 30 permutations of (0, ±1, ±phi)
    # There are 3 positions for 0.
    # For each, 2 signs for 1, 2 signs for phi.
    # 3 * 2 * 2 = 12? No.
    # The icosidodecahedron vertices are:
    # (0, ±1, ±phi) -> 4
    # (±1, ±phi, 0) -> 4
    # (±phi, 0, ±1) -> 4
    # This gives 12. Where are the other 18?
    # Ah, the icosidodecahedron has 30 vertices.
    # The coordinates are:
    # (0, ±1, ±phi)
    # (±1, ±phi, 0)
    # (±phi, 0, ±1)
    # AND
    # (±1, 0, ±phi) -- Wait, (0, ±1, ±phi) covers x=0.
    # Let's list them systematically.
    # x=0: y, z are (±1, ±phi) -> 4 points
    # y=0: x, z are (±1, ±phi) -> 4 points
    # z=0: x, y are (±1, ±phi) -> 4 points
    # This is only 12 points. These are the vertices of a regular icosahedron? No, these are the same as the icosahedron vertices if we swap 1 and phi?
    # The icosahedron vertices are (0, ±1, ±phi).
    # The icosidodecahedron vertices are the midpoints of the edges.
    
    # Let's calculate the midpoints directly from the vertices.
    verts_norm = verts / np.linalg.norm(verts, axis=1)[:, None]
    edge_mids = []
    # Find all pairs with minimum distance
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(verts[i] - verts[j])
    
    # The edge length is the minimum non-zero distance
    min_dist = np.min(dists[dists > 1e-6])
    
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                mid = (verts[i] + verts[j]) / 2.0
                norm_mid = np.linalg.norm(mid)
                if norm_mid > 1e-6:
                    edge_mids.append(mid / norm_mid)
                    
    edge_mids = np.array(edge_mids)
    return verts_unit, edge_mids

def _lj_energy(coords):
    """Compute total LJ energy for a set of coordinates."""
    n = len(coords)
    if n < 2:
        return 0.0
    # Compute all pairwise distances
    # Using broadcasting
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    dist_sq = dist_sq[mask]
    dist = np.sqrt(dist_sq)
    # LJ potential: 4 * (r^-12 - r^-6)
    r6 = dist ** -6
    r12 = r6 ** 2
    energy = 4.0 * np.sum(r12 - r6)
    return energy

def _energy_from_radii(r1, r2, verts_unit, edge_mids):
    """
    Construct the 55-atom Mackay icosahedron with inner radius r1 and outer radius r2.
    Returns the LJ energy.
    """
    # 1 atom at center
    # 12 atoms at r1 (vertices)
    # 12 atoms at r2 (vertices, same directions)
    # 30 atoms at r2 (edge midpoints)
    
    atoms = []
    atoms.append([0.0, 0.0, 0.0])
    
    # Inner shell
    for v in verts_unit:
        atoms.append(v * r1)
        
    # Outer shell (vertices)
    for v in verts_unit:
        atoms.append(v * r2)
        
    # Outer shell (edge midpoints)
    for e in edge_mids:
        atoms.append(e * r2)
        
    coords = np.array(atoms)
    return _lj_energy(coords)

def construct():
    start_time = time.time()
    deadline = 150.0 # 150s limit, keep margin
    
    # Get symmetry vectors
    verts_unit, edge_mids = _get_icosahedron_vectors()
    
    # Initial guess for radii
    # The LJ minimum is at r = 2^(1/6) ≈ 1.122
    # For the icosahedron, the inner radius is typically around 1.0-1.1, outer around 2.0-2.2
    r1_init = 1.0
    r2_init = 2.1
    
    # Optimize radii
    def obj(params):
        r1, r2 = params
        # Ensure positive radii
        if r1 <= 0.1 or r2 <= 0.1:
            return 1e10
        return _energy_from_radii(r1, r2, verts_unit, edge_mids)
    
    # Use Nelder-Mead for robustness, then BFGS for precision
    result_nm = minimize(obj, [r1_init, r2_init], method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 200})
    
    r1_opt, r2_opt = result_nm.x
    
    # Refine with BFGS
    result_bfgs = minimize(obj, [r1_opt, r2_opt], method='BFGS', options={'gtol': 1e-6, 'maxiter': 100})
    
    r1_final, r2_final = result_bfgs.x
    
    # Construct final coordinates
    atoms = []
    atoms.append([0.0, 0.0, 0.0])
    for v in verts_unit:
        atoms.append(v * r1_final)
    for v in verts_unit:
        atoms.append(v * r2_final)
    for e in edge_mids:
        atoms.append(e * r2_final)
        
    coords = np.array(atoms, dtype=float)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Try to relax with the provided tool if available
        try:
            import science_chemistry
            # The relax function is expected to be available
            relaxed_coords = science_chemistry.relax(coords)
            # Ensure relaxation didn't break anything (e.g., atoms too close)
            # Calculate energy to compare
            e_relaxed = _lj_energy(relaxed_coords)
            e_orig = _lj_energy(coords)
            if e_relaxed < e_orig:
                coords = relaxed_coords
        except Exception:
            pass
            
    return coords
# EVOLVE-BLOCK-END
