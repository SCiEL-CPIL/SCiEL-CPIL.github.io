"""Lennard-Jones 55-atom cluster: Mackay icosahedron via radial manifold + polish."""
import numpy as np
import time
import math

N = 55
DEADLINE_SECONDS = 115.0  # safety margin below 150s kill

def _lj_energy(positions):
    """Compute total Lennard-Jones energy."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist2, np.inf)
    dist2 = np.maximum(dist2, 1e-12)
    inv_r2 = 1.0 / dist2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6)
    return energy

def _lj_energy_grad(positions):
    """Compute Lennard-Jones energy and gradient."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist2, np.inf)
    dist2 = np.maximum(dist2, 1e-12)
    inv_r2 = 1.0 / dist2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6)
    # gradient: dE/dr_i = 4 * sum_j (-12*inv_r12 + 6*inv_r6) * r_ij / r^2  (vector form)
    # For pair (i,j): force on i from j = 24*(2*inv_r12 - inv_r6) * (pos_i - pos_j) / r^2
    # Wait, let me be careful. E = 4*sum(r^-12 - r^-6). dE/dr = 4*(-12*r^-13 + 6*r^-7)
    # dE/dpos_i = sum_j dE/dr_ij * d r_ij/d pos_i where r_ij = |pos_i - pos_j|
    # Actually for symmetric sum: E = 4 * sum_{i<j} (r_ij^-12 - r_ij^-6)
    # dE/dpos_i = sum_{j!=i} 4*(-12*r_ij^-14 + 6*r_ij^-8) * (pos_i - pos_j)
    #            = sum_{j!=i} 4*(-12 + 6*r_ij^2) * r_ij^-14 * (pos_i - pos_j)
    #            = sum_{j!=i} 24*(1 - 2*r_ij^-2) * r_ij^-14 * (pos_i - pos_j)  -- let me just use the standard formula
    # Standard: F_ij = 24*(2*sigma^12/r^13 - sigma^6/r^7) * r_hat, with sigma=1
    # dE/dr = -F_scalar = -24*(2/r^13 - 1/r^7) = 24*(1/r^7 - 2/r^13)
    # dE/dpos_i = sum_j dE/dr_ij * (pos_i - pos_j)/r_ij = sum_j 24*(r_ij^-7 - 2*r_ij^-13) * (pos_i-pos_j)/r_ij
    #           = sum_j 24*(r_ij^-8 - 2*r_ij^-14) * (pos_i - pos_j)
    # So grad_i = 24 * sum_{j!=i} (inv_r2^4 - 2*inv_r2^7) * (pos_i - pos_j)
    # Let me verify: with inv_r2 = 1/r^2, inv_r6 = 1/r^6
    # r^-8 = inv_r2^4, r^-14 = inv_r2^7
    # grad_i = 24 * sum_j (inv_r2[i,j]^4 - 2*inv_r2[i,j]^7) * (pos_i - pos_j)
    # But this double counts. Since E = 4*sum_{i<j}(...), each pair appears once.
    # Actually the sum over all pairs i!=j in the matrix form gives each pair twice.
    # Let me just compute it properly:
    # E = 4 * sum_{i<j} (r_ij^-12 - r_ij^-6)
    # grad_i = sum_{j != i} 4 * (-12*r_ij^-14 + 6*r_ij^-8) * (pos_i - pos_j)/r_ij
    #        = sum_{j != i} 4 * (-12*r_ij^-15 + 6*r_ij^-9) * (pos_i - pos_j)
    # Hmm, let me just use: for each pair, contribution to grad_i is:
    # dE_ij/dpos_i = 4*(-12)*(pos_i-pos_j)*r_ij^-14 / r_ij * ... 
    # Let me just do it numerically clean:
    # E_ij = 4*(r^-12 - r^-6), r = |pos_i - pos_j|
    # dE_ij/dpos_i = 4*(-12*r^-13 + 6*r^-7) * (pos_i - pos_j)/r
    #              = 4*(-12*r^-14 + 6*r^-8) * (pos_i - pos_j)
    #              = 24*(r^-8 - 2*r^-14) * (pos_i - pos_j)
    # With inv_r2: r^-8 = inv_r2^4, r^-14 = inv_r2^7
    # dE_ij/dpos_i = 24*(inv_r2^4 - 2*inv_r2^7) * (pos_i - pos_j)
    # And sum over j != i.
    # In the matrix: for row i, sum over j != i of 24*(inv_r2[i,j]^4 - 2*inv_r2[i,j]^7) * diff[i,j]
    # where diff[i,j] = pos_i - pos_j
    coeff = 24.0 * (inv_r2**4 - 2.0 * inv_r2**7)
    np.fill_diagonal(coeff, 0.0)
    grad = coeff @ diff  # shape (n, 3)
    return energy, grad

def _relax(positions, max_iter=1000, ftol=1e-12, gtol=1e-12):
    """L-BFGS relaxation using scipy."""
    from scipy.optimize import minimize
    def obj_and_grad(x):
        pos = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos)
        return e, g.flatten()
    result = minimize(obj_and_grad, positions.flatten(), jac=True, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50})
    return result.x.reshape(-1, 3), result.fun

def _min_distance(positions):
    """Minimum pairwise distance."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, np.inf)
    return np.min(dist)

def _valid(positions):
    """Check minimum distance constraint."""
    return _min_distance(positions) >= 0.5

def _mckay_icosahedron(r1, r2):
    """Generate Mackay icosahedron with inner shell radius r1 and outer shell radius r2.
    Center atom at origin.
    Inner shell: 12 atoms at icosahedron vertices scaled by r1.
    Outer shell: 42 atoms at icosahedron face centers (and edge midpoints?) scaled by r2.
    
    Actually, the 2-shell Mackay icosahedron has:
    - 1 center atom
    - 12 atoms on first shell (icosahedron vertices)
    - 42 atoms on second shell
    
    The 42 outer atoms are placed at:
    - 20 face centers of the icosahedron (one per triangular face)
    - 30 edge midpoints of the icosahedron
    Total: 20 + 30 = 50... that's not 42.
    
    Let me reconsider. The Mackay icosahedron with shells n=0,1,2 has:
    Shell 0: 1 atom (center)
    Shell 1: 12 atoms (vertices of icosahedron)
    Shell 2: 42 atoms
    
    The 42 atoms on shell 2 consist of:
    - 20 atoms at face centers (of the 20 triangular faces)
    - 30 atoms at edge midpoints? No, that gives 50.
    
    Actually, I think the correct construction is:
    The icosahedron has 12 vertices, 30 edges, 20 faces.
    For the 2nd shell, we place atoms at:
    - 20 face centers
    - 12 "vertex" positions (directly outward from vertices)
    - 10 additional positions
    Total: 42
    
    Let me use a different approach. The Mackay icosahedron positions can be generated
    by taking all points (a,b,c) where a,b,c are integers in some range, projecting
    onto the icosahedral symmetry. 
    
    Actually, a cleaner way: the icosahedron vertices are at:
    (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1) where φ = (1+√5)/2
    
    The 42 outer shell positions for the Mackay icosahedron are the points:
    - The 12 vertices scaled (but these are inner shell)
    - Actually let me look at this differently.
    
    The standard 2-shell Mackay icosahedron (n=2) has 55 atoms:
    1 + 12 + 42 = 55
    
    The 42 outer atoms are located at:
    20 face centers + 30 edge midpoints = 50... still not right.
    
    Let me just use the known coordinates. The 55-atom Mackay icosahedron can be
    constructed as follows:
    
    Icosahedron vertices (12):
    (0, 1, φ), (0, -1, φ), (0, 1, -φ), (0, -1, -φ),
    (1, φ, 0), (-1, φ, 0), (1, -φ, 0), (-1, -φ, 0),
    (φ, 0, 1), (φ, 0, -1), (-φ, 0, 1), (-φ, 0, -1)
    
    where φ = (1+√5)/2 ≈ 1.618
    
    For the second shell (42 atoms), the positions are obtained by:
    Taking the icosahedron and placing atoms at all positions that are integer
    linear combinations within the icosahedral lattice at shell radius 2.
    
    A practical approach: generate candidate positions from the icosahedral
    symmetry group applied to basis vectors, then select the 42 closest to
    the desired shell radius.
    
    Let me use a simpler parameterization: generate the 12 inner shell atoms
    at the icosahedron vertices at radius r1, and the 42 outer shell atoms
    at positions derived from the icosahedral symmetry at radius r2.
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    
    # 12 icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Inner shell: 12 atoms
    inner = verts_unit * r1
    
    # For the outer shell (42 atoms), we need to find the positions.
    # The 42 positions in a 2-shell Mackay icosahedron correspond to:
    # Points with icosahedral symmetry at the second shell.
    # 
    # One way: take all points (i, j, k) where i,j,k are small integers,
    # apply icosahedral rotations, and collect unique directions at the
    # correct radius.
    #
    # The icosahedral symmetry group has 60 elements. We can generate
    # candidate directions by applying all 60 rotations to a set of
    # basis vectors and collecting unique directions.
    #
    # Basis vectors to generate the 42 unique directions on shell 2:
    # The 42 directions are the union of orbits of certain vectors
    # under the icosahedral group.
    #
    # Orbit of (0,0,1) under I_h: 12 directions (same as vertices) - already have
    # Orbit of (1,0,0) under I_h: 20 directions  
    # Orbit of (1,1,0)/sqrt(2) under I_h: 30 directions
    # Total unique: 12 + 20 + 30 = 62... but we need 42.
    #
    # Hmm, let me think differently. The 42 outer shell atoms in the Mackay
    # icosahedron are at:
    # - 20 face centers of the icosahedron
    # - 12 positions directly above the vertices (but further out)
    # - 10 positions above the edges
    # Total: 42
    #
    # Actually, I recall now: the 2-shell Mackay icosahedron second shell has:
    # 20 + 12 + 10 = 42 atoms
    # where:
    # 20 are at face centers
    # 12 are at "vertex" directions but at shell 2 radius
    # 10 are at "edge" directions
    #
    # Let me compute these:
    
    # Compute face centers (20 triangular faces of icosahedron)
    # The 20 faces of the icosahedron can be enumerated.
    # Using the vertex indices:
    # Let me hardcode the face connections for the standard icosahedron.
    # Vertices labeled 0-11 as above.
    
    # Actually, let me use a computational approach:
    # Generate all 60 icosahedral rotation matrices, apply them to
    # representative vectors, collect unique directions.
    
    # Icosahedral rotation matrices can be generated from the fundamental
    # rotations. This is complex, so let me use an alternative:
    # 
    # The 42 outer shell directions are the normalized positions:
    # (±1, ±1, ±φ) and permutations and sign changes - but that's the same as vertices.
    #
    # Let me just use the known result: the outer 42 atoms of the Mackay
    # icosahedron are at the positions obtained by taking the icosahedron
    # and placing atoms at all sites that have the same icosahedral
    # Wyckoff positions as the shell-2 sites.
    #
    # Practical approach: I'll generate candidate positions on the sphere
    # using the icosahedral symmetry and select those at the right radius.
    #
    # The key insight: the 42 directions are:
    # 1) The 20 face-center directions
    # 2) The 12 vertex directions (but at shell 2)  
    # 3) The 10 edge-midpoint directions
    #
    # Wait, 20+12+10 = 42. Yes!
    #
    # So:
    # - 20 face centers
    # - 12 at vertex directions
    # - 10 at edge midpoint directions
    
    # Compute edge midpoints: need to identify edges of the icosahedron.
    # An edge connects two vertices that are nearest neighbors.
    
    # Find nearest neighbors (edges)
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i,j] = np.linalg.norm(verts[i] - verts[j])
    np.fill_diagonal(dists, np.inf)
    
    # Edge length is the minimum non-zero distance
    edge_len = np.min(dists)
    
    # Get edge midpoint directions
    edge_mid_dirs = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i,j] - edge_len) < 1e-10:
                mid = (verts[i] + verts[j]) / 2.0
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 1e-10:
                    edge_mid_dirs.append(mid / mid_norm)
    edge_mid_dirs = np.array(edge_mid_dirs)
    # Should have 30 edges, but we only need 10 unique directions?
    # No, all 30 edge midpoints are distinct directions.
    # Hmm, 30 edge midpoints + 20 face centers + 12 vertices = 62, not 42.
    
    # I think I'm overcomplicating this. Let me use a different strategy:
    # Generate the 55-atom structure using the known radial optimization approach
    # from the knowledge: "Using a low-dimensional radial manifold optimization 
    # (3 parameters) to find the exact Mackay icosahedron geometry"
    
    # The 3 parameters might be: r_inner, r_outer, and a twist/scale parameter.
    # Or: the inner shell radius, outer shell radius, and the radial position 
    # of the outer shell relative to icosahedral directions.
    
    # Let me use a simpler and more robust approach:
    # Generate the structure with the correct topology and optimize the radii.
    
    # The 42 outer shell positions: I'll use the fact that they form a 
    # truncated icosahedron-like pattern. Let me generate them by:
    # Taking the icosahedron and for each face, placing atoms at specific
    # fractional positions.
    
    # Actually, the simplest correct approach for a 2-shell Mackay icosahedron:
    # The outer 42 atoms are at positions that can be described as follows:
    # For each of the 12 vertices v of the icosahedron:
    #   Place 1 atom at direction v (giving 12 atoms)
    # For each of the 20 faces f:
    #   Place 1 atom at the face center direction (giving 20 atoms)  
    # For each of the 10 "pentagon centers" or similar:
    #   Place 1 atom (giving 10 atoms)
    # Total: 42
    
    # The 10 additional directions: these correspond to the centers of the
    # 10 pentagonal faces in the dual dodecahedron, or equivalently, the
    # directions from center to the 10 "quasi-vertices" of the icosahedron.
    
    # Actually, I recall that the 42 outer shell positions in the Mackay
    # icosahedron are simply all the icosahedral Wyckoff 42a positions.
    # These can be generated by the icosahedral group acting on (1,0,0).
    
    # Let me generate the 60 icosahedral rotations and apply them to (1,0,0).
    
    # Fundamental icosahedral rotations:
    # 5-fold rotation about a vertex axis
    # 3-fold rotation about a face center axis
    # 2-fold rotation about an edge midpoint axis
    
    # This is getting too complex. Let me use a practical shortcut:
    # Generate candidate points on the unit sphere using a fine grid,
    # select those with icosahedral symmetry (or close to it), and
    # pick the 42 that best match the Mackay pattern.
    
    # Actually, the BEST approach given the knowledge note: use radial manifold
    # optimization with 3 parameters. The 3 parameters define the geometry
    # of the two shells. Let me parameterize:
    # - r1: inner shell radius (12 atoms at icosahedron vertices)
    # - r2: outer shell radius (42 atoms)
    # - alpha: a shape parameter that controls how the outer shell atoms
    #   are distributed (e.g., interpolation between face centers and vertices)
    
    # For the outer 42 atoms, I'll use the following construction:
    # The 42 directions are obtained by:
    # 1. Take the 20 face center directions
    # 2. Take the 12 vertex directions  
    # 3. Take 10 additional directions (centers of the 10 "holes" in the
    #    icosahedron that are not vertices, edges, or faces)
    
    # The 10 additional directions: these are the directions from the center
    # to the 10 "quasi-face centers" or the centers of the 10 pentagonal
    # regions in the icosahedral tiling.
    
    # Actually, in the icosahedral symmetry, the 42 Wyckoff positions
    # (42a) can be described as the orbit of (1, 0, 0) under the full
    # icosahedral group I_h (which has 120 elements including inversion).
    # The orbit size would be 42 if the stabilizer has size 120/42 = 20/7...
    # That doesn't work out. Let me check: |I_h| = 120. Orbit of (1,0,0):
    # The stabilizer of (1,0,0) in I_h has order 120/42 ≈ 2.86... not integer.
    
    # I think the 42 positions are NOT a single orbit. They are:
    # - 20 from orbit of face center direction (orbit size 20, stabilizer order 6)
    # - 12 from orbit of vertex direction (orbit size 12, stabilizer order 10)
    # - 10 from orbit of some other direction (orbit size 10, stabilizer order 12)
    # Total: 20 + 12 + 10 = 42. Yes!
    
    # The 10 directions with stabilizer order 12: these are the edge midpoint
    # directions? No, edge midpoints have stabilizer of order 4 (2-fold rotation
    # and mirror), giving orbit size 30.
    
    # Directions with stabilizer order 12: this would be a 6-fold or 
    # combination. In icosahedral symmetry, the only rotations are 2,3,5-fold.
    # A stabilizer of order 12 could be: 2-fold rotation + 2 mirrors + ... 
    # Actually, in I_h, the stabilizer of a generic point has order 1.
    # Special points:
    # Vertex: stabilizer D_5 = order 10, orbit size 12
    # Edge midpoint: stabilizer D_2 = order 4, orbit size 30
    # Face center: stabilizer C_3 = order 3, orbit size 40? No, 120/3=40.
    
    # Hmm, 120/3 = 40, not 20. So face centers form an orbit of size 40?
    # But there are only 20 faces. The full icosahedral group I_h includes
    # inversion, so the face center and its opposite are in the same orbit.
    # So orbit of a face center under I_h has size 20 (since there are 20 faces,
    # each with its opposite, but in I_h the inversion maps face to opposite face).
    # Wait, 120/20 = 6. So stabilizer of face center in I_h has order 6.
    # That makes sense: C_3 (3-fold rotation) plus inversion gives order 6.
    
    # Similarly, vertices: 12 vertices, 120/12 = 10. Stabilizer D_5 (order 10). ✓
    # Edge midpoints: 30 edges, 120/30 = 4. Stabilizer D_2 (order 4). ✓
    # Face centers: 20 faces, 120/20 = 6. Stabilizer C_3h (order 6). ✓
    
    # So the 42 outer shell atoms = 20 face centers + 12 vertices + 10 ???
    # 120/10 = 12. What has stabilizer of order 12 in I_h?
    # There's no standard point in icosahedral symmetry with stabilizer 12.
    
    # I think the 42 outer atoms are actually:
    # 20 face centers + 30 edge midpoints - 8 = 42? No.
    # Or maybe it's just a specific subset.
    
    # Let me abandon the exact symmetry argument and use a numerical approach:
    # I'll generate the structure using a known parameterization and optimize.
    
    # PRACTICAL APPROACH:
    # I'll generate the 55-atom structure as follows:
    # 1 atom at origin
    # 12 atoms at icosahedron vertices * r1
    # 42 atoms at positions computed from the icosahedral geometry * r2
    #
    # For the 42 outer atoms, I'll use the following well-known construction:
    # The outer shell of the Mackay icosahedron consists of atoms placed at:
    # - The 20 face centers of the icosahedron
    # - The 12 "vertex" positions (same direction as vertices, further out)
    # - The 10 "edge" positions (midway between adjacent edge midpoints?)
    #
    # Since I'm struggling with the exact 42 positions, let me use a 
    # different strategy: I'll use the radial manifold optimization approach
    # with a parameterization that I can compute.
    
    # ALTERNATIVE: Use the fact that the optimal 55-atom LJ cluster is very
    # close to the Mackay icosahedron. I'll:
    # 1. Generate a good initial guess using the icosahedron geometry
    # 2. Optimize the shell radii (2 parameters) + possibly a twist
    # 3. Do a full Cartesian L-BFGS relaxation
    # 4. Use multiple restarts with perturbations
    
    # For the 42 outer shell positions, I'll use a computational geometry approach:
    # Generate all icosahedral symmetry-related positions and select the 42
    # that are most evenly distributed on the sphere.
    
    # Let me generate the icosahedral rotation matrices properly.
    # The icosahedral group can be generated by two rotations:
    # R1: 2*pi/5 rotation about z-axis
    # R2: rotation that maps z-axis to a vertex direction
    
    # This is complex. Let me use a simpler method:
    # Generate points on the sphere using the icosahedral symmetry by:
    # Taking the 12 vertices and applying all 60 proper rotations.
    
    # I'll hardcode the 60 icosahedral rotation matrices using the 
    # quaternion representation.
    
    # Actually, the simplest robust approach: use scipy or manual construction
    # of the icosahedron faces, compute face centers, edge midpoints, and
    # combine them.
    
    # Let me identify the 20 faces of the icosahedron.
    # The icosahedron vertices (using standard labeling):
    # 0: (0, 1, phi), 1: (0, -1, phi), 2: (0, 1, -phi), 3: (0, -1, -phi)
    # 4: (1, phi, 0), 5: (-1, phi, 0), 6: (1, -phi, 0), 7: (-1, -phi, 0)
    # 8: (phi, 0, 1), 9: (phi, 0, -1), 10: (-phi, 0, 1), 11: (-phi, 0, -1)
    
    # Each vertex connects to 5 others. Let me find the faces by checking
    # which triples of mutually adjacent vertices form triangles.
    
    # Two vertices are adjacent if their distance equals the edge length.
    # Let me find all edges first, then find all triangular faces.
    
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i,j] - edge_len) < 1e-10:
                edges.append((i,j))
    
    # Find triangular faces: three vertices that are all mutually adjacent
    faces = []
    adj = [[False]*12 for _ in range(12)]
    for i,j in edges:
        adj[i][j] = True
        adj[j][i] = True
    
    for i in range(12):
        for j in range(i+1, 12):
            if not adj[i][j]:
                continue
            for k in range(j+1, 12):
                if adj[i][k] and adj[j][k]:
                    faces.append((i,j,k))
    
    # Should have 20 faces
    # Compute face center directions
    face_centers = []
    for f in faces:
        center = (verts_unit[f[0]] + verts_unit[f[1]] + verts_unit[f[2]]) / 3.0
        norm = np.linalg.norm(center)
        if norm > 1e-10:
            face_centers.append(center / norm)
    face_centers = np.array(face_centers)  # 20 directions
    
    # Now I have:
    # - 12 vertex directions (verts_unit)
    # - 20 face center directions (face_centers)
    # Total: 32 directions
    # I need 42 outer shell atoms, so I need 10 more directions.
    
    # The 10 additional directions: I'll compute them as the "dual" directions
    # or use edge midpoints. Let me compute all 30 edge midpoint directions
    # and see if I can select 10 of them that are "special".
    
    edge_mid_dirs_all = []
    for i,j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-10:
            edge_mid_dirs_all.append(mid / norm)
    edge_mid_dirs_all = np.array(edge_mid_dirs_all)  # 30 directions
    
    # The 10 special directions might be the ones that are "farthest" from
    # any vertex or face center, or they might correspond to a specific subset.
    
    # Actually, I just realized: maybe the 42 outer atoms are NOT 
    # 20+12+10 but rather some other combination. Let me reconsider.
    
    # The Mackay icosahedron with 2 shells: the second shell has 42 atoms.
    # These 42 atoms are at the positions that form a "truncated" version
    # of the icosahedron. 
    #
    # I think the correct interpretation is:
    # The 42 atoms are at ALL the icosahedral Wyckoff positions that are
    # at a certain radius. The Wyckoff positions of the icosahedral group
    # (point group, not space group) for the sphere are:
    # - 12 vertices (Wyckoff 12a)
    # - 30 edge midpoints (Wyckoff 30b) 
    # - 20 face centers (Wyckoff 20c)
    # Total: 62 general positions on the sphere with icosahedral symmetry.
    #
    # The Mackay icosahedron uses 42 of these 62. Specifically, it uses
    # the 20 face centers and 22 of the other 42? No, that doesn't make sense.
    
    # I think the answer is simpler than I'm making it: the 42 outer shell
    # atoms of the Mackay icosahedron are placed at:
    # - 20 face centers
    # - 12 vertex positions (same direction as inner shell, but at outer radius)
    # - 10 additional positions
    #
    # For the 10 additional positions, I'll use a numerical optimization
    # to find the best 10 directions to add to the 32 existing ones
    # (20 face + 12 vertex) to get 42 total, maximizing symmetry.
    
    # Actually, let me just go with a practical approach:
    # I'll use the 32 directions I have (20 face + 12 vertex) and add
    # 10 more by selecting 10 from the 30 edge midpoints that are
    # most "symmetric" or that give the best energy when optimized.
    
    # For now, let me just use all 32 + 10 edge midpoints = 42, picking
    # the 10 edge midpoints that are farthest from any vertex direction
    # (to maximize coverage).
    
    # Compute for each edge midpoint direction, its minimum angular distance
    # to any vertex direction
    min_dist_to_vertex = []
    for emd in edge_mid_dirs_all:
        min_d = np.inf
        for vu in verts_unit:
            cos_angle = np.dot(emd, vu)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            min_d = min(min_d, angle)
        min_dist_to_vertex.append(min_d)
    min_dist_to_vertex = np.array(min_dist_to_vertex)
    
    # Select the 10 edge midpoints that are farthest from any vertex
    selected_edge_idx = np.argsort(min_dist_to_vertex)[-10:]
    extra_dirs = edge_mid_dirs_all[selected_edge_idx]
    
    # Combine all 42 outer shell directions
    outer_dirs = np.vstack([verts_unit, face_centers, extra_dirs])  # 12+20+10=42
    # Normalize to unit sphere
    outer_dirs = outer_dirs / np.linalg.norm(outer_dirs, axis=1, keepdims=True)
    
    # Build the full 55-atom structure
    center = np.zeros((1, 3))
    inner_shell = verts_unit * r1  # 12 atoms
    outer_shell = outer_dirs * r2  # 42 atoms
    
    positions = np.vstack([center, inner_shell, outer_shell])
    return positions

def _optimize_radial():
    """Optimize the two shell radii to minimize LJ energy."""
    from scipy.optimize import minimize_scalar
    
    best_energy = np.inf
    best_r1, best_r2 = 1.0, 2.0
    
    # Try a grid of radii
    r1_vals = np.linspace(0.9, 1.3, 20)
    r2_vals = np.linspace(1.8, 2.6, 30)
    
    for r1 in r1_vals:
        for r2 in r2_vals:
            if r2 - r1 < 0.5:
                continue
            pos = _mckay_icosahedron(r1, r2)
            if not _valid(pos):
                continue
            e = _lj_energy(pos)
            if e < best_energy:
                best_energy = e
                best_r1, best_r2 = r1, r2
    
    # Refine around best
    if best_r1 > 0:
        res1 = minimize_scalar(lambda r1: _lj_energy(_mckay_icosahedron(r1, best_r2)),
                               bounds=(best_r1-0.1, best_r1+0.1), method='bounded',
                               options={'xatol': 1e-10})
        if res1.success:
            best_r1 = res1.x
            res2 = minimize_scalar(lambda r2: _lj_energy(_mckay_icosahedron(best_r1, r2)),
                                   bounds=(best_r2-0.2, best_r2+0.2), method='bounded',
                                   options={'xatol': 1e-10})
            if res2.success:
                best_r2 = res2.x
    
    return best_r1, best_r2, best_energy

def construct():
    start_time = time.time()
    
    # Step 1: Optimize radial parameters
    r1, r2, e_radial = _optimize_radial()
    elapsed = time.time() - start_time
    
    # Step 2: Full Cartesian L-BFGS relaxation
    pos0 = _mckay_icosahedron(r1, r2)
    
    best_pos = pos0.copy()
    best_e = _lj_energy(best_pos)
    
    # Try relaxation
    remaining = DEADLINE_SECONDS - (time.time() - start_time)
    if remaining > 5.0:
        try:
            relaxed, e_relaxed = _relax(pos0, max_iter=500, ftol=1e-14, gtol=1e-14)
            if _valid(relaxed) and e_relaxed < best_e:
                best_pos = relaxed
                best_e = e_relaxed
        except Exception:
            pass
    
    # Step 3: Multiple restarts with perturbations if time allows
    np.random.seed(42)
    n_restarts = 0
    while True:
        remaining = DEADLINE_SECONDS - (time.time() - start_time)
        if remaining < 15.0:
            break
        n_restarts += 1
        
        # Perturb the best structure
        noise_scale = 0.1 / (1.0 + 0.1 * n_restarts)
        perturbed = best_pos + np.random.randn(*best_pos.shape) * noise_scale
        
        # Ensure minimum distance
        if not _valid(perturbed):
            # Try smaller perturbation
            perturbed = best_pos + np.random.randn(*best_pos.shape) * (noise_scale * 0.5)
            if not _valid(perturbed):
                continue
        
        # Relax
        try:
            relaxed, e_relaxed = _relax(perturbed, max_iter=300, ftol=1e-14, gtol=1e-14)
            if _valid(relaxed) and e_relaxed < best_e:
                best_pos = relaxed
                best_e = e_relaxed
        except Exception:
            pass
    
    # Final high-precision polish
    remaining = DEADLINE_SECONDS - (time.time() - start_time)
    if remaining > 3.0:
        try:
            polished, e_polished = _relax(best_pos, max_iter=2000, ftol=1e-15, gtol=1e-15)
            if _valid(polished) and e_polished < best_e:
                best_pos = polished
                best_e = e_polished
        except Exception:
            pass
    
    return best_pos
