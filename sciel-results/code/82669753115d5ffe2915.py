"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time

def _lj_energy_grad(pos):
    """Compute LJ energy and gradient for a set of positions."""
    n = len(pos)
    rij = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(rij * rij, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r2_safe = np.where(r2 < 1e-12, 1e-12, r2)
    
    r6 = r2_safe ** (-3)
    r12 = r6 * r6
    
    # Energy: 4 * sum_{i<j} (r^-12 - r^-6)
    energy = 2.0 * np.sum(4.0 * (r12 - r6))  # factor 2 for double counting
    
    # Gradient: dE/dpos_i = sum_{j != i} 4 * (-12*r^-13 + 6*r^-7) * (pos_i - pos_j) / r
    # = sum_j 4 * (-12/r^13 + 6/r^7) * (pos_i - pos_j)
    # In terms of r2: r = sqrt(r2), r^7 = r2^(3.5), r^13 = r2^(6.5)
    r = np.sqrt(r2_safe)
    r7 = r ** 7
    r13 = r ** 13
    
    # Force on i due to j: 4 * (12/r^13 - 6/r^7) * (pos_i - pos_j)
    # Gradient of energy w.r.t. pos_i = -sum_j force_ij
    # Actually: dE/dr = 4*(-12*r^-13 + 6*r^-7), and dr/dpos_i = (pos_i - pos_j)/r
    # So dE/dpos_i = sum_j 4*(-12*r^-13 + 6*r^-7) * (pos_i - pos_j)/r
    # = sum_j 4*(-12*r^-14 + 6*r^-8) * (pos_i - pos_j)
    
    coeff = 4.0 * (-12.0 / (r2_safe * r7) + 6.0 / (r2_safe * r))  # 4*(-12/r^13 + 6/r^7)/r = 4*(-12/r^14 + 6/r^8)
    # Wait, let me redo this more carefully.
    # E = 4 * sum_{i<j} (r_ij^-12 - r_ij^-6)
    # dE/dpos_i = sum_{j != i} 4 * (-12 * r_ij^-13 * dr_ij/dpos_i + 6 * r_ij^-7 * dr_ij/dpos_i)
    # dr_ij/dpos_i = (pos_i - pos_j) / r_ij
    # So dE/dpos_i = sum_{j != i} 4 * (-12 * r_ij^-14 + 6 * r_ij^-8) * (pos_i - pos_j)
    
    coeff2 = 4.0 * (-12.0 / (r2_safe ** 3.0 * r) + 6.0 / (r2_safe ** 4.0))
    # r^-14 = r2^-7, r^-8 = r2^-4
    coeff2 = 4.0 * (-12.0 / (r2_safe ** 7) + 6.0 / (r2_safe ** 4))
    
    grad = np.sum(coeff2[:, :, np.newaxis] * rij, axis=1)
    return energy, grad


def _icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron (unnormalized)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    return verts


def _make_icosahedron_55(r1, r2):
    """Construct 55-atom two-shell Mackay icosahedron with given shell radii."""
    verts = _icosahedron_vertices()
    norms = np.linalg.norm(verts, axis=1)
    verts_norm = verts / norms[:, np.newaxis]
    
    # 1 center atom
    center = np.array([[0.0, 0.0, 0.0]])
    
    # 12 inner shell atoms
    inner = verts_norm * r1
    
    # 42 outer shell atoms: 12 face-center directions + 30 edge-midpoint directions
    # Face centers: average of 3 vertices forming each face of icosahedron
    # Icosahedron has 20 faces. We need to identify them.
    # Alternative: use known construction.
    
    # The 42 positions in the second shell of a Mackay icosahedron are:
    # - 12 along face-center directions
    # - 30 along edge-midpoint directions
    # All at the same radius r2.
    
    # Let's compute face centers and edge midpoints from the icosahedron geometry.
    # Icosahedron faces (20 triangles):
    # This is complex. Let's use a simpler approach:
    # Generate all pairs and triples, find which form faces.
    
    # Actually, let's use a well-known set of 42 directions.
    # The 42 directions can be generated as follows:
    # Take the 12 icosahedron vertices, and for each pair that forms an edge,
    # the midpoint direction is an edge-midpoint direction.
    # For face centers, take each face (triangle) and use the normalized sum.
    
    # Icosahedron edges: two vertices are connected if their distance is 2 (in the unit icosahedron with vertices at (0,±1,±φ) etc., edge length is 2)
    edge_pairs = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if abs(d - 2.0) < 1e-9:
                edge_pairs.append((i, j))
    
    # Edge midpoint directions (30 edges)
    edge_dirs = []
    for i, j in edge_pairs:
        mid = (verts[i] + verts[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-9:
            edge_dirs.append(mid / norm)
    edge_dirs = np.array(edge_dirs)
    
    # Face centers: find all triangles of 3 mutually connected vertices
    face_dirs = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                # Check if i-j, j-k, i-k are all edges
                di_j = np.linalg.norm(verts[i] - verts[j])
                dj_k = np.linalg.norm(verts[j] - verts[k])
                di_k = np.linalg.norm(verts[i] - verts[k])
                if abs(di_j - 2.0) < 1e-9 and abs(dj_k - 2.0) < 1e-9 and abs(di_k - 2.0) < 1e-9:
                    face_center = (verts[i] + verts[j] + verts[k]) / 3.0
                    norm = np.linalg.norm(face_center)
                    if norm > 1e-9:
                        face_dirs.append(face_center / norm)
    face_dirs = np.array(face_dirs)
    
    # Should have 30 edge dirs and 20 face dirs, but we only need 12 face-center directions
    # Wait, the Mackay icosahedron second shell has 42 atoms: 12 + 30
    # The 12 are along the same directions as the 1st shell (face centers of the dual dodecahedron?)
    # Let me reconsider.
    
    # Actually, in the Mackay icosahedron:
    # Shell 1: 12 atoms at icosahedron vertices
    # Shell 2: 42 atoms at:
    #   - 12 positions along the same directions as shell 1 but further out? No.
    #   - The 42 are: 12 face-centers of the icosahedron + 30 edge-midpoints
    # But that gives 20 + 30 = 50, not 42.
    
    # Let me look at this differently. The Mackay icosahedron with shells n=1,2:
    # n=1: 1 + 12 = 13 atoms
    # n=2: 1 + 12 + 42 = 55 atoms
    # The 42 atoms in the second shell are at positions:
    # - 12 at the icosahedron vertex directions (same as shell 1)
    # - 30 at edge midpoint directions
    # That's 12 + 30 = 42. Yes!
    
    # So shell 2 = 12 atoms at icosahedron vertex directions (radius r2) + 30 atoms at edge midpoint directions (radius r2)
    
    outer_vertex = verts_norm * r2
    outer_edge = edge_dirs * r2
    
    positions = np.vstack([center, inner, outer_vertex, outer_edge])
    return positions


def _optimize_radii():
    """Optimize shell radii using Nelder-Mead on the relaxed energy."""
    from scipy.optimize import minimize
    
    def objective(params):
        r1, r2 = params
        if r1 < 0.5 or r2 < 0.5 or r1 > 3.0 or r2 > 3.0:
            return 1e10
        pos = _make_icosahedron_55(r1, r2)
        # Check minimum distance
        rij = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2_arr = np.sum(rij * rij, axis=2)
        np.fill_diagonal(r2_arr, 1e10)
        min_dist = np.sqrt(np.min(r2_arr))
        if min_dist < 0.5:
            return 1e10
        try:
            from science_chemistry import relax
            relaxed = relax(pos)
            e, _ = _lj_energy_grad(relaxed)
            return e
        except:
            return 1e10
    
    # Initial guess: literature values approximately r1=1.12, r2=2.0
    x0 = np.array([1.12, 2.0])
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 50, 'xatol': 0.01, 'fatol': 0.1})
        if res.success or res.fun < 1e9:
            return res.x
    except:
        pass
    return x0


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Step 1: Optimize shell radii
    try:
        radii = _optimize_radii()
        r1, r2 = radii
    except:
        r1, r2 = 1.12, 2.0
    
    # Step 2: Construct the icosahedron
    pos = _make_icosahedron_55(r1, r2)
    
    # Ensure minimum distance constraint
    rij = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2_arr = np.sum(rij * rij, axis=2)
    np.fill_diagonal(r2_arr, 1e10)
    min_dist = np.sqrt(np.min(r2_arr))
    if min_dist < 0.5:
        # Scale up slightly
        scale = 0.5 / min_dist * 1.01
        pos = pos * scale
    
    # Step 3: Relax with multiple perturbations if time permits
    best_pos = pos
    best_energy = _lj_energy_grad(pos)[0]
    
    # Try relaxing the initial configuration
    try:
        from science_chemistry import relax
        relaxed = relax(pos)
        e, _ = _lj_energy_grad(relaxed)
        if e < best_energy:
            best_energy = e
            best_pos = relaxed
    except:
        pass
    
    # Try perturbations if time remains
    rng = np.random.RandomState(42)
    for trial in range(20):
        if time.time() > deadline - 15.0:
            break
        
        # Add small random perturbation
        pert = rng.normal(0, 0.1, size=pos.shape)
        perturbed = pos + pert
        
        # Check minimum distance
        rij = perturbed[:, np.newaxis, :] - perturbed[np.newaxis, :, :]
        r2_arr = np.sum(rij * rij, axis=2)
        np.fill_diagonal(r2_arr, 1e10)
        min_dist = np.sqrt(np.min(r2_arr))
        if min_dist < 0.5:
            continue
        
        try:
            from science_chemistry import relax
            relaxed = relax(perturbed)
            e, _ = _lj_energy_grad(relaxed)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed
        except:
            continue
    
    # Final check
    if time.time() > deadline:
        return best_pos
    
    # Final high-precision polish
    try:
        from science_chemistry import relax
        polished = relax(best_pos)
        e, _ = _lj_energy_grad(polished)
        if e < best_energy:
            best_pos = polished
    except:
        pass
    
    return best_pos
# EVOLVE-BLOCK-END
