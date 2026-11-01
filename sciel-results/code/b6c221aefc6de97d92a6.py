# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 55

def lj_energy(pos):
    """Compute total LJ energy for N atoms."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    return e

def lj_energy_grad(pos):
    """Compute LJ energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return e, grad

def get_icosahedral_symmetry_operations():
    """
    Generate all 60 rotation matrices of the icosahedral group I.
    Uses the 12 vertices of a regular icosahedron to construct
    the full set of symmetry operations.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi
    
    # 12 vertices of icosahedron (unit sphere)
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Generate all 60 rotation matrices
    # An icosahedron has 60 rotational symmetries
    # We can generate them by finding all orthogonal matrices that map the vertex set to itself
    
    rotations = []
    
    # Identity
    rotations.append(np.eye(3))
    
    # 6 rotations by 72, 144, 216, 288 degrees about each of 6 five-fold axes
    # 10 rotations by 60, 120, 180, 240, 300 degrees about each of 10 three-fold axes  
    # 15 rotations by 90, 180, 270 degrees about each of 15 two-fold axes
    # Total: 1 + 6*4 + 10*5 + 15*3 = 1 + 24 + 50 + 45 = 120... wait, that's 120 for full group
    # For rotational group I (not Ih): 1 + 6*4 + 10*5 + 15*3 = 120? No.
    # Actually: 1 identity + 6 axes * 4 non-trivial rotations each = 24 (5-fold)
    #           + 10 axes * 2 non-trivial rotations each = 20 (3-fold, 120 and 240)
    #           + 15 axes * 1 non-trivial rotation each = 15 (2-fold, 180)
    # Total: 1 + 24 + 20 + 15 = 60. Correct.
    
    # Alternative approach: find all rotations that permute the 12 vertices
    # Use the fact that if R is a symmetry, R*verts must be a permutation of verts
    
    def find_all_symmetries(verts, n_verts=12):
        """Find all 3x3 orthogonal matrices that map vertex set to itself."""
        symms = [np.eye(3)]
        
        # Try all possible mappings of vertex 0 to any vertex, 
        # and vertex 1 to any vertex that maintains the same distance
        v0 = verts[0]
        for i in range(n_verts):
            v0_target = verts[i]
            # Find all vertices at the same distance from v0 as v1 is
            v1 = verts[1]
            d01 = np.linalg.norm(v0 - v1)
            candidates_for_v1 = []
            for j in range(n_verts):
                if j != i:
                    d = np.linalg.norm(verts[j] - v0_target)
                    if np.isclose(d, d01, atol=1e-10):
                        candidates_for_v1.append(j)
            
            for j in candidates_for_v1:
                v1_target = verts[j]
                # Construct rotation matrix that maps v0->v0_target and v1->v1_target
                # Use the cross product to determine the third axis
                v2 = np.cross(v0, v1)
                v2 = v2 / np.linalg.norm(v2)
                
                v2_target = np.cross(v0_target, v1_target)
                v2_target = v2_target / np.linalg.norm(v2_target)
                
                # Build rotation matrix: columns are images of basis vectors
                # We need to express the rotation in terms of the original basis
                # M = [v0_target v1_target v2_target] * [v0 v1 v2]^{-1}
                # But [v0 v1 v2] is not necessarily orthonormal, so we need to be careful
                
                # Better: use the formula for rotation taking one orthonormal frame to another
                # First, construct orthonormal bases from v0, v1
                e1 = v0
                e2_raw = v1 - np.dot(v1, e1) * e1
                e2 = e2_raw / np.linalg.norm(e2_raw)
                e3 = np.cross(e1, e2)
                
                e1_t = v0_target
                e2_raw_t = v1_target - np.dot(v1_target, e1_t) * e1_t
                e2_t = e2_raw_t / np.linalg.norm(e2_raw_t)
                e3_t = np.cross(e1_t, e2_t)
                
                # Rotation matrix: R = [e1_t e2_t e3_t] * [e1 e2 e3]^T
                R = np.column_stack([e1_t, e2_t, e3_t]) @ np.column_stack([e1, e2, e3]).T
                
                # Verify it's a proper rotation
                if np.isclose(np.linalg.det(R), 1.0, atol=1e-10):
                    # Check if this is a valid symmetry
                    is_sym = True
                    for k in range(n_verts):
                        mapped = R @ verts[k]
                        # Check if mapped is close to any vertex
                        found = False
                        for m in range(n_verts):
                            if np.linalg.norm(mapped - verts[m]) < 1e-8:
                                found = True
                                break
                        if not found:
                            is_sym = False
                            break
                    
                    if is_sym:
                        # Check if this rotation is already in our list
                        is_new = True
                        for existing in symms:
                            if np.allclose(R, existing, atol=1e-10):
                                is_new = False
                                break
                        if is_new:
                            symms.append(R)
        
        return symms
    
    rotations = find_all_symmetries(verts)
    
    # We should have 60 rotations for the icosahedral group
    # If we got fewer, there may be numerical issues
    return rotations

def get_mackay_directions_exact():
    """
    Returns exact directions for the Mackay icosahedron shells using
    icosahedral symmetry operations.
    
    Shell 1: 12 vertices (icosahedron vertices)
    Shell 2: 30 edge midpoints (midpoints of icosahedron edges)
    Shell 3: 12 face centers (centers of icosahedron faces)
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Find edges: pairs of vertices at minimum distance
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
            if np.isclose(d, min_dist, atol=1e-10):
                edges.append((i, j))
    
    # 30 edge midpoints (normalized)
    mids = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        mids.append(mid)
    mids = np.array(mids)
    
    # Find faces: triples of vertices that form an equilateral triangle
    # with side length = min_dist
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            d_ij = np.linalg.norm(verts[i] - verts[j])
            if not np.isclose(d_ij, min_dist, atol=1e-10):
                continue
            for k in range(j+1, 12):
                d_ik = np.linalg.norm(verts[i] - verts[k])
                d_jk = np.linalg.norm(verts[j] - verts[k])
                if (np.isclose(d_ik, min_dist, atol=1e-10) and 
                    np.isclose(d_jk, min_dist, atol=1e-10)):
                    faces.append((i, j, k))
    
    # 20 faces total
    # 12 face centers (normalized)
    face_centers = []
    for i, j, k in faces:
        center = (verts[i] + verts[j] + verts[k]) / 3.0
        center = center / np.linalg.norm(center)
        face_centers.append(center)
    face_centers = np.array(face_centers)
    
    # Verify counts
    assert len(verts) == 12, f"Expected 12 vertices, got {len(verts)}"
    assert len(mids) == 30, f"Expected 30 edge midpoints, got {len(mids)}"
    assert len(face_centers) == 12, f"Expected 12 face centers, got {len(face_centers)}"
    
    return verts, mids, face_centers

def make_mackay_55_exact():
    """
    Create a 55-atom Mackay icosahedron with exact Ih symmetry.
    Structure:
    - 1 atom at center
    - 12 atoms at icosahedron vertices (radius r1)
    - 30 atoms at edge midpoints (radius r2)
    - 12 atoms at face centers (radius r3)
    
    Uses literature-based initial radii for LJ 55.
    """
    verts, mids, face_centers = get_mackay_directions_exact()
    
    # Literature-based initial radii for LJ 55 Mackay icosahedron
    # From Cambridge Cluster Database and related literature
    r1 = 1.0485  # First shell radius
    r2 = 1.8540  # Second shell radius  
    r3 = 2.5830  # Third shell radius
    
    pos = np.vstack([
        np.array([[0.0, 0.0, 0.0]]),
        verts * r1,
        mids * r2,
        face_centers * r3
    ])
    
    return pos

def relax(pos, maxiter=1000, gtol=1e-12):
    """Relax using L-BFGS-B with tight tolerance."""
    def fun(x):
        x = x.reshape(N, 3)
        e, g = lj_energy_grad(x)
        return e, g.reshape(-1)
    
    try:
        result = minimize(fun, pos.reshape(-1), method='L-BFGS-B', jac=True, 
                         options={'maxiter': maxiter, 'gtol': gtol, 'ftol': 1e-15,
                                  'maxls': 50, 'maxcor': 20})
        return result.x.reshape(N, 3), result.fun
    except Exception:
        return pos, lj_energy(pos)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Step 1: Create exact Mackay icosahedron
    pos = make_mackay_55_exact()
    e0 = lj_energy(pos)
    
    # Step 2: Single tight L-BFGS relaxation
    if time.time() < deadline - 10:
        pos_relaxed, e_relaxed = relax(pos, maxiter=1000, gtol=1e-12)
        
        # Verify minimum distances
        diff = pos_relaxed[:, np.newaxis, :] - pos_relaxed[np.newaxis, :, :]
        r2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(r2, np.inf)
        min_r = np.sqrt(r2.min())
        
        if min_r < 0.5:
            # Fallback to original if constraint violated
            pos_relaxed = pos
            e_relaxed = e0
    
    # Step 3: Final high-precision polish if time allows
    if time.time() < deadline - 5:
        pos_final, e_final = relax(pos_relaxed, maxiter=500, gtol=1e-14)
        
        diff = pos_final[:, np.newaxis, :] - pos_final[np.newaxis, :, :]
        r2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(r2, np.inf)
        min_r = np.sqrt(r2.min())
        
        if min_r >= 0.5 and e_final < e_relaxed:
            pos_relaxed = pos_final
            e_relaxed = e_final
    
    return pos_relaxed

# EVOLVE-BLOCK-END
