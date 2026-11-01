"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _lj_energy(pos):
    """Compute Lennard-Jones energy for a set of positions."""
    n = len(pos)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = pos[i] - pos[j]
            r2 = np.dot(d, d)
            r = np.sqrt(r2)
            if r < 1e-10:
                energy += 1e10
            else:
                inv_r2 = 1.0 / r2
                inv_r6 = inv_r2 * inv_r2 * inv_r2
                energy += 4.0 * (inv_r6 * inv_r6 - inv_r6)
    return energy


def _mackay_icosahedron_55(r1, r2):
    """
    Construct a 55-atom Mackay icosahedron.
    r1: radius of inner shell (12 atoms at icosahedron vertices)
    r2: radius of outer shell (42 atoms)
    Center atom at origin.
    """
    # Center atom
    atoms = [np.array([0.0, 0.0, 0.0])]
    
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # Icosahedron vertices (12 atoms)
    verts = np.array([
        [0, 1, phi],
        [0, 1, -phi],
        [0, -1, phi],
        [0, -1, -phi],
        [1, phi, 0],
        [1, -phi, 0],
        [-1, phi, 0],
        [-1, -phi, 0],
        [phi, 0, 1],
        [phi, 0, -1],
        [-phi, 0, 1],
        [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Inner shell: 12 atoms at radius r1
    for v in verts:
        atoms.append(v * r1)
    
    # Outer shell: 42 atoms
    # The 42 atoms of the second shell in a Mackay icosahedron are at:
    # - 20 face centers of the icosahedron (triangular faces)
    # - 30 edge midpoints of the icosahedron
    
    # First, find all edges of the icosahedron
    # An icosahedron has 30 edges
    # Two vertices are connected if their distance equals the edge length
    
    # Edge length for unit icosahedron vertices
    # Distance between [0,1,phi] and [1,phi,0]:
    # sqrt((1-0)^2 + (phi-1)^2 + (0-phi)^2) = sqrt(1 + (phi-1)^2 + phi^2)
    # phi^2 = phi + 1
    # = sqrt(1 + (phi-1)^2 + phi+1) = sqrt(2 + (phi-1)^2 + phi)
    # (phi-1) = 1/phi = phi - 1
    # Let's just compute numerically
    
    edge_len_sq = 2.0  # For unit icosahedron, edge length squared is 2.0
    # Verify: [0,1,phi] to [1,phi,0]: (1)^2 + (phi-1)^2 + (-phi)^2 = 1 + (phi-1)^2 + phi^2
    # phi^2 = phi + 1
    # (phi-1)^2 = phi^2 - 2phi + 1 = (phi+1) - 2phi + 1 = 2 - phi
    # So total = 1 + 2 - phi + phi + 1 = 4 - phi
    # phi = 1.618..., so 4 - phi = 2.382
    # Hmm, let me recalculate. Actually for a regular icosahedron with vertices as above,
    # the edge length squared is 2.0 when vertices are at distance 1 from origin?
    # Let me just compute it properly.
    
    d2 = (verts[0] - verts[4])**2  # [0,1,phi] to [1,phi,0]
    # = 1 + (phi-1)^2 + phi^2
    # phi = (1+sqrt(5))/2 ≈ 1.618
    # phi^2 ≈ 2.618
    # (phi-1) ≈ 0.618, (phi-1)^2 ≈ 0.382
    # So d2 ≈ 1 + 0.382 + 2.618 = 4.0
    # Wait, that doesn't seem right for edge length.
    
    # Let me just find all pairs at minimum distance
    d2_matrix = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            d2_matrix[i,j] = np.dot(verts[i] - verts[j], verts[i] - verts[j])
    
    min_d2 = d2_matrix.min()
    
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(d2_matrix[i,j] - min_d2) < 1e-10:
                edges.append((i, j))
    
    # There should be 30 edges
    # Add edge midpoints (30 atoms)
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid_norm = np.linalg.norm(mid)
        if mid_norm > 1e-10:
            mid_unit = mid / mid_norm
            atoms.append(mid_unit * r2)
    
    # Now add face centers (20 atoms)
    # An icosahedron has 20 triangular faces
    # Each face is a set of 3 vertices that form a triangle
    # We need to find all faces
    
    # For each edge, find the third vertex that forms a face
    faces = set()
    for i, j in edges:
        # Find all vertices k such that (i,j,k) forms a face
        # This means k is connected to both i and j
        candidates = []
        for k in range(12):
            if k != i and k != j:
                if abs(d2_matrix[i,k] - min_d2) < 1e-10 and abs(d2_matrix[j,k] - min_d2) < 1e-10:
                    candidates.append(k)
        for k in candidates:
            # Sort to create canonical face
            face = tuple(sorted([i, j, k]))
            faces.add(face)
    
    # There should be 20 unique faces
    for face in faces:
        i, j, k = face
        center = (verts[i] + verts[j] + verts[k]) / 3.0
        center_norm = np.linalg.norm(center)
        if center_norm > 1e-10:
            center_unit = center / center_norm
            atoms.append(center_unit * r2)
    
    return np.array(atoms[:55], dtype=float)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill
    
    # Known optimal shell radii for Mackay icosahedron
    # Inner shell (12 atoms) and outer shell (42 atoms)
    # For LJ potential, optimal radii are approximately:
    # r1 ≈ 1.12 (equilibrium distance)
    # r2 ≈ 2.2 * r1 or similar
    
    # Let's try a few configurations and relax them
    best_energy = np.inf
    best_pos = None
    
    # Optimal shell radii for 55-atom Mackay icosahedron
    # Based on literature, the inner shell radius is around 1.0-1.2 and outer around 2.0-2.5
    # Let's try several combinations
    
    r1_candidates = [1.0, 1.05, 1.1, 1.12, 1.15, 1.2]
    r2_candidates = [2.0, 2.1, 2.2, 2.3, 2.4, 2.5]
    
    for r1 in r1_candidates:
        for r2 in r2_candidates:
            if time.time() > deadline:
                break
            try:
                pos = _mackay_icosahedron_55(r1, r2)
                if len(pos) != 55:
                    continue
                
                # Check minimum distance
                min_dist = np.inf
                for i in range(55):
                    for j in range(i+1, 55):
                        d = np.linalg.norm(pos[i] - pos[j])
                        if d < min_dist:
                            min_dist = d
                
                if min_dist < 0.5:
                    continue
                
                # Relax using L-BFGS
                def energy_func(x):
                    pos_flat = x.reshape(-1, 3)
                    return _lj_energy(pos_flat)
                
                result = minimize(energy_func, pos.flatten(), method='L-BFGS-B', 
                                 options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if result.success and result.fun < best_energy:
                    pos_flat = result.x.reshape(-1, 3)
                    # Verify minimum distance
                    valid = True
                    for i in range(55):
                        for j in range(i+1, 55):
                            if np.linalg.norm(pos_flat[i] - pos_flat[j]) < 0.5:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        best_energy = result.fun
                        best_pos = pos_flat.copy()
            except Exception:
                continue
    
    if best_pos is None:
        # Fallback: return cubic lattice
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(points[:N], dtype=float) * 1.12
    
    return best_pos
# EVOLVE-BLOCK-END
