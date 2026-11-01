"""Lennard-Jones 55-atom cluster: Mackay icosahedron construction + L-BFGS polish."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _lj_energy_and_grad(pos):
    """Compute total LJ energy and gradient for a set of positions."""
    pos = np.asarray(pos, dtype=float).reshape(-1, 3)
    n = pos.shape[0]
    # Compute pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    # Mask self-interactions
    np.fill_diagonal(r2, np.inf)
    r = np.sqrt(r2)
    
    # LJ energy: 4 * sum(r^-12 - r^-6) over all pairs (each pair counted once)
    r6 = r2 ** 3  # r^6
    r12 = r6 ** 2  # r^12
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    # Each pair counted twice, so divide by 2
    energy = 0.5 * np.sum(pair_energy)
    
    # Gradient: dE/dr * dr/dpos
    # dE/dr = 4 * (-12/r^13 + 6/r^7) = 24/r^7 * (1 - 2/r^6)
    # dE/dpos_i = sum_j dE/dr_ij * (pos_i - pos_j) / r_ij
    # In terms of r2: dE/dr2 = 4 * (-6/r^14 + 3/r^8) = 12/r^8 * (0.5 - 1/r^6) * ... 
    # Let's use: dE/dr = 24/r^7 - 48/r^13
    # dE/dpos_i = sum_{j != i} (dE/dr_ij) * (pos_i - pos_j) / r_ij
    # = sum_{j != i} (24/r^7 - 48/r^13) * diff_ij / r
    # = sum_{j != i} (24/r^8 - 48/r^14) * diff_ij
    # = sum_{j != i} (24/r2^4 - 48/r2^7) * diff_ij
    
    # More carefully:
    # E = 4 * sum_{i<j} (r_ij^-12 - r_ij^-6)
    # dE/dpos_i = 4 * sum_{j != i} d(r_ij^-12 - r_ij^-6)/dpos_i
    # d(r^-12)/dpos_i = -12 * r^-13 * (pos_i - pos_j)/r = -12 * r^-14 * (pos_i - pos_j)
    # d(r^-6)/dpos_i = -6 * r^-7 * (pos_i - pos_j)/r = -6 * r^-8 * (pos_i - pos_j)
    # So dE/dpos_i = 4 * sum_{j != i} (-12 * r^-14 + 6 * r^-8) * (pos_i - pos_j)
    # = 4 * sum_{j != i} (6/r^8 - 12/r^14) * (pos_i - pos_j)
    # = 24 * sum_{j != i} (1/r^8 - 2/r^14) * (pos_i - pos_j)
    
    # Using r2: 1/r^8 = 1/r2^4, 1/r^14 = 1/r2^7
    coeff = 24.0 * (1.0 / r2**4 - 2.0 / r2**7)
    # coeff is n x n, zero on diagonal
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad


def _valid_check(pos, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    pos = np.asarray(pos, dtype=float).reshape(-1, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    return min_r2 >= min_dist ** 2


def _mckay_icosahedron():
    """Construct the Mackay icosahedron with 55 atoms (1 + 12 + 42).
    
    The Mackay icosahedron for n=2 has:
    - 1 atom at center
    - 12 atoms on the first shell (vertices of icosahedron) at distance r1
    - 42 atoms on the second shell at distance r2
    
    The second shell consists of:
    - 30 atoms at face centers of the icosahedron (scaled)
    - 12 atoms at edge midpoints of the icosahedron (scaled)
    
    Actually, the standard Mackay icosahedron n=2:
    - Shell 1: 12 vertices of regular icosahedron
    - Shell 2: 42 atoms = 30 (face-center type) + 12 (edge-midpoint type)
    
    Let me use the standard construction.
    """
    # Golden ratio
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 vertices of a regular icosahedron
    # Standard vertices: (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # First shell: 12 atoms at distance r1 from center
    # Second shell: 42 atoms at distance r2 from center
    # 
    # The optimal ratio r2/r1 for Mackay icosahedron is determined by minimizing energy.
    # Known value: r2/r1 ≈ 1.7013 (for the 55-atom cluster)
    # 
    # Let's parameterize: r1 = 1.0, r2 = ratio
    # Then optimize ratio and overall scale.
    
    # Face centers: for each face of the icosahedron, the center is the average of 3 vertices
    # The icosahedron has 20 faces. We need to find them.
    
    # Edges of icosahedron: pairs of vertices that are closest
    # Compute all pairwise distances among the 12 vertices
    vdiff = verts_unit[:, np.newaxis, :] - verts_unit[np.newaxis, :, :]
    vdist = np.sqrt(np.sum(vdiff * vdiff, axis=2))
    np.fill_diagonal(vdist, np.inf)
    
    # Find edges: minimum distance
    min_dist = np.min(vdist)
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(vdist[i, j] - min_dist) < 1e-10:
                edges.append((i, j))
    
    # Find faces: triples of vertices that form triangles
    # A face is a set of 3 vertices where each pair is an edge
    edge_set = set()
    for e in edges:
        edge_set.add((e[0], e[1]))
        edge_set.add((e[1], e[0]))
    
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            if (i, j) not in edge_set:
                continue
            for k in range(j+1, 12):
                if (j, k) in edge_set and (i, k) in edge_set:
                    faces.append((i, j, k))
    
    # Should have 20 faces
    # Face centers (normalized)
    face_centers = []
    for f in faces:
        fc = (verts_unit[f[0]] + verts_unit[f[1]] + verts_unit[f[2]]) / 3.0
        fc_norm = np.linalg.norm(fc)
        face_centers.append(fc / fc_norm)
    face_centers = np.array(face_centers)
    
    # Edge midpoints (normalized)
    edge_mids = []
    for e in edges:
        em = (verts_unit[e[0]] + verts_unit[e[1]]) / 2.0
        em_norm = np.linalg.norm(em)
        edge_mids.append(em / em_norm)
    edge_mids = np.array(edge_mids)
    
    # Second shell: 30 face-center type + 12 edge-midpoint type
    # But wait, 30 + 12 = 42. Good.
    # However, each face center should appear once, and we have 20 faces.
    # Actually for the Mackay icosahedron, the second shell has:
    # - 12 atoms along the 12 edge directions (but offset)
    # - 30 atoms... hmm let me reconsider.
    
    # Standard Mackay icosahedron n=2:
    # Shell 1: 12 atoms at icosahedron vertices
    # Shell 2: 42 atoms
    #   - 12 atoms: one per icosahedron vertex, pushed outward along the vertex direction
    #   - 30 atoms: one per icosahedron edge, at the edge midpoint direction, pushed outward
    #   
    # Actually, I think the correct construction is:
    # Shell 2 has atoms at:
    # - 12 positions: along each vertex direction, at distance r2
    # - 30 positions: along each edge-midpoint direction, at distance r2
    # Total: 42
    
    # So the directions for shell 2 are:
    # - 12 vertex directions (same as shell 1 directions)
    # - 30 edge-midpoint directions
    # All at the same radius r2.
    
    # Let me verify: 12 + 30 = 42. Yes.
    
    # So shell 2 directions:
    shell2_dirs = np.vstack([verts_unit, edge_mids])  # 12 + 30 = 42 directions
    
    # Now construct the full structure with parameters r1, r2
    def build(r1, r2):
        center = np.array([[0.0, 0.0, 0.0]])
        shell1 = verts_unit * r1
        shell2 = shell2_dirs * r2
        return np.vstack([center, shell1, shell2])
    
    # Optimize r1 and r2 to minimize energy
    # Use a simple 2D optimization
    def energy_func(params):
        r1, r2 = params
        if r1 < 0.3 or r2 < 0.3:
            return 1e10
        pos = build(r1, r2)
        # Check minimum distance
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2mat = np.sum(diff * diff, axis=2)
        np.fill_diagonal(r2mat, np.inf)
        if np.min(r2mat) < 0.25:  # 0.5^2
            return 1e10
        e, _ = _lj_energy_and_grad(pos)
        return e
    
    # Initial guess: r1 = 1.0, r2 = 1.7
    x0 = np.array([1.0, 1.7])
    res = minimize(energy_func, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
    
    r1_opt, r2_opt = res.x
    pos = build(r1_opt, r2_opt)
    
    return pos


def construct():
    """Construct the 55-atom LJ cluster near the Mackay icosahedron minimum."""
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin before 120s limit
    
    # Step 1: Build Mackay icosahedron
    pos = _mckay_icosahedron()
    
    # Step 2: L-BFGS relaxation
    def obj_and_grad(x):
        pos = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos)
        return e, g.reshape(-1)
    
    x0 = pos.reshape(-1)
    
    # Check time
    if time.time() > deadline - 30:
        # Not enough time for relaxation, return current
        return pos
    
    # Run L-BFGS
    try:
        result = minimize(
            obj_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 500,
                'ftol': 1e-14,
                'gtol': 1e-10,
                'maxls': 50
            }
        )
        pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        return pos
    
    # Step 3: Final high-precision polish
    try:
        result2 = minimize(
            obj_and_grad,
            pos.reshape(-1),
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 200,
                'ftol': 1e-16,
                'gtol': 1e-12,
                'maxls': 30
            }
        )
        pos = result2.x.reshape(-1, 3)
    except Exception:
        pass
    
    # Validate
    if not _valid_check(pos, min_dist=0.5):
        # Fallback: return the Mackay structure
        pos = _mckay_icosahedron()
    
    return pos

# EVOLVE-BLOCK-END
