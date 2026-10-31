"""LJ38 cluster global minimum search."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations


def _lj_energy(pos):
    """Compute total LJ energy for positions (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero on diagonal
    r6 = r2 ** 3
    r12 = r6 ** 2
    mask = ~np.eye(n, dtype=bool)
    return 4.0 * np.sum((r12 - r6)[mask]) / 2.0


def _lj_grad(pos):
    """Compute gradient of LJ energy."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** 3
    r12 = r6 ** 2
    # d/dr2 of 4*(r^-12 - r^-6) = 4*(-12*r^-14*2 + 6*r^-8*2) ... 
    # Actually d/dr of 4*(r^-12 - r^-6) = 4*(-12*r^-13 + 6*r^-7)
    # d/dr2 = (1/(2r)) * d/dr = 4*(-12*r^-13 + 6*r^-7)/(2r) = 4*(-6*r^-14 + 3*r^-8)
    # = 4*(-6/r^14 + 3/r^8)
    # In terms of r2: r^14 = r2^7, r^8 = r2^4
    # dE/dr2_ij = 4*(-6/r2^7 + 3/r2^4) for i != j
    # grad_i = sum_{j!=i} dE/dr2_ij * d(r2_ij)/d(pos_i) = sum_{j!=i} dE/dr2_ij * 2*(pos_i - pos_j)
    # But we need to be careful: the energy is sum over i<j, so each pair contributes once.
    # Let me use the standard formula:
    # E = sum_{i<j} 4*(r_ij^-12 - r_ij^-6)
    # dE/d(pos_i) = sum_{j != i} 4*(-12*r_ij^-14*(pos_i - pos_j)/r_ij + 6*r_ij^-8*(pos_i - pos_j)/r_ij)
    #             = sum_{j != i} 4*(-12/r_ij^13 + 6/r_ij^7) * (pos_i - pos_j)
    # In terms of r2: r_ij^13 = r2_ij^6.5, r_ij^7 = r2_ij^3.5
    # Let's just compute numerically or use the r2 formulation carefully.
    
    # Using r2: dE/dr2 for pair (i,j) where E_pair = 4*(r2^-6 - r2^-3)
    # dE_pair/dr2 = 4*(-6*r2^-7 + 3*r2^-4)
    # dE/d(pos_i) = sum_{j!=i} dE_pair/dr2 * 2*(pos_i - pos_j)
    
    dE_dr2 = 4.0 * (-6.0 / r2 ** 7 + 3.0 / r2 ** 4)
    np.fill_diagonal(dE_dr2, 0.0)
    # grad_i = 2 * sum_j dE_dr2_ij * (pos_i - pos_j)
    # = 2 * (dE_dr2 @ diff) where diff[i,j] = pos_i - pos_j
    grad = 2.0 * (dE_dr2 * diff).sum(axis=1)
    return grad


def _fcc_truncated_octahedron():
    """
    Construct an FCC truncated octahedron for 38 atoms.
    The ideal LJ38 global minimum is an FCC truncated octahedron with energy ~-173.93.
    
    FCC truncated octahedron has:
    - 1 central atom
    - 12 atoms at nearest neighbor positions (cuboctahedron vertices)
    - 14 atoms at next-nearest neighbor positions (octahedron face centers + cube face centers)
    - 11 atoms at the outer shell
    
    Actually, let me think more carefully. The truncated octahedron in FCC:
    Layer 1 (z=0): center + 6 face-centered = 7 atoms? No...
    
    Let me use the known structure. The FCC truncated octahedron for 38 atoms:
    It's like a small FCC cluster. Let me construct it from FCC lattice points.
    
    FCC lattice: points at (i,j,k) where i+j+k is even, scaled by a/2.
    
    For a truncated octahedron, we take FCC points within a certain radius.
    """
    # FCC lattice basis vectors
    # Generate FCC points: (a/2)*(i, j, k) where i, j, k are all integers and i+j+k is even
    # Or equivalently: integer lattice points where i+j+k is even, scaled by 1.
    
    # Let's generate candidate FCC points
    a = 1.0  # will scale later
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                if (i + j + k) % 2 == 0:
                    points.append((i * a, j * a, k * a))
    
    points = np.array(points)
    # Center the points
    center = points.mean(axis=0)
    points -= center
    
    # Sort by distance from center
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order]
    
    # Take the first 38
    return points[:38].copy()


def _icosahedral_structure():
    """
    Construct an icosahedral structure for 38 atoms.
    This is a local minimum but not the global minimum.
    """
    # Icosahedral cluster: 1 center + 12 first shell + 20 second shell + 5 third shell? 
    # Actually 1+12+20 = 33, need 5 more.
    # Let's just use a rough icosahedral arrangement.
    
    # Icosahedron vertices (golden ratio)
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # 12 first shell atoms
    shell1 = ico_verts * 1.0
    
    # 20 second shell atoms: at midpoints of icosahedron edges, pushed out
    # Or use the icosahedron of the dual (dodecahedron)
    # Let's place 20 atoms at positions corresponding to dodecahedron vertices
    # Dodecahedron vertices: (±1,±1,±1), (0,±1/phi,±phi), (±1/phi,±phi,0), (±phi,0,±1/phi)
    dodec_verts = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                dodec_verts.append([sx, sy, sz])
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            dodec_verts.append([0, sx / phi, sy * phi])
            dodec_verts.append([sx / phi, sy * phi, 0])
            dodec_verts.append([sx * phi, 0, sy / phi])
    
    dodec_verts = np.array(dodec_verts)
    dodec_verts = dodec_verts / np.linalg.norm(dodec_verts, axis=1, keepdims=True)
    
    shell2 = dodec_verts * 1.8
    
    # 5 more atoms for the third shell - place them at some reasonable positions
    # Let's add 5 atoms at the poles of a pentagonal arrangement
    shell3 = []
    for i in range(5):
        angle = 2 * np.pi * i / 5
        shell3.append([np.cos(angle) * 2.5, np.sin(angle) * 2.5, 0.5])
    shell3 = np.array(shell3)
    
    # Combine: 1 center + 12 + 20 + 5 = 38
    center = np.array([[0, 0, 0]])
    all_pts = np.vstack([center, shell1, shell2, shell3])
    return all_pts


def _relax(pos, max_iter=2000, gtol=1e-10, ftol=1e-12):
    """Relax positions using L-BFGS."""
    result = minimize(
        _lj_energy,
        pos,
        jac=_lj_grad,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'gtol': gtol, 'ftol': ftol}
    )
    return result.x, result.fun


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    best_pos = None
    best_energy = np.inf
    
    # Try multiple initial structures
    initial_structures = []
    
    # 1. FCC truncated octahedron
    fcc = _fcc_truncated_octahedron()
    # Scale to reasonable spacing
    fcc = fcc * 1.12
    initial_structures.append(fcc)
    
    # 2. Icosahedral structure
    ico = _icosahedral_structure()
    initial_structures.append(ico)
    
    # 3. Random perturbations of FCC
    rng = np.random.default_rng(42)
    for trial in range(5):
        fcc_pert = fcc + rng.normal(0, 0.3, size=fcc.shape)
        initial_structures.append(fcc_pert)
    
    # 4. Random perturbations of icosahedral
    for trial in range(5):
        ico_pert = ico + rng.normal(0, 0.3, size=ico.shape)
        initial_structures.append(ico_pert)
    
    # 5. Simple cubic lattice (original seed)
    side = int(np.ceil(N ** (1 / 3)))
    points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
    cubic = np.array(points[:N], dtype=float) * 1.12
    initial_structures.append(cubic)
    
    # 6. More diverse random starts
    for trial in range(3):
        random_start = rng.uniform(-3, 3, size=(N, 3))
        # Ensure minimum distance
        for _ in range(100):
            min_dist = np.inf
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(random_start[i] - random_start[j])
                    if d < min_dist:
                        min_dist = d
            if min_dist > 0.5:
                break
        initial_structures.append(random_start)
    
    # Relax each initial structure
    for idx, init_pos in enumerate(initial_structures):
        if time.time() > deadline - 15:
            break
        
        try:
            relaxed, energy = _relax(init_pos, max_iter=1000, gtol=1e-10, ftol=1e-12)
            
            # Check minimum distance constraint
            min_dist = np.inf
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(relaxed[i] - relaxed[j])
                    if d < min_dist:
                        min_dist = d
            
            if min_dist >= 0.5 and energy < best_energy:
                best_energy = energy
                best_pos = relaxed.copy()
        except Exception:
            continue
    
    # If we have time, do a high-precision polish on the best result
    if best_pos is not None and time.time() < deadline - 5:
        try:
            polished, polished_energy = _relax(
                best_pos, max_iter=5000, gtol=1e-13, ftol=1e-15
            )
            # Check minimum distance
            min_dist = np.inf
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(polished[i] - polished[j])
                    if d < min_dist:
                        min_dist = d
            if min_dist >= 0.5 and polished_energy < best_energy:
                best_pos = polished
                best_energy = polished_energy
        except Exception:
            pass
    
    if best_pos is None:
        # Fallback to cubic lattice
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(points[:N], dtype=float) * 1.12
    
    return best_pos
# EVOLVE-BLOCK-END
