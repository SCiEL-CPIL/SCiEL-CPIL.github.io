"""
LJ38 cluster global minimum search.

Strategy:
  1. Generate the FCC truncated-octahedron geometry (the known global minimum
     for LJ38, energy ≈ -173.928427).
  2. Generate several other plausible starting geometries (icosahedral shell,
     random perturbed FCC, etc.).
  3. For each starting geometry, perform L-BFGS local relaxation using
     scipy.optimize.minimize with analytic gradient.
  4. Keep the best (lowest-energy) configuration found.
  5. Use a deadline guard to ensure we return well before the 120 s limit.
"""
import time
import numpy as np
from itertools import combinations

N = 38


def _lj_energy_and_grad(pos):
    """Compute total LJ energy and gradient for a set of positions.

    pos: (N, 3) array of Cartesian coordinates.
    Returns (energy, grad) where grad has same shape as pos.
    """
    n = len(pos)
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 3)
    r2 = np.sum(diff ** 2, axis=2)  # (N, N)
    # Avoid division by zero on diagonal
    np.fill_diagonal(r2, 1.0)
    r = np.sqrt(r2)
    r6 = r2 ** 3
    r12 = r6 ** 2

    # Energy: sum over i<j of 4*(r^-12 - r^-6)
    # Using full matrix and dividing by 2
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    # Gradient: dE/dr_i = sum_{j != i} 4 * (-12*r^-13*(r_i - r_j)/r + 6*r^-7*(r_i - r_j)/r)
    #           = sum_{j != i} 4 * (-12/r^14 + 6/r^8) * (r_i - r_j)
    # Let c = 4 * (-12/r^14 + 6/r^8) = 4*(6/r^8 - 12/r^14) = 24/r^8 - 48/r^14
    # grad_i = sum_j c_ij * (r_i - r_j)
    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)  # (N, N)
    np.fill_diagonal(coeff, 0.0)
    # grad_i = sum_j coeff[i,j] * (pos[i] - pos[j])
    #        = sum_j coeff[i,j]*pos[i] - sum_j coeff[i,j]*pos[j]
    #        = pos[i] * sum_j coeff[i,j] - (coeff @ pos)[i]
    row_sum = np.sum(coeff, axis=1, keepdims=True)  # (N, 1)
    grad = row_sum * pos - coeff @ pos

    return energy, grad


def _make_fcc_truncated_octahedron():
    """
    Construct the FCC truncated octahedron for N=38.

    The FCC truncated octahedron for 38 atoms has the following structure:
    - 6 face-centered atoms (centers of the 6 square faces)
    - 12 edge atoms (on the 12 edges of the truncated octahedron)
    - 8 corner atoms (at the 8 corners)
    - 12 additional atoms on the 12 hexagonal faces? 

    Actually, let me use the known coordinates. The standard FCC truncated 
    octahedron for LJ38 can be generated from FCC lattice points within a 
    truncated octahedral cutoff.

    Alternative: use the known structure. The truncated octahedron has:
    - 24 vertices (but we only need 38 total)
    
    Let me use a different approach: generate FCC lattice points and select 
    the 38 closest to the origin that form a truncated octahedron shape.
    
    Actually, the simplest robust approach: use the known analytic coordinates.
    The FCC truncated octahedron for N=38 consists of:
    - Layer 1 (z=0): 1 center + 6 surrounding = 7 atoms? 
    
    Let me just use a well-known parametrization. The truncated octahedron 
    from FCC for 38 atoms:
    
    Using the standard construction: take all FCC lattice points (h,k,l) 
    where h+k+l is even (FCC condition) within a certain radius, and the 
    shape naturally becomes a truncated octahedron.
    
    FCC points: (i, j, k) where i+j+k is even, in units of a/2 where a is 
    the lattice constant. For LJ, the nearest-neighbor distance is a/√2 = 1.0 
    (at equilibrium), so a = √2.
    
    Let me generate FCC points and pick the 38 nearest to origin.
    """
    # FCC lattice: points (i, j, k) * (a/2) where i+j+k is even
    # Nearest neighbor distance = a/√2. For LJ sigma=1, equilibrium r_min = 2^(1/6) ≈ 1.1225
    # But the task says sigma=1, so the potential minimum is at r=2^(1/6).
    # However, the cluster geometry should be built at the optimal spacing.
    # Let's build at unit spacing first and scale later, or just build directly.
    
    # Generate FCC points in a box
    a = 1.0  # we'll scale at the end
    max_coord = 3
    fcc_points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    fcc_points.append((i * a, j * a, k * a))
    
    fcc_points = np.array(fcc_points, dtype=float)
    # Distance from origin
    dists = np.sum(fcc_points ** 2, axis=1)
    # Sort by distance
    order = np.argsort(dists)
    fcc_points = fcc_points[order]
    
    # Take the 38 closest
    pts = fcc_points[:N].copy()
    
    # Center the cluster
    pts -= np.mean(pts, axis=0)
    
    # Scale: the nearest-neighbor distance in this FCC fragment should be 
    # approximately 2^(1/6) for optimal LJ packing. Let's compute the typical 
    # nearest-neighbor distance and scale accordingly.
    # Actually, let's just return as-is and let the optimizer adjust.
    # But we want a good starting point. Let's scale so that the average 
    # nearest-neighbor distance is 2^(1/6).
    
    # Compute nearest-neighbor distances
    from scipy.spatial.distance import cdist
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    nn_dists = np.min(D, axis=1)
    avg_nn = np.mean(nn_dists)
    target_nn = 2.0 ** (1.0 / 6.0)  # ≈ 1.1225
    scale = target_nn / avg_nn if avg_nn > 0 else 1.0
    pts *= scale
    
    return pts


def _make_icosahedral_structure():
    """
    Generate an icosahedral-like structure for 38 atoms.
    Icosahedral shell: 12 vertices + 1 center + additional layers.
    For 38: 12 (outer) + 12 (middle) + 12 (inner) + 1 (center) = 37... 
    Let's do 12+12+12+2 or similar.
    
    Actually, a common icosahedral arrangement for ~38:
    - 1 center
    - 12 at radius r1 (icosahedron vertices)
    - 12 at radius r2 (edge midpoints or face centers)
    - 12 at radius r3
    - 1 extra? 
    
    Let's just make a reasonable icosahedral-like starting point.
    """
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(ico_verts, axis=1, keepdims=True)
    ico_verts /= norms
    
    # Build layers
    r1 = 1.0
    r2 = 1.8
    r3 = 2.6
    r4 = 3.4
    
    pts = []
    # Center
    pts.append([0, 0, 0])
    # Layer 1: icosahedron vertices at r1
    pts.extend(ico_verts * r1)
    # Layer 2: icosahedron vertices at r2 (slightly rotated)
    # Use the same directions but different radius
    pts.extend(ico_verts * r2)
    # Layer 3: icosahedron vertices at r3
    pts.extend(ico_verts * r3)
    # That's 1 + 12 + 12 + 12 = 37. Add one more near the center.
    pts.append([0.3, 0.3, 0.3])
    
    pts = np.array(pts[:N], dtype=float)
    # Scale
    from scipy.spatial.distance import cdist
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    nn_dists = np.min(D, axis=1)
    avg_nn = np.mean(nn_dists)
    target_nn = 2.0 ** (1.0 / 6.0)
    scale = target_nn / avg_nn if avg_nn > 0 else 1.0
    pts *= scale
    
    return pts


def _make_random_perturbed_fcc(rng):
    """Generate FCC truncated octahedron with random perturbations."""
    base = _make_fcc_truncated_octahedron()
    perturbation = 0.1 * rng.standard_normal(base.shape)
    return base + perturbation


def _relax_lbfgs(pos, max_iter=500, deadline=None):
    """
    Relax positions using L-BFGS-B with analytic gradient.
    Returns (positions, energy).
    """
    from scipy.optimize import minimize
    
    def fun_and_grad(x):
        pos = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    result = minimize(
        fun_and_grad,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-12}
    )
    
    pos_opt = result.x.reshape(-1, 3)
    e_opt, _ = _lj_energy_and_grad(pos_opt)
    
    return pos_opt, e_opt


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110 s to be safe within 120 s limit
    
    rng = np.random.default_rng(42)
    
    # Generate multiple starting geometries
    starts = []
    
    # 1. FCC truncated octahedron (best known)
    try:
        fcc = _make_fcc_truncated_octahedron()
        starts.append(fcc)
    except Exception:
        pass
    
    # 2. Icosahedral structure
    try:
        ico = _make_icosahedral_structure()
        starts.append(ico)
    except Exception:
        pass
    
    # 3. Randomly perturbed FCC (a few variants)
    for i in range(5):
        try:
            if time.time() > deadline - 30:
                break
            perturbed = _make_random_perturbed_fcc(rng)
            starts.append(perturbed)
        except Exception:
            pass
    
    # 4. Simple cubic (original seed) as fallback
    try:
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        cubic = np.array(points[:N], dtype=float) * 1.12
        starts.append(cubic)
    except Exception:
        pass
    
    best_pos = None
    best_energy = np.inf
    
    for idx, start in enumerate(starts):
        if time.time() > deadline - 20:
            break
        
        try:
            # Ensure minimum separation
            from scipy.spatial.distance import cdist
            D = cdist(start, start)
            np.fill_diagonal(D, np.inf)
            if np.min(D) < 0.5:
                # Scale up slightly
                scale = 0.5 / np.min(D) * 1.01
                start = start * scale
            
            pos_opt, e_opt = _relax_lbfgs(start, max_iter=300, deadline=deadline)
            
            # Verify minimum separation
            D = cdist(pos_opt, pos_opt)
            np.fill_diagonal(D, np.inf)
            if np.min(D) < 0.5:
                # Invalid, skip
                continue
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_pos = pos_opt.copy()
        except Exception:
            continue
    
    # If no valid result, return the FCC structure scaled
    if best_pos is None:
        try:
            best_pos = _make_fcc_truncated_octahedron()
        except Exception:
            # Last resort: simple cubic
            side = int(np.ceil(N ** (1 / 3)))
            points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
            points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
            best_pos = np.array(points[:N], dtype=float) * 1.12
    
    # Final check: ensure minimum separation
    from scipy.spatial.distance import cdist
    D = cdist(best_pos, best_pos)
    np.fill_diagonal(D, np.inf)
    if np.min(D) < 0.5:
        scale = 0.5 / np.min(D) * 1.01
        best_pos = best_pos * scale
    
    return best_pos
