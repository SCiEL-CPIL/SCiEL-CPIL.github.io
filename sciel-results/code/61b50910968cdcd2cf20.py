"""Lennard-Jones 55-atom global minimum: Corrected Mackay Icosahedron (Face Centers) + Radial Optimization + BFGS Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START
def _lj_energy_grad(pos):
    """Compute total LJ energy and gradient (vectorized, numerically stable)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 * r2 * r2
    r12 = r6 * r6

    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0

    with np.errstate(divide='ignore', invalid='ignore'):
        # Gradient of 4(r^-12 - r^-6) wrt r_i
        # dE/dr = 4(-12r^-13 + 6r^-7)
        # dE/dx_i = (dE/dr) * (x_i / r) = 4(-12/r^14 + 6/r^8) * x_i
        # Let's use the form: 24/r^8 - 48/r^14
        # factor = 4 * (-12/r^14 + 6/r^8) * (1/r) ? No.
        # Force on i from j: F_ij = -grad_i E_ij
        # grad_i (4(r^-12 - r^-6)) = 4(-12 r^-14 (x_i - x_j) + 6 r^-8 (x_i - x_j))
        # = 24 r^-8 (x_i - x_j) - 48 r^-14 (x_i - x_j)
        # = (24/r^8 - 48/r^14) * (x_i - x_j)
        
        # In our vectorized code:
        # diff = pos_i - pos_j
        # We want sum_j (24/r_ij^8 - 48/r_ij^14) * (pos_i - pos_j)
        
        # r2 = |diff|^2
        # 1/r^8 = 1/r2^4
        # 1/r^14 = 1/r2^7
        
        factor = 24.0 / (r2 ** 4) - 48.0 / (r2 ** 7)
    np.fill_diagonal(factor, 0.0)

    # grad_i = sum_j factor_ij * (pos_i - pos_j)
    # = pos_i * sum_j factor_ij - sum_j (factor_ij * pos_j)
    
    sum_factor = np.sum(factor, axis=1, keepdims=True)
    sum_factor_pos = np.sum(factor[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    grad = pos * sum_factor - sum_factor_pos

    return energy, grad

def _make_mackay_icosahedron_face_centers(r1=1.0, r2=1.7, r3=2.6):
    """
    Construct a Mackay icosahedron for 55 atoms.
    1 center + 12 vertices (shell 1) + 30 edge midpoints (shell 2) + 12 face centers (shell 3).
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms_v = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms_v
    
    # 30 edge midpoints
    # Find edges by distance
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(verts_unit[i] - verts_unit[j])
            
    min_dist = np.min(dists[dists > 1e-6])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))
                
    edge_mids = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            edge_mids.append(mid / norm)
    edge_mids = np.array(edge_mids)
    
    # 12 face centers (Dodecahedron vertices)
    # The face centers of an icosahedron are the vertices of a dodecahedron.
    # Dodecahedron vertices can be generated from (0, +/-1, +/-phi), etc.
    face_centers = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # Wait, the set of points (0, +/-1, +/-phi) is the same as the icosahedron vertices?
    # No. Icosahedron: (0, +/-1, +/-phi). Dodecahedron: (0, +/-1, +/-1/phi)?
    # Let's check the coordinates of a dodecahedron.
    # Dodecahedron vertices:
    # (±1, ±1, ±1)
    # (0, ±1/φ, ±φ)
    # (±1/φ, ±φ, 0)
    # (±φ, 0, ±1/φ)
    
    # Let's generate them properly.
    inv_phi = 1.0 / phi
    fc = []
    # (±1, ±1, ±1)
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                fc.append([x, y, z])
    # (0, ±1/φ, ±φ)
    for y in [-inv_phi, inv_phi]:
        for z in [-phi, phi]:
            fc.append([0, y, z])
    # (±1/φ, ±φ, 0)
    for x in [-inv_phi, inv_phi]:
        for y in [-phi, phi]:
            fc.append([x, y, 0])
    # (±φ, 0, ±1/φ)
    for x in [-phi, phi]:
        for z in [-inv_phi, inv_phi]:
            fc.append([x, 0, z])
            
    face_centers = np.array(fc, dtype=float)
    norms_fc = np.linalg.norm(face_centers, axis=1, keepdims=True)
    face_centers_unit = face_centers / norms_fc
    
    center = np.array([[0.0, 0.0, 0.0]])
    first_shell = verts_unit * r1
    second_inner = edge_mids * r2
    third_shell = face_centers_unit * r3

    all_pos = np.vstack([center, first_shell, second_inner, third_shell])
    return all_pos

def _relax_lbfgs(pos, deadline=None, maxiter=1000, gtol=1e-10):
    """Relax using L-BFGS-B with deadline guard."""
    best_energy, _ = _lj_energy_grad(pos)
    best_pos = pos.copy()

    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()

    x0 = pos.flatten()

    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-14}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass

    return best_pos, best_energy

def construct():
    """Construct 55-atom LJ cluster near global minimum."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Optimize radial parameters r1, r2, r3
    # Initial guess based on literature values for Mackay icosahedron
    # r1 ~ 1.0, r2 ~ 1.7, r3 ~ 2.6
    
    # We will perform a coarse grid search over r1, r2, r3 to find the best basin,
    # then relax locally.
    
    # To save time, we will sample a few promising points and relax them.
    # The global minimum is very close to the ideal Mackay geometry with slightly relaxed radii.
    
    candidates = []
    
    # Ideal Mackay radii (approximate)
    # The 55-atom cluster is often cited with radii around 1.0, 1.7, 2.6
    # Let's try a small grid around these values.
    
    r1_vals = [0.95, 1.0, 1.05]
    r2_vals = [1.65, 1.7, 1.75]
    r3_vals = [2.55, 2.6, 2.65]
    
    # Generate candidates
    for r1 in r1_vals:
        for r2 in r2_vals:
            for r3 in r3_vals:
                if time.time() > deadline - 30.0:
                    break
                cand = _make_mackay_icosahedron_face_centers(r1, r2, r3)
                candidates.append(cand)
    
    # Also add a perturbed version of the center candidate
    base_cand = _make_mackay_icosahedron_face_centers(1.0, 1.7, 2.6)
    np.random.seed(42)
    for _ in range(3):
        perturbed = base_cand + np.random.randn(55, 3) * 0.05
        candidates.append(perturbed)
        
    results = []
    
    # Relax each candidate
    for cand in candidates:
        if time.time() > deadline - 10.0:
            break
        relaxed, energy = _relax_lbfgs(cand, deadline=deadline)
        results.append((energy, relaxed))
        
    if not results:
        # Fallback if no relaxation completed
        return _make_mackay_icosahedron_face_centers(1.0, 1.7, 2.6)
        
    # Sort by energy
    results.sort(key=lambda x: x[0])
    
    # Take the best one and do a final high-precision polish
    best_energy, best_pos = results[0]
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5.0:
        polished, pol_energy = _relax_lbfgs(best_pos, deadline=deadline, maxiter=5000, gtol=1e-12)
        if pol_energy < best_energy:
            best_energy = pol_energy
            best_pos = polished
            
    # Ensure minimum distance constraint
    min_dist_found = float('inf')
    for i in range(55):
        for j in range(i + 1, 55):
            d = np.linalg.norm(best_pos[i] - best_pos[j])
            if d < min_dist_found:
                min_dist_found = d
                
    if min_dist_found < 0.5:
        scale = 0.5 / min_dist_found * 1.01
        best_pos = best_pos * scale
        
    return best_pos
# EVOLVE-BLOCK-END
