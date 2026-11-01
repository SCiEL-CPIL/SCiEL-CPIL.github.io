"""Improved program: construct the 55-atom Mackay icosahedron, optimize shell radii,
then apply L-BFGS relaxation with multi-start refinement and deadline guard."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55
DEADLINE_SECONDS = 110.0

def _lj_energy(pos):
    """Compute total Lennard-Jones energy for N atoms."""
    n = len(pos)
    # Use vectorized pairwise distance computation
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    # Avoid division by zero on diagonal
    r2 = np.where(r2 < 1e-12, 1.0, r2)
    r6 = r2**(-3)
    r12 = r6**2
    energy = 4.0 * np.sum(r12 - r6) - 4.0 * n  # subtract diagonal self-terms
    return energy

def _lj_energy_grad(pos):
    """Compute gradient of Lennard-Jones energy."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    r2 = np.where(r2 < 1e-12, 1.0, r2)
    r6 = r2**(-3)
    # dE/dr2 for each pair = 4*(-12*r2^-7 + 6*r2^-4)
    coeff = 4.0 * (-12.0 * r2**(-7) + 6.0 * r2**(-4))
    np.fill_diagonal(coeff, 0.0)
    grad = 2.0 * np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return grad

def _get_icosahedral_directions():
    """Get 12 vertex directions and 30 edge-midpoint directions of regular icosahedron."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 vertices of regular icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit radius
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find edges: pairs of vertices at minimum distance
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            dists[i, j] = d
            dists[j, i] = d
    
    min_dist = np.min(dists[dists > 0])
    edge_mask = dists < min_dist * 1.001
    
    # Get edge midpoints (30 edges)
    edge_midpoints = []
    for i in range(12):
        for j in range(i+1, 12):
            if edge_mask[i, j]:
                mid = (verts_unit[i] + verts_unit[j]) / 2.0
                mid_unit = mid / np.linalg.norm(mid)
                edge_midpoints.append(mid_unit)
    
    edge_midpoints = np.array(edge_midpoints)  # shape (30, 3)
    
    return verts_unit, edge_midpoints

def _mackay_icosahedron_55(r_inner, r_outer):
    """
    Construct 55-atom Mackay icosahedron.
    - 1 atom at center
    - 12 atoms at inner shell radius r_inner (vertices of regular icosahedron)
    - 30 atoms at outer shell radius r_outer (edge midpoints)
    - 12 atoms at outer shell radius r_outer * vertex_ratio (vertex directions)
    """
    verts_unit, edge_midpoints = _get_icosahedral_directions()
    
    # Vertex-to-edge-midpoint radius ratio for Mackay icosahedron
    vertex_ratio = 2.0 / 1.755  # ≈ 1.1397
    
    points = [np.zeros(3)]  # center atom
    
    # Inner shell: 12 vertices at radius r_inner
    for v in verts_unit:
        points.append(r_inner * v)
    
    # Outer shell: 30 edge-midpoint directions at radius r_outer
    for v in edge_midpoints:
        points.append(r_outer * v)
    
    # Outer shell: 12 vertex directions at radius r_outer * vertex_ratio
    for v in verts_unit:
        points.append(r_outer * vertex_ratio * v)
    
    return np.array(points, dtype=float)

def _optimize_radii():
    """Optimize the two shell radii to minimize LJ energy."""
    def energy_from_radii(params):
        r_inner, r_outer = params
        if r_inner < 0.3 or r_outer < 0.5 or r_outer < r_inner:
            return 1e10
        pos = _mackay_icosahedron_55(r_inner, r_outer)
        return _lj_energy(pos)
    
    # Initial guess based on known Mackay geometry
    r0 = [1.054, 1.755]
    result = minimize(energy_from_radii, r0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14})
    return result.x, result.fun

def _relax(pos0, maxiter=5000):
    """Apply L-BFGS relaxation to a starting position."""
    def energy_func(x):
        return _lj_energy(x.reshape(-1, 3))
    
    def grad_func(x):
        return _lj_energy_grad(x.reshape(-1, 3)).flatten()
    
    try:
        result = minimize(
            energy_func, 
            pos0.flatten(), 
            jac=grad_func,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': 1e-12, 'ftol': 1e-16}
        )
        return result.x.reshape(-1, 3), result.fun
    except Exception:
        return pos0, _lj_energy(pos0)

def construct():
    start_time = time.time()
    
    # Step 1: Optimize shell radii
    try:
        radii, radius_energy = _optimize_radii()
    except Exception:
        radii = [1.054, 1.755]
        pos_tmp = _mackay_icosahedron_55(1.054, 1.755)
        radius_energy = _lj_energy(pos_tmp)
    
    pos0 = _mackay_icosahedron_55(radii[0], radii[1])
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 20:
        return pos0
    
    # Step 2: Primary L-BFGS relaxation
    pos_relaxed, energy_relaxed = _relax(pos0, maxiter=10000)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 10:
        return pos_relaxed
    
    # Step 3: Multi-start refinement with Gaussian perturbations
    best_pos = pos_relaxed
    best_energy = energy_relaxed
    
    np.random.seed(42)
    remaining = DEADLINE_SECONDS - elapsed - 5
    if remaining > 10:
        # Try multiple perturbation scales
        scales = [0.05, 0.1, 0.15, 0.2, 0.3]
        for scale in scales:
            if time.time() - start_time > DEADLINE_SECONDS - 5:
                break
            pos_perturbed = pos_relaxed + np.random.normal(0, scale, pos_relaxed.shape)
            
            pos_new, energy_new = _relax(pos_perturbed, maxiter=3000)
            if energy_new < best_energy:
                best_energy = energy_new
                best_pos = pos_new
    
    # Step 4: Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < DEADLINE_SECONDS - 5:
        pos_final, energy_final = _relax(best_pos, maxiter=20000)
        if energy_final < best_energy:
            best_pos = pos_final
            best_energy = energy_final
    
    return best_pos
