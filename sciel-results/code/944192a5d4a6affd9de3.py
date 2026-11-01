"""Lennard-Jones 55-atom global minimum: Mackay icosahedron seed + L-BFGS relaxation."""
import numpy as np
import time
import sys

N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _lj_energy(pos):
    """Compute total LJ energy for a set of positions (vectorized)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3
    e = 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return np.sum(e) / 2.0

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
        factor = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    np.fill_diagonal(factor, 0.0)
    
    sum_factor = np.sum(factor, axis=1, keepdims=True)
    sum_factor_pos = np.sum(factor[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    grad = pos * sum_factor - sum_factor_pos
    
    return energy, grad

def _get_mackay_icosahedron_coords():
    """
    Returns unit vectors for the 55 positions in a Mackay icosahedron.
    1 center (0,0,0)
    12 first shell (icosahedron vertices)
    30 second shell (icosahedron edge midpoints)
    12 third shell (icosahedron vertices, same direction as first shell)
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find edges of the icosahedron
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
    
    # Edge midpoints (30 atoms)
    edge_mids = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            edge_mids.append(mid / norm)
    edge_mids = np.array(edge_mids)
    
    # Construct full unit vectors
    # Order: Center, Shell1 (12), Shell2 (30), Shell3 (12)
    center = np.array([[0.0, 0.0, 0.0]])
    shell1_dir = verts_unit
    shell2_dir = edge_mids
    shell3_dir = verts_unit
    
    return center, shell1_dir, shell2_dir, shell3_dir

def _build_direction_manifold():
    """Build the 54 unit direction vectors for non-center atoms."""
    center, s1, s2, s3 = _get_mackay_icosahedron_coords()
    # 54 directions: 12 (shell1) + 30 (shell2) + 12 (shell3)
    dirs = np.vstack([s1, s2, s3])
    return dirs

def _radial_to_positions(radii, dirs):
    """Convert 54 radial distances to 54 positions (plus center at origin)."""
    pos_noncenter = dirs * radii[:, np.newaxis]
    center = np.array([[0.0, 0.0, 0.0]])
    return np.vstack([center, pos_noncenter])

def _radial_energy_and_grad(radii, dirs):
    """Energy and gradient w.r.t. radial parameters on the direction manifold."""
    pos = _radial_to_positions(radii, dirs)
    e, grad_pos = _lj_energy_grad(pos)
    # grad_radial[i] = dot(grad_pos[i+1], dirs[i])  (center is fixed at 0)
    grad_rad = np.sum(grad_pos[1:] * dirs, axis=1)
    return e, grad_rad

def _optimize_radial_manifold(dirs, r1_init, r2_init, r3_init, deadline):
    """Optimize all 54 individual radial distances on the icosahedral manifold."""
    # Initial guess: replicate shell radii
    x0 = np.zeros(54)
    x0[:12] = r1_init      # shell1
    x0[12:42] = r2_init    # shell2 (30 edge midpoints)
    x0[42:54] = r3_init    # shell3
    
    def obj_and_grad(x):
        return _radial_energy_and_grad(x, dirs)
    
    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(0.3, 5.0)] * 54,
            options={'maxiter': 2000, 'gtol': 1e-12, 'ftol': 1e-14, 'maxls': 200}
        )
        return result.x, result.fun
    except Exception:
        return x0, obj_and_grad(x0)[0]

def _relax_lbfgs(pos, deadline, maxiter=1000, gtol=1e-10):
    """Relax using L-BFGS-B with deadline guard."""
    best_energy = _lj_energy(pos)
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
    """Construct 55-atom LJ cluster using 3-shell Mackay seed + full 3D relaxation + multi-start."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (evaluator kills at 150s)
    
    # 1. Quick 3-shell radius optimization for good initial guess
    center, s1, s2, s3 = _get_mackay_icosahedron_coords()
    
    try:
        if time.time() < deadline - 120.0:
            def radii_energy(radii):
                r1, r2, r3 = radii
                pos = np.vstack([center, s1 * r1, s2 * r2, s3 * r3])
                return _lj_energy(pos)
            result = minimize(radii_energy, [1.05, 1.75, 1.95], method='Nelder-Mead',
                              options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 2000, 'maxfev': 5000})
            r1_init, r2_init, r3_init = result.x
        else:
            r1_init, r2_init, r3_init = 1.05, 1.75, 1.95
    except Exception:
        r1_init, r2_init, r3_init = 1.05, 1.75, 1.95
    
    # Build initial positions from optimized shell radii
    initial_pos = np.vstack([center, s1 * r1_init, s2 * r2_init, s3 * r3_init])
    
    best_pos = initial_pos.copy()
    best_energy = _lj_energy(initial_pos)
    
    # 2. Full 3D L-BFGS relaxation from Mackay seed
    if time.time() < deadline - 30.0:
        pos_relaxed, energy_relaxed = _relax_lbfgs(initial_pos, deadline, maxiter=5000, gtol=1e-13)
        if energy_relaxed < best_energy:
            best_energy = energy_relaxed
            best_pos = pos_relaxed
    
    # 3. Multi-start with diverse perturbations around best structure
    rng = np.random.default_rng(42)
    search_deadline = deadline - 20.0
    
    # Try multiple perturbation scales and seeds for diversity
    scales = [0.02, 0.04, 0.06, 0.08, 0.12, 0.18]
    for scale in scales:
        if time.time() >= search_deadline - 8.0:
            break
        noise = rng.normal(0, scale, size=best_pos.shape)
        perturbed = best_pos + noise
        # Enforce min distance constraint
        diff = perturbed[:, np.newaxis, :] - perturbed[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        if min_d < 0.5:
            perturbed = perturbed * (0.5 / min_d * 1.02)
        
        pos_relaxed, energy_relaxed = _relax_lbfgs(perturbed, search_deadline, maxiter=2000, gtol=1e-12)
        if energy_relaxed < best_energy:
            best_energy = energy_relaxed
            best_pos = pos_relaxed
    
    # 4. Additional multi-start from slightly different shell radii
    if time.time() < search_deadline - 5.0:
        alt_radii_sets = [
            [1.02, 1.72, 1.92],
            [1.08, 1.78, 1.98],
            [1.05, 1.70, 1.90],
        ]
        for alt_r in alt_radii_sets:
            if time.time() >= search_deadline - 5.0:
                break
            alt_pos = np.vstack([center, s1 * alt_r[0], s2 * alt_r[1], s3 * alt_r[2]])
            pos_relaxed, energy_relaxed = _relax_lbfgs(alt_pos, search_deadline, maxiter=2000, gtol=1e-12)
            if energy_relaxed < best_energy:
                best_energy = energy_relaxed
                best_pos = pos_relaxed
    
    # 5. Final high-precision polish with very tight tolerance
    if time.time() < deadline - 5.0:
        pos_polished, energy_polished = _relax_lbfgs(best_pos, deadline, maxiter=8000, gtol=1e-15)
        if energy_polished < best_energy:
            best_energy = energy_polished
            best_pos = pos_polished
    
    # Final min-distance check
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    if min_dist_found < 0.5:
        best_pos = best_pos * (0.5 / min_dist_found * 1.02)
    
    return best_pos
# EVOLVE-BLOCK-END
