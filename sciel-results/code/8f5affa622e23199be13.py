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
    
    center = np.array([[0.0, 0.0, 0.0]])
    shell1_dir = verts_unit
    shell2_dir = edge_mids
    shell3_dir = verts_unit
    
    return center, shell1_dir, shell2_dir, shell3_dir

def _build_direction_manifold():
    """Build the 54 unit direction vectors for non-center atoms."""
    center, s1, s2, s3 = _get_mackay_icosahedron_coords()
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
    grad_rad = np.sum(grad_pos[1:] * dirs, axis=1)
    return e, grad_rad

def _optimize_radial_manifold(dirs, r1_init, r2_init, r3_init, maxiter=3000, gtol=1e-12):
    """Optimize all 54 individual radial distances on the icosahedral manifold."""
    x0 = np.zeros(54)
    x0[:12] = r1_init
    x0[12:42] = r2_init
    x0[42:54] = r3_init
    
    def obj_and_grad(x):
        return _radial_energy_and_grad(x, dirs)
    
    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(0.3, 5.0)] * 54,
            options={'maxiter': maxiter, 'gtol': gtol, 'ftol': 1e-14, 'maxls': 200}
        )
        return result.x, result.fun
    except Exception:
        return x0, obj_and_grad(x0)[0]

def _relax_lbfgs(pos, maxiter=5000, gtol=1e-12):
    """Relax using L-BFGS-B with analytical gradient."""
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
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 200, 'ftol': 1e-15}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return best_pos, best_energy

def _check_min_dist(pos, min_d=0.5):
    """Check if all pairwise distances are >= min_d."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_d

def construct():
    """Construct 55-atom LJ cluster using radial manifold + full relaxation + tight polish."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Build direction manifold
    dirs = _build_direction_manifold()
    center, s1, s2, s3 = _get_mackay_icosahedron_coords()
    
    # 1. Use known Mackay icosahedron radii as initial guess (more accurate than Nelder-Mead)
    # For a 2-shell Mackay icosahedron with 55 atoms:
    # Shell 1 (12 atoms): r1 ≈ 1.05
    # Shell 2 (30 atoms): r2 ≈ 1.75  
    # Shell 3 (12 atoms): r3 ≈ 1.95 (same direction as shell 1, slightly larger)
    r1_init, r2_init, r3_init = 1.05, 1.75, 1.95
    
    # Refine with Nelder-Mead for better initial radii
    if time.time() < deadline - 80.0:
        try:
            def radii_energy(radii):
                r1, r2, r3 = radii
                pos = np.vstack([center, s1 * r1, s2 * r2, s3 * r3])
                return _lj_energy(pos)
            result = minimize(radii_energy, [r1_init, r2_init, r3_init], method='Nelder-Mead',
                              options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 3000})
            r1_init, r2_init, r3_init = result.x
        except Exception:
            pass
    
    # 2. Full 54D radial manifold optimization
    if time.time() < deadline - 50.0:
        try:
            radii_opt, e_radial = _optimize_radial_manifold(dirs, r1_init, r2_init, r3_init, maxiter=5000, gtol=1e-12)
            manifold_pos = _radial_to_positions(radii_opt, dirs)
        except Exception:
            radii_opt = np.zeros(54)
            radii_opt[:12] = r1_init
            radii_opt[12:42] = r2_init
            radii_opt[42:54] = r3_init
            manifold_pos = _radial_to_positions(radii_opt, dirs)
    else:
        radii_opt = np.zeros(54)
        radii_opt[:12] = r1_init
        radii_opt[12:42] = r2_init
        radii_opt[42:54] = r3_init
        manifold_pos = _radial_to_positions(radii_opt, dirs)
    
    best_pos = manifold_pos.copy()
    best_energy = _lj_energy(manifold_pos)
    
    # 3. Full 3D L-BFGS relaxation from manifold result
    if time.time() < deadline - 25.0:
        pos_relaxed, energy_relaxed = _relax_lbfgs(manifold_pos, maxiter=8000, gtol=1e-12)
        if energy_relaxed < best_energy:
            best_energy = energy_relaxed
            best_pos = pos_relaxed
    
    # 4. Multi-scale perturbation search (diverse scales for better basin exploration)
    rng = np.random.default_rng(42)
    search_deadline = deadline - 20.0
    # Use a range of scales: small for fine-tuning, medium for basin hopping
    scales = [0.005, 0.01, 0.02, 0.03, 0.05]
    for scale in scales:
        if time.time() >= search_deadline:
            break
        # Try 2 perturbations per scale for diversity
        for _ in range(2):
            if time.time() >= search_deadline:
                break
            noise = rng.normal(0, scale, size=best_pos.shape)
            perturbed_pos = best_pos + noise
            if not _check_min_dist(perturbed_pos):
                diff = perturbed_pos[:, np.newaxis, :] - perturbed_pos[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff ** 2, axis=2))
                np.fill_diagonal(dists, np.inf)
                min_d = np.min(dists)
                if min_d > 0:
                    perturbed_pos = perturbed_pos * (0.5 / min_d * 1.01)
            
            pos_relaxed, energy_relaxed = _relax_lbfgs(perturbed_pos, maxiter=3000, gtol=1e-10)
            if energy_relaxed < best_energy:
                best_energy = energy_relaxed
                best_pos = pos_relaxed
    
    # 5. Final high-precision polish
    remaining = deadline - time.time()
    if remaining > 12.0:
        pos_polished, energy_polished = _relax_lbfgs(best_pos, maxiter=15000, gtol=1e-14)
        if energy_polished < best_energy:
            best_energy = energy_polished
            best_pos = pos_polished
    
    # 6. Ultra-tight final polish if time allows
    remaining = deadline - time.time()
    if remaining > 6.0:
        pos_ultra, energy_ultra = _relax_lbfgs(best_pos, maxiter=30000, gtol=1e-15)
        if energy_ultra < best_energy:
            best_energy = energy_ultra
            best_pos = pos_ultra
    
    # Final minimum distance check
    if not _check_min_dist(best_pos):
        diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        if min_d > 0:
            best_pos = best_pos * (0.5 / min_d * 1.01)
    
    return best_pos
# EVOLVE-BLOCK-END
