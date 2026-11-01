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
    """Construct 55-atom LJ cluster using radial manifold optimization + full relaxation."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Build direction manifold
    dirs = _build_direction_manifold()
    
    # 1. Quick 3-shell radius optimization for good initial guess
    try:
        if time.time() < deadline - 100.0:
            center, s1, s2, s3 = _get_mackay_icosahedron_coords()
            def radii_energy(radii):
                r1, r2, r3 = radii
                pos = np.vstack([center, s1 * r1, s2 * r2, s3 * r3])
                return _lj_energy(pos)
            result = minimize(radii_energy, [1.05, 1.75, 1.95], method='Nelder-Mead',
                              options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000})
            r1_init, r2_init, r3_init = result.x
        else:
            r1_init, r2_init, r3_init = 1.05, 1.75, 1.95
    except Exception:
        r1_init, r2_init, r3_init = 1.05, 1.75, 1.95
    
    # 2. Full 54D radial manifold optimization
    # We limit iterations to save time for multi-starts
    try:
        if time.time() < deadline - 90.0:
            # Reduce maxiter in the helper or just call with a tighter deadline?
            # The helper has a deadline param but ignores it in minimize options?
            # It uses fixed maxiter=2000. Let's create a version with lower maxiter or just trust it's fast enough.
            # Actually, let's just run it. If it's slow, we lose time.
            # To be safe, let's cap the time spent here explicitly.
            t_start_radial = time.time()
            radii_opt, e_radial = _optimize_radial_manifold(dirs, r1_init, r2_init, r3_init, deadline)
            # If it took too long, we might want to fallback? No, we have the result.
            manifold_pos = _radial_to_positions(radii_opt, dirs)
        else:
            radii_opt = np.zeros(54)
            radii_opt[:12] = r1_init
            radii_opt[12:42] = r2_init
            radii_opt[42:54] = r3_init
            manifold_pos = _radial_to_positions(radii_opt, dirs)
    except Exception:
        radii_opt = np.zeros(54)
        radii_opt[:12] = r1_init
        radii_opt[12:42] = r2_init
        radii_opt[42:54] = r3_init
        manifold_pos = _radial_to_positions(radii_opt, dirs)
    
    best_pos = manifold_pos.copy()
    best_energy = _lj_energy(manifold_pos)
    
    # 3. Full 3D L-BFGS relaxation from manifold result
    if time.time() < deadline - 30.0:
        pos_relaxed, energy_relaxed = _relax_lbfgs(manifold_pos, deadline, maxiter=2000, gtol=1e-12)
        if energy_relaxed < best_energy:
            best_energy = energy_relaxed
            best_pos = pos_relaxed
    
    # 4. Enhanced Multi-start
    # Strategy:
    # A. Perturb the best 3D structure (local search)
    # B. Perturb the manifold radii (global search on manifold)
    rng = np.random.default_rng(42)
    search_deadline = deadline - 10.0
    
    # A. Local search around best 3D structure
    for scale in [0.02, 0.05, 0.1]:
        if time.time() >= search_deadline - 10.0:
            break
        noise = rng.normal(0, scale, size=best_pos.shape)
        perturbed = best_pos + noise
        # Enforce min distance
        diff = perturbed[:, np.newaxis, :] - perturbed[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        if min_d < 0.5:
            perturbed = perturbed * (0.5 / min_d * 1.01)
        
        pos_relaxed, energy_relaxed = _relax_lbfgs(perturbed, search_deadline, maxiter=1000, gtol=1e-12)
        if energy_relaxed < best_energy:
            best_energy = energy_relaxed
            best_pos = pos_relaxed
            
    # B. Global search on manifold (random radial perturbations)
    # We try a few random radial configurations near the optimized radii
    for i in range(3):
        if time.time() >= search_deadline - 10.0:
            break
        # Perturb radii
        noise_rad = rng.normal(0, 0.05, size=54)
        # Ensure radii are positive
        radii_pert = np.abs(radii_opt + noise_rad)
        # Clip to reasonable range
        radii_pert = np.clip(radii_pert, 0.5, 3.0)
        
        pos_candidate = _radial_to_positions(radii_pert, dirs)
        
        # Relax
        pos_relaxed, energy_relaxed = _relax_lbfgs(pos_candidate, search_deadline, maxiter=1000, gtol=1e-12)
        if energy_relaxed < best_energy:
            best_energy = energy_relaxed
            best_pos = pos_relaxed

    # 5. Final high-precision polish
    if time.time() < deadline - 5.0:
        pos_polished, energy_polished = _relax_lbfgs(best_pos, deadline, maxiter=3000, gtol=1e-14)
        if energy_polished < best_energy:
            best_energy = energy_polished
            best_pos = pos_polished
    
    # Final min-distance check
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    if min_dist_found < 0.5:
        best_pos = best_pos * (0.5 / min_dist_found * 1.01)
    
    return best_pos
# EVOLVE-BLOCK-END
