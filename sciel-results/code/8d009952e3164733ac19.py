"""Lennard-Jones 55-atom global minimum: Mackay icosahedron seed + L-BFGS relaxation."""
import numpy as np
import time

N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time

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
    
    # Precompute powers for efficiency
    r4 = r2 * r2
    r6 = r4 * r2
    r12 = r6 * r6
    
    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0
    
    # factor_ij = 4 * (-12/r2^7 + 6/r2^4) = 24*(r6 - 2) / r12
    # This is cleaner: 4*(-12/r2^7 + 6/r2^4) = 4*(6*r2^3 - 12)/r2^7 = 24*(r2^3-2)/r2^7 = 24*(r6-2)/r12
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = 24.0 * (r6 - 2.0) / r12
    np.fill_diagonal(factor, 0.0)
    
    # grad_i = sum_j factor_ij * (pos_i - pos_j)
    sum_factor = np.sum(factor, axis=1, keepdims=True)
    sum_factor_pos = np.sum(factor[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    grad = pos * sum_factor - sum_factor_pos
    
    return energy, grad

def _make_mackay_icosahedron(r1=1.05, r2=1.75, r3=1.95):
    """Construct a two-shell Mackay icosahedron for 55 atoms.
    1 center + 12 first shell + 30 edge-midpoints + 12 vertex-like = 55
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find edges of the icosahedron (12 vertices, 30 edges)
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
    
    # Face centers (20 faces) - but we only need 12 more for the outer shell
    # Use vertices for the outer shell (12 atoms)
    center = np.array([[0.0, 0.0, 0.0]])
    first_shell = verts_unit * r1
    second_inner = edge_mids * r2
    second_outer = verts_unit * r3
    
    all_pos = np.vstack([center, first_shell, second_inner, second_outer])
    return all_pos

def _make_face_center_icosahedron(r1=1.05, r2=1.75, r3=1.95):
    """Alternative: use face centers for some of the second shell."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
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
    
    center = np.array([[0.0, 0.0, 0.0]])
    first_shell = verts_unit * r1
    second_inner = edge_mids * r2
    second_outer = verts_unit * r3
    
    all_pos = np.vstack([center, first_shell, second_inner, second_outer])
    return all_pos

def _relax_lbfgs(pos, deadline=None, maxiter=500, gtol=1e-10):
    """Relax using L-BFGS-B with deadline guard."""
    from scipy.optimize import minimize
    
    start_time = time.time()
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
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100,
                     'ftol': 1e-14}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    if deadline is not None and time.time() > deadline - 5.0:
        return best_pos, best_energy
    
    return best_pos, best_energy

def _polish(pos, deadline=None):
    """High-precision polish pass with tighter tolerance and more iterations."""
    from scipy.optimize import minimize
    
    start_time = time.time()
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    # Multi-stage polish: progressively tighter
    stages = [
        {'maxiter': 2000, 'gtol': 1e-12, 'maxls': 200, 'ftol': 1e-16},
        {'maxiter': 5000, 'gtol': 1e-14, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 10000, 'gtol': 1e-16, 'maxls': 1000, 'ftol': 1e-18},
        {'maxiter': 20000, 'gtol': 1e-18, 'maxls': 2000, 'ftol': 1e-18},
    ]
    
    for stage_opts in stages:
        if deadline is not None and time.time() > deadline - 5.0:
            break
        try:
            result = minimize(
                obj_and_grad, x0, jac=True, method='L-BFGS-B',
                options=stage_opts
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3)
                x0 = result.x
        except Exception:
            pass
    
    return best_pos, best_energy

def _optimize_radial_manifold(pos, deadline=None):
    """Optimize per-atom radial distances on fixed icosahedral directions.
    This is a 54D optimization (all atoms except center) that finds optimal radii
    before full 3D relaxation."""
    from scipy.optimize import minimize
    
    n = pos.shape[0]
    # Directions from center (atom 0 is center)
    dirs = pos[1:].copy()
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    dirs_unit = dirs / norms
    
    def radial_to_pos(radii):
        """Convert radii array (54,) to full positions (55,3)."""
        center = np.array([[0.0, 0.0, 0.0]])
        shell = dirs_unit * radii[:, np.newaxis]
        return np.vstack([center, shell])
    
    def radial_obj_and_grad(radii):
        pos = radial_to_pos(radii)
        e, g = _lj_energy_grad(pos)
        # g[0] is gradient wrt center (should be ~0 by symmetry)
        # g[1:] is gradient wrt each shell atom
        # dE/d(r_i) = dot(g_i, dir_i)
        grad_radial = np.sum(g[1:] * dirs_unit, axis=1)
        return e, grad_radial
    
    r0 = np.linalg.norm(pos[1:], axis=1)
    
    best_energy = _lj_energy(pos)
    best_radii = r0.copy()
    
    try:
        result = minimize(
            radial_obj_and_grad, r0, jac=True, method='L-BFGS-B',
            options={'maxiter': 2000, 'gtol': 1e-14, 'maxls': 200, 'ftol': 1e-16}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_radii = result.x
    except Exception:
        pass
    
    best_pos = radial_to_pos(best_radii)
    return best_pos, best_energy

def construct():
    """Construct 55-atom LJ cluster near global minimum."""
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    
    # Generate a focused set of Mackay variants
    candidates = []
    
    # Curated radius combinations - more diverse
    radius_combos = [
        (1.05, 1.75, 1.95),  # default Mackay
        (1.0, 1.70, 1.90),
        (1.10, 1.80, 2.00),
        (0.95, 1.65, 1.85),
        (1.15, 1.85, 2.05),
        (1.05, 1.80, 2.00),
        (1.0, 1.75, 1.95),
        (1.02, 1.72, 1.92),
        (1.08, 1.78, 1.98),
    ]
    
    for r1, r2, r3 in radius_combos:
        cand = _make_mackay_icosahedron(r1, r2, r3)
        candidates.append(cand)
    
    # Add random perturbations of the default Mackay
    np.random.seed(42)
    base = _make_mackay_icosahedron()
    for scale in [0.02, 0.03, 0.05, 0.08, 0.12, 0.18]:
        perturbed = base + np.random.randn(55, 3) * scale
        candidates.append(perturbed)
    
    # Sort candidates by initial energy
    cand_energies = [(c, _lj_energy(c)) for c in candidates]
    cand_energies.sort(key=lambda x: x[1])
    candidates = [c for c, _ in cand_energies]
    
    best_energy = float('inf')
    best_pos = candidates[0].copy()
    
    # Phase 1: Radial manifold optimization on top candidates
    # This is fast (54D) and finds good radial configurations
    top_radial_n = min(6, len(candidates))
    for i in range(top_radial_n):
        if time.time() > deadline - 100.0:
            break
        radial_pos, radial_energy = _optimize_radial_manifold(candidates[i], deadline=deadline)
        if radial_energy < best_energy:
            best_energy = radial_energy
            best_pos = radial_pos.copy()
    
    # Phase 2: Full 3D relaxation on best radial result + top candidates
    # Create a small pool for full relaxation
    relax_pool = [best_pos.copy()]
    for i in range(min(3, len(candidates))):
        relax_pool.append(candidates[i].copy())
    
    for pos in relax_pool:
        if time.time() > deadline - 50.0:
            break
        relaxed, energy = _relax_lbfgs(pos, deadline=deadline, maxiter=1000, gtol=1e-10)
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
    
    # Phase 3: Aggressive polish on the best result
    if time.time() < deadline - 25.0:
        polished, polished_energy = _polish(best_pos, deadline=deadline)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_pos = polished
    
    # Phase 4: Micro-perturbations around the best to escape shallow local min
    if time.time() < deadline - 15.0:
        np.random.seed(999)
        for scale in [0.003, 0.005, 0.01, 0.015]:
            if time.time() > deadline - 12.0:
                break
            perturbed = best_pos + np.random.randn(55, 3) * scale
            relaxed, energy = _relax_lbfgs(perturbed, deadline=deadline, maxiter=800, gtol=1e-12)
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed.copy()
        
        # Final ultra-tight polish
        if time.time() < deadline - 8.0:
            polished, polished_energy = _polish(best_pos, deadline=deadline)
            if polished_energy < best_energy:
                best_energy = polished_energy
                best_pos = polished
    
    # Final check: ensure minimum distance >= 0.5
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    
    if min_dist_found < 0.5:
        scale = 0.5 / min_dist_found * 1.01
        best_pos = best_pos * scale
    
    return best_pos
# EVOLVE-BLOCK-END
