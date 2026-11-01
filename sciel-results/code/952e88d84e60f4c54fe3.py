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
    
    r6 = r2 * r2 * r2
    r12 = r6 * r6
    
    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = 24.0 * (r6 - 2.0) / (r2 ** 7)
    np.fill_diagonal(factor, 0.0)
    
    sum_factor = np.sum(factor, axis=1, keepdims=True)
    sum_factor_pos = np.sum(factor[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    grad = pos * sum_factor - sum_factor_pos
    
    return energy, grad

def _make_mackay_icosahedron(r1=1.05, r2=1.75, r3=1.95):
    """Construct a two-shell Mackay icosahedron for 55 atoms."""
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
                     'ftol': 1e-14, 'gtol': gtol}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
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
    
    stages = [
        {'maxiter': 2000, 'gtol': 1e-13, 'maxls': 300, 'ftol': 1e-15},
        {'maxiter': 5000, 'gtol': 1e-15, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 10000, 'gtol': 1e-16, 'maxls': 1000, 'ftol': 1e-16},
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

def _basin_hopping_step(pos, deadline, rng):
    """
    Perform one basin-hopping step: perturb, relax, accept/reject.
    Returns (new_pos, new_energy, accepted).
    """
    # Perturb with random displacement
    scale = 0.15
    perturbed = pos + rng.standard_normal((55, 3)) * scale
    
    # Relax
    relaxed, energy = _relax_lbfgs(perturbed, deadline=deadline, maxiter=200, gtol=1e-8)
    return relaxed, energy, True

def _optimize_radii(base_pos, deadline):
    """
    Optimize the three shell radii using coordinate descent.
    """
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
    
    # Index groups
    center_idx = [0]
    first_shell_idx = list(range(1, 13))
    second_inner_idx = list(range(13, 43))
    second_outer_idx = list(range(43, 55))
    
    best_r = [1.05, 1.75, 1.95]
    best_energy = _lj_energy(base_pos)
    best_pos = base_pos.copy()
    
    # Coordinate descent on radii
    for iteration in range(30):
        if time.time() > deadline - 30.0:
            break
        improved = False
        for param_idx in range(3):
            lo = best_r[param_idx] - 0.15
            hi = best_r[param_idx] + 0.15
            # Golden section search
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            a, b = lo, hi
            x1 = b - gr * (b - a)
            x2 = a + gr * (b - a)
            
            for _ in range(8):
                r1_test = best_r[0] if param_idx != 0 else x1
                r2_test = best_r[1] if param_idx != 1 else x1
                r3_test = best_r[2] if param_idx != 2 else x1
                pos1 = _make_mackay_icosahedron(r1_test, r2_test, r3_test)
                e1 = _lj_energy(pos1)
                
                r1_test = best_r[0] if param_idx != 0 else x2
                r2_test = best_r[1] if param_idx != 1 else x2
                r3_test = best_r[2] if param_idx != 2 else x2
                pos2 = _make_mackay_icosahedron(r1_test, r2_test, r3_test)
                e2 = _lj_energy(pos2)
                
                if e1 < e2:
                    b = x2
                    x2 = x1
                    e2 = e1
                    x1 = b - gr * (b - a)
                else:
                    a = x1
                    x1 = x2
                    e1 = e2
                    x2 = a + gr * (b - a)
            
            new_val = (a + b) / 2.0
            r1_f = best_r[0] if param_idx != 0 else new_val
            r2_f = best_r[1] if param_idx != 1 else new_val
            r3_f = best_r[2] if param_idx != 2 else new_val
            pos_f = _make_mackay_icosahedron(r1_f, r2_f, r3_f)
            e_f = _lj_energy(pos_f)
            
            if e_f < best_energy:
                best_energy = e_f
                best_r[param_idx] = new_val
                best_pos = pos_f
                improved = True
        
        if not improved:
            break
    
    return best_pos, best_energy

def construct():
    """Construct 55-atom LJ cluster near global minimum."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    rng = np.random.default_rng(42)
    
    # Step 1: Generate diverse initial candidates
    candidates = []
    
    radius_combos = [
        (1.05, 1.75, 1.95),
        (1.0, 1.70, 1.90),
        (1.10, 1.80, 2.00),
        (0.95, 1.65, 1.85),
        (1.15, 1.85, 2.05),
        (1.05, 1.80, 2.00),
        (1.0, 1.75, 1.95),
        (1.05, 1.87, 2.10),
        (1.00, 1.80, 2.05),
        (1.10, 1.90, 2.15),
        (1.05, 1.75, 2.10),
        (1.00, 1.70, 2.00),
    ]
    
    for r1, r2, r3 in radius_combos:
        cand = _make_mackay_icosahedron(r1, r2, r3)
        candidates.append(cand)
    
    base = _make_mackay_icosahedron()
    for scale in [0.03, 0.05, 0.10, 0.15, 0.20]:
        perturbed = base + rng.standard_normal((55, 3)) * scale
        candidates.append(perturbed)
    
    # Sort by initial energy
    cand_energies = [(c, _lj_energy(c)) for c in candidates]
    cand_energies.sort(key=lambda x: x[1])
    candidates = [c for c, _ in cand_energies]
    
    # Step 2: Relax top candidates
    best_energy = float('inf')
    best_pos = candidates[0].copy()
    
    top_n = min(8, len(candidates))
    for i in range(top_n):
        if time.time() > deadline - 60.0:
            break
        max_iter = max(200, int((deadline - time.time()) * 2))
        relaxed, energy = _relax_lbfgs(candidates[i], deadline=deadline, maxiter=max_iter, gtol=1e-10)
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
    
    # Step 3: Basin hopping from the best configuration
    if time.time() < deadline - 40.0:
        current_pos = best_pos.copy()
        current_energy = best_energy
        T0 = 2.0
        T1 = 0.05
        n_steps = 30
        
        for step in range(n_steps):
            if time.time() > deadline - 25.0:
                break
            # Annealing temperature
            T = T0 * (T1 / T0) ** (step / max(n_steps - 1, 1))
            
            new_pos, new_energy, _ = _basin_hopping_step(current_pos, deadline, rng)
            
            # Metropolis acceptance
            delta_e = new_energy - current_energy
            if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-10)):
                current_pos = new_pos
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pos = current_pos.copy()
    
    # Step 4: Optimize radii around best structure
    if time.time() < deadline - 30.0:
        # Extract approximate radii from best_pos
        norms = np.linalg.norm(best_pos, axis=1)
        # Cluster norms into 4 groups
        sorted_norms = np.sort(norms)
        # Just use the optimize_radii function which does coordinate descent
        opt_pos, opt_energy = _optimize_radii(best_pos, deadline)
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_pos = opt_pos.copy()
    
    # Step 5: Aggressive polish
    if time.time() < deadline - 15.0:
        polished, polished_energy = _polish(best_pos, deadline=deadline)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_pos = polished
    
    # Step 6: Micro-perturbations
    if time.time() < deadline - 10.0:
        for scale in [0.005, 0.01, 0.02]:
            if time.time() > deadline - 8.0:
                break
            perturbed = best_pos + rng.standard_normal((55, 3)) * scale
            relaxed, energy = _relax_lbfgs(perturbed, deadline=deadline, maxiter=500, gtol=1e-12)
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed.copy()
        if time.time() < deadline - 5.0:
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
