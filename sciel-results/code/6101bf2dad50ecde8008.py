"""
LJ38 cluster global minimum search.

Strategy:
1. Generate diverse starting geometries from FCC truncated octahedron and icosahedral structures.
2. Fast local relaxation via L-BFGS-B with analytical gradient.
3. Basin hopping with adaptive temperature and topology-aware perturbation.
4. Low-dimensional symmetry-adapted optimization on the FCC manifold.
5. High-precision final polish.
"""
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 38
DEADLINE_MARGIN = 12.0  # seconds before 150s kill

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** 3
    r12 = r6 ** 2

    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad

def _lj_energy_only(pos):
    """Just the energy."""
    e, _ = _lj_energy_and_grad(pos)
    return e

def _generate_fcc_points(max_c=5):
    """Generate FCC lattice points using primitive vectors."""
    a1 = np.array([0.0, 1.0, 1.0])
    a2 = np.array([1.0, 0.0, 1.0])
    a3 = np.array([1.0, 1.0, 0.0])
    
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                p = i * a1 + j * a2 + k * a3
                d2 = np.dot(p, p)
                if d2 <= 30.0:
                    pts.append([p[0], p[1], p[2], d2])
    
    pts = np.array(pts)
    pts_sorted = pts[np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))]
    unique_pts = [pts_sorted[0]]
    for p in pts_sorted[1:]:
        if np.linalg.norm(p[:3] - unique_pts[-1][:3]) > 1e-10:
            unique_pts.append(p)
    return np.array(unique_pts)

def _get_fcc_truncated_octahedron():
    """Return 38 closest FCC points scaled to LJ equilibrium."""
    pts = _generate_fcc_points(max_c=5)
    pts = pts[np.argsort(pts[:, 3])]
    selected = pts[:38, :3].copy()
    selected -= np.mean(selected, axis=0)
    
    # Scale so nearest neighbor distance = 2^(1/6)
    diff = selected[:, np.newaxis, :] - selected[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        selected *= scale
    return selected

def _get_fcc_shell_structure(n_atoms=38):
    """Return base FCC positions and shell assignments."""
    pts = _generate_fcc_points(max_c=6)
    pts = pts[np.argsort(pts[:, 3])]
    selected = pts[:n_atoms, :3].copy()
    coords = selected.copy()
    
    radii = np.linalg.norm(coords, axis=1)
    unique_r = []
    for r in np.sort(radii):
        if not unique_r or abs(r - unique_r[-1]) > 0.15:
            unique_r.append(r)
    
    shell_ids = np.zeros(n_atoms, dtype=int)
    for i, r in enumerate(radii):
        for sid, ur in enumerate(unique_r):
            if abs(r - ur) < 0.15:
                shell_ids[i] = sid
                break
    
    return coords, shell_ids, np.array(unique_r)

def _make_icosahedral_structure():
    """Icosahedral starting geometry for 38 atoms."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=np.float64)
    norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico /= norms

    pts = [np.zeros(3)]
    pts.extend(ico * 1.0)
    pts.extend(ico * 1.65)
    pts.extend(ico * 2.3)
    pts.append(np.array([0.3, 0.3, 0.3]))
    pts = np.array(pts[:N], dtype=np.float64)
    pts -= np.mean(pts, axis=0)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        pts *= scale
    return pts

def _relax(pos, max_iter=500, gtol=1e-10, ftol=1e-14):
    """L-BFGS-B local relaxation."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    res = minimize(
        obj_grad,
        pos.ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": gtol, "ftol": ftol, "maxls": 50},
    )
    p_out = res.x.reshape(-1, 3)
    e_out, _ = _lj_energy_and_grad(p_out)
    return p_out, e_out

def _optimize_scale(pos):
    """1D scale optimization to minimize energy."""
    def energy_at_scale(s):
        scaled_pos = pos * s
        e, _ = _lj_energy_and_grad(scaled_pos)
        return e

    res = minimize_scalar(energy_at_scale, bounds=(0.7, 1.6), method='bounded', options={'xatol': 1e-10})
    if res.success:
        optimal_scale = res.x
        optimal_pos = pos * optimal_scale
        optimal_energy = res.fun
        return optimal_pos, optimal_energy
    else:
        return pos, energy_at_scale(1.0)

def _ensure_min_sep(pos, min_sep=0.5):
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = min_sep / min_d * 1.02
        pos = pos * scale
    return pos

def _symmetry_adapted_energy_and_grad(params):
    """Compute energy and gradient for symmetry-adapted parameters (global scale + shell breathing)."""
    n_atoms = N
    global_scale = params[0]
    shell_params = params[1:]
    
    base_coords, shell_ids, unique_r = _get_fcc_shell_structure(n_atoms)
    n_shells = len(unique_r)
    
    positions = np.zeros((n_atoms, 3))
    breath = np.ones(n_atoms)
    for i in range(n_atoms):
        s = shell_ids[i]
        if s < len(shell_params):
            breath[i] = 1.0 + shell_params[s]
        positions[i] = base_coords[i] * breath[i] * global_scale
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    
    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    pos_grad = row_sum * positions - coeff @ positions
    
    grad = np.zeros(len(params))
    for i in range(n_atoms):
        grad[0] += np.dot(pos_grad[i], base_coords[i] * breath[i])
    
    for s in range(n_shells):
        if s < len(shell_params):
            mask = (shell_ids == s)
            for i in np.where(mask)[0]:
                grad[1 + s] += np.dot(pos_grad[i], base_coords[i] * global_scale)
    
    return energy, grad

def _optimize_symmetry_adapted(initial_scale=None, maxiter=1000, deadline=None):
    """Optimize the low-dimensional symmetry-adapted parameterization."""
    base_coords, shell_ids, unique_r = _get_fcc_shell_structure(N)
    n_shells = len(unique_r)
    
    if initial_scale is None:
        initial_scale = 2.0 ** (1.0/6.0)
    
    x0 = np.array([initial_scale] + [0.0] * n_shells)
    
    def objective(x):
        if deadline is not None and time.time() > deadline:
            return 1e20, np.zeros_like(x)
        e, g = _symmetry_adapted_energy_and_grad(x)
        return e, g
    
    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'gtol': 1e-14, 'maxls': 200, 'ftol': 1e-18},
        )
        if result.success and result.fun < 1e19:
            return result.x, result.fun
    except Exception:
        pass
    
    return x0, _symmetry_adapted_energy_and_grad(x0)[0]

def _basin_hop(pos, rng, max_iters=80, T0=0.35, T1=0.02, deadline=None):
    """Basin hopping with adaptive temperature and topology-aware perturbation."""
    current_pos = pos.copy()
    current_energy, _ = _lj_energy_and_grad(current_pos)
    best_pos = current_pos.copy()
    best_energy = current_energy
    stuck_count = 0
    last_energy = current_energy

    for i in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        frac = i / max(max_iters - 1, 1)
        # Faster cooling with occasional reheating
        T_curr = T0 * (T1 / T0) ** (frac ** 0.6)
        if i > 10 and i % 20 == 0 and stuck_count > 5:
            T_curr = max(T_curr, 0.15)  # reheat
        
        # Adaptive perturbation: larger when stuck
        if stuck_count > 8:
            sigma = T_curr * 2.0
        elif stuck_count > 4:
            sigma = T_curr * 1.5
        else:
            sigma = T_curr
        
        perturbed = current_pos + rng.normal(0, sigma, size=current_pos.shape)
        
        # Occasionally add a topology-breaking perturbation (move a random atom more)
        if rng.random() < 0.15:
            idx = rng.integers(0, N)
            perturbed[idx] += rng.normal(0, 0.4, size=3)
        
        relaxed, e_relaxed = _relax(perturbed, max_iter=250, gtol=1e-9)
        e_relaxed, _ = _lj_energy_and_grad(relaxed)
        
        if e_relaxed < current_energy:
            accept = True
            stuck_count = 0
        else:
            delta_e = e_relaxed - current_energy
            accept = rng.random() < np.exp(-delta_e / max(T_curr, 1e-10))
            if not accept:
                stuck_count += 1
        
        if accept:
            current_pos = relaxed.copy()
            current_energy = e_relaxed
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
                last_energy = best_energy
    
    return best_pos, best_energy

def construct():
    t0 = time.time()
    deadline = t0 + 138.0  # Safety margin before 150s kill

    rng = np.random.default_rng(42)
    
    # Generate starting geometries
    starts = []
    
    # 1. FCC truncated octahedron
    try:
        to_start = _get_fcc_truncated_octahedron()
        starts.append(to_start)
    except Exception:
        pass
    
    # 2. Icosahedral start
    try:
        ico = _make_icosahedral_structure()
        starts.append(ico)
    except Exception:
        pass
    
    # 3. Perturbed FCC starts with varying magnitudes
    base_fcc = _get_fcc_truncated_octahedron()
    for i in range(30):
        if time.time() > deadline - 50.0:
            break
        sigma = 0.02 + 0.25 * (i / 30.0) ** 1.5
        perturbed = base_fcc + rng.normal(0, sigma, size=(N, 3))
        starts.append(perturbed)
    
    # 4. Perturbed icosahedral starts
    base_ico = _make_icosahedral_structure()
    for i in range(15):
        if time.time() > deadline - 50.0:
            break
        sigma = 0.03 + 0.2 * (i / 15.0)
        perturbed = base_ico + rng.normal(0, sigma, size=(N, 3))
        starts.append(perturbed)
    
    best_pos = None
    best_energy = np.inf
    relaxed_results = []

    # Phase 1: Local relaxation of all starts
    for start in starts:
        if time.time() > deadline - 35.0:
            break
        try:
            p, e = _relax(start, max_iter=400, gtol=1e-10)
            relaxed_results.append((e, p))
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            continue

    # Fallback
    if best_pos is None:
        best_pos = starts[0]
        best_energy, _ = _lj_energy_and_grad(best_pos)

    # Sort relaxed results
    relaxed_results.sort(key=lambda x: x[0])

    # Phase 2: Scale optimization on best
    if time.time() < deadline - 28.0:
        try:
            p_scaled, e_scaled = _optimize_scale(best_pos)
            if e_scaled < best_energy:
                best_energy = e_scaled
                best_pos = p_scaled
        except Exception:
            pass

    # Phase 3: Basin hopping from top local minima
    top_k = min(6, len(relaxed_results))
    for i in range(top_k):
        if time.time() > deadline - 18.0:
            break
        e, p = relaxed_results[i]
        try:
            p_bh, e_bh = _basin_hop(p, rng, max_iters=70, T0=0.3, T1=0.015, deadline=deadline - 15.0)
            if e_bh < best_energy:
                best_energy = e_bh
                best_pos = p_bh
        except Exception:
            continue

    # Phase 4: Symmetry-adapted refinement
    if time.time() < deadline - 10.0:
        try:
            diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
            r2 = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(r2, np.inf)
            nn = np.sqrt(np.min(r2, axis=1))
            avg_nn = np.mean(nn)
            if avg_nn > 1e-12:
                init_scale = (2.0 ** (1.0 / 6.0)) / avg_nn
            else:
                init_scale = 2.0 ** (1.0 / 6.0)
            
            params_opt, e_sym = _optimize_symmetry_adapted(initial_scale=init_scale, maxiter=1500, deadline=deadline - 8.0)
            
            if e_sym < best_energy:
                base_coords, shell_ids, unique_r = _get_fcc_shell_structure(N)
                global_scale = params_opt[0]
                shell_params = params_opt[1:]
                
                new_pos = np.zeros((N, 3))
                for i in range(N):
                    s = shell_ids[i]
                    if s < len(shell_params):
                        breath = 1.0 + shell_params[s]
                    else:
                        breath = 1.0
                    new_pos[i] = base_coords[i] * breath * global_scale
                
                e_verify, _ = _lj_energy_and_grad(new_pos)
                if e_verify < best_energy:
                    best_energy = e_verify
                    best_pos = new_pos
        except Exception:
            pass

    # Phase 5: Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            p, e = _relax(best_pos, max_iter=10000, gtol=1e-14, ftol=1e-16)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass
    
    # Phase 6: Nelder-Mead polish
    if time.time() < deadline - 4.0:
        try:
            def nm_obj(x):
                p = x.reshape(-1, 3)
                e, _ = _lj_energy_and_grad(p)
                return e
            
            res_nm = minimize(
                nm_obj,
                best_pos.ravel(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True},
            )
            if res_nm.success:
                p_nm = res_nm.x.reshape(-1, 3)
                e_nm, _ = _lj_energy_and_grad(p_nm)
                if e_nm < best_energy:
                    best_energy = e_nm
                    best_pos = p_nm
        except Exception:
            pass
    
    # Final scale optimization
    if time.time() < deadline - 2.0:
        try:
            p_scaled, e_scaled = _optimize_scale(best_pos)
            if e_scaled < best_energy:
                best_energy = e_scaled
                best_pos = p_scaled
        except Exception:
            pass

    best_pos = _ensure_min_sep(best_pos)
    return best_pos

if __name__ == "__main__":
    t0 = time.time()
    pos = construct()
    e, _ = _lj_energy_and_grad(pos)
    print(f"Energy: {e:.6f}, Time: {time.time()-t0:.2f}s")
