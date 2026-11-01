"""Improved Lennard-Jones cluster global minimum, 55 atoms."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist

_DEADLINE = 145.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _DEADLINE - _elapsed()


def _lj_energy(pos):
    """Compute total LJ energy: 4 * sum(r^-12 - r^-6) over all pairs."""
    d = pdist(pos)
    d = np.maximum(d, 1e-10)
    r6 = d ** -6
    return 4.0 * np.sum(r6 ** 2 - r6)


def _lj_energy_grad(pos):
    """Vectorized LJ energy and gradient."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        r6 = r2 ** -3
        r12 = r6 ** 2
        e_pair = 4.0 * (r12 - r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0
    
    # grad_i = sum_{j!=i} 4*(-12*(pos_i-pos_j)/r^14 + 6*(pos_i-pos_j)/r^8)
    #        = sum_{j!=i} 4*(pos_i-pos_j)*(-12/r2^7 + 6/r2^4)
    #        = sum_{j!=i} 4*(pos_i-pos_j)*(6/r2^4 - 12/r2^7)
    #        = sum_{j!=i} 4*(pos_i-pos_j)*6*(1/r2^4 - 2/r2^7)
    #        = sum_{j!=i} 24*(pos_i-pos_j)*(r2^3 - 2)/r2^7
    with np.errstate(divide='ignore', invalid='ignore'):
        # factor_ij = 4 * (-12/r2^7 + 6/r2^4) for i != j
        factor = 4.0 * (-12.0 * r2 ** -7 + 6.0 * r2 ** -4)
    np.fill_diagonal(factor, 0.0)
    
    # grad_i = sum_j factor_ij * (pos_i - pos_j)
    sum_factor = np.sum(factor, axis=1, keepdims=True)
    sum_factor_pos = np.sum(factor[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    grad = pos * sum_factor - sum_factor_pos
    
    return energy, grad


def _get_mackay_directions():
    """Return 55 unit direction vectors for the Mackay icosahedron."""
    phi = (1 + np.sqrt(5)) / 2.0
    
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    D = cdist(verts, verts)
    np.fill_diagonal(D, np.inf)
    min_dist = D.min()
    
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(D[i, j] - min_dist) < 1e-10:
                edges.append((i, j))
    
    edge_dirs = []
    for i, j in edges:
        mid = verts[i] + verts[j]
        mid = mid / np.linalg.norm(mid)
        edge_dirs.append(mid)
    edge_dirs = np.array(edge_dirs)
    
    center = np.zeros((1, 3))
    directions = np.vstack([center, verts, verts, edge_dirs])
    return directions


def _radii_to_positions(radii, directions):
    """Convert 55 radii to 55x3 positions along fixed directions."""
    return radii[:, np.newaxis] * directions


def _radial_energy(radii, directions):
    """Energy as function of radii only."""
    pos = _radii_to_positions(radii, directions)
    return _lj_energy(pos)


def _check_min_dist(pos, min_d=0.5):
    """Check all pair distances >= min_d."""
    d = pdist(pos)
    return d.min() >= min_d - 1e-9


def _optimize_radial(directions, r_init, maxiter_nm=5000):
    """Optimize 55 radii using Nelder-Mead."""
    def obj(r_sub):
        r_full = np.concatenate([[0.0], r_sub])
        return _radial_energy(r_full, directions)
    
    x0 = r_init[1:].copy()
    
    result = minimize(obj, x0, method='Nelder-Mead',
                      options={'maxiter': maxiter_nm, 'xatol': 1e-10, 'fatol': 1e-12,
                               'adaptive': True})
    
    r_opt = np.concatenate([[0.0], result.x])
    return r_opt


def _relax(positions, maxiter=500, gtol=1e-12):
    """L-BFGS-B local relaxation with vectorized gradient."""
    if _remaining() < 5.0:
        return positions
    
    def fun(x):
        x = x.reshape(-1, 3)
        e, g = _lj_energy_grad(x)
        return e, g.ravel()
    
    x0 = positions.reshape(-1)
    result = minimize(fun, x0, method='L-BFGS-B', jac=True,
                      options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': gtol, 'maxls': 100})
    return result.x.reshape(-1, 3)


def _polish(positions, deadline=None):
    """High-precision polish with progressively tighter tolerances."""
    if deadline is None:
        deadline = _DEADLINE
    
    best_energy = _lj_energy(positions)
    best_pos = positions.copy()
    
    def fun(x):
        x = x.reshape(-1, 3)
        e, g = _lj_energy_grad(x)
        return e, g.ravel()
    
    x0 = positions.reshape(-1)
    
    stages = [
        {'maxiter': 2000, 'gtol': 1e-12, 'maxls': 200, 'ftol': 1e-14},
        {'maxiter': 4000, 'gtol': 1e-14, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 6000, 'gtol': 1e-16, 'maxls': 1000, 'ftol': 1e-18},
    ]
    
    for stage_opts in stages:
        if time.time() > deadline - 5.0:
            break
        try:
            result = minimize(fun, x0, jac=True, method='L-BFGS-B', options=stage_opts)
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3)
                x0 = result.x
        except Exception:
            pass
    
    return best_pos


def construct():
    global _start_time
    _start_time = time.time()
    
    directions = _get_mackay_directions()
    
    best_pos = None
    best_energy = np.inf
    
    # Reference Mackay radii (standard two-shell)
    r_mackay = np.zeros(55)
    r_mackay[1:13] = 1.12
    r_mackay[13:25] = 2.24
    r_mackay[25:55] = 2.24
    
    # === Phase 1: Radial optimization from multiple starts ===
    initial_configs = []
    
    # Standard Mackay
    initial_configs.append(r_mackay.copy())
    
    # Slightly compressed
    r_c = r_mackay.copy()
    r_c[1:13] = 1.05
    r_c[13:55] = 2.15
    initial_configs.append(r_c)
    
    # Slightly expanded
    r_e = r_mackay.copy()
    r_e[1:13] = 1.20
    r_e[13:55] = 2.35
    initial_configs.append(r_e)
    
    # Different inner/outer ratio
    r_d = r_mackay.copy()
    r_d[1:13] = 1.10
    r_d[13:25] = 2.10
    r_d[25:55] = 2.40
    initial_configs.append(r_d)
    
    # Random perturbations
    rng = np.random.default_rng(123)
    r_r = r_mackay.copy()
    r_r[1:] += rng.normal(0, 0.05, 54)
    r_r[1:] = np.maximum(r_r[1:], 0.5)
    initial_configs.append(r_r)
    
    r_r2 = r_mackay.copy()
    r_r2[1:] += rng.normal(0, 0.10, 54)
    r_r2[1:] = np.maximum(r_r2[1:], 0.5)
    initial_configs.append(r_r2)
    
    # Uniform outer shell
    r_u = r_mackay.copy()
    r_u[1:13] = 1.12
    r_u[13:55] = 2.20
    initial_configs.append(r_u)
    
    # Larger outer
    r_l = r_mackay.copy()
    r_l[1:13] = 1.15
    r_l[13:55] = 2.50
    initial_configs.append(r_l)
    
    # Smaller inner
    r_s = r_mackay.copy()
    r_s[1:13] = 1.00
    r_s[13:55] = 2.30
    initial_configs.append(r_s)
    
    for idx, r_init in enumerate(initial_configs):
        if _remaining() < 20.0:
            break
        try:
            r_opt = _optimize_radial(directions, r_init, maxiter_nm=5000)
            pos = _radii_to_positions(r_opt, directions)
            e = _lj_energy(pos)
            
            if _check_min_dist(pos) and e < best_energy:
                best_energy = e
                best_pos = pos.copy()
        except Exception:
            continue
    
    # === Phase 2: Full Cartesian relaxation from best radial result ===
    if best_pos is not None and _remaining() > 15.0:
        try:
            relaxed = _relax(best_pos, maxiter=500, gtol=1e-12)
            e = _lj_energy(relaxed)
            if _check_min_dist(relaxed) and e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
        except Exception:
            pass
    
    # === Phase 3: Cartesian multi-start from Mackay with various noise levels ===
    mackay_pos = _radii_to_positions(r_mackay, directions)
    rng2 = np.random.default_rng(99)
    
    noise_levels = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    for noise in noise_levels:
        if _remaining() < 15.0:
            break
        try:
            perturbed = mackay_pos + rng2.normal(0, noise, mackay_pos.shape)
            # Ensure minimum distance constraint
            if not _check_min_dist(perturbed):
                # Scale up slightly until constraint is met
                scale = 1.0
                for _ in range(100):
                    if _check_min_dist(perturbed * scale):
                        break
                    scale *= 1.01
                perturbed = perturbed * scale
            
            relaxed = _relax(perturbed, maxiter=300, gtol=1e-12)
            e = _lj_energy(relaxed)
            if _check_min_dist(relaxed) and e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
        except Exception:
            continue
    
    # === Phase 4: Additional starts from best radial solution with noise ===
    if best_pos is not None and _remaining() > 15.0:
        rng3 = np.random.default_rng(42)
        for noise in [0.02, 0.05, 0.10]:
            if _remaining() < 15.0:
                break
            try:
                perturbed = best_pos + rng3.normal(0, noise, best_pos.shape)
                if not _check_min_dist(perturbed):
                    scale = 1.0
                    for _ in range(100):
                        if _check_min_dist(perturbed * scale):
                            break
                        scale *= 1.01
                    perturbed = perturbed * scale
                
                relaxed = _relax(perturbed, maxiter=300, gtol=1e-12)
                e = _lj_energy(relaxed)
                if _check_min_dist(relaxed) and e < best_energy:
                    best_energy = e
                    best_pos = relaxed.copy()
            except Exception:
                continue
    
    # === Phase 5: Final high-precision polish ===
    if best_pos is not None and _remaining() > 10.0:
        try:
            polished = _polish(best_pos, deadline=_DEADLINE)
            e = _lj_energy(polished)
            if _check_min_dist(polished) and e < best_energy:
                best_energy = e
                best_pos = polished.copy()
        except Exception:
            pass
    
    if best_pos is None:
        best_pos = _radii_to_positions(r_mackay, directions)
    
    return best_pos
# EVOLVE-BLOCK-END
