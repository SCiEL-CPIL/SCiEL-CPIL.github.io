"""Combined: 54D radial manifold optimization for 55-atom Mackay icosahedron,
with multi-start perturbation exploration and high-precision L-BFGS polish."""
import numpy as np
import time
from scipy.optimize import minimize

try:
    from science_chemistry import relax as _relax_science
    _HAS_RELAX = True
except ImportError:
    _HAS_RELAX = False

N = 55
DEADLINE_SECONDS = 112.0

_PHI = (1.0 + np.sqrt(5.0)) / 2.0

def _get_icosahedral_directions():
    """Return 54 unit direction vectors: 12 vertex, 30 edge-mid, 12 vertex."""
    verts = np.array([
        [0, 1, _PHI], [0, -1, _PHI], [0, 1, -_PHI], [0, -1, -_PHI],
        [1, _PHI, 0], [-1, _PHI, 0], [1, -_PHI, 0], [-1, -_PHI, 0],
        [_PHI, 0, 1], [-_PHI, 0, 1], [_PHI, 0, -1], [-_PHI, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms

    # Vectorized pairwise distances
    diff = verts_unit[:, np.newaxis, :] - verts_unit[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    edge_mask = dists < min_dist * 1.001
    np.fill_diagonal(edge_mask, False)

    # Collect edge midpoints
    edge_mids = []
    for i in range(12):
        for j in range(i + 1, 12):
            if edge_mask[i, j]:
                mid = (verts_unit[i] + verts_unit[j]) / 2.0
                mid_unit = mid / np.linalg.norm(mid)
                edge_mids.append(mid_unit)
    edge_mids = np.array(edge_mids)

    dirs = np.vstack([verts_unit, edge_mids, verts_unit])
    return dirs

_DIRS = _get_icosahedral_directions()

def _radial_to_cartesian(radii):
    return radii[:, np.newaxis] * _DIRS

def _cartesian_to_full(pos54):
    return np.vstack([np.zeros(3), pos54])

def _lj_energy(pos):
    n = len(pos)
    # Vectorized: compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2**(-3)
    r12 = r6**2
    # Sum only upper triangle (each pair counted once)
    iu = np.triu_indices(n, k=1)
    return 4.0 * np.sum(r12[iu] - r6[iu])

def _lj_energy_grad(pos):
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2**(-3)
    coeff = 4.0 * (-12.0 * r2**(-7) + 6.0 * r2**(-4))
    np.fill_diagonal(coeff, 0.0)
    # grad_i = sum_j coeff_ij * (pos_i - pos_j) * 2 (since d/dr2 * 2*pos_i)
    # Actually: dE/dr2_ij = 4*(-12*r2^-7 + 6*r2^-4) for each pair
    # dE/dpos_i = sum_j 2 * coeff_ij * (pos_i - pos_j)
    grad = 2.0 * np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return grad

def _radial_energy(radii):
    pos54 = _radial_to_cartesian(radii)
    pos_full = _cartesian_to_full(pos54)
    return _lj_energy(pos_full)

def _radial_grad(radii):
    pos54 = _radial_to_cartesian(radii)
    pos_full = _cartesian_to_full(pos54)
    cart_grad = _lj_energy_grad(pos_full)
    return np.sum(cart_grad[1:] * _DIRS, axis=1)

def _optimize_radial_manifold(r0=None, maxiter=10000):
    if r0 is None:
        # Known approximate Mackay icosahedron radii
        r_inner = 1.060
        r_outer_mid = 1.760
        r_outer_vert = 2.010
        r0 = np.concatenate([
            np.full(12, r_inner),
            np.full(30, r_outer_mid),
            np.full(12, r_outer_vert)
        ])
    lb = np.concatenate([np.full(12, 0.5), np.full(30, 0.5), np.full(12, 0.5)])
    ub = np.concatenate([np.full(12, 3.0), np.full(30, 3.5), np.full(12, 3.5)])
    result = minimize(
        _radial_energy, r0, jac=_radial_grad,
        bounds=list(zip(lb, ub)),
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'gtol': 1e-16, 'ftol': 1e-20, 'maxls': 200}
    )
    return result.x, result.fun

def _polish_cartesian(pos0, maxiter=10000, gtol=1e-15, ftol=1e-18):
    def energy_func(x):
        return _lj_energy(x.reshape(-1, 3))
    def grad_func(x):
        return _lj_energy_grad(x.reshape(-1, 3)).flatten()
    result = minimize(
        energy_func, pos0.flatten(), jac=grad_func,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol, 'maxls': 100}
    )
    return result.x.reshape(-1, 3), result.fun

def construct():
    start_time = time.time()
    best_pos = None
    best_energy = np.inf

    # Step 1: Radial manifold optimization
    try:
        radii_opt, rad_energy = _optimize_radial_manifold()
        best_pos = _cartesian_to_full(_radial_to_cartesian(radii_opt))
        best_energy = rad_energy
    except Exception:
        r0 = np.concatenate([
            np.full(12, 1.060), np.full(30, 1.760), np.full(12, 2.010)
        ])
        best_pos = _cartesian_to_full(_radial_to_cartesian(r0))
        best_energy = _lj_energy(best_pos)

    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 25:
        return best_pos

    # Step 2: Cartesian L-BFGS polish (main)
    try:
        pos_polished, e_polished = _polish_cartesian(best_pos, maxiter=20000, gtol=1e-16, ftol=1e-20)
        if e_polished < best_energy:
            best_energy = e_polished
            best_pos = pos_polished
    except Exception:
        pass

    # Step 2b: Try science_chemistry relax if available
    if _HAS_RELAX:
        try:
            pos_relaxed = _relax_science(best_pos.copy())
            e_relaxed = _lj_energy(pos_relaxed)
            if e_relaxed < best_energy:
                best_energy = e_relaxed
                best_pos = pos_relaxed
        except Exception:
            pass

    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 15:
        return best_pos

    # Step 3: Multi-start perturbation exploration
    np.random.seed(42)
    remaining = DEADLINE_SECONDS - elapsed - 10
    if remaining > 5:
        # Structured perturbations: break icosahedral symmetry in targeted ways
        n_attempts = min(8, max(1, int(remaining / 8)))
        
        # Different perturbation strategies
        strategies = []
        # Small random noise on all atoms
        strategies.append(('noise_all', 0.01, None))
        # Medium noise on outer shell vertices
        strategies.append(('noise_outer_v', 0.03, 'outer_v'))
        # Medium noise on outer shell edge-mids
        strategies.append(('noise_outer_m', 0.03, 'outer_m'))
        # Small noise on inner shell
        strategies.append(('noise_inner', 0.015, 'inner'))
        # Larger noise on random subset
        strategies.append(('noise_rand_sub', 0.05, 'rand_sub'))
        # Very small uniform radial perturbation
        strategies.append(('radial_tiny', 0.005, 'radial'))
        # Slightly larger radial perturbation
        strategies.append(('radial_med', 0.02, 'radial'))
        # Mixed: inner + one outer vertex
        strategies.append(('mixed', 0.02, 'mixed'))

        for attempt in range(n_attempts):
            if time.time() - start_time > DEADLINE_SECONDS - 6:
                break
            
            strat_name, scale, strat_type = strategies[attempt % len(strategies)]
            pos_perturbed = best_pos.copy()
            
            if strat_type is None:
                # All atoms
                pos_perturbed += np.random.normal(0, scale, (N, 3))
            elif strat_type == 'outer_v':
                # Outer vertices are indices 42-53 in full pos (12+30+12 layout)
                idx = np.arange(42, 55)
                pos_perturbed[idx] += np.random.normal(0, scale, (len(idx), 3))
            elif strat_type == 'outer_m':
                # Outer edge-mids are indices 12-41
                idx = np.arange(12, 42)
                pos_perturbed[idx] += np.random.normal(0, scale, (len(idx), 3))
            elif strat_type == 'inner':
                # Inner shell is indices 1-12
                idx = np.arange(1, 13)
                pos_perturbed[idx] += np.random.normal(0, scale, (len(idx), 3))
            elif strat_type == 'rand_sub':
                n_perturb = np.random.randint(3, 10)
                indices = np.random.choice(N, size=n_perturb, replace=False)
                pos_perturbed[indices] += np.random.normal(0, scale, (n_perturb, 3))
            elif strat_type == 'radial':
                # Perturb along radial directions
                radii = np.linalg.norm(pos_perturbed, axis=1)
                radii[0] = 0  # center
                radii += np.random.normal(0, scale, N)
                radii = np.maximum(radii, 0.1)
                if np.linalg.norm(pos_perturbed) > 1e-10:
                    dirs = pos_perturbed / np.maximum(radii[:, np.newaxis], 1e-10)
                    dirs[0] = 0
                    pos_perturbed = radii[:, np.newaxis] * dirs
            elif strat_type == 'mixed':
                idx_inner = np.arange(1, 13)
                pos_perturbed[idx_inner] += np.random.normal(0, scale * 0.5, (len(idx_inner), 3))
                # Pick one outer vertex
                ov_idx = np.random.choice(np.arange(42, 55), size=3, replace=False)
                pos_perturbed[ov_idx] += np.random.normal(0, scale, (3, 3))
            
            try:
                pos_trial, e_trial = _polish_cartesian(
                    pos_perturbed, maxiter=5000, gtol=1e-15, ftol=1e-18
                )
                if e_trial < best_energy - 1e-10:
                    best_energy = e_trial
                    best_pos = pos_trial
            except Exception:
                pass

    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 5:
        return best_pos

    # Step 4: Final ultra-precise polish
    try:
        pos_final, e_final = _polish_cartesian(best_pos, maxiter=15000, gtol=1e-16, ftol=1e-20)
        if e_final < best_energy:
            best_energy = e_final
            best_pos = pos_final
    except Exception:
        pass

    return best_pos
