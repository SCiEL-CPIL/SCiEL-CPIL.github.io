import numpy as np
import time
import sys

# Try to import science_chemistry
try:
    import science_chemistry
    HAS_RELAX = True
except ImportError:
    HAS_RELAX = False
    science_chemistry = None

from scipy.optimize import minimize

N = 55
DEADLINE_SECONDS = 140.0  # Increased from 112 to allow more work

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

    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            dists[i, j] = d
            dists[j, i] = d
    min_dist = np.min(dists[dists > 0])
    edge_mask = dists < min_dist * 1.001

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
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2**(-3)
    r12 = r6**2
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
    grad = 2.0 * np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return grad

def _polish_scipy(pos0, maxiter=20000, gtol=1e-15, ftol=1e-18):
    """Fallback high-precision L-BFGS using SciPy."""
    def energy_func(x):
        return _lj_energy(x.reshape(-1, 3))
    def grad_func(x):
        return _lj_energy_grad(x.reshape(-1, 3)).flatten()
    try:
        result = minimize(
            energy_func, pos0.flatten(), jac=grad_func,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol, 'maxls': 100}
        )
        return result.x.reshape(-1, 3), result.fun
    except Exception:
        return pos0, _lj_energy(pos0)

def _polish_external(pos0):
    """Use science_chemistry.relax if available."""
    if not HAS_RELAX:
        return pos0, _lj_energy(pos0)
    try:
        # The API is relax(positions) -> positions? Or does it return energy?
        # "science_chemistry.relax(positions) (L-BFGS local relaxation) is available"
        # Usually returns relaxed positions.
        relaxed_pos = science_chemistry.relax(pos0)
        if relaxed_pos is not None:
            e = _lj_energy(relaxed_pos)
            return relaxed_pos, e
    except Exception:
        pass
    return pos0, _lj_energy(pos0)

def _generate_mackay_seed():
    """Generate the ideal Mackay icosahedron with optimized radii."""
    # Ideal radii for LJ 55 Mackay Icosahedron
    # These are approximate values known to be close to the minimum.
    # r_inner (12 atoms), r_outer_vert (12 atoms), r_outer_edge (30 atoms)
    # Note: The order in _DIRS is [12 verts, 30 edges, 12 verts]
    # So radii array should be [r_v1]*12 + [r_e]*30 + [r_v2]*12
    
    # Start with known good values
    r_v1 = 1.054
    r_e = 1.755
    r_v2 = 2.0
    
    radii = np.concatenate([
        np.full(12, r_v1),
        np.full(30, r_e),
        np.full(12, r_v2)
    ])
    
    pos54 = _radial_to_cartesian(radii)
    pos_full = _cartesian_to_full(pos54)
    return pos_full

def construct():
    start_time = time.time()
    
    # 1. Generate initial Mackay structure
    pos_init = _generate_mackay_seed()
    e_init = _lj_energy(pos_init)
    
    best_pos = pos_init
    best_energy = e_init
    
    # 2. Try to relax using external library
    if HAS_RELAX:
        try:
            pos_relaxed, e_relaxed = _polish_external(pos_init)
            if e_relaxed < best_energy:
                best_energy = e_relaxed
                best_pos = pos_relaxed
        except Exception:
            pass
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 10:
        return best_pos
    
    # 3. If external relax wasn't used or didn't improve, use high-precision SciPy L-BFGS
    # We always do this to ensure we are at a local minimum, even if external was used, 
    # in case external was less precise or failed silently.
    # However, if external succeeded and energy is very low, we might skip to save time?
    # No, L-BFGS is cheap if already converged.
    
    try:
        pos_polished, e_polished = _polish_scipy(best_pos, maxiter=20000, gtol=1e-15, ftol=1e-18)
        if e_polished < best_energy - 1e-12:
            best_energy = e_polished
            best_pos = pos_polished
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 5:
        return best_pos
    
    # 4. Multi-start perturbation to ensure global minimum
    # The Mackay icosahedron is the global minimum, but let's try a few perturbations
    # to see if we can find something even lower (unlikely, but good practice).
    # Or just to refine the local minimum if we started slightly off.
    
    np.random.seed(42)
    remaining = DEADLINE_SECONDS - elapsed - 2
    if remaining > 5:
        n_attempts = min(5, int(remaining / 5)) # Keep it small, we are likely at global min
        for attempt in range(n_attempts):
            if time.time() - start_time > DEADLINE_SECONDS - 2:
                break
            
            # Small perturbation
            pos_perturbed = best_pos + np.random.normal(0, 0.01, best_pos.shape)
            
            # Relax
            if HAS_RELAX:
                pos_trial, e_trial = _polish_external(pos_perturbed)
            else:
                pos_trial, e_trial = _polish_scipy(pos_perturbed, maxiter=5000)
            
            if e_trial < best_energy - 1e-10:
                best_energy = e_trial
                best_pos = pos_trial
    
    # 5. Final high-precision polish
    try:
        pos_final, e_final = _polish_scipy(best_pos, maxiter=10000, gtol=1e-16, ftol=1e-18)
        if e_final < best_energy:
            best_energy = e_final
            best_pos = pos_final
    except Exception:
        pass

    return best_pos
