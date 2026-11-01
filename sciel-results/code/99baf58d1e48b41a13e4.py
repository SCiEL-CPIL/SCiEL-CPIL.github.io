import numpy as np
import time
from scipy.optimize import minimize

N = 55
DEADLINE_SECONDS = 115.0

_PHI = (1.0 + np.sqrt(5.0)) / 2.0

def _get_icosahedral_directions():
    """Return 54 unit direction vectors: 12 vertex dirs, 30 edge-mid dirs, 12 vertex dirs again."""
    verts = np.array([
        [0, 1, _PHI], [0, -1, _PHI], [0, 1, -_PHI], [0, -1, -_PHI],
        [1, _PHI, 0], [-1, _PHI, 0], [1, -_PHI, 0], [-1, -_PHI, 0],
        [_PHI, 0, 1], [-_PHI, 0, 1], [_PHI, 0, -1], [-_PHI, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms

    # Find edges
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
    edge_mids = np.array(edge_mids)  # (30, 3)

    # Order: 12 inner-vertex dirs, 30 edge-mid dirs, 12 outer-vertex dirs
    dirs = np.vstack([verts_unit, edge_mids, verts_unit])
    return dirs  # shape (54, 3)

_DIRS = _get_icosahedral_directions()

def _make_mackay_icosahedron():
    """Construct the standard Mackay icosahedron 55-atom cluster."""
    # Standard radii for Mackay icosahedron N=55
    r1 = 1.054
    r2 = 1.755
    r3 = 2.0
    
    radii = np.concatenate([
        np.full(12, r1),
        np.full(30, r2),
        np.full(12, r3)
    ])
    pos54 = radii[:, np.newaxis] * _DIRS
    pos_full = np.vstack([np.zeros(3), pos54])
    return pos_full

def _lj_energy(pos):
    """Compute total Lennard-Jones energy for N atoms."""
    n = len(pos)
    rij = np.zeros((n, n, 3))
    for i in range(n):
        rij[i, i + 1:] = pos[i] - pos[i + 1:]
        rij[i + 1:, i] = -rij[i, i + 1:]
    r2 = np.sum(rij**2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2**(-3)
    r12 = r6**2
    energy = 4.0 * np.sum(r12 - r6)
    return energy

def _lj_energy_grad(pos):
    """Compute gradient of Lennard-Jones energy w.r.t. Cartesian coords."""
    n = len(pos)
    rij = np.zeros((n, n, 3))
    for i in range(n):
        rij[i, i + 1:] = pos[i] - pos[i + 1:]
        rij[i + 1:, i] = -rij[i, i + 1:]
    r2 = np.sum(rij**2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2**(-3)
    r12 = r6**2
    coeff = 4.0 * (-12.0 * r2**(-7) + 6.0 * r2**(-4))
    np.fill_diagonal(coeff, 0.0)
    grad = 2.0 * np.sum(coeff[:, :, np.newaxis] * rij, axis=1)
    return grad

def _is_valid(pos):
    """Check if all pairwise distances are >= 0.5."""
    n = len(pos)
    min_dist = 0.5
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < min_dist:
                return False
    return True

def _relax_lbfgs(pos0, maxiter=2000, gtol=1e-10):
    """Relax structure using L-BFGS-B."""
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
            options={'maxiter': maxiter, 'gtol': gtol, 'ftol': 1e-15, 'maxls': 50}
        )
        return result.x.reshape(-1, 3), result.fun
    except Exception:
        return pos0, _lj_energy(pos0)

def _multi_start_search():
    """Perform multi-start search from Mackay icosahedron with perturbations."""
    start_time = time.time()
    best_pos = None
    best_energy = np.inf
    
    base_pos = _make_mackay_icosahedron()
    
    # Try the un-perturbed structure first
    pos_relaxed, e_relaxed = _relax_lbfgs(base_pos)
    if _is_valid(pos_relaxed):
        best_pos = pos_relaxed
        best_energy = e_relaxed
    
    # Multi-start with Gaussian noise
    noise_scales = [0.01, 0.05, 0.1, 0.2, 0.5]
    seeds = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200]
    
    rng = np.random.default_rng(42)
    
    attempts = 0
    max_attempts = 50 # Limit total attempts to ensure speed
    
    for scale in noise_scales:
        for seed in seeds:
            if time.time() - start_time > DEADLINE_SECONDS - 10:
                break
            if attempts >= max_attempts:
                break
            
            attempts += 1
            
            # Generate perturbed structure
            pos_perturbed = base_pos.copy()
            noise = rng.normal(0, scale, pos_perturbed.shape)
            pos_perturbed[1:] += noise # Perturb only outer 54
            
            # Relax
            pos_relaxed, e_relaxed = _relax_lbfgs(pos_perturbed)
            
            if _is_valid(pos_relaxed) and e_relaxed < best_energy:
                best_energy = e_relaxed
                best_pos = pos_relaxed
        
        if time.time() - start_time > DEADLINE_SECONDS - 10:
            break

    # Final high-precision polish on the best candidate
    if best_pos is not None and time.time() - start_time < DEADLINE_SECONDS - 5:
        def energy_func(x):
            return _lj_energy(x.reshape(-1, 3))
        def grad_func(x):
            return _lj_energy_grad(x.reshape(-1, 3)).flatten()
        
        try:
            result = minimize(
                energy_func,
                best_pos.flatten(),
                jac=grad_func,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-15, 'maxls': 50}
            )
            pos_polished = result.x.reshape(-1, 3)
            e_polished = result.fun
            if _is_valid(pos_polished) and e_polished < best_energy:
                best_pos = pos_polished
                best_energy = e_polished
        except Exception:
            pass

    return best_pos, best_energy

def construct():
    start_time = time.time()
    
    try:
        best_pos, best_energy = _multi_start_search()
        if best_pos is None:
            # Fallback to base structure if something went wrong
            best_pos = _make_mackay_icosahedron()
        return best_pos
    except Exception:
        # Absolute fallback
        return _make_mackay_icosahedron()
