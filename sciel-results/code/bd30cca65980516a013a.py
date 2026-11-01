import numpy as np
from scipy.optimize import minimize
import time

N = 55

# Cache for icosahedral directions
_ICO_CACHE = None

def _get_ico_data():
    """Get icosahedral vertex and face-center unit vectors (cached)."""
    global _ICO_CACHE
    if _ICO_CACHE is not None:
        return _ICO_CACHE
    
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of regular icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts_unit = verts / np.linalg.norm(verts, axis=1)[:, np.newaxis]
    
    # 20 face centers: each face is a triangle of 3 mutually adjacent vertices
    # Find all triangles where all 3 edges have the same (minimum) length
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(verts[i] - verts[j])
    min_edge = np.min(dists[dists > 1e-8])
    
    face_centers = []
    seen = set()
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_edge) > 1e-6:
                continue
            for k in range(j+1, 12):
                if abs(dists[j, k] - min_edge) > 1e-6 or abs(dists[i, k] - min_edge) > 1e-6:
                    continue
                key = tuple(sorted([i, j, k]))
                if key not in seen:
                    seen.add(key)
                    fc = (verts[i] + verts[j] + verts[k]) / 3.0
                    fc_unit = fc / np.linalg.norm(fc)
                    face_centers.append(fc_unit)
    
    face_centers = np.array(face_centers)
    _ICO_CACHE = (verts_unit, face_centers)
    return _ICO_CACHE

# EVOLVE-BLOCK-START
def _lj_energy(pos):
    """Compute total LJ energy for a set of positions (vectorized)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3
    e = 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return np.sum(e) / 2.0

def _lj_energy_fast(pos):
    """Faster energy computation using upper triangle only."""
    n = pos.shape[0]
    # Use upper triangle to avoid double counting
    idx = np.triu_indices(n, k=1)
    diff = pos[idx[0]] - pos[idx[1]]
    r2 = np.sum(diff ** 2, axis=1)
    r6 = r2 ** 3
    e = 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return np.sum(e)

def _lj_energy_grad(pos):
    """Compute total LJ energy and gradient (vectorized, numerically stable)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    
    # Use r6 and r12 for stability
    r6 = r2 * r2 * r2
    r12 = r6 * r6
    
    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0
    
    # dE/dxi = sum_j 4*(-12*(xi-xj)/r^14 + 6*(xi-xj)/r^8)
    # = sum_j 4*(xi-xj)*(-12/r2^7 + 6/r2^4)
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    np.fill_diagonal(factor, 0.0)
    
    # grad_i = sum_j factor_ij * (pos_i - pos_j)
    #        = pos_i * sum_j factor_ij - sum_j factor_ij * pos_j
    sum_factor = np.sum(factor, axis=1, keepdims=True)
    sum_factor_pos = np.sum(factor[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    grad = pos * sum_factor - sum_factor_pos
    
    return energy, grad

def _lj_energy_grad_fast(pos):
    """Faster gradient using upper triangle only."""
    n = pos.shape[0]
    idx_i, idx_j = np.triu_indices(n, k=1)
    diff = pos[idx_i] - pos[idx_j]
    r2 = np.sum(diff ** 2, axis=1)
    r6 = r2 ** 3
    r12 = r6 * r6
    
    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    energy = np.sum(e_pair)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    
    grad = np.zeros_like(pos)
    np.add.at(grad, idx_i, factor[:, np.newaxis] * diff)
    np.add.at(grad, idx_j, -factor[:, np.newaxis] * diff)
    
    return energy, grad

def _get_icosahedron_vectors():
    """
    Returns the 12 unit vectors to the vertices of a regular icosahedron
    and the 30 unit vectors to the midpoints of the edges.
    """
    verts_unit, face_centers = _get_ico_data()
    
    # 30 edge midpoints
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(verts[i] - verts[j])
    
    min_dist = np.min(dists[dists > 1e-6])
    
    edge_mids = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                mid = (verts[i] + verts[j]) / 2.0
                norm_mid = np.linalg.norm(mid)
                if norm_mid > 1e-6:
                    edge_mids.append(mid / norm_mid)
                    
    edge_mids = np.array(edge_mids)
    return verts_unit, edge_mids

def _make_mackay_icosahedron(r1, r2, r3):
    """
    Construct a two-shell Mackay icosahedron for 55 atoms.
    1 center + 12 first shell (vertices) + 30 second shell (edge midpoints) + 12 third shell (vertices)
    r1: radius of inner shell (12 atoms)
    r2: radius of middle shell (30 atoms)
    r3: radius of outer shell (12 atoms)
    """
    verts_unit, edge_mids = _get_icosahedron_vectors()
    
    center = np.array([[0.0, 0.0, 0.0]])
    first_shell = verts_unit * r1
    second_shell = edge_mids * r2
    third_shell = verts_unit * r3
    
    all_pos = np.vstack([center, first_shell, second_shell, third_shell])
    return all_pos

def _make_face_center_variant(r1, r2, r3):
    """
    Alternative 55-atom structure using face centers instead of outer vertices.
    1 center + 12 vertices (r1) + 30 edge mids (r2) + 12 face centers (r3)
    """
    verts_unit, face_centers = _get_ico_data()
    _, edge_mids = _get_icosahedron_vectors()
    
    center = np.array([[0.0, 0.0, 0.0]])
    first_shell = verts_unit * r1
    second_shell = edge_mids * r2
    third_shell = face_centers * r3
    
    all_pos = np.vstack([center, first_shell, second_shell, third_shell])
    return all_pos

def _relax_lbfgs(pos, deadline=None, maxiter=1000, gtol=1e-10):
    """Relax using L-BFGS-B with deadline guard."""
    from scipy.optimize import minimize
    
    start_time = time.time()
    best_energy = _lj_energy_fast(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad_fast(pos_flat)
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
    best_energy = _lj_energy_fast(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad_fast(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    # Multi-stage polish: progressively tighter
    stages = [
        {'maxiter': 3000, 'gtol': 1e-12, 'maxls': 200, 'ftol': 1e-16},
        {'maxiter': 5000, 'gtol': 1e-14, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 10000, 'gtol': 1e-16, 'maxls': 1000, 'ftol': 1e-18},
        {'maxiter': 20000, 'gtol': 1e-18, 'maxls': 2000, 'ftol': 1e-18},
    ]
    
    for stage_opts in stages:
        if deadline is not None and time.time() > deadline - 3.0:
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

def _optimize_mackay_radii(initial_guess, deadline):
    """Optimize the radii of the Mackay icosahedron shells."""
    from scipy.optimize import minimize
    
    def objective(params):
        if time.time() > deadline - 5.0:
            return 1e10
        r1, r2, r3 = params
        if r1 < 0.5 or r2 < 0.5 or r3 < 0.5:
            return 1e10
        pos = _make_mackay_icosahedron(r1, r2, r3)
        return _lj_energy_fast(pos)
    
    try:
        result = minimize(
            objective, 
            initial_guess, 
            method='Nelder-Mead',
            options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 2000, 'maxfev': 2000}
        )
        if time.time() > deadline - 5.0:
             return initial_guess
        return result.x
    except Exception:
        return initial_guess

def construct():
    """Construct 55-atom LJ cluster near global minimum."""
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    
    # Initial guess for Mackay radii
    initial_radii = np.array([1.05, 1.75, 1.95])
    
    # Optimize the radii in the 3D symmetric manifold
    opt_radii = _optimize_mackay_radii(initial_radii, deadline)
    
    # Construct the symmetric configuration
    best_pos = _make_mackay_icosahedron(*opt_radii)
    best_energy = _lj_energy_fast(best_pos)
    
    # Relax the full 55-atom system to allow for any symmetry breaking
    # This is crucial to reach the true global minimum if it's slightly distorted
    if time.time() < deadline - 10.0:
        best_pos, best_energy = _relax_lbfgs(best_pos, deadline=deadline, maxiter=500, gtol=1e-10)
    
    # High-precision polish
    if time.time() < deadline - 3.0:
        best_pos, best_energy = _polish(best_pos, deadline=deadline)
        
    return best_pos
# EVOLVE-BLOCK-END
