"""Lennard-Jones 55-atom global minimum: Parameterized Mackay icosahedron with modal optimization."""
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

def _get_icosahedron_geometry():
    """
    Returns:
        v_inner: 12 unit vectors for the first shell (vertices).
        v_edge: 30 unit vectors for the second shell (edge midpoints).
        v_outer: 12 unit vectors for the second shell (vertices, same directions as inner).
        edges: list of edge indices (i,j) for the icosahedron connectivity
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    v_inner = verts / norms
    
    # Find edges
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(v_inner[i] - v_inner[j])
    
    min_dist = np.min(dists[dists > 1e-6])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))
    
    v_edge = []
    for i, j in edges:
        mid = (v_inner[i] + v_inner[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            v_edge.append(mid / norm)
    v_edge = np.array(v_edge)
    
    # Outer vertices have same directions as inner vertices in the standard Mackay model
    v_outer = v_inner.copy()
    
    return v_inner, v_edge, v_outer, edges

V_INNER, V_EDGE, V_OUTER, _ICOSA_EDGES = _get_icosahedron_geometry()

def _construct_from_params(params):
    """
    params: [r1, r2, r3]
    r1: radius for 12 inner vertices
    r2: radius for 30 edge midpoints
    r3: radius for 12 outer vertices
    """
    r1, r2, r3 = params
    center = np.array([[0.0, 0.0, 0.0]])
    shell1 = V_INNER * r1
    shell2a = V_EDGE * r2
    shell2b = V_OUTER * r3
    
    pos = np.vstack([center, shell1, shell2a, shell2b])
    return pos

def _build_graph_laplacian(pos):
    """
    Build a graph Laplacian from the Mackay icosahedron connectivity.
    Returns the Laplacian matrix and its eigenvectors (lowest frequency modes).
    """
    n = pos.shape[0]
    
    # Build connectivity based on nearest neighbors
    # For Mackay icosahedron:
    # - Atom 0 is center
    # - Atoms 1-12 are inner shell (icosahedron vertices)
    # - Atoms 13-42 are edge midpoints (30 atoms)
    # - Atoms 43-54 are outer vertices (12 atoms, same directions as inner)
    
    # Define edges based on known Mackay structure
    edges = set()
    
    # Center to all inner shell atoms
    for i in range(1, 13):
        edges.add((0, i))
        edges.add((i, 0))
    
    # Inner shell icosahedron edges
    for i, j in _ICOSA_EDGES:
        edges.add((1 + i, 1 + j))
        edges.add((1 + j, 1 + i))
    
    # Edge midpoints connect to their two parent vertices
    # The edge midpoints are in the same order as _ICOSA_EDGES
    for idx, (i, j) in enumerate(_ICOSA_EDGES):
        edge_atom = 13 + idx  # 13-42
        v1 = 1 + i  # inner vertex index
        v2 = 1 + j  # inner vertex index
        edges.add((edge_atom, v1))
        edges.add((v1, edge_atom))
        edges.add((edge_atom, v2))
        edges.add((v2, edge_atom))
    
    # Outer vertices connect to center and their corresponding inner vertex
    for i in range(12):
        outer_atom = 43 + i
        inner_atom = 1 + i
        edges.add((outer_atom, 0))
        edges.add((0, outer_atom))
        edges.add((outer_atom, inner_atom))
        edges.add((inner_atom, outer_atom))
    
    # Also connect outer vertices to nearby edge midpoints
    # Each outer vertex is connected to the 5 edge midpoints of the pentagon face it's above
    # This is complex, so we'll use distance-based connectivity as a fallback
    # For now, just use the basic connectivity above
    
    # Build adjacency matrix
    adj = np.zeros((n, n))
    for i, j in edges:
        if i < n and j < n:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    
    # Build degree matrix
    degree = np.sum(adj, axis=1)
    D = np.diag(degree)
    
    # Graph Laplacian: L = D - A
    L = D - adj
    
    # Compute eigenvalues and eigenvectors
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except:
        # Fallback: use identity if eigendecomposition fails
        eigenvectors = np.eye(n)
        eigenvalues = np.ones(n)
    
    # Sort by eigenvalue (lowest first)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return L, eigenvalues, eigenvectors

def _project_to_modes(pos, n_modes=20):
    """
    Project atomic displacements from the reference Mackay structure onto
    the lowest-frequency vibrational modes.
    Returns modal coefficients.
    """
    # Reference structure: optimal Mackay icosahedron
    ref_params = np.array([1.04, 1.76, 1.98])
    ref_pos = _construct_from_params(ref_params)
    
    # Compute displacements
    disp = pos - ref_pos
    
    # Build Laplacian and get modes
    L, eigenvalues, eigenvectors = _build_graph_laplacian(ref_pos)
    
    # Use lowest n_modes modes (excluding the zero mode which is translation)
    # The zero eigenvalue corresponds to rigid translation
    mode_vectors = eigenvectors[:, 1:n_modes+1]  # Skip the first (zero) mode
    
    # Project displacements onto modes
    # disp is (n, 3), mode_vectors is (n, n_modes)
    # For each dimension, project
    coeffs = np.zeros((3, n_modes))
    for d in range(3):
        coeffs[d] = mode_vectors.T @ disp[:, d]
    
    return coeffs.flatten(), ref_pos, mode_vectors

def _reconstruct_from_modes(coeffs, ref_pos, mode_vectors):
    """
    Reconstruct positions from modal coefficients.
    coeffs: flattened array of shape (3 * n_modes,)
    """
    n_modes = mode_vectors.shape[1]
    coeffs = coeffs.reshape(3, n_modes)
    
    disp = np.zeros((ref_pos.shape[0], 3))
    for d in range(3):
        disp[:, d] = mode_vectors @ coeffs[d]
    
    pos = ref_pos + disp
    return pos

def _objective(params):
    pos = _construct_from_params(params)
    # Check for minimum distance violation to penalize if needed
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    if min_dist < 0.5:
        return 1e10
    return _lj_energy(pos)

def _relax_lbfgs(pos, deadline=None, maxiter=500, gtol=1e-10):
    """Relax using L-BFGS-B with deadline guard."""
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
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-14}
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
    """High-precision polish pass."""
    start_time = time.time()
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    stages = [
        {'maxiter': 1000, 'gtol': 1e-12, 'maxls': 200, 'ftol': 1e-16},
        {'maxiter': 5000, 'gtol': 1e-14, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 10000, 'gtol': 1e-16, 'maxls': 1000, 'ftol': 1e-18},
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

def _optimize_modal_coefficients(pos_initial, n_modes=20, deadline=None):
    """
    Optimize modal coefficients to find better local minima.
    This allows symmetry-breaking distortions while maintaining structural integrity.
    """
    # Project initial position onto modes
    coeffs, ref_pos, mode_vectors = _project_to_modes(pos_initial, n_modes)
    
    def modal_objective(c):
        pos = _reconstruct_from_modes(c, ref_pos, mode_vectors)
        # Check minimum distance
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        if min_dist < 0.5:
            return 1e10
        return _lj_energy(pos)
    
    # Optimize modal coefficients using Nelder-Mead
    try:
        result = minimize(
            modal_objective, 
            coeffs, 
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        pos_optimized = _reconstruct_from_modes(result.x, ref_pos, mode_vectors)
        return pos_optimized, result.fun
    except Exception:
        return pos_initial, _lj_energy(pos_initial)

def construct():
    """
    Construct 55-atom LJ cluster using:
    1. Radial parameter optimization of Mackay icosahedron
    2. Modal coefficient optimization for symmetry-breaking distortions
    3. Full 3D L-BFGS relaxation and high-precision polish
    """
    start_time = time.time()
    deadline = start_time + 145.0

    # Step 1: Optimize the 3 radial parameters (r1, r2, r3)
    starts = [
        [1.04, 1.76, 1.98],
        [1.05, 1.75, 1.95],
        [1.00, 1.70, 1.90],
        [1.10, 1.80, 2.00],
        [0.95, 1.65, 1.85],
        [1.20, 1.90, 2.10],
    ]
    
    best_params = None
    best_param_energy = float('inf')
    
    for start in starts:
        if time.time() > deadline - 30.0:
            break
        try:
            res = minimize(
                _objective, 
                start, 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.fun < best_param_energy:
                best_param_energy = res.fun
                best_params = res.x
        except Exception:
            continue
            
    if best_params is None:
        best_params = np.array([1.04, 1.76, 1.98])
        best_param_energy = _objective(best_params)
        
    # Construct the initial structure from optimized parameters
    pos_0 = _construct_from_params(best_params)
    
    # Step 2: Modal coefficient optimization for symmetry-breaking
    if time.time() < deadline - 40.0:
        pos_1, e_modal = _optimize_modal_coefficients(pos_0, n_modes=15, deadline=deadline)
        if e_modal < best_param_energy:
            pos_0 = pos_1
            best_param_energy = e_modal
    
    # Step 3: Full 3D L-BFGS relaxation
    pos_2, e_2 = _relax_lbfgs(pos_0, deadline=deadline, maxiter=500, gtol=1e-8)
    
    # Step 4: High precision polish
    pos_final, e_final = _polish(pos_2, deadline=deadline)
    
    # Verify constraints
    diff = pos_final[:, np.newaxis, :] - pos_final[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    
    if min_dist_found < 0.5:
        scale = 0.5 / min_dist_found * 1.01
        pos_final = pos_final * scale
        pos_final, e_final = _relax_lbfgs(pos_final, deadline=deadline, maxiter=200, gtol=1e-10)

    return pos_final
# EVOLVE-BLOCK-END
