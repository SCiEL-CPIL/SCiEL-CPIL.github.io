"""Lennard-Jones 55-atom cluster: 54-parameter radial manifold optimization + L-BFGS relaxation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START

def _lj_energy_and_grad(pos):
    """Compute Lennard-Jones energy and gradient for a set of positions."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r2_safe = np.maximum(r2, 1e-12)
    
    r6 = r2_safe ** (-3)
    r12 = r6 ** 2
    
    # Energy: 4 * sum_{i<j} (r^-12 - r^-6)
    # Matrix has each pair twice, so: E = 2 * sum(r12 - r6)
    E = 2.0 * np.sum(r12 - r6)
    
    # Gradient
    # dE/dpos_i = 4 * sum_{j!=i} (-12*r^-14 + 6*r^-8) * (pos_i - pos_j)
    # In terms of r2: r^-14 = r2^(-7), r^-8 = r2^(-4)
    # coeff = 24 * (2*r2^(-7) - r2^(-4))
    coeff = 24.0 * (2.0 * r2_safe ** (-7) - r2_safe ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return E, grad

def _get_mackay_directions():
    """
    Returns unit vectors for the Mackay icosahedron shells.
    Returns:
        verts1: 12 unit vectors to icosahedron vertices (Shell 1)
        verts2: 12 unit vectors to icosahedron vertices (Shell 2 vertices)
        mids2:  30 unit vectors to edge midpoints (Shell 2 edges)
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # Icosahedron vertices (unit length)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Find edges: pairs of vertices that are nearest neighbors
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            dists[i, j] = d
            dists[j, i] = d
    
    min_dist = np.min(dists[dists > 0])
    tol = min_dist * 1.01
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if dists[i, j] < tol:
                edges.append((i, j))
    
    # Edge midpoints (30 atoms)
    edge_mids = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid /= np.linalg.norm(mid)
        edge_mids.append(mid)
    edge_mids = np.array(edge_mids)
    
    # Shell 1 uses the same vertex directions
    # Shell 2 uses the same vertex directions for the 12 vertices
    # Shell 2 uses the edge midpoints for the 30 edges
    return verts, verts, edge_mids

def _build_mackay_54r(radii):
    """
    Build the 55-atom Mackay icosahedron with 54 independent radii.
    The center atom is fixed at (0,0,0).
    radii: array of shape (54,)
    """
    verts1, verts2, mids2 = _get_mackay_directions()
    
    # Order: 12 shell1, 12 shell2_verts, 30 shell2_mids
    dirs = np.vstack([verts1, verts2, mids2])
    
    # radii corresponds to these 54 directions
    # pos = dirs * radii[:, np.newaxis]
    pos_outer = dirs * radii[:, np.newaxis]
    
    center = np.array([[0.0, 0.0, 0.0]])
    all_points = np.vstack([center, pos_outer])
    return all_points

def _energy_for_radii_54r(radii):
    """Objective function for optimizing 54 radii."""
    pos = _build_mackay_54r(radii)
    e, _ = _lj_energy_and_grad(pos)
    return e

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    best_pos = None
    best_energy = np.inf
    
    # Get directions to set up initial radii
    verts1, verts2, mids2 = _get_mackay_directions()
    
    # Initial guess based on standard Mackay radii
    # Shell 1: ~1.12, Shell 2 Verts: ~2.24, Shell 2 Mids: ~2.24
    r1_guess = 1.12
    r2v_guess = 2.24
    r2m_guess = 2.24
    
    initial_radii = np.concatenate([
        np.full(12, r1_guess),
        np.full(12, r2v_guess),
        np.full(30, r2m_guess)
    ])
    
    # 1. Optimize the 54 radii using L-BFGS-B
    # Bounds: prevent atoms from crossing the center or each other too closely
    # Min radius 0.5, Max radius 3.5
    bounds = [(0.5, 3.5)] * 54
    
    try:
        res_radii = minimize(_energy_for_radii_54r, initial_radii, method='L-BFGS-B', 
                             bounds=bounds, 
                             options={'maxiter': 5000, 'gtol': 1e-6, 'ftol': 1e-6})
        radii_opt = res_radii.x
    except Exception:
        radii_opt = initial_radii
        
    # Build the optimized radial structure
    pos_seed = _build_mackay_54r(radii_opt)
    
    # 2. Full Cartesian relaxation with L-BFGS to break any remaining radial constraints
    try:
        result = minimize(_lj_energy_and_grad, pos_seed, method='L-BFGS-B',
                         options={'maxiter': 5000, 'gtol': 1e-8, 'ftol': 1e-8})
        
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.copy()
    except Exception:
        pass
    
    # 3. If the first relaxation didn't reach a good energy, try a second polish
    # or a small perturbation if we are stuck in a local minimum near the Mackay basin.
    if best_energy > -275.0 and time.time() < deadline:
        # Small perturbation to escape potential symmetry locks
        np.random.seed(42)
        perturbed = pos_seed + np.random.randn(55, 3) * 0.01
        try:
            res = minimize(_lj_energy_and_grad, perturbed, method='L-BFGS-B',
                           options={'maxiter': 5000, 'gtol': 1e-8, 'ftol': 1e-8})
            if res.fun < best_energy:
                best_energy = res.fun
                best_pos = res.x.copy()
        except Exception:
            pass

    if best_pos is None:
        # Absolute fallback: return the radially optimized seed
        return pos_seed
        
    return best_pos

# EVOLVE-BLOCK-END
