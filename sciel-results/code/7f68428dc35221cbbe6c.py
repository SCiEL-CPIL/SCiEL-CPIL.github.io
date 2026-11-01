import numpy as np
import time
from scipy.optimize import minimize

N = 55
DEADLINE = 140.0  # Safety margin before 150s kill

def _lj_energy(pos):
    """Compute total LJ energy for N atoms (vectorized)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    return e

def _lj_energy_grad(pos):
    """Compute LJ energy and gradient (vectorized)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    # dE/dpos_i = sum_{j!=i} 4 * (-12*r_ij^-14 + 6*r_ij^-8) * (pos_i - pos_j)
    coeff = 4.0 * (-12.0 * r6 * r2 ** (-2) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return e, grad

def _min_distance(pos):
    """Compute minimum pairwise distance."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    return np.sqrt(np.min(r2))

def _get_mackay_directions():
    """Get unit vectors for Mackay icosahedron directions.
    Returns (inner_vecs, outer_vecs) where:
    - inner_vecs: 12 vectors (icosahedron vertices)
    - outer_vecs: 42 vectors (12 face centers * 2 + 30 edge centers)
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of a regular icosahedron
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find adjacent pairs (edges) - 5 neighbors for each vertex
    dots = verts_unit @ verts_unit.T
    edges = []
    for i in range(12):
        d = dots[i].copy()
        d[i] = -np.inf
        neighbor_indices = np.argsort(d)[-5:]
        for j in neighbor_indices:
            if i < j:
                edges.append((i, j))
    
    # Outer shell: 42 atoms
    # 12 atoms: 2 * v_i (face centers of icosahedron, one per face)
    # 30 atoms: v_i + v_j (edge centers, one per edge)
    outer_vecs = []
    for i in range(12):
        v = 2 * verts_unit[i]
        outer_vecs.append(v / np.linalg.norm(v))
        
    for i, j in edges:
        v = verts_unit[i] + verts_unit[j]
        outer_vecs.append(v / np.linalg.norm(v))
    
    outer_vecs = np.array(outer_vecs)
    inner_vecs = verts_unit
    
    return inner_vecs, outer_vecs

def _make_mackay_from_radii(r_inner, r_outer, inner_vecs, outer_vecs):
    """Build positions from Mackay directions and radii."""
    pos = np.zeros((1, 3))  # center atom
    pos_in = r_inner * inner_vecs
    pos_out = r_outer * outer_vecs
    positions = np.vstack([pos, pos_in, pos_out])
    return positions

def _relax(positions, deadline):
    """Relax positions using L-BFGS-B with analytical gradient."""
    def fun(x):
        coords = x.reshape(N, 3)
        return _lj_energy(coords)
    
    def jac(x):
        coords = x.reshape(N, 3)
        _, grad = _lj_energy_grad(coords)
        return grad.flatten()
    
    try:
        result = minimize(fun, positions.flatten(), jac=jac, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        return result.x.reshape(N, 3)
    except:
        return positions

def _optimize_radii(inner_vecs, outer_vecs, deadline):
    """Optimize inner and outer radii using Nelder-Mead."""
    def radius_energy(radii):
        r_inner, r_outer = radii
        if r_inner < 0.5 or r_outer < 1.0:
            return 1e10
        pos = _make_mackay_from_radii(r_inner, r_outer, inner_vecs, outer_vecs)
        if _min_distance(pos) < 0.5:
            return 1e10
        return _lj_energy(pos)
    
    # Initial guess
    r0 = np.array([1.12, 2.2])
    
    try:
        res = minimize(radius_energy, r0, method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
        r_opt = res.x
    except:
        r_opt = r0
    
    # Ensure valid radii
    r_inner, r_outer = r_opt
    if r_inner < 0.5:
        r_inner = 0.5
    if r_outer < 1.0:
        r_outer = 1.0
    
    return r_inner, r_outer

def _perturb_positions(positions, magnitude=0.1):
    """Add random perturbations to positions."""
    rng = np.random.default_rng()
    perturbed = positions + rng.normal(0, magnitude, size=positions.shape)
    return perturbed

def construct():
    start_time = time.time()
    inner_vecs, outer_vecs = _get_mackay_directions()
    
    # Optimize radii using Nelder-Mead
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 20:
        r_inner, r_outer = _optimize_radii(inner_vecs, outer_vecs, DEADLINE)
    else:
        r_inner, r_outer = 1.12, 2.2
    
    # Build initial positions
    positions = _make_mackay_from_radii(r_inner, r_outer, inner_vecs, outer_vecs)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 10:
        return positions
    
    # Final relaxation
    final_positions = _relax(positions, DEADLINE)
    
    # Check if minimum distance constraint is satisfied
    if _min_distance(final_positions) < 0.5:
        # Try to fix by scaling
        center = final_positions.mean(axis=0)
        final_positions = final_positions - center
        scale = 1.0
        while _min_distance(final_positions) < 0.5 and scale < 2.0:
            scale *= 1.01
            final_positions = final_positions * scale
        final_positions = final_positions + center
    
    return final_positions
