"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import itertools


def _lj_energy(pos):
    """Compute total Lennard-Jones energy and check minimum distance."""
    d = pdist(pos)
    r2 = d ** 2
    r6 = r2 ** 3
    r12 = r6 ** 2
    # Energy = 4 * sum(r^-12 - r^-6)
    e = 4.0 * np.sum(r12 ** -1 - r6 ** -1)
    # Check minimum distance constraint
    min_d = np.min(d)
    if min_d < 0.5:
        return e + 1e6 * (0.5 - min_d) ** 2, False
    return e, True


def _lj_energy_grad(pos):
    """Compute total Lennard-Jones energy and gradient."""
    n = pos.shape[0]
    d = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(d ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r6 = r2 ** 3
    r12 = r6 ** 2
    # Energy: 4 * sum_{i<j} (r^-12 - r^-6)
    # Gradient: -24 * sum_{j != i} (2*r^-14 - r^-8) * (r_i - r_j)
    # More carefully: for pair (i,j), dE/dr = -24*(2*r^-14 - r^-8)
    # dE/dx_i = -24*(2*r^-14 - r^-8) * (x_i - x_j)
    
    # Vectorized gradient
    # r^-14 = r^-12 * r^-2 = r12^-1 * r2^-1
    # r^-8 = r^-6 * r^-2 = r6^-1 * r2^-1
    inv_r14 = r12 ** -1 * r2 ** -1
    inv_r8 = r6 ** -1 * r2 ** -1
    coeff = -24.0 * (2.0 * inv_r14 - inv_r8)
    # coeff has shape (n, n), zero diagonal
    np.fill_diagonal(coeff, 0.0)
    grad = -2.0 * (coeff * d).sum(axis=1)
    
    # Compute energy
    r2_upper = r2[np.triu_indices(n, k=1)]
    r6_upper = r2_upper ** 3
    r12_upper = r6_upper ** 2
    e = 4.0 * np.sum(r12_upper ** -1 - r6_upper ** -1)
    
    # Check minimum distance
    min_d = np.sqrt(np.min(r2_upper))
    if min_d < 0.5:
        e += 1e6 * (0.5 - min_d) ** 2
        return e, grad, False
    return e, grad, True


def _mckay_icosahedron_55():
    """
    Generate the Mackay icosahedron with 55 atoms (two shells).
    The Mackay icosahedron has:
    - Shell 0: 1 atom at center (r=0)
    - Shell 1: 12 atoms at vertices of icosahedron (r = 1)
    - Shell 2: 42 atoms (30 edge midpoints + 12 face centers) at r = sqrt(5)/2 ≈ 1.118
    
    For LJ, the optimal spacing between nearest neighbors is 2^(1/6) ≈ 1.1225.
    We'll scale so that the first shell is at distance 2^(1/6) from center.
    """
    # Icosahedron vertices (12 points)
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    # Standard icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit sphere
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Find edges (pairs of vertices at distance 2 from each other on unit icosahedron)
    # On a unit icosahedron, edge length is 2/phi = 2*(sqrt(5)-1)/2 = sqrt(5)-1 ≈ 1.236
    edge_len_target = 2.0 / phi  # ≈ 1.23607
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, edge_len_target, atol=1e-10):
                edges.append((i, j))
    
    # Edge midpoints (30 points)
    edge_mids = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid /= np.linalg.norm(mid)
        edge_mids.append(mid)
    edge_mids = np.array(edge_mids)
    
    # Face centers (12 points)
    # Find triangular faces
    faces = []
    for i in range(12):
        for j in range(12):
            for k in range(12):
                if i < j < k:
                    dij = np.linalg.norm(verts[i] - verts[j])
                    djk = np.linalg.norm(verts[j] - verts[k])
                    dki = np.linalg.norm(verts[k] - verts[i])
                    if (np.isclose(dij, edge_len_target, atol=1e-10) and
                        np.isclose(djk, edge_len_target, atol=1e-10) and
                        np.isclose(dki, edge_len_target, atol=1e-10)):
                        faces.append((i, j, k))
    
    face_centers = []
    for i, j, k in faces:
        center = (verts[i] + verts[j] + verts[k]) / 3.0
        center /= np.linalg.norm(center)
        face_centers.append(center)
    face_centers = np.array(face_centers)
    
    # Scale: first shell at r = 2^(1/6)
    scale = 2.0 ** (1.0/6.0)
    
    positions = [np.zeros(3)]  # center
    positions.extend(verts * scale)  # 12 vertices
    positions.extend(edge_mids * scale * np.sqrt(5)/2)  # 30 edge midpoints, r = sqrt(5)/2 * scale
    positions.extend(face_centers * scale * np.sqrt(5)/2)  # 12 face centers, r = sqrt(5)/2 * scale
    
    positions = np.array(positions)
    assert positions.shape[0] == 55, f"Expected 55 atoms, got {positions.shape[0]}"
    return positions


def _relax(pos, max_iter=2000, tol=1e-10):
    """Relax using L-BFGS with custom energy/gradient."""
    def fun(x):
        p = x.reshape(-1, 3)
        e, grad, valid = _lj_energy_grad(p)
        if not valid:
            # Penalize violations
            d = pdist(p)
            min_d = np.min(d)
            if min_d < 0.5:
                e += 1e8 * (0.5 - min_d) ** 2
        return e
    
    def jac(x):
        p = x.reshape(-1, 3)
        e, grad, valid = _lj_energy_grad(p)
        return grad.ravel()
    
    x0 = pos.ravel()
    try:
        res = minimize(fun, x0, jac=jac, method='L-BFGS-B', 
                       options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxls': 20})
        return res.x.reshape(-1, 3), res.fun
    except Exception:
        return pos, _lj_energy(pos)[0]


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Start from Mackay icosahedron
    pos = _mckay_icosahedron_55()
    
    # Verify minimum distance
    d = pdist(pos)
    min_d = np.min(d)
    if min_d < 0.5:
        # Scale up slightly
        scale = 0.5 / min_d * 1.001
        pos = pos * scale
    
    # Relax
    pos_relaxed, e_relaxed = _relax(pos)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return pos_relaxed
    
    # Try a few different initializations and keep the best
    best_pos = pos_relaxed
    best_e = e_relaxed
    
    # Try small perturbations
    rng = np.random.default_rng(42)
    for trial in range(10):
        if time.time() > deadline - 10.0:
            break
        perturbation = rng.normal(0, 0.01, size=(55, 3))
        pos_pert = pos + perturbation
        # Ensure minimum distance
        d = pdist(pos_pert)
        min_d = np.min(d)
        if min_d < 0.5:
            continue
        pos_r, e_r = _relax(pos_pert, max_iter=1000)
        if e_r < best_e:
            best_e = e_r
            best_pos = pos_r
    
    return best_pos
# EVOLVE-BLOCK-END
