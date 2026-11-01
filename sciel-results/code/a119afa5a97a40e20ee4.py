"""Lennard-Jones 55-atom cluster: Flexible Radial Mackay Icosahedron Optimization."""
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

def _get_mackay_unit_vectors():
    """
    Returns 55 unit vectors corresponding to the Mackay icosahedron geometry.
    Index 0: Center (0,0,0) - direction is irrelevant for radius optimization, but we store a dummy.
    Indices 1-12: Shell 1 vertices.
    Indices 13-24: Shell 2 vertices.
    Indices 25-54: Shell 2 edge midpoints.
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
    
    # Construct the list of 55 unit vectors
    # 0: Center
    center = np.array([0.0, 0.0, 0.0])
    # 1-12: Shell 1
    shell1 = verts
    # 13-24: Shell 2 Vertices
    shell2_v = verts
    # 25-54: Shell 2 Mids
    shell2_m = edge_mids
    
    all_vectors = np.vstack([center.reshape(1,3), shell1, shell2_v, shell2_m])
    return all_vectors

def _build_from_radii(radii):
    """
    Build positions from 54 radii (for atoms 1-54). Atom 0 is at origin.
    radii: array of shape (54,)
    """
    vectors = _get_mackay_unit_vectors()
    # Atom 0 is at origin
    pos = np.zeros((55, 3))
    # Atoms 1-54
    pos[1:] = vectors[1:] * radii[:, np.newaxis]
    return pos

def _energy_and_grad_manifold(radii):
    """
    Objective function for the manifold optimization.
    Returns energy and gradient with respect to radii.
    """
    pos = _build_from_radii(radii)
    E, grad_pos = _lj_energy_and_grad(pos)
    
    # Gradient with respect to radii
    # dE/dr_i = grad_pos[i] . unit_vector[i]
    vectors = _get_mackay_unit_vectors()
    # Atom 0 has radius 0, so we don't include it in the manifold parameters.
    # But grad_pos[0] is the force on the center.
    # The parameters are radii[0]..radii[53] corresponding to atoms 1..54.
    grad_radii = np.sum(grad_pos[1:] * vectors[1:], axis=1)
    
    return E, grad_radii

def _enforce_min_distance(pos, min_dist=0.5):
    """
    Enforce minimum distance constraint by pushing overlapping atoms apart.
    """
    pos = pos.copy()
    n = pos.shape[0]
    for _ in range(50):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist < min_dist:
                    if dist < 1e-10:
                        d = np.random.randn(3) * 0.1
                        dist = np.linalg.norm(d)
                    overlap = min_dist - dist
                    if overlap > 0:
                        direction = d / dist
                        pos[i] += 0.5 * overlap * direction
                        pos[j] -= 0.5 * overlap * direction
                        moved = True
        if not moved:
            break
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial guess for radii
    # Shell 1 (12 atoms): ~1.12
    # Shell 2 (42 atoms): ~2.24
    initial_radii = np.array([1.12] * 12 + [2.24] * 42)
    
    best_energy = np.inf
    best_pos = _build_from_radii(initial_radii)
    
    # 1. Optimize the 54 radial coordinates
    # This allows each atom to adjust its distance from the center independently
    # while maintaining the icosahedral directions.
    try:
        if time.time() < deadline:
            # Bounds: ensure atoms don't cross each other or go too far
            # Min radius ~0.5 (to avoid center collision for shell 1? No, center is 0.
            # Shell 1 atoms must be > 0.5 from center.
            # Shell 2 atoms must be > 0.5 from Shell 1 atoms and each other.
            # Conservative bounds:
            # Shell 1: [0.8, 1.5]
            # Shell 2: [1.8, 2.8]
            lower_bounds = [0.8] * 12 + [1.8] * 42
            upper_bounds = [1.5] * 12 + [2.8] * 42
            bounds = list(zip(lower_bounds, upper_bounds))
            
            res_manifold = minimize(_energy_and_grad_manifold, initial_radii, 
                                    jac=True, method='L-BFGS-B', bounds=bounds,
                                    options={'maxiter': 2000, 'gtol': 1e-8})
            
            if res_manifold.fun < best_energy:
                best_energy = res_manifold.fun
                best_pos = _build_from_radii(res_manifold.x)
                
    except Exception:
        pass
    
    # 2. Final full relaxation in Cartesian space
    # This allows breaking the radial symmetry if it leads to lower energy
    # (though for the global min, the symmetry is likely preserved or nearly so).
    try:
        if time.time() < deadline - 5.0:
            # Ensure min distance before relaxation to avoid singularities
            best_pos = _enforce_min_distance(best_pos, 0.5)
            
            result = minimize(_lj_energy_and_grad, best_pos, method='L-BFGS-B',
                             jac=True,
                             options={'maxiter': 3000, 'gtol': 1e-8})
            
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x
    except Exception:
        pass
    
    # 3. If time remains, try a few random restarts in the manifold
    if time.time() < deadline - 10.0:
        np.random.seed(42)
        for trial in range(5):
            if time.time() > deadline:
                break
            
            # Perturb the radii
            noise_scale = 0.1
            perturbed_radii = best_radii = res_manifold.x if 'res_manifold' in locals() and res_manifold.success else initial_radii
            perturbed_radii = perturbed_radii + np.random.randn(54) * noise_scale
            
            # Clip to bounds
            perturbed_radii[0:12] = np.clip(perturbed_radii[0:12], 0.8, 1.5)
            perturbed_radii[12:54] = np.clip(perturbed_radii[12:54], 1.8, 2.8)
            
            try:
                res_restart = minimize(_energy_and_grad_manifold, perturbed_radii, 
                                       jac=True, method='L-BFGS-B', bounds=bounds,
                                       options={'maxiter': 1000, 'gtol': 1e-6})
                
                if res_restart.fun < best_energy:
                    best_energy = res_restart.fun
                    candidate_pos = _build_from_radii(res_restart.x)
                    candidate_pos = _enforce_min_distance(candidate_pos, 0.5)
                    
                    # Quick full relaxation check
                    if time.time() < deadline - 2.0:
                         res_full = minimize(_lj_energy_and_grad, candidate_pos, method='L-BFGS-B',
                                             jac=True, options={'maxiter': 500, 'gtol': 1e-6})
                         if res_full.fun < best_energy:
                             best_energy = res_full.fun
                             best_pos = res_full.x
                    else:
                         best_pos = candidate_pos
            except Exception:
                pass

    # Final sanity check
    best_pos = _enforce_min_distance(best_pos, 0.5)
    
    return best_pos

# EVOLVE-BLOCK-END
