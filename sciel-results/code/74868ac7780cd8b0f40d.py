"""Lennard-Jones 55-atom cluster: Mackay icosahedron with optimized radii + L-BFGS relaxation."""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import product

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

def _get_mackay_vectors():
    """
    Returns the 12 unit vectors to icosahedron vertices and 
    the 30 unit vectors to edge midpoints.
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
    
    return verts, edge_mids

def _build_mackay(r1, r2):
    """
    Build the 55-atom Mackay icosahedron with given shell radii.
    Shell 0: 1 atom at origin.
    Shell 1: 12 atoms at r1 * verts.
    Shell 2: 42 atoms (12 at r2 * verts, 30 at r2 * edge_mids).
    """
    verts, edge_mids = _get_mackay_vectors()
    
    center = np.array([[0.0, 0.0, 0.0]])
    shell1 = verts * r1
    shell2_verts = verts * r2
    shell2_mids = edge_mids * r2
    
    all_points = np.vstack([center, shell1, shell2_verts, shell2_mids])
    return all_points

def _energy_for_radii(params):
    """Objective function for optimizing radii r1, r2."""
    r1, r2 = params
    pos = _build_mackay(r1, r2)
    e, _ = _lj_energy_and_grad(pos)
    return e

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Optimize the radii of the symmetric Mackay icosahedron
    # Initial guess: r1 ~ 1.12, r2 ~ 2.24
    # We use L-BFGS on the 2 parameters r1, r2
    initial_guess = [1.12, 2.24]
    
    try:
        # Minimize radii
        res_radii = minimize(_energy_for_radii, initial_guess, method='L-BFGS-B', 
                             bounds=[(1.0, 1.5), (2.0, 2.5)], 
                             options={'maxiter': 1000, 'gtol': 1e-6})
        r1_opt, r2_opt = res_radii.x
    except Exception:
        r1_opt, r2_opt = initial_guess
        
    # Build the optimized symmetric structure
    pos_seed = _build_mackay(r1_opt, r2_opt)
    
    # 2. Full relaxation with L-BFGS
    # This should converge very quickly as the seed is structurally correct
    try:
        result = minimize(_lj_energy_and_grad, pos_seed, method='L-BFGS-B',
                         options={'maxiter': 5000, 'gtol': 1e-8})
        
        if result.fun < -200: # Sanity check
            return result.x
    except Exception:
        pass
    
    # Fallback: Multi-start with small perturbations if the main relaxation failed
    np.random.seed(42)
    best_energy = np.inf
    best_pos = pos_seed
    
    for scale in [0.01, 0.05, 0.1]:
        if time.time() > deadline:
            break
        perturbed = pos_seed + np.random.randn(55, 3) * scale
        # Enforce min distance 0.5
        # Simple rejection or push-apart
        valid = True
        for i in range(55):
            for j in range(i+1, 55):
                d = np.linalg.norm(perturbed[i] - perturbed[j])
                if d < 0.5:
                    valid = False
                    break
            if not valid: break
            
        if not valid:
            continue
            
        try:
            res = minimize(_lj_energy_and_grad, perturbed, method='L-BFGS-B',
                           options={'maxiter': 1000, 'gtol': 1e-6})
            if res.fun < best_energy:
                best_energy = res.fun
                best_pos = res.x
        except Exception:
            pass
            
    return best_pos

# EVOLVE-BLOCK-END
