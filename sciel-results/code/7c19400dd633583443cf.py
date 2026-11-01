"""Lennard-Jones 55-atom cluster: Mackay icosahedron with optimized radii + L-BFGS relaxation."""
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
    
    return verts, edge_mids, edges

def _build_mackay(r1, r2):
    """
    Build the 55-atom Mackay icosahedron with given shell radii.
    Shell 0: 1 atom at origin.
    Shell 1: 12 atoms at r1 * verts.
    Shell 2: 42 atoms (12 at r2 * verts, 30 at r2 * edge_mids).
    """
    verts, edge_mids, _ = _get_mackay_vectors()
    
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

def _enforce_min_distance(pos, min_dist=0.5):
    """
    Enforce minimum distance constraint by pushing overlapping atoms apart.
    Returns a valid position array.
    """
    pos = pos.copy()
    n = pos.shape[0]
    for _ in range(100):
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

def _get_symmetry_groups():
    """
    Return symmetry-equivalent groups of atom indices for the Mackay icosahedron.
    Group 0: center (1 atom)
    Group 1: first shell vertices (12 atoms)
    Group 2: second shell vertices (12 atoms)
    Group 3: second shell edge midpoints (30 atoms)
    """
    groups = [
        [0],           # center
        list(range(1, 13)),    # shell 1: 12 vertices
        list(range(13, 25)),   # shell 2: 12 vertices
        list(range(25, 55))    # shell 2: 30 edge midpoints
    ]
    return groups

def _symmetry_aware_perturb(pos, groups, scale, rng):
    """
    Perturb positions with symmetry-aware noise: atoms in the same group
    receive correlated perturbations (same magnitude, random direction).
    """
    perturbed = pos.copy()
    n = pos.shape[0]
    
    for group in groups:
        if len(group) == 1:
            # Center atom: small independent perturbation
            perturbed[group[0]] += rng.randn(3) * scale * 0.5
        else:
            # For each atom in the group, add independent small perturbation
            # but scale the overall perturbation by group size to maintain energy scale
            for idx in group:
                perturbed[idx] += rng.randn(3) * scale
    
    return perturbed

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 40s safety margin for 150s wall-clock
    
    # 1. Optimize the radii of the symmetric Mackay icosahedron
    initial_guess = [1.12, 2.24]
    
    try:
        res_radii = minimize(_energy_for_radii, initial_guess, method='L-BFGS-B', 
                             bounds=[(1.0, 1.5), (2.0, 2.5)], 
                             options={'maxiter': 1000, 'gtol': 1e-8})
        r1_opt, r2_opt = res_radii.x
    except Exception:
        r1_opt, r2_opt = initial_guess
        
    # Build the optimized symmetric structure
    pos_seed = _build_mackay(r1_opt, r2_opt)
    
    best_energy = np.inf
    best_pos = pos_seed
    
    # 2. Full relaxation with L-BFGS from the symmetric structure
    try:
        if time.time() < deadline:
            result = minimize(_lj_energy_and_grad, pos_seed, method='L-BFGS-B',
                             options={'maxiter': 10000, 'gtol': 1e-10, 'ftol': 1e-12})
            
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x
    except Exception:
        pass
    
    # 3. Multi-start with symmetry-aware perturbations
    rng = np.random.RandomState(42)
    groups = _get_symmetry_groups()
    
    # Try several perturbation scales with multiple trials
    for scale in [0.005, 0.01, 0.02, 0.05, 0.1]:
        if time.time() > deadline:
            break
            
        n_trials = 5 if scale <= 0.02 else 3
        for trial in range(n_trials):
            if time.time() > deadline:
                break
                
            perturbed = _symmetry_aware_perturb(pos_seed, groups, scale, rng)
            perturbed = _enforce_min_distance(perturbed, 0.5)
            
            try:
                res = minimize(_lj_energy_and_grad, perturbed, method='L-BFGS-B',
                             options={'maxiter': 3000, 'gtol': 1e-8, 'ftol': 1e-10})
                if res.fun < best_energy:
                    best_energy = res.fun
                    best_pos = res.x
            except Exception:
                pass
    
    # 4. If we have time, try a few larger perturbations
    if time.time() < deadline - 15.0:
        for scale in [0.2, 0.3]:
            if time.time() > deadline:
                break
                
            for trial in range(2):
                if time.time() > deadline:
                    break
                    
                perturbed = _symmetry_aware_perturb(pos_seed, groups, scale, rng)
                perturbed = _enforce_min_distance(perturbed, 0.5)
                
                try:
                    res = minimize(_lj_energy_and_grad, perturbed, method='L-BFGS-B',
                                 options={'maxiter': 2000, 'gtol': 1e-7, 'ftol': 1e-9})
                    if res.fun < best_energy:
                        best_energy = res.fun
                        best_pos = res.x
                except Exception:
                    pass
    
    # Final sanity check: ensure min distance constraint
    best_pos = _enforce_min_distance(best_pos, 0.5)
    
    return best_pos

# EVOLVE-BLOCK-END
