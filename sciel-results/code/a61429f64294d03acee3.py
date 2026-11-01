"""Lennard-Jones 55-atom cluster: edge-fraction manifold optimization for Mackay icosahedron."""
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
    
    E = 2.0 * np.sum(r12 - r6)
    
    coeff = 24.0 * (2.0 * r2_safe ** (-7) - r2_safe ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return E, grad

def _get_icosahedron_data():
    """
    Precompute icosahedron vertices, edge list, and edge geometry.
    Returns:
        verts: (12, 3) unit vectors to icosahedron vertices
        edges: list of 30 (i, j) vertex index pairs
        edge_starts: (30, 3) unit vectors to vertex i of each edge
        edge_dirs: (30, 3) unit vectors from vertex i toward vertex j for each edge
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Find edges (nearest-neighbor pairs)
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
    
    # Precompute edge geometry: for each edge (i,j), store start direction and
    # the direction from i to j (both unit vectors)
    edge_starts = np.zeros((30, 3))
    edge_dirs = np.zeros((30, 3))
    for k, (i, j) in enumerate(edges):
        edge_starts[k] = verts[i]
        d = verts[j] - verts[i]
        d /= np.linalg.norm(d)
        edge_dirs[k] = d
    
    return verts, edges, edge_starts, edge_dirs

def _build_from_manifold(params):
    """
    Build 55-atom positions from manifold parameters.
    params = [r1, r2, t_0, ..., t_29]
    - r1: radius of inner shell (12 atoms at icosahedron vertices)
    - r2: radius of outer shell vertex atoms (12 atoms)
    - t_k: fraction along edge k for the 30 outer-shell edge atoms
    
    Outer edge atom k is at: r2 * (edge_starts[k] + t_k * edge_dirs[k])
    This places it between the two inner-shell vertices of edge k, scaled by r2.
    """
    verts, edges, edge_starts, edge_dirs = _ICOSA_DATA
    r1 = params[0]
    r2 = params[1]
    t = params[2:32]
    
    pos = np.zeros((55, 3))
    # Atom 0: center
    pos[0] = 0.0
    # Atoms 1-12: inner shell
    pos[1:13] = verts * r1
    # Atoms 13-24: outer shell vertices
    pos[13:25] = verts * r2
    # Atoms 25-54: outer shell edge atoms
    for k in range(30):
        pos[25 + k] = r2 * (edge_starts[k] + t[k] * edge_dirs[k])
    
    return pos

def _manifold_energy_and_grad(params):
    """
    Energy and gradient on the 32D manifold.
    params = [r1, r2, t_0, ..., t_29]
    """
    pos = _build_from_manifold(params)
    E, grad_pos = _lj_energy_and_grad(pos)
    
    # Chain rule: dE/dparams = sum_i (dE/dpos_i) * (dpos_i/dparams)
    grad = np.zeros(32)
    verts, edges, edge_starts, edge_dirs = _ICOSA_DATA
    r1 = params[0]
    r2 = params[1]
    
    # dpos_i/dr1 for inner shell atoms (1-12): verts[i]
    for i in range(12):
        grad[0] += np.dot(grad_pos[i + 1], verts[i])
    
    # dpos_i/dr2 for outer shell vertices (13-24): verts[i]
    for i in range(12):
        grad[1] += np.dot(grad_pos[i + 13], verts[i])
    
    # dpos_i/dr2 for outer shell edge atoms (25-54): (edge_starts[k] + t[k]*edge_dirs[k])
    t = params[2:32]
    for k in range(30):
        direction = edge_starts[k] + t[k] * edge_dirs[k]
        grad[1] += np.dot(grad_pos[25 + k], direction)
    
    # dpos_i/dt_k for outer shell edge atom k: r2 * edge_dirs[k]
    for k in range(30):
        grad[2 + k] = r2 * np.dot(grad_pos[25 + k], edge_dirs[k])
    
    return E, grad

def _enforce_min_distance(pos, min_dist=0.5):
    """
    Enforce minimum distance constraint by pushing overlapping atoms apart.
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

# Precompute icosahedron data once at module level
_ICOSA_DATA = _get_icosahedron_data()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_energy = np.inf
    best_pos = None
    
    # Phase 1: Optimize on the 32D manifold (r1, r2, 30 edge fractions)
    # Start from symmetric Mackay: r1=1.12, r2=2.24, all t=0.5
    initial_params = np.zeros(32)
    initial_params[0] = 1.12
    initial_params[1] = 2.24
    initial_params[2:] = 0.5
    
    bounds = [(1.0, 1.5), (2.0, 2.5)] + [(0.0, 1.0)] * 30
    
    try:
        if time.time() < deadline:
            res = minimize(_manifold_energy_and_grad, initial_params, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 5000, 'gtol': 1e-10, 'ftol': 1e-12})
            if res.fun < best_energy:
                best_energy = res.fun
                best_pos = _build_from_manifold(res.x)
    except Exception:
        pass
    
    # Phase 2: Multi-start on the manifold with perturbed initial conditions
    rng = np.random.RandomState(42)
    
    for trial in range(10):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb the manifold parameters
        pert_params = initial_params.copy()
        pert_params[0] += rng.randn() * 0.05
        pert_params[1] += rng.randn() * 0.1
        pert_params[2:] += rng.randn(30) * 0.1
        # Clamp to bounds
        pert_params[0] = np.clip(pert_params[0], 1.0, 1.5)
        pert_params[1] = np.clip(pert_params[1], 2.0, 2.5)
        pert_params[2:] = np.clip(pert_params[2:], 0.0, 1.0)
        
        try:
            res = minimize(_manifold_energy_and_grad, pert_params, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 3000, 'gtol': 1e-9, 'ftol': 1e-11})
            if res.fun < best_energy:
                best_energy = res.fun
                best_pos = _build_from_manifold(res.x)
        except Exception:
            pass
    
    # Phase 3: Full 3D relaxation from the best manifold result
    if best_pos is not None and time.time() < deadline - 5.0:
        try:
            result = minimize(_lj_energy_and_grad, best_pos, method='L-BFGS-B',
                             options={'maxiter': 5000, 'gtol': 1e-10, 'ftol': 1e-12})
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x
        except Exception:
            pass
    
    # Phase 4: If time remains, try full 3D multi-start from manifold results
    if best_pos is not None and time.time() < deadline - 10.0:
        for trial in range(5):
            if time.time() > deadline - 5.0:
                break
            
            perturbed = best_pos + rng.randn(55, 3) * 0.05
            perturbed = _enforce_min_distance(perturbed, 0.5)
            
            try:
                res = minimize(_lj_energy_and_grad, perturbed, method='L-BFGS-B',
                             options={'maxiter': 2000, 'gtol': 1e-9, 'ftol': 1e-11})
                if res.fun < best_energy:
                    best_energy = res.fun
                    best_pos = res.x
            except Exception:
                pass
    
    # Fallback: if nothing worked, use the symmetric Mackay
    if best_pos is None:
        best_pos = _build_from_manifold(initial_params)
    
    # Final sanity check
    best_pos = _enforce_min_distance(best_pos, 0.5)
    
    return best_pos

# EVOLVE-BLOCK-END
