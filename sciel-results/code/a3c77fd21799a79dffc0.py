"""
Lennard-Jones 55-atom cluster: Mackay icosahedron with optimized shell radii
and full Cartesian relaxation.

Strategy:
  1. Build an initial guess from icosahedral geometry (1 center, 12 vertices,
     30 edge-midpoints, 12 face-centers) with optimized shell radii.
  2. Optimize shell radii (r1, r2) via a small Nelder-Mead search to get a
     better starting point.
  3. Full Cartesian L-BFGS-B relaxation.
  4. Multi-start with structured perturbations.
  5. Final high-precision polish.
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 55
DEADLINE = 142.0  # safety margin before 150 s kill


# ---------------------------------------------------------------------------
# Lennard-Jones energy and gradient
# ---------------------------------------------------------------------------

def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient for N atoms. pos shape (N,3)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** -3
    r12 = r6 ** 2
    energy = 2.0 * np.sum(np.triu(r12 - r6, k=1))
    # Gradient
    coeff = (24.0 * r2 ** -4 - 48.0 * r2 ** -7)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1)
    grad = row_sum[:, np.newaxis] * pos - coeff @ pos
    return energy, grad


# ---------------------------------------------------------------------------
# Icosahedral geometry
# ---------------------------------------------------------------------------

def _icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron at unit circumradius."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  phi, 0], [1,  phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1,  phi], [0, 1,  phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts /= norms
    return verts


def _find_edges_and_faces(ico):
    """Find edges and faces of the icosahedron from vertices."""
    n = len(ico)
    dists = np.linalg.norm(ico[:, np.newaxis, :] - ico[np.newaxis, :, :], axis=2)
    edge_threshold = 1.2
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if dists[i, j] < edge_threshold:
                edges.append((i, j))
    # Find faces
    adj = {i: set() for i in range(n)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    faces = set()
    for i in range(n):
        for j in adj[i]:
            for k in adj[j]:
                if k != i and k in adj[i]:
                    face = tuple(sorted((i, j, k)))
                    faces.add(face)
    return edges, list(faces)


def _mackay_positions(r1=1.1225, r2=2.051):
    """Build 55-atom Mackay icosahedron positions.
    
    Structure:
      - 1 atom at origin
      - 12 atoms at icosahedral vertices at radius r1
      - 30 atoms at edge midpoints at radius r2
      - 12 atoms at selected face centers at radius r2
    """
    ico = _icosahedron_vertices()
    edges, faces = _find_edges_and_faces(ico)
    
    # Edge midpoints (30)
    edge_mids = []
    for i, j in edges:
        mid = (ico[i] + ico[j]) / 2.0
        mid /= np.linalg.norm(mid)
        edge_mids.append(mid)
    edge_mids = np.array(edge_mids)
    
    # Face centers (20), select 12
    face_centers = []
    for f in faces:
        c = (ico[f[0]] + ico[f[1]] + ico[f[2]]) / 3.0
        c /= np.linalg.norm(c)
        face_centers.append(c)
    face_centers = np.array(face_centers)
    
    # Select 12 face centers (use deterministic selection)
    # Pick face centers that are most "spread out" - use a simple greedy approach
    # or just take every other one. For determinism, take indices 0,2,4,...,22
    selected_fc = face_centers[::2][:12]
    
    center = np.zeros((1, 3))
    shell1 = ico * r1
    shell2 = np.vstack([edge_mids, selected_fc]) * r2
    
    pos = np.vstack([center, shell1, shell2])
    return pos


def _optimize_shell_radii():
    """Optimize r1 and r2 for the Mackay icosahedron."""
    def energy_fn(params):
        r1, r2 = params
        pos = _mackay_positions(r1, r2)
        # Check minimum distance
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 0.5:
            return 1e6
        e, _ = _lj_energy_and_grad(pos)
        return e
    
    # Nelder-Mead for 2D optimization
    x0 = np.array([1.12, 2.05])
    res = minimize(energy_fn, x0, method='Nelder-Mead',
                   options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
    return res.x[0], res.x[1]


# ---------------------------------------------------------------------------
# Relaxation
# ---------------------------------------------------------------------------

def _relax(pos, high_precision=False, max_iter=5000):
    """Relax positions using L-BFGS-B. Returns (positions, energy)."""
    def obj_and_grad(p):
        pos_2d = p.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos_2d)
        return e, g.ravel()
    
    if high_precision:
        options = {'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-12, 'maxfun': 20000}
    else:
        options = {'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 10000}
    
    res = minimize(obj_and_grad, pos.ravel(), method='L-BFGS-B', jac=True, options=options)
    if hasattr(res, 'x'):
        new_pos = res.x.reshape(-1, 3)
        e, _ = _lj_energy_and_grad(new_pos)
        return new_pos, e
    return pos, _lj_energy_and_grad(pos)[0]


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    t0 = time.time()
    
    # Step 1: Optimize shell radii
    r1_opt, r2_opt = _optimize_shell_radii()
    pos0 = _mackay_positions(r1_opt, r2_opt)
    
    # Step 2: Full Cartesian relaxation
    pos_best, e_best = _relax(pos0)
    
    elapsed = time.time() - t0
    if elapsed > DEADLINE - 25.0:
        # Final polish
        if time.time() - t0 < DEADLINE - 3.0:
            pos_final, e_final = _relax(pos_best, high_precision=True)
            if e_final < e_best:
                pos_best, e_best = pos_final, e_final
        return pos_best
    
    # Step 3: Multi-start with perturbations
    rng = np.random.RandomState(42)
    
    for s in range(6):
        if time.time() - t0 > DEADLINE - 15.0:
            break
        sigma = 0.05 + 0.05 * s  # increasing perturbation
        noise = rng.normal(0, sigma, size=(N, 3))
        pos_pert = pos_best + noise
        pos_new, e_new = _relax(pos_pert)
        if e_new < e_best:
            pos_best, e_best = pos_new, e_new
    
    # Step 4: Try perturbations from the original optimized structure
    if time.time() - t0 < DEADLINE - 10.0:
        for s in range(4):
            if time.time() - t0 > DEADLINE - 5.0:
                break
            sigma = 0.1 + 0.1 * s
            noise = rng.normal(0, sigma, size=(N, 3))
            pos_pert = pos0 + noise
            pos_new, e_new = _relax(pos_pert)
            if e_new < e_best:
                pos_best, e_best = pos_new, e_new
    
    # Step 5: Final high-precision polish
    if time.time() - t0 < DEADLINE - 3.0:
        pos_final, e_final = _relax(pos_best, high_precision=True, max_iter=20000)
        if e_final < e_best:
            pos_best, e_best = pos_final, e_final
    
    return pos_best
