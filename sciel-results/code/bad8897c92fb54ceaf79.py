"""Lennard-Jones 55-atom cluster: edge-fraction manifold + full relaxation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

def _lj_energy(pos):
    """Compute total LJ energy."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    return e

def _lj_energy_grad(pos):
    """Compute LJ energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return e, grad

def _get_icosahedron_data():
    """Return icosahedron vertices, edges, and face centers."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    edge_len = np.linalg.norm(verts[0] - verts[1])
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, edge_len, atol=1e-10):
                edges.append((i, j))
    
    midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        midpoints.append(mid)
    midpoints = np.array(midpoints)
    
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if (np.isclose(d1, edge_len, atol=1e-10) and 
                    np.isclose(d2, edge_len, atol=1e-10) and 
                    np.isclose(d3, edge_len, atol=1e-10)):
                    faces.append((i, j, k))
    
    face_centers = []
    for i, j, k in faces:
        fc = (verts[i] + verts[j] + verts[k]) / 3.0
        fc = fc / np.linalg.norm(fc)
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    
    return verts, edges, midpoints, face_centers

def _build_mackay_with_edge_fractions(r_inner, r_outer, edge_fracs):
    """Build Mackay structure with outer atoms positioned by edge fractions.
    
    edge_fracs: array of 30 values in [0,1], one per edge.
    Each outer atom on edge i is at: (1-t_i)*verts[i] + t_i*verts[j], then scaled.
    """
    verts, edges, midpoints, face_centers = _get_icosahedron_data()
    
    inner = np.array([[0.0, 0.0, 0.0]])
    shell1 = verts * r_inner
    
    # 30 outer atoms on edges with variable fractions
    outer_edge_atoms = []
    for idx, (i, j) in enumerate(edges):
        t = edge_fracs[idx]
        t = np.clip(t, 0.0, 1.0)
        pos = (1.0 - t) * verts[i] + t * verts[j]
        pos = pos / np.linalg.norm(pos) * r_outer
        outer_edge_atoms.append(pos)
    outer_edge_atoms = np.array(outer_edge_atoms)
    
    # 12 outer atoms at face centers (fixed)
    outer_face_atoms = face_centers[:12] * r_outer
    
    pos = np.vstack([inner, shell1, outer_edge_atoms, outer_face_atoms])
    return pos

def _relax_lbfgs(pos, maxiter=500, gtol=1e-10, ftol=1e-12):
    """Relax using L-BFGS-B with gradient."""
    def fun(x):
        x = x.reshape(N, 3)
        e, g = _lj_energy_grad(x)
        return e, g.reshape(-1)
    
    try:
        result = minimize(fun, pos.reshape(-1), method='L-BFGS-B', jac=True, 
                         options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol})
        return result.x.reshape(N, 3), result.fun
    except Exception:
        return pos, _lj_energy(pos)

def _min_distance(pos):
    """Compute minimum pairwise distance."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    return np.sqrt(r2.min())

def _optimize_edge_fractions(r_inner, r_outer, deadline):
    """Optimize the 30 edge fractions using L-BFGS on the relaxed energy."""
    def objective(fracs):
        pos = _build_mackay_with_edge_fractions(r_inner, r_outer, fracs)
        if _min_distance(pos) < 0.5:
            return 1e10
        pos_r, e_r = _relax_lbfgs(pos, maxiter=100, gtol=1e-8, ftol=1e-10)
        if _min_distance(pos_r) < 0.5:
            return 1e10
        return e_r
    
    x0 = np.full(30, 0.5)
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 50, 'xatol': 1e-5, 'fatol': 1e-7,
                                  'adaptive': True})
        if time.time() > deadline:
            return x0
        if result.fun < objective(x0):
            return np.clip(result.x, 0, 1)
    except Exception:
        pass
    return x0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    verts, edges, midpoints, face_centers = _get_icosahedron_data()
    
    # Phase 1: Optimize shell radii
    def radial_energy(radii):
        r_in, r_out = radii
        if r_in < 0.5 or r_out < 1.0 or r_out < r_in + 0.3:
            return 1e10
        pos = _build_mackay_with_edge_fractions(r_in, r_out, np.full(30, 0.5))
        if _min_distance(pos) < 0.5:
            return 1e10
        pos_r, e_r = _relax_lbfgs(pos, maxiter=100, gtol=1e-8, ftol=1e-10)
        if _min_distance(pos_r) < 0.5:
            return 1e10
        return e_r
    
    r_init = np.array([1.0, 2.0])
    try:
        result = minimize(radial_energy, r_init, method='Nelder-Mead',
                         options={'maxiter': 30, 'xatol': 1e-4, 'fatol': 1e-6,
                                  'adaptive': True})
        if result.fun < radial_energy(r_init):
            r_in_opt, r_out_opt = result.x
        else:
            r_in_opt, r_out_opt = r_init
    except Exception:
        r_in_opt, r_out_opt = r_init
    
    if time.time() > deadline:
        pos = _build_mackay_with_edge_fractions(r_in_opt, r_out_opt, np.full(30, 0.5))
        pos, e = _relax_lbfgs(pos, maxiter=500, gtol=1e-12, ftol=1e-14)
        return pos
    
    # Phase 2: Optimize edge fractions
    edge_fracs = _optimize_edge_fractions(r_in_opt, r_out_opt, deadline)
    
    if time.time() > deadline:
        pos = _build_mackay_with_edge_fractions(r_in_opt, r_out_opt, edge_fracs)
        pos, e = _relax_lbfgs(pos, maxiter=500, gtol=1e-12, ftol=1e-14)
        return pos
    
    # Phase 3: Build and relax the optimized structure
    pos_base = _build_mackay_with_edge_fractions(r_in_opt, r_out_opt, edge_fracs)
    pos_relaxed, e_relaxed = _relax_lbfgs(pos_base, maxiter=1000, gtol=1e-12, ftol=1e-14)
    best_pos = pos_relaxed.copy()
    best_e = e_relaxed
    
    if time.time() > deadline - 20:
        return best_pos
    
    # Phase 4: Multi-start with perturbations
    rng = np.random.RandomState(42)
    n_trials = 0
    while time.time() < deadline - 20:
        n_trials += 1
        scale = 0.01 + 0.02 * (n_trials % 8)
        pos_pert = pos_base + rng.randn(N, 3) * scale
        
        if _min_distance(pos_pert) < 0.5:
            continue
        
        pos_r, e_r = _relax_lbfgs(pos_pert, maxiter=200, gtol=1e-10, ftol=1e-12)
        
        if _min_distance(pos_r) < 0.5:
            continue
        
        if e_r < best_e:
            best_e = e_r
            best_pos = pos_r.copy()
    
    # Phase 5: Final high-precision polish
    if time.time() < deadline - 5:
        pos_final, e_final = _relax_lbfgs(best_pos, maxiter=2000, gtol=1e-13, ftol=1e-15)
        if _min_distance(pos_final) >= 0.5 and e_final < best_e:
            best_e = e_final
            best_pos = pos_final.copy()
    
    return best_pos
