"""Lennard-Jones cluster global minimum, 55 atoms.

Strategy:
- Construct the exact Mackay icosahedron (two shells: 12 + 42) using 3 radial parameters.
- Optimize those 3 parameters via L-BFGS to find the best symmetric geometry.
- Then perform a full Cartesian relaxation with L-BFGS to polish.
- Use a deadline guard to ensure we return within the time limit.
"""
N = 55

import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

# Deadline: kill at 150s, return best by 145s
_DEADLINE = 145.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _lj_energy(pos):
    """Compute total LJ energy for positions (N,3)."""
    n = len(pos)
    if n < 2:
        return 0.0
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    # Avoid self-interaction (diagonal)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    energy = 4.0 * np.sum(r12 - r6)
    return energy

def _lj_energy_grad(pos):
    """Compute total LJ energy and gradient for positions (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    energy = 4.0 * np.sum(r12 - r6)
    
    # Gradient: dE/dr_i = 4 * sum_j (12*r_i^-13 - 6*r_i^-7) * (r_i/r_i) ... 
    # Actually: d/dr (4*(r^-12 - r^-6)) = 4*(-12*r^-13 + 6*r^-7)
    # dE/dpos_i = sum_j 4*(-12*r_ij^-13 + 6*r_ij^-7) * (pos_i - pos_j)/r_ij
    # = sum_j 4*(-12*r_ij^-14 + 6*r_ij^-8) * (pos_i - pos_j)
    # Wait, let me be careful.
    # E = 4 * sum_{i<j} (r_ij^-12 - r_ij^-6)
    # dE/dx_i = 4 * sum_{j != i} (-12*r_ij^-13 + 6*r_ij^-7) * (x_i - x_j)/r_ij
    # = 4 * sum_{j != i} (-12*r_ij^-14 + 6*r_ij^-8) * (x_i - x_j)
    
    # Using vectorized approach
    # coeff[i,j] = 4 * (-12 * r_ij^-14 + 6 * r_ij^-8)
    r14 = r2 ** (-7)  # r^-14
    r8 = r2 ** (-4)    # r^-8
    coeff = 4.0 * (-12.0 * r14 + 6.0 * r8)
    np.fill_diagonal(coeff, 0.0)
    
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad

def _mckay_icosahedron(r1, r2, r3):
    """
    Generate Mackay icosahedron positions for 55 atoms.
    
    Shell 1: 12 atoms (vertices of icosahedron) at radius r1
    Shell 2: 42 atoms at radius r2 (or split into sub-shells)
    
    Actually, the Mackay icosahedron for n=55 has:
    - 12 atoms at the icosahedron vertices (radius R1)
    - 30 atoms at edge midpoints of the icosahedron (radius R2) 
    - 12 atoms at face centers (radius R3)
    - 1 atom at center? No, that's 55 = 1+12+30+12
    
    Wait: 1 + 12 + 30 + 12 = 55. Yes!
    - 1 center atom
    - 12 vertices
    - 30 edge centers
    - 12 face centers
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices (normalized to unit sphere)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Center atom
    center = np.array([[0.0, 0.0, 0.0]])
    
    # Shell 1: 12 vertices at radius r1
    shell1 = r1 * verts
    
    # Shell 2: 30 edge midpoints at radius r2
    # Edges of icosahedron: each vertex connects to 5 others
    # We need to find all unique edges
    # For a regular icosahedron, the edge length in terms of the circumradius is known.
    # But we can compute edge midpoints by finding pairs of vertices that are nearest neighbors.
    
    # Compute pairwise distances between vertices
    vdiff = verts[:, np.newaxis, :] - verts[np.newaxis, :, :]
    vd2 = np.sum(vdiff * vdiff, axis=2)
    np.fill_diagonal(vd2, np.inf)
    
    # Find the minimum distance (edge length squared)
    min_d2 = np.min(vd2)
    # Tolerance for identifying edges
    edge_mask = (vd2 < min_d2 * 1.001) & (vd2 > 0)
    
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if edge_mask[i, j]:
                edges.append((i, j))
    
    # Should have 30 edges
    edge_mids = []
    for i, j in edges:
        mid = 0.5 * (verts[i] + verts[j])
        mid /= np.linalg.norm(mid)
        edge_mids.append(mid)
    edge_mids = np.array(edge_mids)
    shell2 = r2 * edge_mids
    
    # Shell 3: 12 face centers at radius r3
    # Faces of icosahedron: each face is a triangle of 3 mutually connected vertices
    # Find all triangular faces
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            if edge_mask[i, j]:
                for k in range(j+1, 12):
                    if edge_mask[i, k] and edge_mask[j, k]:
                        faces.append((i, j, k))
    
    face_centers = []
    for i, j, k in faces:
        fc = verts[i] + verts[j] + verts[k]
        fc /= np.linalg.norm(fc)
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    shell3 = r3 * face_centers
    
    # Combine: 1 + 12 + 30 + 12 = 55
    pos = np.vstack([center, shell1, shell2, shell3])
    return pos

def _mckay_energy(params):
    """Energy as a function of the 3 shell radii."""
    r1, r2, r3 = params
    pos = _mckay_icosahedron(r1, r2, r3)
    return _lj_energy(pos)

def _mckay_energy_grad(params):
    """Energy and gradient as a function of the 3 shell radii (finite diff for grad)."""
    pos = _mckay_icosahedron(params[0], params[1], params[2])
    e = _lj_energy(pos)
    # Numerical gradient
    grad = np.zeros(3)
    h = 1e-6
    for i in range(3):
        p_plus = params.copy()
        p_plus[i] += h
        p_minus = params.copy()
        p_minus[i] -= h
        e_plus = _lj_energy(_mckay_icosahedron(*p_plus))
        e_minus = _lj_energy(_mckay_icosahedron(*p_minus))
        grad[i] = (e_plus - e_minus) / (2 * h)
    return e, grad

def construct():
    global _start_time
    _start_time = time.time()
    
    # Initial guess for shell radii
    # Typical LJ minimum distance is 2^(1/6) ≈ 1.1225
    # For Mackay icosahedron, reasonable starting values
    r1_init = 1.1225 * 1.0
    r2_init = 1.1225 * 1.4
    r3_init = 1.1225 * 1.7
    
    best_pos = None
    best_energy = np.inf
    
    # Phase 1: Optimize the 3 radial parameters
    x0 = np.array([r1_init, r2_init, r3_init])
    
    try:
        result = minimize(
            _mckay_energy_grad,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        if result.success and _elapsed() < _DEADLINE - 30:
            pos_mckay = _mckay_icosahedron(*result.x)
            e_mckay = _lj_energy(pos_mckay)
            if e_mckay < best_energy:
                best_energy = e_mckay
                best_pos = pos_mckay
    except Exception:
        pass
    
    if _elapsed() > _DEADLINE - 10:
        if best_pos is not None:
            return best_pos
        # Fallback to initial guess
        return _mckay_icosahedron(r1_init, r2_init, r3_init)
    
    # Phase 2: Full Cartesian relaxation from the optimized Mackay geometry
    if best_pos is None:
        best_pos = _mckay_icosahedron(r1_init, r2_init, r3_init)
        best_energy = _lj_energy(best_pos)
    
    # L-BFGS relaxation in Cartesian coordinates
    try:
        result2 = minimize(
            _lj_energy_grad,
            best_pos,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        if result2.success and _elapsed() < _DEADLINE - 5:
            e2 = _lj_energy(result2.x)
            if e2 < best_energy:
                best_energy = e2
                best_pos = result2.x
    except Exception:
        pass
    
    # Phase 3: Try a few different initial radial parameters and relax
    if _elapsed() < _DEADLINE - 20:
        # Try slightly perturbed radii
        perturbations = [
            [1.10, 1.55, 1.85],
            [1.15, 1.45, 1.75],
            [1.12, 1.50, 1.80],
            [1.08, 1.60, 1.90],
        ]
        for p_init in perturbations:
            if _elapsed() > _DEADLINE - 15:
                break
            try:
                res_p = minimize(
                    _mckay_energy_grad,
                    np.array(p_init),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if res_p.success:
                    pos_p = _mckay_icosahedron(*res_p.x)
                    e_p = _lj_energy(pos_p)
                    if e_p < best_energy:
                        best_energy = e_p
                        best_pos = pos_p
                        
                        # Polish in Cartesian
                        if _elapsed() < _DEADLINE - 10:
                            res_c = minimize(
                                _lj_energy_grad,
                                best_pos,
                                jac=True,
                                method='L-BFGS-B',
                                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
                            )
                            if res_c.success:
                                e_c = _lj_energy(res_c.x)
                                if e_c < best_energy:
                                    best_energy = e_c
                                    best_pos = res_c.x
            except Exception:
                pass
    
    # Final check
    if best_pos is None:
        best_pos = _mckay_icosahedron(r1_init, r2_init, r3_init)
    
    return best_pos
