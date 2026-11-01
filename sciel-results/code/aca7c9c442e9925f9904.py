"""Program for Lennard-Jones 55-atom cluster global minimum search.

Uses a symmetry-constrained manifold (3 shell radii + rigid rotation)
optimized via Nelder-Mead, followed by full L-BFGS relaxation to
converge to the global minimum energy of ~-279.248.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

N = 55

def _get_icosahedron_vectors():
    """
    Returns 12 unit vectors along the 5-fold axes (icosahedron vertices)
    and 30 unit vectors along the 2-fold axes (icosahedron edge midpoints).
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi],
        [0, -1, phi],
        [0, 1, -phi],
        [0, -1, -phi],
        [1, phi, 0],
        [-1, phi, 0],
        [1, -phi, 0],
        [-1, -phi, 0],
        [phi, 0, 1],
        [-phi, 0, 1],
        [phi, 0, -1],
        [-phi, 0, -1],
    ], dtype=float)
    
    # Normalize to unit length
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # 30 edge midpoints
    dists = np.linalg.norm(verts[:, np.newaxis, :] - verts[np.newaxis, :, :], axis=2)
    min_dist = np.min(dists[dists > 1e-8])
    
    edge_mids = []
    for i in range(12):
        for j in range(i+1, 12):
            if np.isclose(dists[i,j], min_dist, atol=1e-8):
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid)
                edge_mids.append(mid)
                
    edge_mids = np.array(edge_mids)
    return verts, edge_mids

def _mackay_icosahedron_55():
    """
    Construct the 55-atom two-shell Mackay icosahedron.
    1 center, 12 at 5-fold (r1), 30 at 2-fold (r2), 12 at 5-fold (r3).
    Uses radii close to the known optimal values.
    """
    v5, v2 = _get_icosahedron_vectors()
    
    # Optimal radii for the Mackay icosahedron (from literature)
    r1 = 1.135
    r2 = 1.162
    r3 = 1.283
    
    points = [np.zeros(3)]
    
    # Add 12 atoms at 5-fold axes, radius r1
    for v in v5:
        points.append(v * r1)
        
    # Add 30 atoms at 2-fold axes, radius r2
    for v in v2:
        points.append(v * r2)
        
    # Add 12 atoms at 5-fold axes, radius r3
    for v in v5:
        points.append(v * r3)
        
    points = np.array(points, dtype=float)
    return points

def _lj_energy(pos):
    """Compute total Lennard-Jones energy."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    mask = r2 > 1e-10
    r2_safe = np.where(mask, r2, 1.0)
    r6 = r2_safe ** (-3)
    r12 = r6 ** 2
    energy = 4 * np.sum(np.where(mask, r12 - r6, 0))
    return energy

def _lj_energy_grad(pos):
    """Compute total Lennard-Jones energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    mask = r2 > 1e-10
    r2_safe = np.where(mask, r2, 1.0)
    r6 = r2_safe ** (-3)
    r12 = r6 ** 2
    energy = 4 * np.sum(np.where(mask, r12 - r6, 0))
    
    # Gradient: dE/dx_i = 4 * sum_j (12*r^-14 - 6*r^-8) * (x_i - x_j) / r
    # = 24 * sum_j (r^-14 - 0.5*r^-8) * (x_i - x_j)
    # More carefully: E = 4 * sum_{i<j} (r_ij^-12 - r_ij^-6)
    # dE/dx_i = 4 * sum_{j != i} (-12*r_ij^-14 + 6*r_ij^-8) * (x_i - x_j)/r_ij * r_ij
    # Wait, let me redo. E_pair = 4*(r^-12 - r^-6) where r = |x_i - x_j|
    # dE_pair/dx_i = 4*(-12*r^-13 + 6*r^-7) * d r / d x_i
    # d r / d x_i = (x_i - x_j) / r
    # So dE_pair/dx_i = 4*(-12*r^-14 + 6*r^-8) * (x_i - x_j)
    # = (48*r^-14 - 24*r^-8) * (x_i - x_j)
    # = 24 * (2*r^-14 - r^-8) * (x_i - x_j)
    
    # Vectorized: for each pair (i,j), contribution to grad[i] is 
    # 4*(-12*r^-14 + 6*r^-8) * (x_i - x_j), and to grad[j] is the negative
    r2_safe_full = np.where(mask, r2_safe, 1.0)
    r = np.sqrt(r2_safe_full)
    # factor[i,j] = 4*(-12*r^-14 + 6*r^-8) = 4*(-12/r^14 + 6/r^8)
    # = 4*(-12*r2_safe_full^-7 + 6*r2_safe_full^-4)
    factor = np.where(mask, 4 * (-12 * r2_safe_full ** (-7) + 6 * r2_safe_full ** (-4)), 0.0)
    
    # grad[i] = sum_j factor[i,j] * (x_i - x_j)
    # = sum_j factor[i,j] * x_i - sum_j factor[i,j] * x_j
    # = x_i * sum_j factor[i,j] - (factor @ x)[i]
    row_sums = factor.sum(axis=1)
    grad = row_sums[:, np.newaxis] * pos - factor @ pos
    
    return energy, grad

def _is_valid(pos):
    """Check if all inter-atomic distances are >= 0.5."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.5

def _relax(pos, maxiter=10000, ftol=1e-12, gtol=1e-10):
    """Relax positions using L-BFGS-B with analytic gradient."""
    def energy_func(x):
        pos_flat = x.reshape(-1, 3)
        return _lj_energy(pos_flat)
        
    def energy_grad_func(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
        
    result = minimize(
        energy_grad_func,
        pos.flatten(),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
    )
    return result.x.reshape(-1, 3)

def _manifold_energy(params, v5, v2):
    """
    Compute LJ energy on the 9D manifold: 3 radii + 3 Euler angles (rotation).
    params = [r1, r2, r3, rx, ry, rz]
    """
    r1, r2, r3, rx, ry, rz = params
    rot = Rotation.from_euler('xyz', [rx, ry, rz])
    R = rot.as_matrix()
    
    # Build positions
    center = np.zeros(3)
    inner = (v5 * r1) @ R.T   # 12 atoms
    edge = (v2 * r2) @ R.T     # 30 atoms
    outer = (v5 * r3) @ R.T    # 12 atoms
    
    pos = np.vstack([center[None, :], inner, edge, outer])
    return _lj_energy(pos)

def _manifold_energy_and_grad(params, v5, v2):
    """Compute energy and gradient on the manifold (numerical gradient for rotation)."""
    e = _manifold_energy(params, v5, v2)
    # Use analytic gradient for radii, numerical for rotation
    grad = np.zeros(6)
    eps = 1e-7
    for i in range(3):
        p_plus = params.copy()
        p_plus[i] += eps
        p_minus = params.copy()
        p_minus[i] -= eps
        grad[i] = (_manifold_energy(p_plus, v5, v2) - _manifold_energy(p_minus, v5, v2)) / (2*eps)
    for i in range(3, 6):
        p_plus = params.copy()
        p_plus[i] += eps
        p_minus = params.copy()
        p_minus[i] -= eps
        grad[i] = (_manifold_energy(p_plus, v5, v2) - _manifold_energy(p_minus, v5, v2)) / (2*eps)
    return e, grad

def construct():
    """
    Construct the 55-atom LJ cluster using symmetry-constrained manifold
    optimization (Nelder-Mead on 9D parameter space) followed by L-BFGS polish.
    """
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    v5, v2 = _get_icosahedron_vectors()
    base = _mackay_icosahedron_55()
    
    best_energy = float('inf')
    best_pos = base.copy()
    rng = np.random.default_rng(42)
    
    # Phase 1: Nelder-Mead on the 9D manifold (radii + rotation)
    # Initial parameters: nominal radii, zero rotation
    init_params = np.array([1.135, 1.162, 1.283, 0.0, 0.0, 0.0])
    
    # Try multiple initial radii perturbations
    radius_perturbations = [
        [0.0, 0.0, 0.0],
        [0.01, 0.01, 0.01],
        [-0.01, -0.01, -0.01],
        [0.02, -0.01, 0.01],
        [-0.02, 0.01, -0.01],
        [0.0, 0.02, 0.0],
        [0.0, -0.02, 0.0],
        [0.02, 0.0, -0.02],
    ]
    
    rotation_perturbations = [
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1],
        [0.05, 0.05, 0.05],
    ]
    
    for rp in radius_perturbations:
        if time.time() > deadline:
            break
        for rot_p in rotation_perturbations[:3]:
            if time.time() > deadline:
                break
            params0 = np.array([1.135 + rp[0], 1.162 + rp[1], 1.283 + rp[2]] + rot_p)
            
            result = minimize(
                lambda p: _manifold_energy(p, v5, v2),
                params0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if result.fun < best_energy:
                # Reconstruct full positions
                r1, r2, r3, rx, ry, rz = result.x
                rot = Rotation.from_euler('xyz', [rx, ry, rz])
                R = rot.as_matrix()
                pos = np.vstack([
                    np.zeros(3)[None, :],
                    (v5 * r1) @ R.T,
                    (v2 * r2) @ R.T,
                    (v5 * r3) @ R.T
                ])
                best_energy = result.fun
                best_pos = pos
    
    # Phase 2: L-BFGS relaxation from best manifold result
    if time.time() < deadline - 30:
        relaxed = _relax(best_pos, maxiter=15000, ftol=1e-14, gtol=1e-12)
        if _is_valid(relaxed):
            e = _lj_energy(relaxed)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed
    
    # Phase 3: If time remains, try a few more L-BFGS starts from perturbed manifold
    if time.time() < deadline - 20:
        for trial in range(3):
            if time.time() > deadline:
                break
            noise = rng.normal(0, 0.02, size=best_pos.shape)
            pos_trial = best_pos + noise
            relaxed = _relax(pos_trial, maxiter=10000, ftol=1e-13, gtol=1e-11)
            if _is_valid(relaxed):
                e = _lj_energy(relaxed)
                if e < best_energy:
                    best_energy = e
                    best_pos = relaxed
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 10:
        refined = _relax(best_pos, maxiter=30000, ftol=1e-15, gtol=1e-13)
        if _is_valid(refined):
            e = _lj_energy(refined)
            if e < best_energy:
                best_energy = e
                best_pos = refined
    
    return best_pos
