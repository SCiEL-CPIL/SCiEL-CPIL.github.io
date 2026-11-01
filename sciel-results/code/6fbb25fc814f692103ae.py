"""Program for Lennard-Jones 55-atom cluster global minimum search.

Uses a hierarchical multi-scale approach: first rigid-body optimization
(translate + rotate) to find the global minimum basin, then full Cartesian
L-BFGS relaxation for fine-grained polishing. Multi-start with Gaussian
noise around the Mackay icosahedron geometry.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

N = 55
_DEADLINE = 150.0  # seconds
_START_TIME = time.time()

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

def _mackay_icosahedron_55(r1=1.135, r2=1.162, r3=1.283):
    """
    Construct the 55-atom two-shell Mackay icosahedron.
    1 center, 12 at 5-fold (r1), 30 at 2-fold (r2), 12 at 5-fold (r3).
    """
    v5, v2 = _get_icosahedron_vectors()
    
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

def _lj_energy_and_grad(pos):
    """Compute total Lennard-Jones energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    mask = r2 > 1e-10
    r2_safe = np.where(mask, r2, 1.0)
    r6 = r2_safe ** (-3)
    r12 = r6 ** 2
    energy = 4 * np.sum(np.where(mask, r12 - r6, 0))
    
    # Gradient: dE/dx_i = sum_j 4*(-12*r^-14 + 6*r^-8) * (x_i - x_j)/r
    # Simplified: grad = 4 * sum_j (-12*r2^-7 + 6*r2^-4) * diff
    r2_inv4 = r2_safe ** (-4)
    r2_inv7 = r2_safe ** (-7)
    coeff = 4 * (-12 * r2_inv7 + 6 * r2_inv4)
    coeff = np.where(mask, coeff, 0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad

def _is_valid(pos):
    """Check if all inter-atomic distances are >= 0.5."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.5

def _relax(pos, maxiter=5000, ftol=1e-12, gtol=1e-10):
    """Relax positions using L-BFGS-B with analytical gradient."""
    def energy_func(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos_flat)
        return e, g.flatten()
        
    result = minimize(
        energy_func,
        pos.flatten(),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
    )
    return result.x.reshape(-1, 3)

def _rigid_body_relax(pos, maxiter=200, ftol=1e-10, gtol=1e-8):
    """
    Hierarchical step 1: Optimize only translation (3 DOF) and rotation
    (3 DOF, as Euler angles) of the entire cluster as a rigid body.
    This reduces the effective dimensionality from 165 to 6, making it
    much easier to find the global minimum basin.
    """
    # Center of mass
    com = np.mean(pos, axis=0)
    centered = pos - com
    
    def rigid_energy(params):
        """params = [tx, ty, tz, rx, ry, rz]"""
        tx, ty, tz = params[0], params[1], params[2]
        rx, ry, rz = params[3], params[4], params[5]
        rot = Rotation.from_euler('xyz', [rx, ry, rz])
        rotated = rot.apply(centered)
        translated = rotated + com + np.array([tx, ty, tz])
        e = _lj_energy(translated)
        return e
    
    # Initial parameters: zero translation, zero rotation
    x0 = np.zeros(6)
    result = minimize(
        rigid_energy,
        x0,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
    )
    
    # Apply the optimal rigid transformation
    tx, ty, tz = result.x[0], result.x[1], result.x[2]
    rx, ry, rz = result.x[3], result.x[4], result.x[5]
    rot = Rotation.from_euler('xyz', [rx, ry, rz])
    rotated = rot.apply(centered)
    translated = rotated + com + np.array([tx, ty, tz])
    
    return translated

def construct():
    """
    Construct the 55-atom LJ cluster using hierarchical multi-scale search:
    1. Rigid-body optimization (translate + rotate) to find global basin
    2. Full Cartesian L-BFGS relaxation for fine-grained polishing
    Multi-start with Gaussian noise around the Mackay icosahedron geometry.
    """
    rng = np.random.default_rng(42)
    
    # Generate the exact Mackay icosahedron geometry
    base = _mackay_icosahedron_55(r1=1.135, r2=1.162, r3=1.283)
    
    best_energy = float('inf')
    best_pos = base.copy()
    
    def _time_remaining():
        return _DEADLINE - (time.time() - _START_TIME)
    
    # First, try the exact geometry with hierarchical relaxation
    if _time_remaining() > 5:
        # Step 1: Rigid body relaxation
        rigid_relaxed = _rigid_body_relax(base)
        # Step 2: Full Cartesian relaxation
        relaxed = _relax(rigid_relaxed)
        if _is_valid(relaxed):
            e = _lj_energy(relaxed)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
    
    # Multi-start stochastic search with hierarchical relaxation
    noise_scales = [
        0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5
    ]
    
    for scale in noise_scales:
        if _time_remaining() < 15:
            break
        for trial in range(5):
            if _time_remaining() < 10:
                break
            
            noise = rng.normal(0, scale, size=base.shape)
            pos = base + noise
            
            # Hierarchical: rigid body first, then full Cartesian
            if _time_remaining() > 8:
                rigid_relaxed = _rigid_body_relax(pos)
            else:
                rigid_relaxed = pos
            
            if _time_remaining() < 5:
                break
                
            relaxed = _relax(rigid_relaxed)
            
            if not _is_valid(relaxed):
                continue
                
            energy = _lj_energy(relaxed)
            
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed.copy()
    
    # Final high-precision relaxation
    if best_energy < float('inf') and _time_remaining() > 5:
        refined = _relax(best_pos, maxiter=20000, ftol=1e-14, gtol=1e-12)
        if _is_valid(refined):
            e = _lj_energy(refined)
            if e < best_energy:
                best_energy = e
                best_pos = refined
                
    return best_pos
