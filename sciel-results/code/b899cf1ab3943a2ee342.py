"""Program for Lennard-Jones 55-atom cluster global minimum search.

Uses the known Mackay icosahedron geometry with optimized shell radii,
applies diverse perturbations (Gaussian, radial, angular), and relaxes
with L-BFGS to converge to the global minimum energy of ~-279.248.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 55
_DEADLINE = 140.0  # seconds, safety margin before 150s kill
_START_TIME = None

def _elapsed():
    """Return elapsed time since start."""
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME

def _time_remaining():
    """Return time remaining before deadline."""
    return max(0.0, _DEADLINE - _elapsed())

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
    energy = 4.0 * np.sum(np.where(mask, r12 - r6, 0.0))
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
    energy = 4.0 * np.sum(np.where(mask, r12 - r6, 0.0))
    
    # Gradient: dE/dx_i = sum_j 4*(-12*r^-14 + 6*r^-8) * (x_i - x_j)/r
    # Simplified: grad = 4 * sum_j (-12*r2^-7 + 6*r2^-4) * diff
    r2_inv4 = r2_safe ** (-4)
    r2_inv7 = r2_safe ** (-7)
    coeff = 4.0 * (-12.0 * r2_inv7 + 6.0 * r2_inv4)
    coeff = np.where(mask, coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad

def _is_valid(pos):
    """Check if all inter-atomic distances are >= 0.5."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    return np.min(r2) >= 0.25

def _relax(pos, maxiter=10000, ftol=1e-12, gtol=1e-10):
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

def _optimize_radii(r1_init, r2_init, r3_init, base_dirs):
    """
    Optimize the three shell radii to minimize LJ energy of the Mackay geometry.
    base_dirs: tuple (v5, v2) of unit direction vectors.
    """
    v5, v2 = base_dirs
    
    def objective(radii):
        r1, r2, r3 = radii
        points = [np.zeros(3)]
        for v in v5:
            points.append(v * r1)
        for v in v2:
            points.append(v * r2)
        for v in v5:
            points.append(v * r3)
        pos = np.array(points)
        return _lj_energy(pos)
    
    # Use Nelder-Mead for radius optimization (small 3D problem)
    result = minimize(
        objective,
        np.array([r1_init, r2_init, r3_init]),
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
    )
    
    return result.x

def construct():
    """
    Construct the 55-atom LJ cluster by perturbing the Mackay icosahedron
    and relaxing with L-BFGS.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    rng = np.random.default_rng(42)
    
    v5, v2 = _get_icosahedron_vectors()
    
    # Initial guess for radii
    r1_init = 1.14
    r2_init = 1.18
    r3_init = 1.29
    
    # Step 1: Optimize radii to get a better starting geometry
    if _time_remaining() > 10.0:
        optimal_radii = _optimize_radii(r1_init, r2_init, r3_init, (v5, v2))
        r1_opt, r2_opt, r3_opt = optimal_radii
    else:
        r1_opt, r2_opt, r3_opt = r1_init, r2_init, r3_init
    
    base = _mackay_icosahedron_55(r1_opt, r2_opt, r3_opt)
    
    best_energy = float('inf')
    best_pos = base.copy()
    
    # Step 2: Relax the base geometry
    if _time_remaining() > 5.0:
        relaxed = _relax(base, maxiter=10000)
        if _is_valid(relaxed):
            e = _lj_energy(relaxed)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
    
    # Step 3: Multi-start with diverse perturbations
    if _time_remaining() > 10.0:
        # Radial perturbations (vary shell radii)
        radial_perturbations = [
            (1.0, 1.0, 1.0),   # uniform scale
            (1.02, 1.0, 1.0),  # inner shell out
            (0.98, 1.0, 1.0),  # inner shell in
            (1.0, 1.02, 1.0),  # middle shell out
            (1.0, 0.98, 1.0),  # middle shell in
            (1.0, 1.0, 1.02),  # outer shell out
            (1.0, 1.0, 0.98),  # outer shell in
        ]
        
        for scale_f1, scale_f2, scale_f3 in radial_perturbations:
            if _time_remaining() < 5.0:
                break
            pos = base.copy()
            # Apply radial scaling per shell
            for i in range(1, 13):  # inner 5-fold shell
                pos[i] *= scale_f1
            for i in range(13, 43):  # 2-fold shell
                pos[i] *= scale_f2
            for i in range(43, 55):  # outer 5-fold shell
                pos[i] *= scale_f3
            
            if _is_valid(pos):
                relaxed = _relax(pos, maxiter=5000)
                if _is_valid(relaxed):
                    energy = _lj_energy(relaxed)
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed.copy()
    
    # Step 4: Gaussian noise multi-start
    if _time_remaining() > 10.0:
        noise_scales = [0.02, 0.05, 0.1, 0.15, 0.2]
        for scale in noise_scales:
            if _time_remaining() < 5.0:
                break
            for trial in range(2):
                if _time_remaining() < 3.0:
                    break
                noise = rng.normal(0, scale, size=base.shape)
                pos = base + noise
                
                if not _is_valid(pos):
                    continue
                    
                relaxed = _relax(pos, maxiter=5000)
                
                if not _is_valid(relaxed):
                    continue
                
                energy = _lj_energy(relaxed)
                
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed.copy()
    
    # Step 5: Multi-stage high-precision polish
    if best_energy < float('inf') and _time_remaining() > 5.0:
        # Stage 1: moderate tolerance
        if _time_remaining() > 10.0:
            polished = _relax(best_pos, maxiter=10000, ftol=1e-14, gtol=1e-12)
            if _is_valid(polished):
                e = _lj_energy(polished)
                if e < best_energy:
                    best_energy = e
                    best_pos = polished.copy()
        
        # Stage 2: tight tolerance
        if _time_remaining() > 5.0:
            polished = _relax(best_pos, maxiter=10000, ftol=1e-16, gtol=1e-14)
            if _is_valid(polished):
                e = _lj_energy(polished)
                if e < best_energy:
                    best_energy = e
                    best_pos = polished.copy()
    
    return best_pos
