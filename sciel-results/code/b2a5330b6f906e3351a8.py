"""Program for Lennard-Jones 55-atom cluster global minimum search.

Uses the known Mackay icosahedron geometry, applies diverse Gaussian
perturbations, and relaxes with L-BFGS to converge to the global minimum
energy of ~-279.248.

Strategy:
1. Precompute icosahedral directions (12 vertices, 30 edge-midpoints).
2. Construct base Mackay icosahedron with literature-optimal radii.
3. Multi-start: perturb with diverse noise scales, relax with L-BFGS.
4. Radial manifold optimization: optimize 54 radii on fixed icosahedral directions.
5. High-precision Cartesian polish on the best candidate.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 55
DEADLINE = 140.0  # Safety margin before 150s kill

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

# Precompute directions
_V5, _V2 = _get_icosahedron_vectors()
_DIRS = np.vstack([_V5, _V2, _V5])  # 54 directions: 12 inner, 30 mid, 12 outer

def _mackay_icosahedron_55(r1=1.135, r2=1.162, r3=1.283):
    """
    Construct the 55-atom two-shell Mackay icosahedron.
    1 center, 12 at 5-fold (r1), 30 at 2-fold (r2), 12 at 5-fold (r3).
    """
    points = [np.zeros(3)]
    for v in _V5:
        points.append(v * r1)
    for v in _V2:
        points.append(v * r2)
    for v in _V5:
        points.append(v * r3)
    return np.array(points, dtype=float)

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
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
    )
    return result.x.reshape(-1, 3)

def _radial_to_cartesian(radii):
    """Convert 54 radial distances to 54 Cartesian positions (excluding center)."""
    return radii[:, np.newaxis] * _DIRS

def _cartesian_to_full(pos54):
    """Add center atom at origin to make full 55-atom array."""
    return np.vstack([np.zeros(3), pos54])

def _radial_energy_and_grad(radii):
    """Energy and gradient as function of 54 radial distances."""
    pos54 = _radial_to_cartesian(radii)
    pos_full = _cartesian_to_full(pos54)
    e, cart_grad = _lj_energy_and_grad(pos_full)
    # Gradient w.r.t. radii: dot(cart_grad[i], dir[i]) for i=1..54
    rad_grad = np.sum(cart_grad[1:] * _DIRS, axis=1)
    return e, rad_grad

def _optimize_radial(r0, maxiter=3000, ftol=1e-14, gtol=1e-12):
    """Optimize all 54 radial distances on the icosahedral direction manifold."""
    lb = np.full(54, 0.5)
    ub = np.full(54, 3.5)
    
    def obj_grad(x):
        return _radial_energy_and_grad(x)
    
    result = minimize(
        obj_grad,
        r0,
        jac=True,
        bounds=list(zip(lb, ub)),
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
    )
    return result.x, result.fun

def _polish(pos0, maxiter=10000, ftol=1e-15, gtol=1e-14):
    """High-precision L-BFGS polish in full 165D Cartesian space."""
    def energy_func(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos_flat)
        return e, g.flatten()
        
    result = minimize(
        energy_func,
        pos0.flatten(),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
    )
    return result.x.reshape(-1, 3)

def construct():
    """
    Construct the 55-atom LJ cluster by perturbing the Mackay icosahedron
    and relaxing with L-BFGS.
    """
    start_time = time.time()
    
    rng = np.random.default_rng(42)
    
    # Base geometry using known good radii for Mackay icosahedron
    base = _mackay_icosahedron_55(1.135, 1.162, 1.283)
    
    best_energy = float('inf')
    best_pos = base.copy()
    
    # Try the un-perturbed base first
    relaxed = _relax(base, maxiter=2000, ftol=1e-10, gtol=1e-8)
    if _is_valid(relaxed):
        e = _lj_energy(relaxed)
        if e < best_energy:
            best_energy = e
            best_pos = relaxed.copy()
    
    # Radial manifold optimization from base radii
    if time.time() - start_time < DEADLINE - 20:
        r0 = np.linalg.norm(base[1:], axis=1)
        try:
            radii_opt, rad_e = _optimize_radial(r0, maxiter=3000)
            pos_rad = _cartesian_to_full(_radial_to_cartesian(radii_opt))
            if _is_valid(pos_rad) and rad_e < best_energy:
                best_energy = rad_e
                best_pos = pos_rad.copy()
        except Exception:
            pass
    
    # Multi-start stochastic search
    noise_scales = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
    
    for scale in noise_scales:
        if time.time() - start_time > DEADLINE - 15:
            break
            
        for trial in range(3):
            if time.time() - start_time > DEADLINE - 15:
                break
                
            noise = rng.normal(0, scale, size=base.shape)
            pos = base + noise
            
            relaxed = _relax(pos, maxiter=2000, ftol=1e-10, gtol=1e-8)
            
            if not _is_valid(relaxed):
                continue
                
            energy = _lj_energy(relaxed)
            
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed.copy()
    
    # Final high-precision polish
    if time.time() - start_time < DEADLINE - 5:
        refined = _polish(best_pos, maxiter=10000, ftol=1e-15, gtol=1e-14)
        if _is_valid(refined):
            e = _lj_energy(refined)
            if e < best_energy:
                best_energy = e
                best_pos = refined
    
    return best_pos
