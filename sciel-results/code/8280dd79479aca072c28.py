"""Program for Lennard-Jones 55-atom cluster global minimum search.

Uses the known Mackay icosahedron geometry, optimizes shell radii via Nelder-Mead,
applies diverse perturbations (Gaussian, radial), and relaxes with L-BFGS to converge
to the global minimum energy of ~-279.248.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 55
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin for 150s limit

def _time_remaining():
    return TIME_LIMIT - (time.time() - START_TIME)

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

def _lj_energy_and_grad(pos):
    """Compute total Lennard-Jones energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    mask = r2 > 1e-10
    r2_safe = np.where(mask, r2, 1.0)
    
    # r^-6 and r^-12
    r6 = r2_safe ** (-3)
    r12 = r6 ** 2
    
    # Energy: 4 * sum(r^-12 - r^-6)
    energy = 4 * np.sum(np.where(mask, r12 - r6, 0))
    
    # Gradient:
    # dE/dx_i = sum_j 4 * (-12*r^-14 + 6*r^-8) * (x_i - x_j)
    # Note: r^-14 = r2^-7, r^-8 = r2^-4
    r2_inv4 = r2_safe ** (-4)
    r2_inv7 = r2_safe ** (-7)
    
    coeff = 4 * (-12 * r2_inv7 + 6 * r2_inv4)
    coeff = np.where(mask, coeff, 0)
    
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad

def _lj_energy(pos):
    """Compute total Lennard-Jones energy."""
    e, _ = _lj_energy_and_grad(pos)
    return e

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

def _optimize_radii(radii, maxiter_inner=1000, ftol_inner=1e-8, gtol_inner=1e-6):
    """Optimize the three shell radii by relaxing the full geometry."""
    def objective(r):
        # Check time
        if _time_remaining() < 5.0:
            return 1e10
        pos = _mackay_icosahedron_55(r[0], r[1], r[2])
        # Use faster relaxation for the objective function of radii optimization
        relaxed = _relax(pos, maxiter=maxiter_inner, ftol=ftol_inner, gtol=gtol_inner)
        if not _is_valid(relaxed):
            return 1e10
        return _lj_energy(relaxed)
    
    # Nelder-Mead is good for low-dimensional black-box optimization
    result = minimize(objective, radii, method='Nelder-Mead', 
                      options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-8, 'adaptive': True})
    return result.x

def construct():
    """
    Construct the 55-atom LJ cluster by perturbing the Mackay icosahedron
    and relaxing with L-BFGS.
    """
    rng = np.random.default_rng(42)
    
    # Step 1: Optimize the shell radii to find the precise Mackay geometry
    # Start with known approximate values
    initial_radii = np.array([1.135, 1.162, 1.283])
    
    # Only optimize if we have enough time
    if _time_remaining() > 30.0:
        optimal_radii = _optimize_radii(initial_radii)
    else:
        optimal_radii = initial_radii
        
    # Construct base geometry with optimized radii
    base = _mackay_icosahedron_55(*optimal_radii)
    
    best_energy = float('inf')
    best_pos = base.copy()
    
    # Strategy 1: Relax the exact optimized geometry
    if _time_remaining() > 10.0:
        relaxed = _relax(base)
        if _is_valid(relaxed):
            e = _lj_energy(relaxed)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
                
    # Strategy 2: Multi-start with perturbations
    # We try diverse noise scales and radial perturbations
    # to ensure robustness against local minima or symmetry breaking issues.
    
    # Pre-define perturbation strategies
    noise_scales = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    radial_perts = [1.0, 0.98, 1.02, 0.95, 1.05]
    
    # Combine strategies in a loop, checking time
    # We prioritize small perturbations first as they are likely to stay in the basin
    attempts = 0
    max_attempts = 50
    
    for scale in noise_scales:
        for rp in radial_perts:
            if _time_remaining() < 5.0:
                break
            if attempts >= max_attempts:
                break
                
            # Perturb radii slightly
            r1 = optimal_radii[0] * rp
            r2 = optimal_radii[1] * rp
            r3 = optimal_radii[2] * rp
            pos = _mackay_icosahedron_55(r1, r2, r3)
            
            # Add Gaussian noise
            if scale > 0:
                noise = rng.normal(0, scale, size=pos.shape)
                pos = pos + noise
            else:
                # If scale is 0, just use the radial perturbed geometry
                pass
                
            # Relax
            relaxed = _relax(pos)
            
            if not _is_valid(relaxed):
                continue
                
            e = _lj_energy(relaxed)
            
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
                
            attempts += 1
        if _time_remaining() < 5.0:
            break

    # Strategy 3: Random radial perturbations (more diverse)
    for i in range(20):
        if _time_remaining() < 5.0:
            break
            
        r1_pert = optimal_radii[0] * (1 + rng.normal(0, 0.05))
        r2_pert = optimal_radii[1] * (1 + rng.normal(0, 0.05))
        r3_pert = optimal_radii[2] * (1 + rng.normal(0, 0.05))
        pos = _mackay_icosahedron_55(r1_pert, r2_pert, r3_pert)
        
        # Add small Gaussian noise
        noise = rng.normal(0, 0.02, size=pos.shape)
        pos = pos + noise
        
        relaxed = _relax(pos)
        
        if not _is_valid(relaxed):
            continue
            
        e = _lj_energy(relaxed)
        
        if e < best_energy:
            best_energy = e
            best_pos = relaxed.copy()

    # Final fine-tuning pass with tighter tolerances on the best candidate
    if best_energy < float('inf') and _time_remaining() > 5.0:
        # Use very tight tolerances for the final polish
        refined = _relax(best_pos, maxiter=20000, ftol=1e-15, gtol=1e-13)
        if _is_valid(refined):
            e = _lj_energy(refined)
            if e < best_energy:
                best_energy = e
                best_pos = refined

    return best_pos
