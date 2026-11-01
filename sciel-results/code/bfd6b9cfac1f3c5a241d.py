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
TIME_LIMIT = 145.0  # Safety margin for 150s limit

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

def _relax(pos, maxiter=5000, ftol=1e-12, gtol=1e-10, time_budget=None):
    """Relax positions using L-BFGS-B with analytical gradient and optional time budget."""
    if time_budget is None:
        time_budget = max(1.0, _time_remaining() * 0.8)
    
    start = time.time()
    
    def energy_func(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos_flat)
        return e, g.flatten()
    
    def callback(xk):
        if time.time() - start > time_budget:
            raise RuntimeError("time_budget_exceeded")
        return False
    
    try:
        result = minimize(
            energy_func,
            pos.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol},
            callback=callback
        )
        return result.x.reshape(-1, 3)
    except RuntimeError:
        # Return the last known position if time budget exceeded
        # We need to track the best position during optimization
        # For simplicity, return the input if we can't track intermediate states
        # Actually, let's use a different approach: reduce maxiter dynamically
        pass
    
    # Fallback: run with reduced iterations
    elapsed = time.time() - start
    remaining = max(0.5, time_budget - elapsed)
    # Estimate iterations per second from what we've done
    # Just use a conservative maxiter
    reduced_maxiter = max(100, int(maxiter * remaining / max(time_budget, 1e-8)))
    result = minimize(
        energy_func,
        pos.flatten(),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': reduced_maxiter, 'ftol': ftol, 'gtol': gtol}
    )
    return result.x.reshape(-1, 3)

def _optimize_radii(radii, maxiter_inner=500, ftol_inner=1e-6, gtol_inner=1e-5):
    """Optimize the three shell radii by relaxing the full geometry."""
    def objective(r):
        # Check time
        if _time_remaining() < 10.0:
            return 1e10
        pos = _mackay_icosahedron_55(r[0], r[1], r[2])
        # Use faster relaxation for the objective function of radii optimization
        relaxed = _relax(pos, maxiter=maxiter_inner, ftol=ftol_inner, gtol=gtol_inner,
                        time_budget=5.0)
        if not _is_valid(relaxed):
            return 1e10
        return _lj_energy(relaxed)
    
    # Nelder-Mead is good for low-dimensional black-box optimization
    # Limit total time spent here to ~20s
    result = minimize(objective, radii, method='Nelder-Mead', 
                      options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True})
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
    if _time_remaining() > 25.0:
        optimal_radii = _optimize_radii(initial_radii)
    else:
        optimal_radii = initial_radii
        
    # Construct base geometry with optimized radii
    base = _mackay_icosahedron_55(*optimal_radii)
    
    best_energy = float('inf')
    best_pos = base.copy()
    
    # Strategy 1: Relax the exact optimized geometry with a tight time budget
    if _time_remaining() > 8.0:
        relaxed = _relax(base, maxiter=3000, ftol=1e-10, gtol=1e-8, time_budget=8.0)
        if _is_valid(relaxed):
            e = _lj_energy(relaxed)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
                
    # Strategy 2: Multi-start with diverse perturbations
    # Use adaptive time allocation: spend less time per trial to fit more trials
    # Prioritize small perturbations that stay in the basin of attraction
    
    # Define perturbation schedule: (noise_scale, radial_factor, time_budget_per_trial)
    # Small perturbations get more time, larger ones get less
    perturbations = [
        # (noise_scale, radial_scale, time_budget)
        (0.0,  1.0,   6.0),   # Exact optimized geometry
        (0.01, 1.0,   4.0),   # Very small noise
        (0.02, 1.0,   4.0),   # Small noise
        (0.05, 1.0,   3.0),   # Moderate noise
        (0.0,  0.98,  3.0),   # Radial compression
        (0.0,  1.02,  3.0),   # Radial expansion
        (0.05, 0.98,  3.0),   # Combined
        (0.05, 1.02,  3.0),   # Combined
        (0.1,  1.0,   2.5),   # Larger noise
        (0.1,  0.95,  2.5),   # Larger noise + compression
        (0.1,  1.05,  2.5),   # Larger noise + expansion
        (0.2,  1.0,   2.0),   # Large noise
        (0.2,  0.95,  2.0),   # Large noise + compression
        (0.2,  1.05,  2.0),   # Large noise + expansion
        (0.5,  1.0,   1.5),   # Very large noise
        (0.5,  0.9,   1.5),   # Very large noise + strong compression
        (0.5,  1.1,   1.5),   # Very large noise + strong expansion
    ]
    
    for noise_scale, radial_scale, tb in perturbations:
        if _time_remaining() < tb + 2.0:
            break
            
        r1 = optimal_radii[0] * radial_scale
        r2 = optimal_radii[1] * radial_scale
        r3 = optimal_radii[2] * radial_scale
        pos = _mackay_icosahedron_55(r1, r2, r3)
        
        if noise_scale > 0:
            noise = rng.normal(0, noise_scale, size=pos.shape)
            pos = pos + noise
        
        relaxed = _relax(pos, maxiter=2000, ftol=1e-9, gtol=1e-7, time_budget=tb)
        
        if not _is_valid(relaxed):
            continue
            
        e = _lj_energy(relaxed)
        
        if e < best_energy:
            best_energy = e
            best_pos = relaxed.copy()

    # Strategy 3: Random perturbations with varying scales to explore more broadly
    for i in range(15):
        if _time_remaining() < 4.0:
            break
        
        # Random radial perturbation per shell
        r1_pert = optimal_radii[0] * (1 + rng.normal(0, 0.03))
        r2_pert = optimal_radii[1] * (1 + rng.normal(0, 0.03))
        r3_pert = optimal_radii[2] * (1 + rng.normal(0, 0.03))
        pos = _mackay_icosahedron_55(r1_pert, r2_pert, r3_pert)
        
        # Random noise scale between 0.01 and 0.15
        noise_scale = rng.uniform(0.01, 0.15)
        noise = rng.normal(0, noise_scale, size=pos.shape)
        pos = pos + noise
        
        tb = 2.5
        if _time_remaining() < tb + 2.0:
            break
            
        relaxed = _relax(pos, maxiter=1500, ftol=1e-9, gtol=1e-7, time_budget=tb)
        
        if not _is_valid(relaxed):
            continue
            
        e = _lj_energy(relaxed)
        
        if e < best_energy:
            best_energy = e
            best_pos = relaxed.copy()

    # Final fine-tuning pass with tighter tolerances on the best candidate
    if best_energy < float('inf') and _time_remaining() > 8.0:
        # Use very tight tolerances for the final polish
        tb_final = min(15.0, _time_remaining() - 3.0)
        refined = _relax(best_pos, maxiter=5000, ftol=1e-14, gtol=1e-12, time_budget=tb_final)
        if _is_valid(refined):
            e = _lj_energy(refined)
            if e < best_energy:
                best_energy = e
                best_pos = refined

    return best_pos
