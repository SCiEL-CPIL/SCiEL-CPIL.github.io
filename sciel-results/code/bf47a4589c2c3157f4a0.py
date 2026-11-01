import numpy as np
from scipy.optimize import minimize
import time

N = 55
TIME_LIMIT = 145.0  # Safety margin for 150s limit

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

def construct():
    """
    Construct the 55-atom LJ cluster by perturbing the Mackay icosahedron
    and relaxing with L-BFGS.
    """
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Standard Mackay icosahedron radii
    r1, r2, r3 = 1.135, 1.162, 1.283
    base = _mackay_icosahedron_55(r1, r2, r3)
    
    best_energy = float('inf')
    best_pos = base.copy()
    
    def try_relax(pos, maxiter=5000, ftol=1e-12, gtol=1e-10):
        nonlocal best_energy, best_pos
        if time.time() - start_time > TIME_LIMIT - 5:
            return
        
        relaxed = _relax(pos, maxiter=maxiter, ftol=ftol, gtol=gtol)
        
        if not _is_valid(relaxed):
            return
            
        e = _lj_energy(relaxed)
        if e < best_energy:
            best_energy = e
            best_pos = relaxed.copy()
            
    # 1. Relax the exact Mackay geometry
    try_relax(base, maxiter=10000, ftol=1e-14, gtol=1e-12)
    
    # 2. Multi-start with Gaussian noise
    # We use a range of noise scales to explore the energy landscape
    # Small noise refines, large noise escapes local minima
    noise_scales = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    
    for scale in noise_scales:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        # Multiple trials per scale to improve robustness
        for _ in range(5):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            noise = rng.normal(0, scale, size=base.shape)
            pos = base + noise
            try_relax(pos, maxiter=5000, ftol=1e-12, gtol=1e-10)
            
    # 3. Radial perturbations
    # Perturb the shell radii to find better packing
    for _ in range(20):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        r1_p = r1 * (1 + rng.normal(0, 0.05))
        r2_p = r2 * (1 + rng.normal(0, 0.05))
        r3_p = r3 * (1 + rng.normal(0, 0.05))
        pos = _mackay_icosahedron_55(r1_p, r2_p, r3_p)
        # Add small noise to break symmetry
        noise = rng.normal(0, 0.01, size=pos.shape)
        pos = pos + noise
        try_relax(pos, maxiter=5000, ftol=1e-12, gtol=1e-10)
        
    # 4. Angular perturbations
    for _ in range(10):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        pos = base.copy()
        for i in range(1, len(pos)):
            angle = rng.normal(0, 0.05)
            axis = rng.normal(size=3)
            axis = axis / (np.linalg.norm(axis) + 1e-10)
            c = np.cos(angle)
            s = np.sin(angle)
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R = c * np.eye(3) + s * K + (1 - c) * np.outer(axis, axis)
            pos[i] = R @ pos[i]
        try_relax(pos, maxiter=5000, ftol=1e-12, gtol=1e-10)
        
    # 5. Final high-precision refinement of the best candidate
    if best_energy < float('inf') and time.time() - start_time < TIME_LIMIT - 5:
        refined = _relax(best_pos, maxiter=50000, ftol=1e-15, gtol=1e-13)
        if _is_valid(refined):
            e = _lj_energy(refined)
            if e < best_energy:
                best_energy = e
                best_pos = refined

    return best_pos
