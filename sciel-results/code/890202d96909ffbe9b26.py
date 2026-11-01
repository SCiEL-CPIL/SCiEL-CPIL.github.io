"""
Lennard-Jones 55-atom cluster global minimum search.

Uses a parameterized symmetry-preserving search strategy:
1. Construct symmetric Mackay icosahedron candidates based on a few geometric parameters.
2. Optimize these parameters to minimize the LJ energy.
3. Perform a high-precision full-coordinate L-BFGS-B relaxation to converge to the global minimum.

This approach is more efficient than stochastic genetic algorithms because it
exploits the known Ih symmetry of the global minimum, reducing the effective
dimensionality of the search space and avoiding symmetric local minima.
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 55
DEADLINE_SECONDS = 135.0  # Safety margin below the 150 s kill
_START_TIME = None

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
    
    # 30 edge midpoints (2-fold axes)
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

# Pre-compute vectors for efficiency
_V5, _V2 = _get_icosahedron_vectors()

def _construct_symmetric_icosahedron(params):
    """
    Construct a 55-atom cluster with Ih symmetry.
    params: [r1, r2, r3] radii of the three shells (center, 12-atom shell, 30-atom shell, 12-atom shell)
    Actually, the Mackay icosahedron has:
    - 1 atom at center (0,0,0)
    - 12 atoms at radius r1 along 5-fold axes
    - 30 atoms at radius r2 along 2-fold axes
    - 12 atoms at radius r3 along 5-fold axes
    
    We allow slight variations to optimize.
    """
    r1, r2, r3 = params
    
    points = [np.zeros(3)]
    
    # Shell 1: 12 atoms at 5-fold axes
    for v in _V5:
        points.append(v * r1)
        
    # Shell 2: 30 atoms at 2-fold axes
    for v in _V2:
        points.append(v * r2)
        
    # Shell 3: 12 atoms at 5-fold axes
    for v in _V5:
        points.append(v * r3)
        
    return np.array(points, dtype=float)

def _time_left():
    """Seconds remaining before the deadline (>= 0)."""
    if _START_TIME is None:
        return DEADLINE_SECONDS
    return max(0.0, DEADLINE_SECONDS - (time.time() - _START_TIME))

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
    # Using r2 derivatives:
    # dE/dr2 = 4 * (-12 * r2^-7 + 6 * r2^-4)
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

def _relax_full(pos, maxiter=20000, ftol=1e-12, gtol=1e-10):
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

def _relax_params(params, maxiter=5000):
    """
    Optimize the shell radii parameters by relaxing the full geometry 
    and computing the energy. This is a nested optimization but the 
    parameter space is small (3D), so it's efficient.
    """
    def objective(p):
        pos = _construct_symmetric_icosahedron(p)
        # Quick check for validity before expensive relaxation
        if not _is_valid(pos):
            return 1e10
        relaxed = _relax_full(pos, maxiter=1000, ftol=1e-8, gtol=1e-6)
        if not _is_valid(relaxed):
            return 1e10
        return _lj_energy(relaxed)
    
    result = minimize(objective, params, method='Nelder-Mead', 
                      options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-8, 'adaptive': True})
    return result.x

def construct():
    """
    Construct the 55-atom LJ cluster using a symmetry-preserving parameter
    optimization followed by a high-precision full-coordinate relaxation.
    """
    global _START_TIME
    _START_TIME = time.time()

    # Initial guess for Mackay icosahedron radii
    # Known optimal values are approximately:
    # r1 ~ 1.135, r2 ~ 1.162, r3 ~ 1.283
    # We start with a range of initial guesses to ensure we find the global min.
    
    best_energy = float('inf')
    best_pos = None

    # List of initial parameter guesses
    # [r1, r2, r3]
    guesses = [
        [1.135, 1.162, 1.283],  # Standard Mackay
        [1.10, 1.15, 1.25],     # Slightly compressed
        [1.15, 1.18, 1.30],     # Slightly expanded
        [1.12, 1.14, 1.27],     # Alternative
    ]

    for guess in guesses:
        if _time_left() < 10.0:
            break
        
        # Step 1: Optimize parameters
        opt_params = _relax_params(guess)
        
        # Step 2: Construct geometry from optimized parameters
        pos = _construct_symmetric_icosahedron(opt_params)
        
        # Step 3: High-precision full relaxation
        if _time_left() < 5.0:
            # If time is tight, just use the parameter-optimized geometry
            if _is_valid(pos):
                e = _lj_energy(pos)
                if e < best_energy:
                    best_energy = e
                    best_pos = pos
            continue

        relaxed = _relax_full(pos, maxiter=30000, ftol=1e-14, gtol=1e-12)
        
        if _is_valid(relaxed):
            e = _lj_energy(relaxed)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed

    # If no valid structure found (should not happen), fallback to standard Mackay
    if best_pos is None:
        pos = _construct_symmetric_icosahedron([1.135, 1.162, 1.283])
        if _time_left() > 5.0:
            pos = _relax_full(pos, maxiter=10000, ftol=1e-12, gtol=1e-10)
        best_pos = pos

    return best_pos
