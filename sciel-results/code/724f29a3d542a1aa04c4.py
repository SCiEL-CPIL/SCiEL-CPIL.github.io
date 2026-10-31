"""Combined program: parameterized FCC construction with multi-start relaxation."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _lj_energy(pos):
    """Compute total Lennard-Jones energy."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    mask = ~np.eye(len(pos), dtype=bool)
    r2 = r2[mask]
    r6 = r2 ** 3
    r12 = r6 ** 2
    energy = 4.0 * np.sum(1.0 / r12 - 1.0 / r6)
    return energy

def _lj_grad(pos):
    """Compute gradient of total Lennard-Jones energy."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    r2 = np.where(r2 < 1e-12, 1e-12, r2)
    r6 = r2 ** 3
    r12 = r6 ** 2
    coeff = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return grad

def _minimize_lbfgs(pos, max_iter=1000, tol=1e-10):
    """L-BFGS minimization."""
    res = minimize(
        _lj_energy,
        pos,
        jac=_lj_grad,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxls': 50}
    )
    return res.x

def _make_fcc_positions():
    """Generate the 38 FCC truncated octahedron positions (unit scale)."""
    max_coord = 4
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    points.append([i, j, k])
    
    pts = np.array(points, dtype=float)
    
    # Sort by distance to origin
    dists = np.sum(pts**2, axis=1)
    sorted_indices = np.argsort(dists)
    
    # Take the first N points
    cluster = pts[sorted_indices[:N]].copy()
    return cluster

def _param_to_pos(params):
    """Convert parameter vector to Cartesian positions.
    
    params[0]: global scale factor
    params[1]: cubic distortion (x-axis stretch)
    params[2]: cubic distortion (y-axis stretch)
    params[3]: cubic distortion (z-axis stretch)
    params[4:]: per-atom radial offsets (N values)
    """
    scale = params[0]
    dx = 1.0 + 0.05 * params[1]
    dy = 1.0 + 0.05 * params[2]
    dz = 1.0 + 0.05 * params[3]
    
    base = _make_fcc_positions()
    pos = base.copy()
    pos[:, 0] *= dx
    pos[:, 1] *= dy
    pos[:, 2] *= dz
    
    # Apply per-atom radial offsets
    radii = np.sqrt(np.sum(pos**2, axis=1))
    radii = np.where(radii < 1e-10, 1e-10, radii)
    radial_offsets = params[4:4+N]
    pos = pos * (1.0 + radial_offsets / radii)[:, np.newaxis]
    
    # Center of mass to origin
    pos -= np.mean(pos, axis=0)
    
    # Apply global scale
    pos *= scale
    
    return pos

def _param_energy(params):
    """Energy as function of parameters."""
    pos = _param_to_pos(params)
    return _lj_energy(pos)

def _param_grad(params):
    """Gradient of energy w.r.t. parameters (numerical)."""
    eps = 1e-7
    grad = np.zeros_like(params)
    base_e = _param_energy(params)
    for i in range(len(params)):
        params[i] += eps
        e_plus = _param_energy(params)
        params[i] -= 2*eps
        e_minus = _param_energy(params)
        params[i] += eps
        grad[i] = (e_plus - e_minus) / (2*eps)
    return grad

def _validate_geometry(pos, energy):
    """Check if the geometry is close to the theoretical FCC minimum.
    
    The theoretical minimum energy is -173.928427.
    We check if we're within a tight tolerance of this value.
    """
    theoretical_min = -173.928427
    return energy < theoretical_min + 1e-6

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Get base FCC positions
    base_pos = _make_fcc_positions()
    
    # The optimal scale for LJ minimum: r_min = 2^(1/6)
    # In FCC, nearest neighbor distance = scale * sqrt(2) (for unit lattice spacing)
    # So scale = r_min / sqrt(2)
    r_min = 2.0 ** (1.0 / 6.0)
    optimal_scale = r_min / np.sqrt(2.0)
    
    # Initial parameters: optimal scale, no distortion, no radial offsets
    init_params = np.zeros(4 + N)
    init_params[0] = optimal_scale
    
    # Optimize the low-dimensional parameter space
    try:
        res = minimize(
            _param_energy,
            init_params,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        best_params = res.x
    except Exception:
        best_params = init_params
    
    # Convert to Cartesian and do full relaxation
    pos = _param_to_pos(best_params)
    
    try:
        relaxed = _minimize_lbfgs(pos, max_iter=500, tol=1e-10)
        energy_relaxed = _lj_energy(relaxed)
        energy_initial = _lj_energy(pos)
        
        if energy_relaxed < energy_initial:
            pos = relaxed
    except Exception:
        pass
    
    if time.time() > deadline:
        return pos
    
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    # Try additional parameter optimizations with different initial conditions
    rng = np.random.RandomState(42)
    
    for attempt in range(15):
        if time.time() > deadline - 15.0:
            break
        
        # Perturb parameters
        perturbed_params = best_params.copy()
        perturbed_params[0] *= (1.0 + rng.normal(0, 0.02))
        perturbed_params[1:4] += rng.normal(0, 0.1, size=3)
        perturbed_params[4:] += rng.normal(0, 0.01, size=N)
        
        try:
            res = minimize(
                _param_energy,
                perturbed_params,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 30}
            )
            candidate_params = res.x
            candidate_pos = _param_to_pos(candidate_params)
            
            # Full relaxation
            relaxed = _minimize_lbfgs(candidate_pos, max_iter=300, tol=1e-10)
            energy = _lj_energy(relaxed)
            
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed
                best_params = candidate_params
        except Exception:
            continue
    
    # Final high-precision polish with tighter tolerance
    if time.time() < deadline - 5.0:
        try:
            polished = _minimize_lbfgs(best_pos, max_iter=2000, tol=1e-12)
            energy_polished = _lj_energy(polished)
            if energy_polished < best_energy:
                best_pos = polished
                best_energy = energy_polished
        except Exception:
            pass
    
    # Validate and return
    return best_pos
# EVOLVE-BLOCK-END
