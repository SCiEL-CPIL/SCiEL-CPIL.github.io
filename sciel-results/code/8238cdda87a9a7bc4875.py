import numpy as np
import time
import math
from scipy.optimize import minimize, minimize_scalar

N = 38
DEADLINE = 145.0  # Safety margin before 150s kill
START_TIME = time.time()

def lj_energy_vectorized(positions):
    """Vectorized LJ energy computation."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            if np.dot(rij, rij) < min_dist * min_dist:
                return False
    return True

def _get_fcc_shell_structure(n_atoms=38):
    """
    Return base FCC positions and shell assignments for LJ38 truncated octahedron.
    The FCC lattice is generated, sorted by distance from origin, and the first n_atoms are taken.
    Shells are identified by clustering the radial distances.
    """
    scale = 1.0 / math.sqrt(2.0)
    max_coord = 8
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = i * scale, j * scale, k * scale
                    r2 = x*x + y*y + z*z
                    points.append((x, y, z, r2))
    
    # Sort by distance from origin
    points.sort(key=lambda p: p[3])
    selected = [p[:3] for p in points[:n_atoms]]
    coords = np.array(selected, dtype=float)
    
    # Assign shells based on radial distance
    radii = np.linalg.norm(coords, axis=1)
    # Cluster radii into shells with a tolerance
    unique_r = np.sort(np.unique(np.round(radii, 4)))
    shell_ids = np.zeros(n_atoms, dtype=int)
    for i, r in enumerate(radii):
        for sid, ur in enumerate(unique_r):
            if abs(r - ur) < 0.05:
                shell_ids[i] = sid
                break
    return coords, shell_ids, unique_r

def make_fcc_truncated_octahedron(n_atoms=38):
    """Generate the standard FCC truncated octahedron structure."""
    coords, _, _ = _get_fcc_shell_structure(n_atoms)
    # Scale to approximate equilibrium distance
    coords *= 2.0 ** (1.0 / 6.0)
    return coords

def _lj_energy_and_grad(pos_flat):
    """Compute LJ energy and gradient vectorized."""
    pos = pos_flat.reshape(N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    
    inv_r14 = inv_r12 * inv_r2
    inv_r8 = inv_r6 * inv_r2
    coeff = -24.0 * inv_r14 + 12.0 * inv_r8
    
    grad_matrix = coeff[:, :, np.newaxis] * diff
    grad_flat = np.sum(grad_matrix, axis=1).flatten()
    
    return energy, grad_flat

def _symmetry_adapted_energy_and_grad(params):
    """
    Compute energy and gradient for symmetry-adapted parameters.
    params: [global_scale, shell_breathing_0, shell_breathing_1, ...]
    """
    n_atoms = N
    global_scale = params[0]
    shell_params = params[1:]
    
    base_coords, shell_ids, unique_r = _get_fcc_shell_structure(n_atoms)
    n_shells = len(unique_r)
    
    # Construct positions from parameters
    # pos_i = global_scale * (1 + shell_breathing[shell_i]) * base_coords_i
    positions = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        s = shell_ids[i]
        if s < n_shells:
            breath = 1.0 + shell_params[s]
        else:
            breath = 1.0
        positions[i] = base_coords[i] * breath * global_scale
    
    # Compute energy
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    
    # Compute gradient numerically for robustness in low-dim space
    # This is fast enough for ~10 parameters
    eps = 1e-7
    grad = np.zeros_like(params)
    for k in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[k] += eps
        params_minus[k] -= eps
        
        # Helper to get energy for a set of params
        def _calc_energy(p):
            gs = p[0]
            sp = p[1:]
            pos = np.zeros((n_atoms, 3))
            for i in range(n_atoms):
                s = shell_ids[i]
                if s < n_shells:
                    b = 1.0 + sp[s]
                else:
                    b = 1.0
                pos[i] = base_coords[i] * b * gs
            
            d = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            r2_calc = np.sum(d * d, axis=-1)
            np.fill_diagonal(r2_calc, 1.0)
            r2_calc = np.maximum(r2_calc, 1e-12)
            ir2 = 1.0 / r2_calc
            ir6 = ir2 ** 3
            ir12 = ir6 ** 2
            return 4.0 * np.sum(ir12 - ir6) / 2.0
        
        e_plus = _calc_energy(params_plus)
        e_minus = _calc_energy(params_minus)
        grad[k] = (e_plus - e_minus) / (2 * eps)
        
    return energy, grad

def optimize_symmetry_adapted():
    """Optimize the low-dimensional symmetry-adapted parameterization."""
    base_coords, shell_ids, unique_r = _get_fcc_shell_structure(N)
    n_shells = len(unique_r)
    
    # Initial guess: scale = 2^(1/6), all breathing = 0
    x0 = np.array([2.0 ** (1.0/6.0)] + [0.0] * n_shells)
    
    try:
        result = minimize(_symmetry_adapted_energy_and_grad, x0, method='L-BFGS-B', 
                          jac=True,
                          options={'maxiter': 500, 'gtol': 1e-10, 'maxls': 100, 'ftol': 1e-14})
        if result.success:
            # Reconstruct positions from optimized params
            params = result.x
            global_scale = params[0]
            shell_params = params[1:]
            positions = np.zeros((N, 3))
            for i in range(N):
                s = shell_ids[i]
                if s < n_shells:
                    breath = 1.0 + shell_params[s]
                else:
                    breath = 1.0
                positions[i] = base_coords[i] * breath * global_scale
            return positions
    except Exception:
        pass
    
    # Fallback: return standard FCC
    return make_fcc_truncated_octahedron(N)

def relax(positions, maxiter=300, gtol=1e-8):
    """Relax using L-BFGS-B with analytic gradient."""
    x0 = positions.flatten()
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B', 
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 50})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def polish(positions, maxiter=500, gtol=1e-10):
    """High-precision final polish pass with analytic gradient."""
    x0 = positions.flatten()
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B', 
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def ultra_polish(positions, maxiter=2000, gtol=1e-14, ftol=1e-16):
    """Ultra-high-precision polish pass for final energy minimization."""
    x0 = positions.flatten()
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B', 
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol, 'maxls': 200})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_energy = float('inf')
    best_positions = None
    
    # 1. Start with symmetry-adapted optimization
    # This finds the optimal "breathing" of the FCC shells, which is a very good approximation
    # of the global minimum without the risk of full-coordinate optimization getting stuck.
    if time.time() - START_TIME < DEADLINE - 30:
        try:
            sym_pos = optimize_symmetry_adapted()
            if check_validity(sym_pos):
                energy = lj_energy_vectorized(sym_pos)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = sym_pos
        except Exception:
            pass
    
    # 2. If symmetry optimization failed or didn't produce a valid result, fallback to standard FCC
    if best_positions is None:
        try:
            fcc = make_fcc_truncated_octahedron(N)
            if check_validity(fcc):
                energy = lj_energy_vectorized(fcc)
                best_energy = energy
                best_positions = fcc
        except Exception:
            pass
    
    # 3. Full-coordinate relaxation from the best starting point
    # This allows the atoms to move slightly out of perfect shell symmetry if needed,
    # though for LJ38 the symmetry is very strong.
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 20:
        try:
            relaxed = relax(best_positions, maxiter=300, gtol=1e-9)
            if check_validity(relaxed):
                energy = lj_energy_vectorized(relaxed)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed
        except Exception:
            pass
    
    # 4. High-precision polish
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 15:
        try:
            polished = polish(best_positions, maxiter=500, gtol=1e-12)
            if check_validity(polished):
                energy = lj_energy_vectorized(polished)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = polished
        except Exception:
            pass
            
    # 5. Ultra-high-precision polish
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 10:
        try:
            ultra = ultra_polish(best_positions, maxiter=1000, gtol=1e-14)
            if check_validity(ultra):
                energy = lj_energy_vectorized(ultra)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = ultra
        except Exception:
            pass

    # Final check and return
    if best_positions is None:
        # Fallback: generate a simple cubic grid if everything else fails
        side = int(np.ceil(N ** (1/3)))
        pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        pts.sort(key=lambda p: (p[0]-(side-1)/2)**2 + (p[1]-(side-1)/2)**2 + (p[2]-(side-1)/2)**2)
        best_positions = np.array(pts[:N], dtype=float) * 1.2

    return best_positions
