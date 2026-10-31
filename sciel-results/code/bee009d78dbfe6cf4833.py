"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 140s safety margin out of 150s

def _remaining_time():
    return _DEADLINE - time.time()


def lj_energy_and_grad(pos):
    """Compute LJ energy and gradient for N atoms."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=-1)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r2 = np.maximum(r2, 1e-12)
    
    r6 = r2**(-3)
    r12 = r6**2
    
    # Energy: 4 * sum_{i<j} (r^-12 - r^-6)
    # The full matrix sum (excluding diagonal) gives 2*E
    full_sum = np.sum(4.0 * (r12 - r6))
    energy = 0.5 * full_sum
    
    # Gradient computation
    r2_safe = np.maximum(r2, 1e-12)
    coeff = -24.0 * r2_safe**(-7) + 12.0 * r2_safe**(-4)
    np.fill_diagonal(coeff, 0.0)
    
    row_sum = coeff.sum(axis=1, keepdims=True)
    grad = 2.0 * (row_sum * pos - coeff @ pos)
    
    return energy, grad


def _make_fcc_cluster(n_atoms, a=None):
    """Generate FCC lattice points and select the n_atoms closest to center."""
    if a is None:
        a = 2**(1/3)  # FCC lattice constant for LJ minimum (nearest neighbor = 1)
    
    # FCC lattice: points at (i,j,k) where i+j+k is even, scaled by a/2
    max_coord = 10
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x = i * a / 2
                    y = j * a / 2
                    z = k * a / 2
                    points.append([x, y, z])
    
    points = np.array(points)
    # Remove duplicates
    points = np.unique(points, axis=0)
    # Sort by distance from center
    dists = np.sum(points**2, axis=1)
    idx = np.argsort(dists)
    return points[idx[:n_atoms]]


def _enforce_min_sep(pos, min_sep=0.5):
    """Ensure all pairs are at least min_sep apart."""
    pos = pos.copy()
    for _ in range(200):
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.sum(diff**2, axis=-1)
        np.fill_diagonal(r2, 1.0)
        min_r2 = np.min(r2)
        if min_r2 >= min_sep**2 - 1e-10:
            break
        # Find violating pairs and push apart
        for i in range(pos.shape[0]):
            for j in range(i + 1, pos.shape[0]):
                if r2[i, j] < min_sep**2 - 1e-10:
                    diff_vec = pos[i] - pos[j]
                    r_ij = np.sqrt(r2[i, j])
                    if r_ij > 1e-10:
                        push = (min_sep - r_ij) / r_ij
                        pos[i] += 0.5 * push * diff_vec
                        pos[j] -= 0.5 * push * diff_vec
                    else:
                        pos[i] += min_sep / 2
        pos = pos - pos.mean(axis=0)
    return pos


def _relax(pos, maxiter=500, gtol=1e-8, ftol=1e-10):
    """Relax positions using L-BFGS."""
    try:
        res = minimize(lj_energy_and_grad, pos.copy(), method='L-BFGS-B',
                      jac=True, options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol})
        return res.x, res.fun
    except Exception:
        return pos.copy(), np.inf


def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 140.0
    
    best_energy = np.inf
    best_pos = None
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: Start from FCC truncated octahedron and relax
    fcc_pos = _make_fcc_cluster(N)
    fcc_pos = _enforce_min_sep(fcc_pos.copy())
    
    if _remaining_time() > 10:
        pos_relaxed, e_relaxed = _relax(fcc_pos.copy(), maxiter=1000, gtol=1e-10, ftol=1e-12)
        pos_relaxed = _enforce_min_sep(pos_relaxed.copy())
        e_check, _ = lj_energy_and_grad(pos_relaxed)
        if e_check < best_energy:
            best_energy = e_check
            best_pos = pos_relaxed
    
    if best_pos is None:
        best_pos = fcc_pos.copy()
        best_energy, _ = lj_energy_and_grad(best_pos)
    
    # Strategy 2: Basin hopping with random perturbations
    current_pos = best_pos.copy()
    current_energy = best_energy
    
    n_basin_steps = 0
    while _remaining_time() > 30 and n_basin_steps < 30:
        n_basin_steps += 1
        # Random perturbation with decreasing step size
        step_size = 0.5 * (1.0 - n_basin_steps / 30.0) + 0.1
        perturbed = current_pos + rng.normal(0, step_size, size=current_pos.shape)
        perturbed = _enforce_min_sep(perturbed)
        
        pos_relaxed, e_relaxed = _relax(perturbed, maxiter=300, gtol=1e-6, ftol=1e-8)
        pos_relaxed = _enforce_min_sep(pos_relaxed.copy())
        e_check, _ = lj_energy_and_grad(pos_relaxed)
        
        if e_check < current_energy - 1e-10:
            current_energy = e_check
            current_pos = pos_relaxed
            best_energy = current_energy
            best_pos = current_pos.copy()
    
    # Final polish with tighter tolerance
    if _remaining_time() > 15:
        pos_final, e_final = _relax(best_pos.copy(), maxiter=2000, gtol=1e-12, ftol=1e-14)
        pos_final = _enforce_min_sep(pos_final.copy())
        e_check, _ = lj_energy_and_grad(pos_final)
        if e_check < best_energy:
            best_energy = e_check
            best_pos = pos_final
    
    # Final check: ensure minimum separation
    best_pos = _enforce_min_sep(best_pos.copy())
    
    return best_pos
# EVOLVE-BLOCK-END
