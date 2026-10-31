import numpy as np
import time
import math
from scipy.optimize import minimize, minimize_scalar

N = 38
DEADLINE = 145.0
START_TIME = time.time()

def lj_energy(positions):
    """Compute total Lennard-Jones energy: 4 * sum(r^-12 - r^-6) over all pairs."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = np.dot(rij, rij)
            if r2 < 1e-12:
                return 1e20
            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 * inv_r2 * inv_r2
            inv_r12 = inv_r6 * inv_r6
            energy += 4.0 * (inv_r12 - inv_r6)
    return energy

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
            r2 = np.dot(rij, rij)
            if r2 < min_dist * min_dist:
                return False
    return True

def make_fcc_truncated_octahedron(n_atoms=38):
    """
    Generate the FCC truncated octahedron cluster for n_atoms atoms.
    This is the known global minimum structure for LJ38.
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
    
    # Select the first n_atoms points
    selected = [p[:3] for p in points[:n_atoms]]
    coords = np.array(selected, dtype=float)
    
    # Scale to equilibrium distance 2^(1/6)
    coords *= 2.0 ** (1.0 / 6.0)
    
    return coords

def _lj_energy_and_grad(pos_flat):
    """Compute LJ energy and gradient vectorized."""
    pos = pos_flat.reshape(N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    
    # Avoid division by zero
    r2 = np.maximum(r2, 1e-12)
    
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    # Energy
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    
    # Gradient
    inv_r14 = inv_r12 * inv_r2
    inv_r8 = inv_r6 * inv_r2
    
    coeff = -24.0 * inv_r14 + 12.0 * inv_r8
    grad_matrix = coeff[:, :, np.newaxis] * diff
    grad_flat = np.sum(grad_matrix, axis=1).flatten()
    
    return energy, grad_flat

def relax(positions, maxiter=500, gtol=1e-10):
    """Relax using L-BFGS-B with analytic gradient."""
    x0 = positions.flatten()
    
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B', 
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-14})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def polish(positions, maxiter=1000, gtol=1e-12):
    """High-precision final polish pass with analytic gradient."""
    x0 = positions.flatten()
    
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B', 
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 200, 'ftol': 1e-15})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def optimize_scale(base_positions):
    """
    Optimize the global scale factor for the given base positions.
    """
    def energy_at_scale(s):
        scaled = base_positions * s
        return lj_energy_vectorized(scaled)
    
    try:
        result = minimize_scalar(energy_at_scale, bounds=(0.9, 1.2), method='bounded',
                                 options={'xatol': 1e-12, 'maxiter': 300})
        if result.success:
            return base_positions * result.x
    except Exception:
        pass
    
    # Fallback
    best_energy = float('inf')
    best_scaled = base_positions
    for s in np.linspace(0.95, 1.15, 41):
        scaled = base_positions * s
        e = lj_energy_vectorized(scaled)
        if e < best_energy:
            best_energy = e
            best_scaled = scaled
            
    return best_scaled

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_energy = float('inf')
    best_positions = None
    
    # 1. Primary Candidate: Ideal FCC with optimized scale
    try:
        fcc = make_fcc_truncated_octahedron(N)
        if check_validity(fcc):
            fcc_scaled = optimize_scale(fcc)
            if check_validity(fcc_scaled):
                relaxed = relax(fcc_scaled, maxiter=500, gtol=1e-10)
                if check_validity(relaxed):
                    energy = lj_energy_vectorized(relaxed)
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = relaxed
    except Exception:
        pass
    
    # 2. Secondary Candidate: Noisy FCC (breaks symmetry, robust against local traps near FCC)
    if time.time() - START_TIME < DEADLINE - 30:
        try:
            rng = np.random.default_rng(42)
            fcc_noisy = make_fcc_truncated_octahedron(N)
            fcc_noisy += rng.normal(0, 0.05, fcc_noisy.shape)
            if check_validity(fcc_noisy):
                relaxed = relax(fcc_noisy, maxiter=500, gtol=1e-10)
                if check_validity(relaxed):
                    energy = lj_energy_vectorized(relaxed)
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = relaxed
        except Exception:
            pass

    # 3. Tertiary Candidate: FCC without explicit scale opt (let 3D optimizer handle it)
    if time.time() - START_TIME < DEADLINE - 20:
        try:
            fcc_raw = make_fcc_truncated_octahedron(N)
            if check_validity(fcc_raw):
                relaxed = relax(fcc_raw, maxiter=800, gtol=1e-10)
                if check_validity(relaxed):
                    energy = lj_energy_vectorized(relaxed)
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = relaxed
        except Exception:
            pass

    # 4. Final High-Precision Polish
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 10:
        try:
            polished = polish(best_positions, maxiter=1000, gtol=1e-12)
            if check_validity(polished):
                energy = lj_energy_vectorized(polished)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = polished
        except Exception:
            pass

    # Fallback if all failed
    if best_positions is None:
        try:
            best_positions = make_fcc_truncated_octahedron(N)
        except Exception:
            best_positions = np.zeros((N, 3))
            for i in range(N):
                best_positions[i] = [i*0.6, (i%3)*0.6, (i//3)%3*0.6]

    return best_positions
