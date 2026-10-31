"""Rewritten program using robust multi-start FCC lattice construction with local relaxation."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
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

def _minimize_lbfgs(pos, max_iter=2000, tol=1e-10):
    """L-BFGS minimization with tight tolerances."""
    try:
        res = minimize(
            _lj_energy,
            pos,
            jac=_lj_grad,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxls': 50}
        )
        return res.x
    except Exception:
        return pos

def _generate_fcc_cluster(n_atoms, r_nn=2**(1/6)):
    """Generate an FCC cluster of n_atoms closest to the origin."""
    scale = r_nn / np.sqrt(2.0)
    max_coord = 10  # Ensure we have enough points
    
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    points.append((i, j, k))
    
    pts = np.array(points, dtype=float)
    pts *= scale
    
    # Sort by distance to origin
    dists = np.sum(pts**2, axis=1)
    sorted_indices = np.argsort(dists)
    
    # Take the first n_atoms points
    cluster = pts[sorted_indices[:n_atoms]].copy()
    
    # Center the cluster at the origin
    center = np.mean(cluster, axis=0)
    cluster -= center
    
    return cluster

def _optimize_fcc_scale(base_cluster, target_energy=-173.928427):
    """Optimize the 1D scale parameter to hit the target energy."""
    def energy_at_scale(s):
        scaled = base_cluster * s
        return _lj_energy(scaled)
    
    # Use golden section search or simple grid + refinement
    # The optimal scale should be near 1.0
    best_s = 1.0
    best_e = energy_at_scale(1.0)
    
    # Grid search first
    for s in np.linspace(0.95, 1.05, 21):
        e = energy_at_scale(s)
        if e < best_e:
            best_e = e
            best_s = s
    
    # Refine around best
    for _ in range(3):
        step = 0.005
        for s in np.linspace(best_s - step, best_s + step, 11):
            e = energy_at_scale(s)
            if e < best_e:
                best_e = e
                best_s = s
        step *= 0.1
    
    return base_cluster * best_s, best_e

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin (150s limit, 35s buffer)
    
    # 1. Generate the ideal FCC truncated octahedron
    base_cluster = _generate_fcc_cluster(N)
    
    # 1a. Optimize the 1D scale to hit the known global minimum energy
    scaled_cluster, scaled_energy = _optimize_fcc_scale(base_cluster)
    best_pos = scaled_cluster.copy()
    best_energy = scaled_energy
    
    # 1b. Full relaxation of the scaled cluster
    relaxed_pos = _minimize_lbfgs(best_pos, max_iter=2000, tol=1e-12)
    relaxed_energy = _lj_energy(relaxed_pos)
    if relaxed_energy < best_energy:
        best_energy = relaxed_energy
        best_pos = relaxed_pos
    
    # 2. Multi-start with perturbations
    rng = np.random.RandomState(42)
    
    # Scales: small to medium
    scales = [0.01, 0.05, 0.1, 0.2, 0.3]
    
    # We will perform up to 4 restarts per scale
    # Total max restarts: 20
    for scale in scales:
        for _ in range(4):
            if time.time() > deadline - 20.0:
                break
            
            perturbed = best_pos + rng.normal(0, scale, size=best_pos.shape)
            
            # Relax the perturbed configuration
            relaxed = _minimize_lbfgs(perturbed, max_iter=1500, tol=1e-12)
            energy = _lj_energy(relaxed)
            
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed
    
    # 3. Final high-precision polish
    # Use a very tight tolerance to ensure we are at the bottom of the well
    if time.time() < deadline - 10.0:
        final_pos = _minimize_lbfgs(best_pos, max_iter=10000, tol=1e-14)
        final_energy = _lj_energy(final_pos)
        if final_energy < best_energy:
            best_pos = final_pos
            best_energy = final_energy

    return best_pos
# EVOLVE-BLOCK-END
