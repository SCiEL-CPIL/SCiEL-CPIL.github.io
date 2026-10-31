"""LJ38 global minimum search: basin hopping with multi-start and deadline guard."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

N = 38
# Deadline: evaluator kills at 150s; use 120s as safe limit
DEADLINE = 120.0

def lj_energy(pos):
    """Compute total LJ energy for N atoms."""
    d = pdist(pos)
    # Avoid division by zero
    d = np.maximum(d, 1e-10)
    r6 = d ** 6
    return 4.0 * np.sum(d ** -12 - d ** -6)

def lj_energy_and_grad(pos):
    """Compute LJ energy and gradient."""
    pos = np.asarray(pos, dtype=float)
    n = pos.shape[0]
    # Pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (n, n, 3)
    r2 = np.sum(diff ** 2, axis=2)  # (n, n)
    np.fill_diagonal(r2, 1.0)  # avoid zero
    r = np.sqrt(r2)
    r6 = r ** 6
    r12 = r6 ** 2
    # Energy: 4 * sum_{i<j} (r^-12 - r^-6)
    # Using full matrix, each pair counted twice
    eij = 4.0 * (r12 ** -1 - r6 ** -1)
    np.fill_diagonal(eij, 0.0)
    energy = 0.5 * np.sum(eij)
    # Gradient: dE/dr * dr/dpos
    # dE/dr = 4 * (-12 * r^-13 + 6 * r^-7) = 24 * r^-7 * (r^-6 - 2)
    # Actually: d/dr [4(r^-12 - r^-6)] = 4(-12 r^-13 + 6 r^-7) = -48 r^-13 + 24 r^-7
    # = 24 r^-7 (1 - 2 r^-6)
    coeff = 24.0 * (r ** -7) * (1.0 - 2.0 * r ** -6)
    np.fill_diagonal(coeff, 0.0)
    # grad[i] = sum_j coeff[i,j] * (pos[i] - pos[j]) / r[i,j]
    # coeff already includes 1/r factor? Let's be careful.
    # E = sum_{i<j} 4(r_ij^-12 - r_ij^-6)
    # dE/dpos_i = sum_{j != i} 4 * (-12 r_ij^-14 + 6 r_ij^-8) * (pos_i - pos_j) / r_ij
    # = sum_{j != i} (48 r_ij^-15 - 24 r_ij^-9) * (pos_i - pos_j)
    # = sum_{j != i} 24 r_ij^-9 * (2 r_ij^-6 - 1) * (pos_i - pos_j)
    # Let's use the standard formula:
    # dE/dpos_i = 48 * sum_{j != i} (r_ij^-14 - 0.5 * r_ij^-8) * (pos_i - pos_j)
    # Actually let me just use the clean form:
    # dE/dr_ij = 4 * (-12 * r_ij^-13 + 6 * r_ij^-7) = 24 * r_ij^-7 * (1 - 2*r_ij^-6)
    # dE/dpos_i = dE/dr_ij * (pos_i - pos_j)/r_ij = 24 * r_ij^-8 * (1 - 2*r_ij^-6) * (pos_i - pos_j)
    # Wait, dE/dr * dr/dpos_i = dE/dr * (pos_i - pos_j)/r
    # So dE/dpos_i = 24 * r^-7 * (1-2*r^-6) * (pos_i - pos_j) / r = 24 * r^-8 * (1-2*r^-6) * (pos_i - pos_j)
    
    # Let me recompute more carefully
    # E_pair = 4*(r^-12 - r^-6)
    # dE_pair/dr = 4*(-12*r^-13 + 6*r^-7) = -48*r^-13 + 24*r^-7
    # dE/dpos_i = dE_pair/dr * (pos_i - pos_j)/r = (-48*r^-13 + 24*r^-7) * (pos_i-pos_j)/r
    # = (-48*r^-14 + 24*r^-8) * (pos_i - pos_j)
    # = 24*r^-8 * (1 - 2*r^-6) * (pos_i - pos_j)
    
    # Using full matrix approach:
    # grad[i] = sum_{j!=i} 24*r_ij^-8 * (1 - 2*r_ij^-6) * (pos_i - pos_j)
    
    # coeff2[i,j] = 24 * r_ij^-8 * (1 - 2*r_ij^-6) for i != j
    r_safe = np.where(r > 1e-10, r, 1e-10)
    r6_safe = r_safe ** 6
    coeff2 = 24.0 * r_safe ** -8 * (1.0 - 2.0 * r6_safe ** -1)
    np.fill_diagonal(coeff2, 0.0)
    
    # grad[i] = sum_j coeff2[i,j] * (pos[i] - pos[j])
    # = sum_j coeff2[i,j] * pos[i] - sum_j coeff2[i,j] * pos[j]
    # = pos[i] * sum_j coeff2[i,j] - sum_j coeff2[i,j] * pos[j]
    grad = np.einsum('ij,jk->ik', coeff2, pos)
    grad -= pos * np.sum(coeff2, axis=1, keepdims=True)
    # Wait, that's not right. Let me think again.
    # grad[i] = sum_j coeff2[i,j] * (pos[i] - pos[j])
    # = sum_j coeff2[i,j] * pos[i] - sum_j coeff2[i,j] * pos[j]
    # = pos[i] * (sum_j coeff2[i,j]) - sum_j (coeff2[i,j] * pos[j])
    
    # The second term: (coeff2 @ pos)[i] = sum_j coeff2[i,j] * pos[j]
    # So grad[i] = pos[i] * row_sum[i] - (coeff2 @ pos)[i]
    
    row_sum = np.sum(coeff2, axis=1)
    grad = pos * row_sum[:, np.newaxis] - coeff2 @ pos
    
    return energy, grad

def basin_hop_step(pos, step_size, rng):
    """Perform one basin hopping step: perturb then relax."""
    # Random perturbation
    noise = rng.normal(0, step_size, size=pos.shape)
    perturbed = pos + noise
    
    # Relax using L-BFGS
    result = minimize(
        lj_energy_and_grad,
        perturbed,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-8}
    )
    return result.x, result.fun

def construct():
    start_time = time.time()
    best_energy = np.inf
    best_pos = None
    rng = np.random.RandomState(42)
    
    # Initial guess: FCC truncated octahedron approximation
    # FCC lattice with spacing ~1.12
    a = 1.12
    # FCC basis positions in unit cell
    fcc_basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ])
    
    # Generate FCC points in a cube and pick closest to center
    side = 5
    all_points = []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                for b in fcc_basis:
                    pt = np.array([i, j, k]) + b
                    all_points.append(pt * a)
    all_points = np.array(all_points)
    
    # Center of the cube
    center = np.array([side * a / 2, side * a / 2, side * a / 2])
    dists = np.sum((all_points - center) ** 2, axis=1)
    idx = np.argsort(dists)[:N]
    initial_pos = all_points[idx].copy()
    
    # Relax the initial position
    result = minimize(
        lj_energy_and_grad,
        initial_pos,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-10}
    )
    best_pos = result.x
    best_energy = result.fun
    
    # Basin hopping
    step_sizes = [0.5, 0.4, 0.3, 0.2, 0.15, 0.1]
    current_pos = best_pos.copy()
    current_energy = best_energy
    
    # Multi-start with different initial conditions
    n_restarts = 0
    while time.time() - start_time < DEADLINE - 5:
        # Try different step sizes
        for step_size in step_sizes:
            if time.time() - start_time > DEADLINE - 5:
                break
            try:
                new_pos, new_energy = basin_hop_step(current_pos, step_size, rng)
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pos = new_pos.copy()
                    current_pos = new_pos.copy()
                    current_energy = new_energy
                else:
                    # Metropolis acceptance: accept with probability exp(-(E_new - E_old)/T)
                    # Use a simple cooling schedule
                    T = 0.5 * max(1.0, abs(best_energy) / 100.0)
                    delta_e = new_energy - current_energy
                    if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                        current_pos = new_pos.copy()
                        current_energy = new_energy
            except Exception:
                continue
        
        # Occasionally try a fresh random restart
        n_restarts += 1
        if n_restarts % 3 == 0 and time.time() - start_time < DEADLINE - 10:
            # Random perturbation of best
            noise = rng.normal(0, 0.8, size=best_pos.shape)
            perturbed = best_pos + noise
            try:
                result = minimize(
                    lj_energy_and_grad,
                    perturbed,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 300, 'gtol': 1e-7, 'ftol': 1e-9}
                )
                if result.fun < best_energy:
                    best_energy = result.fun
                    best_pos = result.x.copy()
            except Exception:
                continue
    
    # Final relaxation of best
    if time.time() - start_time < DEADLINE - 2:
        try:
            result = minimize(
                lj_energy_and_grad,
                best_pos,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'gtol': 1e-10, 'ftol': 1e-12}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.copy()
        except Exception:
            pass
    
    return best_pos

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
