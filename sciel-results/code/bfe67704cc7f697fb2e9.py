"""Coarse-to-fine annealing for Sutton-Chen transition-metal clusters."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import time


def _sc_energy_gradient(positions, n_exp, m_exp, c):
    """Compute Sutton-Chen energy and gradient with given parameters."""
    n = positions.shape[0]
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff**2, axis=-1))
    mask = r > 1e-10
    r_safe = np.where(mask, r, 1.0)
    
    rn = r_safe ** (-n_exp)
    rm = r_safe ** (-m_exp)
    rn[~mask] = 0
    rm[~mask] = 0
    
    phi_i = np.sum(rn, axis=1)
    psi_i = np.sum(rm, axis=1)
    
    E_rep = 0.5 * np.sum(phi_i)
    E_att = -c * np.sum(np.sqrt(psi_i))
    energy = E_rep + E_att
    
    rn2 = r_safe ** (-(n_exp + 2))
    rm2 = r_safe ** (-(m_exp + 2))
    rn2[~mask] = 0
    rm2[~mask] = 0
    
    sum_rn2 = np.sum(rn2, axis=1)
    dot_rn2_x = rn2 @ positions
    
    grad_rep = -0.5 * n_exp * (positions * sum_rn2[:, np.newaxis] - dot_rn2_x)
    
    sum_rm2 = np.sum(rm2, axis=1)
    dot_rm2_x = rm2 @ positions
    
    sqrt_psi = np.sqrt(np.maximum(psi_i, 1e-10))
    grad_att = c * m_exp / (2.0 * sqrt_psi[:, np.newaxis]) * (positions * sum_rm2[:, np.newaxis] - dot_rm2_x)
    
    gradient = grad_rep + grad_att
    
    return energy, gradient


def _sc_energy(positions, n_exp, m_exp, c):
    """Compute Sutton-Chen energy only."""
    n = positions.shape[0]
    if n < 2:
        return 0.0
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff**2, axis=-1))
    mask = r > 1e-10
    r_safe = np.where(mask, r, 1.0)
    
    rn = r_safe ** (-n_exp)
    rm = r_safe ** (-m_exp)
    rn[~mask] = 0
    rm[~mask] = 0
    
    phi_i = np.sum(rn, axis=1)
    psi_i = np.sum(rm, axis=1)
    
    E_rep = 0.5 * np.sum(phi_i)
    E_att = -c * np.sum(np.sqrt(psi_i))
    return E_rep + E_att


def _local_relax(positions, n_exp, m_exp, c, maxiter=300):
    """Local relaxation using L-BFGS-B."""
    def fun(x):
        pos = x.reshape(-1, 3)
        e, g = _sc_energy_gradient(pos, n_exp, m_exp, c)
        return e, g.ravel()
    
    try:
        result = minimize(fun, positions.ravel(), method='L-BFGS-B', jac=True,
                          options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
        return result.fun, result.x.reshape(-1, 3)
    except:
        return _sc_energy(positions, n_exp, m_exp, c), positions


def _generate_fcc_initial(n, rng, spacing=1.4):
    """Generate initial structure from FCC lattice with random perturbations."""
    # Generate FCC lattice points
    a = spacing
    points = []
    
    # FCC unit cell positions
    fcc_offsets = [
        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1.5, 0.5, 0], [1.5, 0, 0.5], [1.5, 0.5, 0.5],
        [0.5, 1, 0], [0, 1, 0.5], [0.5, 1, 0.5],
        [0.5, 0.5, 1], [0, 0.5, 1], [0.5, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1],
        [1.5, 1.5, 0], [1.5, 1, 0.5], [1.5, 1.5, 0.5],
        [1, 1.5, 0], [1, 1, 0.5], [1, 1.5, 0.5],
        [1, 1, 1],
    ]
    
    # Scale and collect points
    for offset in fcc_offsets:
        p = np.array(offset) * a
        points.append(p)
    
    # Add more points if needed by extending the lattice
    if len(points) < n:
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for offset in [[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]:
                        p = (np.array(offset) + np.array([i, j, k])) * a
                        points.append(p)
    
    # Shuffle and take n points
    rng.shuffle(points)
    points = points[:n]
    
    # Center the structure
    center = np.mean(points, axis=0)
    points = np.array(points) - center
    
    # Add small random perturbations to break symmetry
    points += rng.normal(0, 0.05, size=points.shape)
    
    return points


def _generate_random_initial(n, rng, spacing=1.2):
    """Generate random initial cluster with minimum distance constraint."""
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    min_dist = 0.6 * spacing
    attempts = 0
    max_attempts = n * 500
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    
    # Fill remaining with random points
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def _adaptive_basin_hop(positions, n_exp, m_exp, c, temp_start, temp_end, n_hops, maxiter_per_relax, rng, time_budget, start_time):
    """
    Basin hopping with adaptive temperature and perturbation size.
    """
    best_e, best_pos = _local_relax(positions, n_exp, m_exp, c, maxiter=maxiter_per_relax)
    current_pos = best_pos
    current_e = best_e
    
    failed_hops = 0
    max_failed = 15
    
    for hop in range(n_hops):
        if time.time() - start_time > time_budget:
            break
        
        # Adaptive temperature: decrease over hops
        t_frac = hop / max(n_hops - 1, 1)
        temp = temp_start * (1 - t_frac) + temp_end * t_frac
        
        # Adaptive perturbation: start large, decrease
        base_noise = 0.5 + 0.5 * (1 - t_frac)
        noise_scale = temp * base_noise
        
        # If many failed hops, increase perturbation
        if failed_hops > max_failed:
            noise_scale *= 1.5
            failed_hops = 0
        
        perturbed = current_pos + rng.normal(0, noise_scale, size=current_pos.shape)
        
        # Relax
        e_new, pos_new = _local_relax(perturbed, n_exp, m_exp, c, maxiter=maxiter_per_relax)
        
        # Metropolis accept/reject
        delta_e = e_new - current_e
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(temp, 1e-10)):
            current_pos = pos_new
            current_e = e_new
            failed_hops = 0
        else:
            failed_hops += 1
        
        # Track best
        if e_new < best_e:
            best_e = e_new
            best_pos = pos_new.copy()
    
    return best_e, best_pos


def construct(instance):
    """
    Coarse-to-fine annealing for Sutton-Chen clusters with adaptive basin hopping.
    """
    n = instance["n"]
    kind = instance["kind"]
    n_exp_target = instance["n_exp"]
    m_exp_target = instance["m_exp"]
    c_target = instance["c"]
    
    start_time = time.time()
    time_budget = 285.0  # Leave 15s margin
    
    # Coarse-to-fine schedule: (n_exp_frac, m_exp_frac, c_frac, temp_start, temp_end, n_hops, maxiter)
    schedule = [
        # Very soft potential, high temperature - explore broadly
        (0.30, 0.30, 0.10, 1.2, 0.8, 15, 50),
        # Softening still active
        (0.45, 0.45, 0.25, 0.7, 0.5, 18, 60),
        # Medium soft
        (0.60, 0.60, 0.50, 0.5, 0.3, 20, 80),
        # Getting closer to target
        (0.75, 0.75, 0.75, 0.3, 0.15, 20, 100),
        # Nearly target
        (0.90, 0.90, 0.90, 0.15, 0.05, 18, 120),
        # Target potential, low temperature - refine
        (1.00, 1.00, 1.00, 0.08, 0.02, 15, 150),
    ]
    
    # Multiple restarts with different initial structures
    n_restarts = 5
    rng = np.random.default_rng(42 + n)
    
    best_overall_e = np.inf
    best_overall_pos = None
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Alternate between FCC-based and random initial structures
        if restart % 2 == 0:
            pos = _generate_fcc_initial(n, rng)
        else:
            pos = _generate_random_initial(n, rng)
        
        # Quick initial relaxation with target parameters
        e_init, pos = _local_relax(pos, n_exp_target, m_exp_target, c_target, maxiter=80)
        
        # Run coarse-to-fine schedule
        for stage_idx, (ne_frac, me_frac, c_frac, t_start, t_end, n_hops, maxiter) in enumerate(schedule):
            if time.time() - start_time > time_budget:
                break
            
            n_exp_s = n_exp_target * ne_frac
            m_exp_s = m_exp_target * me_frac
            c_s = c_target * c_frac
            
            e_stage, pos = _adaptive_basin_hop(
                pos, n_exp_s, m_exp_s, c_s,
                t_start, t_end, n_hops, maxiter, rng,
                time_budget, start_time
            )
        
        # Final precise relaxation with target parameters
        if time.time() - start_time < time_budget - 10:
            e_final, pos_final = _local_relax(pos, n_exp_target, m_exp_target, c_target, maxiter=400)
            
            if e_final < best_overall_e:
                best_overall_e = e_final
                best_overall_pos = pos_final.copy()
        
        # Early termination if we find a very good solution
        if best_overall_e < -1000 * n:  # Heuristic: very deep minimum
            break
    
    pos_final = best_overall_pos
    
    # Fallback if no valid result
    if pos_final is None or not np.all(np.isfinite(pos_final)):
        rng2 = np.random.default_rng(123 + n)
        pos_final = _generate_random_initial(n, rng2)
        e_final, pos_final = _local_relax(pos_final, n_exp_target, m_exp_target, c_target, maxiter=300)
    
    # Validate: check no atoms too close
    if n > 1:
        min_dist = np.min(pdist(pos_final))
        if min_dist < 0.35:
            # Push apart atoms that are too close
            for _ in range(50):
                diff = pos_final[:, np.newaxis, :] - pos_final[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=-1))
                mask = (dists < 0.35) & (dists > 1e-10)
                if not np.any(mask):
                    break
                for i in range(n):
                    for j in range(i + 1, n):
                        if dists[i, j] < 0.35 and dists[i, j] > 1e-10:
                            direction = diff[i, j] / dists[i, j]
                            push = (0.35 - dists[i, j]) * 0.5
                            pos_final[i] += direction * push
                            pos_final[j] -= direction * push
                _, pos_final = _local_relax(pos_final, n_exp_target, m_exp_target, c_target, maxiter=100)
    
    # Final check
    if n > 1:
        min_dist = np.min(pdist(pos_final))
        if min_dist < 0.35:
            rng3 = np.random.default_rng(999 + n)
            pos_final = _generate_random_initial(n, rng3)
            _, pos_final = _local_relax(pos_final, n_exp_target, m_exp_target, c_target, maxiter=300)
    
    # Ensure finite
    if not np.all(np.isfinite(pos_final)):
        rng4 = np.random.default_rng(777 + n)
        pos_final = _generate_random_initial(n, rng4)
        _, pos_final = _local_relax(pos_final, n_exp_target, m_exp_target, c_target, maxiter=200)
    
    return pos_final
# EVOLVE-BLOCK-END
