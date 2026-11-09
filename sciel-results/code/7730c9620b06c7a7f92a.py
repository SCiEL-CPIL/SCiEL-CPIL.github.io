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


def _local_relax(positions, n_exp, m_exp, c, maxiter=500):
    """Local relaxation using L-BFGS-B with robust convergence."""
    def fun(x):
        pos = x.reshape(-1, 3)
        e, g = _sc_energy_gradient(pos, n_exp, m_exp, c)
        return e, g.ravel()
    
    try:
        result = minimize(fun, positions.ravel(), method='L-BFGS-B', jac=True,
                          options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100})
        return result.fun, result.x.reshape(-1, 3)
    except Exception:
        return _sc_energy(positions, n_exp, m_exp, c), positions


def _generate_icosahedral_initial(n, rng):
    """Generate initial structure based on icosahedral geometry."""
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices
    ico_verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ]
    ico_verts = np.array(ico_verts) / np.linalg.norm(ico_verts[0])
    
    # Generate shells
    points = []
    shell_radii = [1.0, 1.6, 2.2, 2.8, 3.4, 4.0]
    shell_counts = [12, 30, 60, 120, 240, 480]
    
    for radius, count in zip(shell_radii, shell_counts):
        if len(points) >= n:
            break
        needed = min(count, n - len(points))
        
        # Use Fibonacci sphere for even distribution
        for i in range(needed):
            t = i / max(needed - 1, 1)
            y = 1 - 2 * t
            r = np.sqrt(1 - y**2)
            theta = 2 * np.pi * i / needed
            
            # Perturb slightly
            theta += rng.normal(0, 0.1)
            y += rng.normal(0, 0.05)
            y = np.clip(y, -1, 1)
            r = np.sqrt(1 - y**2)
            
            p = radius * np.array([r * np.cos(theta), y, r * np.sin(theta)])
            points.append(p)
    
    # If still not enough, add more
    while len(points) < n:
        p = rng.uniform(-3, 3, 3)
        points.append(p)
    
    points = np.array(points[:n])
    
    # Center
    center = np.mean(points, axis=0)
    points = points - center
    
    # Add small perturbation
    points += rng.normal(0, 0.1, size=points.shape)
    
    return points


def _generate_fcc_initial(n, rng, spacing=1.4):
    """Generate initial structure from FCC lattice."""
    a = spacing
    points = []
    
    # Generate FCC lattice points in a cube
    size = int(np.ceil(n ** (1/3)) + 2)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                # FCC basis
                basis = [
                    [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]
                ]
                for b in basis:
                    p = (np.array([i, j, k]) + np.array(b)) * a
                    points.append(p)
    
    # Shuffle and take n points
    rng.shuffle(points)
    points = np.array(points[:n])
    
    # Center
    center = np.mean(points, axis=0)
    points = points - center
    
    # Add small perturbation
    points += rng.normal(0, 0.08, size=points.shape)
    
    return points


def _generate_random_initial(n, rng, min_dist=0.5):
    """Generate random initial cluster with minimum distance constraint."""
    radius = 1.5 * n ** (1.0 / 3.0)
    points = []
    max_attempts = n * 1000
    
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(points) == 0 or all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    
    # Fill remaining
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def _coarse_to_fine_basin_hop(positions, n_exp_target, m_exp_target, c_target, rng, time_budget, start_time):
    """
    Coarse-to-fine annealing with basin hopping.
    Dynamically softens the potential and cools the temperature.
    """
    best_e, best_pos = _local_relax(positions, n_exp_target, m_exp_target, c_target, maxiter=200)
    current_pos = best_pos.copy()
    current_e = best_e
    
    # Schedule: (n_exp_frac, m_exp_frac, c_frac, temp_start, temp_end, n_hops, maxiter)
    schedule = [
        # Very soft: explore broadly
        (0.25, 0.25, 0.08, 1.5, 1.0, 20, 80),
        # Soft
        (0.40, 0.40, 0.20, 1.0, 0.7, 25, 100),
        # Medium
        (0.55, 0.55, 0.40, 0.7, 0.4, 30, 120),
        # Getting closer
        (0.70, 0.70, 0.65, 0.4, 0.2, 30, 150),
        # Nearly target
        (0.85, 0.85, 0.85, 0.2, 0.08, 25, 200),
        # Target potential, low temp
        (1.00, 1.00, 1.00, 0.1, 0.02, 20, 250),
        # Final refinement
        (1.00, 1.00, 1.00, 0.05, 0.01, 15, 300),
    ]
    
    for stage_idx, (ne_frac, me_frac, c_frac, t_start, t_end, n_hops, maxiter) in enumerate(schedule):
        if time.time() - start_time > time_budget:
            break
        
        n_exp_s = n_exp_target * ne_frac
        m_exp_s = m_exp_target * me_frac
        c_s = c_target * c_frac
        
        failed_hops = 0
        max_failed = 10
        
        for hop in range(n_hops):
            if time.time() - start_time > time_budget:
                break
            
            t_frac = hop / max(n_hops - 1, 1)
            temp = t_start * (1 - t_frac) + t_end * t_frac
            
            # Adaptive perturbation size
            base_noise = 0.4 + 0.6 * (1 - t_frac)
            noise_scale = temp * base_noise
            
            if failed_hops > max_failed:
                noise_scale *= 1.8
                failed_hops = 0
            
            perturbed = current_pos + rng.normal(0, noise_scale, size=current_pos.shape)
            
            e_new, pos_new = _local_relax(perturbed, n_exp_s, m_exp_s, c_s, maxiter=maxiter)
            
            # Metropolis accept/reject
            delta_e = e_new - current_e
            if delta_e < 0 or rng.random() < np.exp(-delta_e / max(temp, 1e-10)):
                current_pos = pos_new
                current_e = e_new
                failed_hops = 0
            else:
                failed_hops += 1
            
            if e_new < best_e:
                best_e = e_new
                best_pos = pos_new.copy()
    
    return best_e, best_pos


def construct(instance):
    """
    Coarse-to-fine annealing for Sutton-Chen clusters.
    """
    n = instance["n"]
    kind = instance["kind"]
    n_exp_target = instance["n_exp"]
    m_exp_target = instance["m_exp"]
    c_target = instance["c"]
    
    start_time = time.time()
    time_budget = 285.0
    
    # Multiple restarts with diverse initial structures
    n_restarts = 6
    rng = np.random.default_rng(42 + n * 7)
    
    best_overall_e = np.inf
    best_overall_pos = None
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Vary initial structure type
        if restart % 3 == 0:
            pos = _generate_icosahedral_initial(n, rng)
        elif restart % 3 == 1:
            pos = _generate_fcc_initial(n, rng)
        else:
            pos = _generate_random_initial(n, rng, min_dist=0.4)
        
        # Quick initial relaxation
        e_init, pos = _local_relax(pos, n_exp_target, m_exp_target, c_target, maxiter=100)
        
        # Run coarse-to-fine basin hopping
        e_stage, pos = _coarse_to_fine_basin_hop(
            pos, n_exp_target, m_exp_target, c_target,
            rng, time_budget, start_time
        )
        
        # Final precise relaxation
        if time.time() - start_time < time_budget - 15:
            e_final, pos_final = _local_relax(pos, n_exp_target, m_exp_target, c_target, maxiter=500)
            
            if e_final < best_overall_e:
                best_overall_e = e_final
                best_overall_pos = pos_final.copy()
        
        # Early termination check
        if best_overall_e < -50 * n:
            break
    
    pos_final = best_overall_pos
    
    # Fallback
    if pos_final is None or not np.all(np.isfinite(pos_final)):
        rng2 = np.random.default_rng(123 + n)
        pos_final = _generate_random_initial(n, rng2)
        _, pos_final = _local_relax(pos_final, n_exp_target, m_exp_target, c_target, maxiter=300)
    
    # Validate: ensure no atoms too close
    if n > 1:
        for _ in range(20):
            min_dist = np.min(pdist(pos_final))
            if min_dist >= 0.35:
                break
            # Push apart close atoms
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
