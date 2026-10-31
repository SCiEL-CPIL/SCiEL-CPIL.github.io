"""Basin-hopping with analytical gradients for LJ38 global minimum."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

DEADLINE = 145.0  # Safety margin before 150s kill


def _lj_energy_and_grad(pos):
    """
    Compute LJ energy and gradient for N atoms.
    pos: (N, 3) array
    Returns (energy, grad) where grad is (N, 3).
    Energy = 4 * sum_{i<j} (r_ij^-12 - r_ij^-6)
    """
    n = pos.shape[0]
    # Pairwise displacement vectors: (N, N, 3)
    diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]  # diff[i,j] = pos[j] - pos[i]
    r2 = np.sum(diff * diff, axis=2)  # (N, N)
    # Avoid division by zero on diagonal
    np.fill_diagonal(r2, 1.0)
    
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    # Pair energy matrix (symmetric, diagonal is zero after fill)
    pair_energy = 4.0 * (inv_r12 - inv_r6)
    np.fill_diagonal(pair_energy, 0.0)
    
    energy = 0.5 * np.sum(pair_energy)
    
    # Gradient: dE/dpos[i] = sum_{j != i} dE/dpos[i]
    # dE_ij/dpos[i] = 4 * (-12 * r^-14 * (pos[i]-pos[j])/r + 6 * r^-8 * (pos[i]-pos[j])/r)
    #                = 4 * (12 * inv_r12 - 6 * inv_r6) / r2 * (pos[i] - pos[j])
    # Actually: dE/dpos[i] for pair (i,j):
    #   E_ij = 4*(r^-12 - r^-6), r^2 = |pos_i - pos_j|^2
    #   dE_ij/dpos_i = 4*(-12*r^-14 * (pos_i-pos_j)/r + 6*r^-8*(pos_i-pos_j)/r)
    #   = 4*(pos_i - pos_j) * (-12*inv_r12 + 6*inv_r6) / r
    #   = 4*(pos_i - pos_j) * (6*inv_r6 - 12*inv_r12) / r
    #   = 24*(pos_i - pos_j) * (inv_r6 - 2*inv_r12) / r
    # Since diff[i,j] = pos[j] - pos[i] = -(pos_i - pos_j):
    #   dE_ij/dpos_i = -24*(diff[i,j]) * (inv_r6 - 2*inv_r12) / r
    #   = 24*(pos_i - pos_j) * (2*inv_r12 - inv_r6) / r
    # Let's use: grad[i] = sum_j 24 * (pos[i] - pos[j]) * (2*inv_r12 - inv_r6) / r
    # In terms of diff: pos[i]-pos[j] = -diff[i,j], r = sqrt(r2)
    # grad[i] = sum_j 24 * (-diff[i,j]) * (2*inv_r12 - inv_r6) / sqrt(r2)
    #         = sum_j 24 * diff[i,j] * (inv_r6 - 2*inv_r12) / sqrt(r2)
    
    # Coefficient for each pair direction
    # coeff[i,j] = 24 * (inv_r6 - 2*inv_r12) / sqrt(r2)
    r = np.sqrt(r2)
    np.fill_diagonal(r, 1.0)
    coeff = 24.0 * (inv_r6 - 2.0 * inv_r12) / r  # (N, N)
    np.fill_diagonal(coeff, 0.0)
    
    # grad[i] = sum_j coeff[i,j] * diff[i,j]
    # diff[i,j] = pos[j] - pos[i]
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad


def _lj_energy_only(pos):
    """Fast energy-only computation."""
    n = pos.shape[0]
    diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    pair_energy = 4.0 * (inv_r12 - inv_r6)
    np.fill_diagonal(pair_energy, 0.0)
    return 0.5 * np.sum(pair_energy)


def _check_validity(pos, min_dist=0.5):
    """Check minimum distance constraint."""
    diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    return min_r2 >= min_dist * min_dist


def _make_fcc_truncated_octahedron():
    """
    Construct FCC truncated octahedron for 38 atoms.
    
    The FCC truncated octahedron (Voronoi cell of FCC) has 24 vertices.
    For 38 atoms, the known global minimum structure is an FCC-based cluster.
    
    Construction: Take FCC lattice points (i+j+k even in cubic grid with spacing 1/sqrt(2))
    and select the 38 closest to the origin, then relax.
    
    The key insight: the optimal FCC cluster for N=38 is essentially the
    38 nearest neighbors in FCC, which forms a shape close to a truncated octahedron.
    """
    # FCC lattice: points at (i, j, k) * s where i+j+k is even
    # Nearest neighbor distance in this lattice = s * sqrt(2)
    # For NN distance = 1 (sigma), s = 1/sqrt(2)
    s = 1.0 / math.sqrt(2.0)
    
    max_c = 5
    points = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = i * s, j * s, k * s
                    r2 = x*x + y*y + z*z
                    points.append((x, y, z, r2))
    
    # Sort by distance from origin
    points.sort(key=lambda p: p[3])
    
    # Take 38 closest
    selected = np.array([(p[0], p[1], p[2]) for p in points[:N]], dtype=float)
    
    # Center of mass to origin
    com = np.mean(selected, axis=0)
    selected -= com
    
    return selected


def _make_fcc_with_shell():
    """
    Alternative FCC construction: core FCC + partial outer shell.
    FCC shell structure for 38:
    - Layer 0: 1 atom (center)
    - Layer 1: 12 atoms (octahedral directions, r = 1)
    - Layer 2: 6 atoms (cubic directions, r = sqrt(2))
    - Layer 3: 8 atoms (body diagonal directions, r = sqrt(3))
    - Layer 4: 11 atoms (next shell)
    Total: 1 + 12 + 6 + 8 + 11 = 38
    """
    s = 1.0 / math.sqrt(2.0)
    
    # Generate FCC points and assign shells by distance
    max_c = 5
    points = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = i * s, j * s, k * s
                    r2 = x*x + y*y + z*z
                    points.append((x, y, z, r2, (i, j, k)))
    
    points.sort(key=lambda p: p[3])
    
    # Take 38
    selected = np.array([(p[0], p[1], p[2]) for p in points[:N]], dtype=float)
    com = np.mean(selected, axis=0)
    selected -= com
    
    return selected


def _local_relax(pos):
    """
    Local minimization using L-BFGS with analytical gradient.
    Returns relaxed positions.
    """
    def fun(x):
        p = x.reshape(N, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.flatten()
    
    try:
        result = minimize(
            fun, pos.flatten(), method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-12, 'maxls': 50}
        )
        relaxed = result.x.reshape(N, 3)
        return relaxed
    except Exception:
        # Fallback: try without gradient
        def fun_nojac(x):
            return _lj_energy_only(x.reshape(N, 3))
        try:
            result = minimize(
                fun_nojac, pos.flatten(), method='L-BFGS-B',
                options={'maxiter': 100, 'gtol': 1e-6}
            )
            return result.x.reshape(N, 3)
        except Exception:
            return pos


def _kick(pos, sigma):
    """
    Apply a random Gaussian perturbation (kick) to positions.
    sigma controls the step size.
    """
    rng = np.random.default_rng()
    displacement = rng.normal(0.0, sigma, size=pos.shape)
    return pos + displacement


def construct():
    """
    Basin-hopping search for LJ38 global minimum.
    
    Strategy:
    1. Start from FCC truncated octahedron geometry.
    2. Local relax with analytical-gradient L-BFGS.
    3. Basin hopping: kick → relax → Metropolis accept/reject.
    4. Track best configuration throughout.
    """
    start_time = time.time()
    
    # Initial structure: FCC truncated octahedron
    pos0 = _make_fcc_truncated_octahedron()
    
    # Ensure validity
    if not _check_validity(pos0):
        # Nudge apart any too-close atoms
        for _ in range(100):
            diff = pos0[np.newaxis, :, :] - pos0[:, np.newaxis, :]
            r2 = np.sum(diff * diff, axis=2)
            np.fill_diagonal(r2, 1.0)
            r = np.sqrt(r2)
            if np.min(r) >= 0.5:
                break
            # Find closest pair and push apart
            idx = np.unravel_index(np.argmin(r2), r2.shape)
            i, j = idx
            direction = pos0[j] - pos0[i]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                direction = np.array([1.0, 0.0, 0.0])
                norm = 1.0
            direction /= norm
            push = (0.5 - norm) / 2.0 + 0.01
            pos0[i] -= direction * push
            pos0[j] += direction * push
    
    # Initial local relaxation
    pos0 = _local_relax(pos0)
    e0 = _lj_energy_only(pos0)
    
    best_pos = pos0.copy()
    best_energy = e0
    current_pos = pos0.copy()
    current_energy = e0
    
    # Basin hopping parameters
    # Start with moderate kick size, adapt based on acceptance rate
    sigma = 0.3
    T = 1.0  # Initial temperature for Metropolis
    T_decay = 0.995
    T_min = 0.05
    
    n_accepted = 0
    n_rejected = 0
    total_kicks = 0
    
    # Adaptive sigma: track acceptance rate
    accept_window = []
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break
        
        # Check if we have time for another kick+relax cycle
        # Estimate: each cycle takes roughly 0.5-2s
        if elapsed > DEADLINE - 5:
            break
        
        # Apply kick
        kicked = _kick(current_pos, sigma)
        
        # Relax
        relaxed = _local_relax(kicked)
        
        # Check validity
        if not _check_validity(relaxed):
            # Try smaller kick
            sigma *= 0.9
            continue
        
        e_new = _lj_energy_only(relaxed)
        
        # Update best
        if e_new < best_energy:
            best_energy = e_new
            best_pos = relaxed.copy()
        
        # Metropolis acceptance
        delta_e = e_new - current_energy
        if delta_e <= 0 or np.random.random() < math.exp(-delta_e / max(T, 1e-10)):
            current_pos = relaxed
            current_energy = e_new
            n_accepted += 1
        else:
            n_rejected += 1
        
        total_kicks += 1
        accept_window.append(1 if n_accepted > (total_kicks - 1) else 0)
        
        # Adaptive sigma: target ~50% acceptance
        if len(accept_window) >= 20:
            recent_accept_rate = np.mean(accept_window[-20:])
            if recent_accept_rate < 0.3:
                sigma *= 0.85
            elif recent_accept_rate > 0.7:
                sigma *= 1.1
            sigma = np.clip(sigma, 0.05, 1.5)
            accept_window = accept_window[-10:]
        
        # Cool temperature
        T = max(T * T_decay, T_min)
        
        # If stuck in a basin (many rejections in a row), increase kick
        if n_rejected > 15 and n_accepted < n_rejected // 3:
            sigma *= 1.3
            n_rejected = 0
            # Also try a larger temperature boost temporarily
            T = min(T * 2.0, 5.0)
        
        # Periodic re-optimization: if we've been at the same energy for a while,
        # try a slightly different initial structure
        if total_kicks % 50 == 0 and total_kicks > 0:
            elapsed = time.time() - start_time
            if elapsed < DEADLINE - 30:
                # Try perturbing the best and relaxing
                perturbed = _kick(best_pos, sigma * 0.5)
                relaxed_p = _local_relax(perturbed)
                if _check_validity(relaxed_p):
                    e_p = _lj_energy_only(relaxed_p)
                    if e_p < best_energy:
                        best_energy = e_p
                        best_pos = relaxed_p.copy()
    
    # Final: ensure we return the best found
    if not _check_validity(best_pos):
        # Fallback: return the initial FCC structure
        best_pos = pos0
    
    return best_pos

# EVOLVE-BLOCK-END
