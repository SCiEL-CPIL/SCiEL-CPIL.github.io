"""Morse cluster global minima: Langevin dynamics with adaptive cooling + periodic relax."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 280.0  # 20s safety margin before 300s kill


def _time_left():
    return _HARD_DEADLINE - time.time()


def _morse_energy(pos, rho):
    """Compute Morse potential energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, 1e10)
    # Only compute for pairs with r > 0 to avoid issues
    x = rho * (1.0 - dist)
    # Avoid overflow for very small distances
    x = np.minimum(x, 50.0)
    exp_x = np.exp(x)
    eij = exp_x * (exp_x - 2.0)
    # Zero out diagonal
    np.fill_diagonal(eij, 0.0)
    return 0.5 * np.sum(eij)


def _morse_grad(pos, rho):
    """Compute Morse potential gradient."""
    n = len(pos)
    if n < 2:
        return np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, 1e10)
    # Avoid division by zero
    safe_dist = np.where(dist > 1e-10, dist, 1e-10)
    x = rho * (1.0 - dist)
    x = np.minimum(x, 50.0)
    exp_x = np.exp(x)
    # dE/dr = rho * exp(x) * (2*exp(x) - 2) * (-1) ... wait let me recompute
    # E = exp(x)(exp(x)-2) where x = rho(1-r)
    # dE/dx = exp(x)(exp(x)-2) + exp(x)*exp(x) = exp(x)(2*exp(x)-2)
    # dE/dr = dE/dx * dx/dr = exp(x)(2*exp(x)-2) * (-rho)
    # Force on i from j: -dE/dr_ij * r_hat_ij where r_ij = r_i - r_j
    # Actually let's use the standard formula:
    # F_i = sum_{j != i} rho * exp(x) * (2*exp(x) - 2) * (r_i - r_j) / r_ij
    # Wait, let me be careful with signs.
    # E = sum_{i<j} exp(x_ij)(exp(x_ij)-2), x_ij = rho(1-r_ij)
    # dE/dr_ij = exp(x)(2exp(x)-2) * (-rho) = -rho * exp(x)(2exp(x)-2)
    # Force on atom i due to atom j: F_i = -dE/dr_ij * (r_i - r_j)/r_ij
    # = rho * exp(x)(2exp(x)-2) * (r_i - r_j)/r_ij
    dE_dr = -rho * exp_x * (2.0 * exp_x - 2.0)
    # F_i = sum_j dE_dr_ij * (r_j - r_i)/r_ij ... let me just use the gradient function
    # Actually, let me compute gradient properly:
    # grad_i E = sum_{j != i} dE/dr_ij * (r_i - r_j)/r_ij
    # dE/dr_ij = -rho * exp(x_ij) * (2*exp(x_ij) - 2)
    # So grad_i = sum_j [-rho*exp(x_ij)*(2exp(x_ij)-2)] * (r_i - r_j)/r_ij
    factor = dE_dr / safe_dist  # shape (n, n)
    np.fill_diagonal(factor, 0.0)
    # grad_i = sum_j factor_ij * (r_i - r_j)
    # = sum_j factor_ij * r_i - sum_j factor_ij * r_j
    # = r_i * sum_j factor_ij - (factor @ r)
    row_sum = np.sum(factor, axis=1)  # shape (n,)
    grad = row_sum[:, np.newaxis] * pos - factor @ pos
    return grad


def _langevin_step(pos, rho, T, dt, rng, mass=1.0):
    """One Langevin dynamics step."""
    n = len(pos)
    grad = _morse_grad(pos, rho)
    # Random force
    noise = rng.normal(0, np.sqrt(2.0 * T * mass / dt), size=pos.shape)
    # Velocity Verlet-like update for Langevin
    # Simplified: pos_new = pos + (-grad/mass + noise) * dt
    pos_new = pos + (-grad / mass + noise) * dt
    return pos_new


def _cooling_schedule(T0, T_end, step, total_steps):
    """Linear cooling schedule."""
    frac = step / total_steps
    return T0 * (1.0 - frac) + T_end * frac


def _anneal_from(pos0, rho, T0, T_end, total_steps, dt, rng, relax_every=200):
    """Run Langevin annealing from initial position."""
    pos = pos0.copy()
    E = _morse_energy(pos, rho)
    best_pos = pos.copy()
    best_E = E
    n = len(pos)
    
    for step in range(total_steps):
        if _time_left() < 15.0:
            break
        T = _cooling_schedule(T0, T_end, step, total_steps)
        # Adaptive dt: smaller when T is low
        dt_eff = dt * min(1.0, T / (T0 * 0.1) + 0.05)
        dt_eff = max(dt_eff, dt * 0.05)
        
        pos = _langevin_step(pos, rho, T, dt_eff, rng)
        
        # Enforce minimum distance constraint
        # Check and fix any pairs too close
        for _fix in range(5):
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dist, 1e10)
            too_close = np.where(dist < 0.5, True, False)
            np.fill_diagonal(too_close, False)
            if not np.any(too_close):
                break
            # Push apart atoms that are too close
            for i in range(n):
                for j in range(i + 1, n):
                    if too_close[i, j]:
                        d = pos[i] - pos[j]
                        norm_d = np.linalg.norm(d)
                        if norm_d < 1e-10:
                            d = rng.normal(0, 0.1, 3)
                            norm_d = np.linalg.norm(d)
                        d_hat = d / norm_d
                        push = (0.5 - norm_d + 0.01) / 2.0
                        pos[i] += push * d_hat
                        pos[j] -= push * d_hat
        
        # Periodic local relaxation
        if step % relax_every == relax_every - 1:
            E_relax, pos_relax = relax(pos, 'morse', rho)
            if E_relax < best_E:
                best_E = E_relax
                best_pos = pos_relax.copy()
            pos = pos_relax.copy()
            E = E_relax
        else:
            E = _morse_energy(pos, rho)
            if E < best_E:
                best_E = E
                best_pos = pos.copy()
    
    return best_pos, best_E


def _make_lattice_seed(n, rng, lattice_type):
    """Generate a compact lattice-like seed."""
    if lattice_type == 0:  # FCC-like
        # Generate points in a sphere with FCC-like spacing
        a = 1.0
        points = []
        # Simple approach: fill a cube with FCC lattice points
        # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
        basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
        # Estimate how many unit cells needed
        # Volume per atom in FCC = a^3/4
        # For n atoms, need n/4 unit cells
        # Cube edge ~ (n/4)^(1/3) * a
        n_cells = max(1, int(np.ceil(n / 4.0) ** (1.0 / 3.0)))
        for ix in range(-n_cells, n_cells + 1):
            for iy in range(-n_cells, n_cells + 1):
                for iz in range(-n_cells, n_cells + 1):
                    for b in basis:
                        p = np.array([ix, iy, iz]) * a + b * a
                        points.append(p)
                    if len(points) >= n:
                        break
            if len(points) >= n:
                break
            if len(points) >= n:
                break
        points = np.array(points[:n])
        # Center
        points -= np.mean(points, axis=0)
        return points
    
    elif lattice_type == 1:  # BCC-like
        a = 1.0
        points = []
        basis = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
        n_cells = max(1, int(np.ceil(n / 2.0) ** (1.0 / 3.0)))
        for ix in range(-n_cells, n_cells + 1):
            for iy in range(-n_cells, n_cells + 1):
                for iz in range(-n_cells, n_cells + 1):
                    for b in basis:
                        p = np.array([ix, iy, iz]) * a + b * a
                        points.append(p)
                    if len(points) >= n:
                        break
            if len(points) >= n:
                break
            if len(points) >= n:
                break
        points = np.array(points[:n])
        points -= np.mean(points, axis=0)
        return points
    
    else:  # Random compact
        radius = n ** (1.0 / 3.0) * 0.6
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                    points.append(p)
        points = np.array(points)
        return points


def construct(instance):
    """Find low-energy Morse cluster configurations using Langevin annealing."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 6))
    
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    best_pos = None
    best_E = np.inf
    
    # Number of restarts based on time budget
    # Each anneal takes roughly total_steps * (gradient cost)
    # For n atoms, gradient is O(n^2)
    # Estimate: ~10-50 ms per step for n=80
    # Budget: ~250s / 6 instances ≈ 40s per instance
    # So ~40s / 0.03s per step ≈ 1300 steps per anneal
    # With multiple restarts...
    
    time_budget = max(5.0, _time_left() / 6.0 - 5.0)  # per instance
    
    # Determine parameters based on n
    if n <= 40:
        total_steps = 800
        relax_every = 100
    elif n <= 60:
        total_steps = 600
        relax_every = 150
    else:
        total_steps = 400
        relax_every = 200
    
    T0 = 5.0
    T_end = 0.01
    dt = 0.01
    
    # Try different lattice types and random restarts
    n_restarts = 3
    
    for restart in range(n_restarts):
        if _time_left() < 20.0:
            break
        
        # Choose initial configuration
        if restart < 2:
            lattice_type = restart
        else:
            lattice_type = 2  # random
        
        pos0 = _make_lattice_seed(n, rng, lattice_type)
        
        # Ensure minimum distance
        # If any pairs are too close, jitter
        for _fix in range(10):
            diff = pos0[:, np.newaxis, :] - pos0[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dist, 1e10)
            too_close = np.where(dist < 0.5, True, False)
            np.fill_diagonal(too_close, False)
            if not np.any(too_close):
                break
            for i in range(n):
                for j in range(i + 1, n):
                    if too_close[i, j]:
                        d = pos0[i] - pos0[j]
                        norm_d = np.linalg.norm(d)
                        if norm_d < 1e-10:
                            d = rng.normal(0, 0.1, 3)
                            norm_d = np.linalg.norm(d)
                        d_hat = d / norm_d
                        push = (0.5 - norm_d + 0.01) / 2.0
                        pos0[i] += push * d_hat
                        pos0[j] -= push * d_hat
        
        # Initial relaxation
        E_init, pos_init = relax(pos0, 'morse', rho)
        
        if E_init < best_E:
            best_E = E_init
            best_pos = pos_init.copy()
        
        # Run Langevin annealing
        pos_anneal, E_anneal = _anneal_from(pos_init, rho, T0, T_end, total_steps, dt, rng, relax_every)
        
        if E_anneal < best_E:
            best_E = E_anneal
            best_pos = pos_anneal.copy()
        
        # Final relaxation
        if _time_left() > 10.0:
            E_final, pos_final = relax(best_pos, 'morse', rho)
            if E_final < best_E:
                best_E = E_final
                best_pos = pos_final.copy()
    
    # Final local relaxation
    if best_pos is not None and _time_left() > 5.0:
        E_final, pos_final = relax(best_pos, 'morse', rho)
        if E_final < best_E:
            best_E = E_final
            best_pos = pos_final.copy()
    
    if best_pos is None:
        # Fallback: random compact
        radius = n ** (1.0 / 3.0) * 0.6
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
        best_pos = np.array(points)
        _, best_pos = relax(best_pos, 'morse', rho)
    
    return best_pos
# EVOLVE-BLOCK-END
