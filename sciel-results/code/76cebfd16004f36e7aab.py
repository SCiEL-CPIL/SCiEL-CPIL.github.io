"""Morse cluster global minima search: structured starts + L-BFGS-B + basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
from scipy.optimize import minimize

# Global deadline: evaluator kills after 330s wall-clock.
# Use 305s as working limit for safety across all six construct() calls.
_DEADLINE = time.time() + 305.0

def _time_left():
    return _DEADLINE - time.time()

def _morse_energy(pos, rho):
    """Compute total Morse energy efficiently."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(n, k=1)
    r = dist[iu]
    x = rho * (1.0 - r)
    ex = np.exp(x)
    return np.sum(ex * (ex - 2.0))

def _morse_energy_gradient(pos, rho):
    """Compute total Morse energy and its gradient efficiently."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    iu = np.triu_indices(n, k=1)
    r = dist[iu]
    x = rho * (1.0 - r)
    ex = np.exp(x)
    e_pair = ex * (ex - 2.0)
    
    # Gradient: dE/dr_i * dr_i/dpos
    # dE/dr = rho * ex * (2*ex - 2)
    de_dr = rho * ex * (2.0 * ex - 2.0)
    
    # dr/dpos for each pair
    # For pair (i,j), r = |pos_i - pos_j|
    # dr/dpos_i = (pos_i - pos_j)/r
    # dr/dpos_j = -(pos_i - pos_j)/r
    
    grad = np.zeros_like(pos)
    for k in range(len(r)):
        i, j = iu[0][k], iu[1][k]
        dr = diff[i, j] / r[k]
        grad[i] += de_dr[k] * dr
        grad[j] -= de_dr[k] * dr
    
    return np.sum(e_pair), grad

def _check_min_dist(pos, min_d=0.5):
    """Check if all pairwise distances are at least min_d."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    return np.min(dist) >= min_d - 1e-9

def _lbfgs_polish(pos, rho, maxiter=200):
    """Polish a configuration using L-BFGS-B with gradient."""
    n = len(pos)
    flat_pos = pos.flatten()
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        e, g = _morse_energy_gradient(p, rho)
        return e, g.flatten()
    
    try:
        result = minimize(objective_and_grad, flat_pos, method='L-BFGS-B',
                         jac=True, options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
        if result.success or result.fun < _morse_energy(pos, rho):
            return result.fun, result.x.reshape(n, 3)
    except Exception:
        pass
    
    # Fallback to relax if L-BFGS-B fails
    try:
        energy, relaxed = relax(pos, "morse", rho)
        return energy, relaxed
    except Exception:
        return _morse_energy(pos, rho), pos

def _build_fibonacci_sphere(n, radius=1.0):
    """Generate n points evenly distributed on a sphere using the Fibonacci spiral."""
    if n == 0:
        return np.zeros((0, 3))
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    theta = np.pi * (1 + np.sqrt(5.0)) * i
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack([x, y, z])

def _build_icosahedral_core(n_atoms, spacing=1.0):
    """Build a configuration with an icosahedral core and outer shells."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])

    points = [np.zeros(3)]  # center

    # First shell: icosahedral vertices
    shell1 = ico_verts * spacing
    for v in shell1:
        if len(points) < n_atoms:
            points.append(v)

    # Second shell: larger icosahedral shell
    if len(points) < n_atoms:
        shell2 = ico_verts * spacing * 1.8
        for v in shell2:
            if len(points) < n_atoms:
                points.append(v)

    # Third shell: Fibonacci sphere
    if len(points) < n_atoms:
        n_remaining = n_atoms - len(points)
        shell3 = _build_fibonacci_sphere(n_remaining, spacing * 2.6)
        for v in shell3:
            if len(points) < n_atoms:
                points.append(v)

    # Fill any remaining with random points in a larger shell
    rng = np.random.default_rng(42)
    while len(points) < n_atoms:
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        r = spacing * (3.0 + rng.uniform(0, 0.5))
        p = np.array([r * np.sin(phi) * np.cos(theta),
                      r * np.sin(phi) * np.sin(theta),
                      r * np.cos(phi)])
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)

    return np.array(points[:n_atoms])

def _build_fcc_lattice(n_atoms, spacing=1.0):
    """Build an FCC-like lattice configuration."""
    points = []
    rng = np.random.default_rng(123)

    # Generate FCC lattice points in a cube
    side = int(np.ceil(n_atoms ** (1.0 / 3.0))) + 2
    fcc_offsets = np.array([
        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]
    ], dtype=float)

    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for off in fcc_offsets:
                    p = np.array([i, j, k], dtype=float) * spacing + off * spacing
                    if len(points) < n_atoms:
                        points.append(p)
                if len(points) >= n_atoms:
                    break
            if len(points) >= n_atoms:
                break
        if len(points) >= n_atoms:
            break

    points = np.array(points[:n_atoms])

    # Center the cluster
    points -= np.mean(points, axis=0)

    # Add small random perturbation to break symmetry
    points += rng.normal(0, 0.05, points.shape)

    return points

def _build_bcc_lattice(n_atoms, spacing=1.0):
    """Build a BCC-like lattice configuration."""
    points = []
    rng = np.random.default_rng(456)

    # Generate BCC lattice points in a cube
    side = int(np.ceil(n_atoms ** (1.0 / 3.0))) + 2
    bcc_offsets = np.array([
        [0, 0, 0], [0.5, 0.5, 0.5]
    ], dtype=float)

    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for off in bcc_offsets:
                    p = np.array([i, j, k], dtype=float) * spacing + off * spacing
                    if len(points) < n_atoms:
                        points.append(p)
                if len(points) >= n_atoms:
                    break
            if len(points) >= n_atoms:
                break
        if len(points) >= n_atoms:
            break

    points = np.array(points[:n_atoms])

    # Center the cluster
    points -= np.mean(points, axis=0)

    # Add small random perturbation to break symmetry
    points += rng.normal(0, 0.05, points.shape)

    return points

def _build_hcp_lattice(n_atoms, spacing=1.0):
    """Build an HCP-like lattice configuration."""
    points = []
    a = spacing
    c = spacing * np.sqrt(8.0/3.0)
    layer_spacing = c / 3.0
    max_layer = int(np.ceil(n_atoms ** (1/3.0))) + 2

    for layer in range(-max_layer, max_layer + 1):
        z = layer * layer_spacing
        for i in range(-max_layer, max_layer + 1):
            for j in range(-max_layer, max_layer + 1):
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0) / 2.0 * j
                if layer % 2 != 0:
                    x += a * 0.5
                p = np.array([x, y, z])
                if len(points) < n_atoms * 2:
                    points.append(p)

    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n_atoms]
    points = points[idx]
    points -= np.mean(points, axis=0)
    return points

def _build_random_compact(n_atoms, spacing=1.0, seed=0):
    """Build a random compact cluster using rejection sampling."""
    rng = np.random.default_rng(seed)
    radius = spacing * n_atoms ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    max_attempts = n_atoms * 200

    while len(points) < n_atoms and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)

    # If not enough, add more loosely
    while len(points) < n_atoms:
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        r = radius * (1.0 + rng.uniform(0, 0.5))
        p = np.array([r * np.sin(phi) * np.cos(theta),
                      r * np.sin(phi) * np.sin(theta),
                      r * np.cos(phi)])
        points.append(p)

    return np.array(points[:n_atoms])

def _simulated_annealing(pos, rho, rng, n_steps=None, t_start=2.0, t_end=0.01, cool_rate=0.995):
    """Direct gradient-free simulated annealing on raw Morse energy."""
    if n_steps is None:
        n_steps = max(200, len(pos) * 10)
    
    current_pos = pos.copy()
    current_energy = _morse_energy(current_pos, rho)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    t = t_start
    n = len(current_pos)
    
    for step in range(n_steps):
        if _time_left() < 3:
            break
        
        # Random perturbation - move one or a few atoms
        n_move = max(1, int(n * 0.05))
        idx = rng.choice(n, size=n_move, replace=False)
        step_size = 0.2 * t / t_start + 0.05
        noise = rng.normal(0, step_size, size=(n_move, 3))
        
        new_pos = current_pos.copy()
        new_pos[idx] += noise
        
        # Check minimum distance constraint
        if not _check_min_dist(new_pos, 0.5):
            # Push apart overlapping atoms
            diff = new_pos[:, np.newaxis, :] - new_pos[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist, 1.0)
            for i in range(n):
                for j in range(i+1, n):
                    if dist[i,j] < 0.5:
                        direction = (new_pos[i] - new_pos[j]) / max(dist[i,j], 1e-10)
                        push = (0.5 - dist[i,j]) / 2.0
                        new_pos[i] += direction * push
                        new_pos[j] -= direction * push
        
        new_energy = _morse_energy(new_pos, rho)
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.uniform() < np.exp(-delta_e / max(t, 1e-10)):
            current_pos = new_pos
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
        
        t *= cool_rate
    
    return best_pos, best_energy

def _basin_hop(pos, rho, rng, n_hops=10, step_scale=0.35):
    """Basin hopping: perturb, L-BFGS-B polish, accept if lower energy."""
    current_pos = pos.copy()
    current_energy = _morse_energy(current_pos, rho)

    for _ in range(n_hops):
        if _time_left() < 3:
            break

        # Perturb
        perturbed = current_pos + rng.normal(0, step_scale, current_pos.shape)

        # Polish using L-BFGS-B
        try:
            energy, polished = _lbfgs_polish(perturbed, rho, maxiter=100)
        except Exception:
            # Fallback to relax
            try:
                energy, polished = relax(perturbed, "morse", rho)
            except Exception:
                continue

        # Check minimum separation
        if len(polished) >= 2:
            diff = polished[:, np.newaxis, :] - polished[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist, np.inf)
            if np.min(dist) < 0.49:
                continue

        # Accept if lower energy
        if energy < current_energy:
            current_energy = energy
            current_pos = polished.copy()

    return current_energy, current_pos

def construct(instance):
    """Find low-energy Morse cluster using structured starts, L-BFGS-B, and basin hopping."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))

    if potential != "morse":
        spacing = 1.0
        rng = np.random.default_rng(size)
        radius = spacing * size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        energy, positions = relax(np.array(points), potential, parameter)
        return positions

    rho = parameter
    spacing = 1.0

    best_energy = np.inf
    best_positions = None

    # Generate diverse initial configurations
    initial_configs = []

    # 1. Icosahedral core with shells
    if _time_left() > 10:
        initial_configs.append(_build_icosahedral_core(size, spacing))

    # 2. FCC lattice
    if _time_left() > 10:
        initial_configs.append(_build_fcc_lattice(size, spacing))

    # 3. HCP lattice
    if _time_left() > 10:
        initial_configs.append(_build_hcp_lattice(size, spacing))

    # 4. BCC lattice (new structural seed)
    if _time_left() > 10:
        initial_configs.append(_build_bcc_lattice(size, spacing))

    # 5-8. Random compact with different seeds
    for seed in [0, 1, 2, 3]:
        if _time_left() > 10:
            initial_configs.append(_build_random_compact(size, spacing, seed))

    rng = np.random.default_rng(999)
    
    for i, config in enumerate(initial_configs):
        if _time_left() < 5:
            break

        try:
            # First try L-BFGS-B directly on the config
            lbfgs_energy, lbfgs_positions = _lbfgs_polish(config, rho, maxiter=150)
            
            if _check_min_dist(lbfgs_positions, 0.5) and lbfgs_energy < best_energy:
                best_energy = lbfgs_energy
                best_positions = lbfgs_positions.copy()
            
            # Then try basin hopping from the L-BFGS-B result
            if _time_left() > 8:
                bh_energy, bh_positions = _basin_hop(lbfgs_positions, rho, rng, 
                                                     n_hops=6, step_scale=0.3)
                
                if bh_energy < best_energy:
                    best_energy = bh_energy
                    best_positions = bh_positions.copy()
            
            # Also try basin hopping directly from original config
            if _time_left() > 8:
                try:
                    energy, positions = relax(config, "morse", rho)
                    
                    if _check_min_dist(positions, 0.5):
                        bh_energy, bh_positions = _basin_hop(positions, rho, rng, 
                                                             n_hops=5, step_scale=0.35)
                        
                        if bh_energy < best_energy:
                            best_energy = bh_energy
                            best_positions = bh_positions.copy()
                        elif energy < best_energy:
                            best_energy = energy
                            best_positions = positions.copy()
                except Exception:
                    continue
        except Exception:
            continue

    # If we still have time, try a few more random restarts with L-BFGS-B + basin hopping
    extra_attempts = 0
    while _time_left() > 15 and extra_attempts < 6:
        extra_attempts += 1
        config = _build_random_compact(size, spacing, 100 + extra_attempts)
        try:
            lbfgs_energy, lbfgs_positions = _lbfgs_polish(config, rho, maxiter=100)
            
            if _check_min_dist(lbfgs_positions, 0.5) and lbfgs_energy < best_energy:
                best_energy = lbfgs_energy
                best_positions = lbfgs_positions.copy()
            
            if _time_left() > 10:
                bh_energy, bh_positions = _basin_hop(lbfgs_positions, rho, rng, 
                                                     n_hops=4, step_scale=0.4)
                if bh_energy < best_energy:
                    best_energy = bh_energy
                    best_positions = bh_positions.copy()
        except Exception:
            continue

    # Final L-BFGS-B polishing on the best configuration
    if best_positions is not None and _time_left() > 5:
        try:
            for _ in range(3):
                if _time_left() < 2:
                    break
                energy, polished = _lbfgs_polish(best_positions, rho, maxiter=50)
                if energy < best_energy and _check_min_dist(polished, 0.5):
                    best_energy = energy
                    best_positions = polished.copy()
        except Exception:
            pass

    if best_positions is None:
        # Fallback
        config = _build_random_compact(size, spacing, 0)
        try:
            energy, best_positions = relax(config, "morse", rho)
        except Exception:
            best_positions = config

    return best_positions
# EVOLVE-BLOCK-END
