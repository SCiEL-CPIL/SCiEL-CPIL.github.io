"""Morse cluster global minima: basin hopping with crystalline starts and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(pos, rho):
    """Compute Morse energy E = sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij)) - 2)."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    x = rho * (1.0 - r_ij)
    e = np.exp(x) * (np.exp(x) - 2.0)
    return float(np.sum(e))


def _morse_energy_and_grad(pos, rho):
    """Compute Morse energy and gradient."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    x = rho * (1.0 - r_ij)
    exp_x = np.exp(x)
    e_ij = exp_x * (exp_x - 2.0)
    energy = float(np.sum(e_ij))
    
    # Gradient: dE/dr_ij = rho * exp_x * (2*exp_x - 2)
    # dr_ij/dpos_i = -(pos_i - pos_j)/r_ij for pair (i,j)
    grad = np.zeros_like(pos)
    i_idx, j_idx = iu
    dr_dr = (pos[i_idx] - pos[j_idx]) / r_ij[:, None]
    de_dr = rho * exp_x * (2.0 * exp_x - 2.0)
    np.add.at(grad, i_idx, de_dr[:, None] * dr_dr)
    np.add.at(grad, j_idx, -de_dr[:, None] * dr_dr)
    return energy, grad


def _is_valid(pos, min_dist=0.5):
    """Check all pairwise distances >= min_dist."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    return bool(np.min(r) >= min_dist)


def _generate_fcc(n, spacing=1.0):
    """Generate FCC lattice points, clipped to sphere."""
    points = []
    # FCC lattice: conventional cubic cell with atoms at corners and face centers
    a = spacing
    # Generate points in a grid
    max_coord = int(np.ceil(n ** 0.5)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # Corner atoms
                points.append(np.array([i * a, j * a, k * a]))
                # Face center atoms
                points.append(np.array([(i + 0.5) * a, (j + 0.5) * a, k * a]))
                points.append(np.array([(i + 0.5) * a, j * a, (k + 0.5) * a]))
                points.append(np.array([i * a, (j + 0.5) * a, (k + 0.5) * a]))
    
    points = np.array(points)
    # Center of mass
    com = np.mean(points, axis=0)
    points -= com
    # Sort by distance from center and take first n
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points


def _generate_bcc(n, spacing=1.0):
    """Generate BCC lattice points, clipped to sphere."""
    points = []
    a = spacing
    max_coord = int(np.ceil(n ** 0.5)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                points.append(np.array([i * a, j * a, k * a]))
                points.append(np.array([(i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a]))
    
    points = np.array(points)
    com = np.mean(points, axis=0)
    points -= com
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points


def _generate_hcp(n, spacing=1.0):
    """Generate HCP lattice points, clipped to sphere."""
    points = []
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)  # HCP c/a ratio
    max_coord = int(np.ceil(n ** 0.5)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # Layer A
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0) / 2.0 * j
                z = c * k
                points.append(np.array([x, y, z]))
                # Layer B
                x_b = a * (i + 0.5 * (j % 2)) + a / 2.0
                y_b = a * np.sqrt(3.0) / 2.0 * j + a * np.sqrt(3.0) / 6.0
                z_b = c * (k + 0.5)
                points.append(np.array([x_b, y_b, z_b]))
    
    points = np.array(points)
    com = np.mean(points, axis=0)
    points -= com
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points


def _generate_random_compact(n, spacing=1.0, rng=None):
    """Generate random compact cluster with minimum distance constraint."""
    if rng is None:
        rng = np.random.default_rng(42)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    for _ in range(10000):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.7 * spacing for q in points):
                points.append(p)
    # If we couldn't get enough, fill with random
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points[:n])


def _local_minimize(pos, rho, max_iter=500):
    """Local minimization using L-BFGS-B."""
    from scipy.optimize import minimize
    try:
        result = minimize(
            lambda x: _morse_energy(x.reshape(-1, 3), rho),
            pos.flatten(),
            jac=lambda x: _morse_energy_and_grad(x.reshape(-1, 3), rho)[1].flatten(),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        new_pos = result.x.reshape(-1, 3)
        new_energy = _morse_energy(new_pos, rho)
        if _is_valid(new_pos):
            return new_energy, new_pos
    except Exception:
        pass
    return _morse_energy(pos, rho), pos


def _basin_hop(pos, rho, rng, max_steps=200, perturb_scale=0.5):
    """Basin hopping: random perturbations + local relaxation."""
    current_energy, current_pos = _local_minimize(pos, rho)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    for step in range(max_steps):
        # Random perturbation
        perturbation = rng.normal(0, perturb_scale, pos.shape)
        perturbed = current_pos + perturbation
        perturbed_energy, perturbed_pos = _local_minimize(perturbed, rho)
        
        if perturbed_energy < best_energy:
            best_energy = perturbed_energy
            best_pos = perturbed_pos.copy()
        
        # Accept with Metropolis criterion
        if perturbed_energy < current_energy:
            current_energy = perturbed_energy
            current_pos = perturbed_pos
        else:
            delta_e = perturbed_energy - current_energy
            # Simple acceptance: accept if better, else reject (greedy basin hopping)
            if rng.random() < np.exp(-delta_e / max(0.01, abs(current_energy) * 0.1)):
                current_energy = perturbed_energy
                current_pos = perturbed_pos
    
    return best_energy, best_pos


def construct(instance):
    """Find global minimum of Morse cluster using basin hopping from crystalline starts."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    rng = np.random.default_rng(size)
    
    # Generate diverse initial configurations
    initial_configs = []
    try:
        fcc = _generate_fcc(size)
        if _is_valid(fcc):
            initial_configs.append(fcc)
    except Exception:
        pass
    
    try:
        bcc = _generate_bcc(size)
        if _is_valid(bcc):
            initial_configs.append(bcc)
    except Exception:
        pass
    
    try:
        hcp = _generate_hcp(size)
        if _is_valid(hcp):
            initial_configs.append(hcp)
    except Exception:
        pass
    
    # Add random compact starts
    for _ in range(3):
        try:
            rand = _generate_random_compact(size, rng=rng)
            if _is_valid(rand):
                initial_configs.append(rand)
        except Exception:
            pass
    
    if not initial_configs:
        # Fallback: simple random
        pos = rng.uniform(-2, 2, (size, 3))
        initial_configs.append(pos)
    
    # Local minimize each initial config
    best_energy = np.inf
    best_pos = None
    
    for config in initial_configs:
        if time.time() > deadline:
            break
        try:
            e, p = _local_minimize(config, rho, max_iter=200)
            if e < best_energy and _is_valid(p):
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass
    
    if best_pos is None:
        best_pos = initial_configs[0]
        best_energy = _morse_energy(best_pos, rho)
    
    # Basin hopping from best configuration
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        try:
            e_bh, pos_bh = _basin_hop(best_pos, rho, rng, max_steps=100, perturb_scale=0.3)
            if e_bh < best_energy and _is_valid(pos_bh):
                best_energy = e_bh
                best_pos = pos_bh
        except Exception:
            pass
    
    # Final refinement
    if time.time() < deadline - 2:
        try:
            e_final, pos_final = _local_minimize(best_pos, rho, max_iter=1000)
            if e_final < best_energy and _is_valid(pos_final):
                best_energy = e_final
                best_pos = pos_final
        except Exception:
            pass
    
    return best_pos
# EVOLVE-BLOCK-END
