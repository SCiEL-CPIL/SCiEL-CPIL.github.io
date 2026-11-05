# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 325.0  # 325s wall clock limit (evaluator kills at 330s)

def _get_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return _DEADLINE - (time.time() - _GLOBAL_START_TIME)

def _morse_energy(x, rho):
    """Compute Morse energy for a set of positions (vectorized)."""
    n = x.shape[0]
    if n < 2:
        return 0.0
    # Compute pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Mask diagonal
    np.fill_diagonal(dist, np.inf)
    
    # Morse term
    er = rho * (1.0 - dist)
    # Avoid overflow for very small r
    er = np.clip(er, -50, 50)
    exp_er = np.exp(er)
    energy_terms = exp_er * (exp_er - 2.0)
    
    # Sum upper triangle (i < j)
    iu = np.triu_indices(n, 1)
    return np.sum(energy_terms[iu])

def _morse_gradient(x, rho):
    """Compute analytical gradient of Morse energy (vectorized)."""
    n = x.shape[0]
    if n < 2:
        return np.zeros_like(x)
    
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    # Avoid division by zero (though dist is inf on diag)
    # dist is 0 on diag, but we mask it.
    
    # Calculate terms
    # dE/dr = -2 * rho * exp(u) * (exp(u) - 1)
    # where u = rho * (1 - r)
    u = rho * (1.0 - dist)
    u = np.clip(u, -50, 50)
    exp_u = np.exp(u)
    
    # dE/dr
    dE_dr = -2.0 * rho * exp_u * (exp_u - 1.0)
    
    # Gradient of distance wrt position i: (r_i - r_j) / r_ij
    # grad_dist[i, j] = (x[i] - x[j]) / dist[i, j]
    # We need to handle dist=0 (diagonal)
    # We can use np.divide with where
    
    # dE/dx_i = sum_j dE_dr[i,j] * (x[i] - x[j]) / dist[i,j]
    
    # Vectorized:
    # dE_dr is (n, n)
    # diff is (n, n, 3)
    # dist is (n, n)
    
    # Factor = dE_dr / dist
    # But dist is 0 on diagonal. We want 0 there.
    # Use np.errstate to ignore divide by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = dE_dr / dist
    np.fill_diagonal(factor, 0.0)
    
    # grad[i] = sum_j factor[i, j] * (x[i] - x[j])
    # factor[i, j] is scalar for pair (i, j)
    # diff[i, j] is vector (x[i] - x[j])
    
    # We can do: grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    return grad

def _min_distance(x):
    """Compute minimum distance between any pair of atoms."""
    n = x.shape[0]
    if n < 2:
        return np.inf
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    return np.min(dist)

def _is_valid(x):
    """Check if the configuration is valid (atoms at least 0.5 apart)."""
    return _min_distance(x) >= 0.5

def _generate_icosahedral(n, seed=42):
    """
    Generate an icosahedral cluster.
    For n <= 13: Icosahedron
    For n <= 55: Icosahedron + Shells
    """
    rng = np.random.default_rng(seed)
    
    # Icosahedron vertices (radius 1)
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)
    ]
    verts = np.array(verts)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    
    positions = [np.array([0.0, 0.0, 0.0])]
    if len(verts) >= 2:
        d_min = np.min(np.linalg.norm(verts[:, np.newaxis, :] - verts[np.newaxis, :, :], axis=2))
        scale = 1.0 / d_min
        verts = verts * scale
        positions.extend([v for v in verts])
    
    remaining = n - len(positions)
    
    if remaining <= 0:
        return np.array(positions[:n])
    
    # Add atoms in the next shell
    shell_radius = 1.8
    for _ in range(remaining):
        u = rng.uniform(-1, 1)
        theta = rng.uniform(0, 2 * np.pi)
        r = shell_radius
        x = r * np.sqrt(1 - u**2) * np.cos(theta)
        y = r * np.sqrt(1 - u**2) * np.sin(theta)
        z = r * u
        positions.append(np.array([x, y, z]))
        
    return np.array(positions[:n])

def _generate_fcc_cluster(n, seed=42):
    """Generate an FCC-like cluster."""
    rng = np.random.default_rng(seed)
    
    a = math.sqrt(2)
    radius = 1.5 * (n ** (1.0/3.0)) * a
    points = []
    
    basis = [(0, 0, 0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)]
    
    max_coord = int(radius / a) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in basis:
                    p = np.array([i * a + dx, j * a + dy, k * a + dz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
    
    points.sort(key=lambda p: np.linalg.norm(p))
    
    if len(points) >= n:
        jitter = rng.uniform(-0.05, 0.05, (n, 3))
        return np.array(points[:n]) + jitter
    else:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
        return np.array(points)

def _generate_bcc_cluster(n, seed=42):
    """Generate a BCC-like cluster."""
    rng = np.random.default_rng(seed)
    
    a = 2.0 / math.sqrt(3)
    radius = 1.5 * (n ** (1.0/3.0)) * a
    points = []
    
    basis = [(0, 0, 0), (a/2, a/2, a/2)]
    
    max_coord = int(radius / a) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in basis:
                    p = np.array([i * a + dx, j * a + dy, k * a + dz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
    
    points.sort(key=lambda p: np.linalg.norm(p))
    
    if len(points) >= n:
        jitter = rng.uniform(-0.05, 0.05, (n, 3))
        return np.array(points[:n]) + jitter
    else:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
        return np.array(points)

def _generate_random_cluster(n, seed=42, min_dist=0.6):
    """Generate a random cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    
    radius = 2.0 * (n ** (1.0/3.0))
    points = []
    attempts = 0
    max_attempts = 50000
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)

def _relax_and_validate(positions, potential, rho):
    """Relax positions and check validity."""
    n = positions.shape[0]
    try:
        # Use the provided relax function first, as it might be optimized
        energy, relaxed = relax(positions, potential, rho)
        if _min_distance(relaxed) >= 0.5:
            return energy, relaxed
        else:
            # Try our own polish if relax failed validity
            polished_pos, polished_energy = _polish(positions, rho)
            if _min_distance(polished_pos) >= 0.5:
                return polished_energy, polished_pos
            return float('inf'), None
    except Exception:
        # Fallback to our own polish
        try:
            polished_pos, polished_energy = _polish(positions, rho)
            if _min_distance(polished_pos) >= 0.5:
                return polished_energy, polished_pos
        except Exception:
            pass
        return float('inf'), None

def _polish(positions, rho, max_time=5.0):
    """
    High-precision polish using L-BFGS-B with analytical gradient.
    """
    n = positions.shape[0]
    x0 = positions.flatten()
    
    def obj(x):
        pos = x.reshape(n, 3)
        return _morse_energy(pos, rho)
    
    def grad(x):
        pos = x.reshape(n, 3)
        return _morse_gradient(pos, rho).flatten()
    
    start_time = time.time()
    try:
        res = minimize(
            obj, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 5000, 'gtol': 1e-6, 'ftol': 1e-8}
        )
        if _min_distance(res.x.reshape(n, 3)) >= 0.5:
            return res.x.reshape(n, 3), res.fun
    except Exception:
        pass
    
    # Fallback: use the library's relax if time permits
    if _get_remaining_time() > 2.0:
        try:
            energy, relaxed = relax(positions, 'morse', rho)
            if _min_distance(relaxed) >= 0.5:
                return relaxed, energy
        except Exception:
            pass
    
    return positions, _morse_energy(positions, rho)

def _basin_hop(start_positions, potential, rho, max_iterations=50, initial_step=0.5, initial_temp=2.0):
    """
    Perform basin hopping optimization.
    Returns (best_energy, best_positions).
    """
    n = start_positions.shape[0]
    current_energy, current_positions = _relax_and_validate(start_positions, potential, rho)
    
    if current_positions is None:
        perturbed = start_positions + np.random.normal(0, 0.1, start_positions.shape)
        current_energy, current_positions = _relax_and_validate(perturbed, potential, rho)
        if current_positions is None:
            return float('inf'), None
    
    best_energy = current_energy
    best_positions = current_positions.copy()
    
    step_size = initial_step
    temp = initial_temp
    
    for it in range(max_iterations):
        if _get_remaining_time() < 5.0:
            break
        
        # Perturb
        noise = np.random.normal(0, step_size, current_positions.shape)
        new_positions = current_positions + noise
        
        # Relax
        new_energy, new_relaxed = _relax_and_validate(new_positions, potential, rho)
        
        if new_relaxed is None:
            noise = np.random.normal(0, step_size * 0.5, current_positions.shape)
            new_positions = current_positions + noise
            new_energy, new_relaxed = _relax_and_validate(new_positions, potential, rho)
        
        if new_relaxed is None:
            continue
        
        # Accept/reject
        delta_e = new_energy - current_energy
        if delta_e < 0 or (temp > 1e-6 and np.random.rand() < math.exp(-delta_e / temp)):
            current_energy = new_energy
            current_positions = new_relaxed
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_positions.copy()
        
        # Anneal
        temp *= 0.95
        step_size *= 0.98
        
        if it > 0 and it % 20 == 0:
            step_size *= 1.2
    
    # Final polish on the best configuration found
    if _get_remaining_time() > 3.0:
        polished_pos, polished_energy = _polish(best_positions, rho)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_positions = polished_pos
    
    return best_energy, best_positions

def construct(instance):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]
    
    if rho is None:
        rho = 6
    
    best_energy = float('inf')
    best_positions = None
    
    initial_guesses = []
    
    try:
        initial_guesses.append(_generate_icosahedral(n, seed=42))
    except Exception:
        pass
    
    try:
        initial_guesses.append(_generate_fcc_cluster(n, seed=42))
    except Exception:
        pass
    
    try:
        initial_guesses.append(_generate_bcc_cluster(n, seed=42))
    except Exception:
        pass
    
    for seed in [1, 2, 3, 4, 5]:
        try:
            initial_guesses.append(_generate_random_cluster(n, seed=seed))
        except Exception:
            pass
    
    if n > 50:
        for seed in [6, 7, 8]:
            try:
                initial_guesses.append(_generate_random_cluster(n, seed=seed))
            except Exception:
                pass
    
    relaxed_results = []
    for guess in initial_guesses:
        if _get_remaining_time() < 10.0:
            break
        
        energy, positions = _relax_and_validate(guess, potential, rho)
        if positions is not None and energy < float('inf'):
            relaxed_results.append((energy, positions))
    
    relaxed_results.sort(key=lambda x: x[0])
    
    if relaxed_results:
        best_energy = relaxed_results[0][0]
        best_positions = relaxed_results[0][1].copy()
    
    num_bh_starts = min(len(relaxed_results), 3)
    
    for idx in range(num_bh_starts):
        if _get_remaining_time() < 15.0:
            break
        
        start_energy, start_pos = relaxed_results[idx]
        
        if rho <= 3:
            initial_step = 0.8
            initial_temp = 5.0
        elif rho <= 6:
            initial_step = 0.5
            initial_temp = 3.0
        else:
            initial_step = 0.3
            initial_temp = 2.0
        
        if n <= 40:
            max_iter = 80
        elif n <= 60:
            max_iter = 50
        else:
            max_iter = 30
        
        bh_energy, bh_positions = _basin_hop(
            start_pos, potential, rho,
            max_iterations=max_iter,
            initial_step=initial_step,
            initial_temp=initial_temp
        )
        
        if bh_positions is not None and bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_positions.copy()
        
        if _get_remaining_time() > 20.0 and idx == 0:
            bh_energy2, bh_positions2 = _basin_hop(
                start_pos, potential, rho,
                max_iterations=max_iter // 2,
                initial_step=initial_step * 1.5,
                initial_temp=initial_temp * 1.5
            )
            if bh_positions2 is not None and bh_energy2 < best_energy:
                best_energy = bh_energy2
                best_positions = bh_positions2.copy()
    
    if best_positions is not None and _get_remaining_time() > 2.0:
        polished_pos, polished_energy = _polish(best_positions, rho)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_positions = polished_pos
    
    if best_positions is None:
        best_positions = _generate_icosahedral(n, seed=0)
        if not _is_valid(best_positions):
            best_positions = best_positions * 1.5
            if not _is_valid(best_positions):
                best_positions = _generate_random_cluster(n, seed=99)
    
    return best_positions
# EVOLVE-BLOCK-END
