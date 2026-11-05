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
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    er = rho * (1.0 - dist)
    er = np.clip(er, -500, 500)
    exp_er = np.exp(er)
    energy = np.sum(exp_er * (exp_er - 2.0)) / 2.0
    return energy

def _min_distance(x):
    """Compute minimum distance between any pair of atoms (vectorized)."""
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

def _generate_icosahedron(center, radius, seed=42):
    """Generate 13-atom icosahedral cluster (1 center + 12 vertices)."""
    phi = (1 + np.sqrt(5)) / 2
    scale = radius / np.sqrt(1 + phi*phi)
    
    verts = [
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    verts = np.array(verts) * scale
    
    positions = [center.copy()]
    for v in verts:
        positions.append(center + v)
    
    return np.array(positions)

def _generate_icosahedral_shell(n, seed=42):
    """
    Generate a cluster based on concentric icosahedral shells.
    Magic numbers: 1, 12, 42, 92, 147...
    We fill shells sequentially.
    """
    rng = np.random.default_rng(seed)
    positions = []
    
    # Approximate shell radii for icosahedral packing
    # r_k ~ k * 1.1
    shell_idx = 0
    current_count = 0
    
    while current_count < n:
        if shell_idx == 0:
            # Center atom
            positions.append(np.zeros(3))
            current_count += 1
            shell_idx += 1
            continue
            
        # Determine how many atoms to put in this shell
        # Capacity of shell k is roughly 12 + 30*(k-1) ... actually icosahedral numbers are 10k^3+15k^2+11k+1
        # Let's just estimate capacity as ~ 4*pi*r^2 / area_per_atom
        # Area per atom ~ 1.0
        # r = shell_idx * 1.1
        r = shell_idx * 1.1
        capacity = int(4 * np.pi * r**2 * 0.8) # 0.8 packing factor
        
        # Ensure we don't overshoot too much, but fill up to n
        atoms_to_add = min(capacity, n - current_count)
        
        # Generate points on sphere of radius r
        # Use Fibonacci sphere for even distribution
        points = []
        if atoms_to_add > 0:
            golden_angle = np.pi * (3 - np.sqrt(5))
            for i in range(atoms_to_add):
                y = 1 - (i / atoms_to_add) * 2
                radius_yz = np.sqrt(1 - y**2)
                theta = golden_angle * i
                
                x = np.cos(theta) * radius_yz
                z = np.sin(theta) * radius_yz
                
                p = np.array([x, y, z]) * r
                # Add small jitter to break symmetry
                p += rng.normal(0, 0.05, 3)
                points.append(p)
        
        positions.extend(points)
        current_count += atoms_to_add
        shell_idx += 1
        
        # Safety break
        if shell_idx > 20:
             break
            
    return np.array(positions[:n])

def _generate_fcc_chunk(n, seed=42):
    """Generate an FCC-like cluster."""
    rng = np.random.default_rng(seed)
    
    # Estimate radius needed for n atoms in FCC
    # Volume ~ n / density. R ~ (3V/4pi)^(1/3).
    # Density of FCC with nearest neighbor distance 1 is ~ 0.74 / (1/sqrt(2))^3 ?
    # Lattice constant a = sqrt(2). Volume per atom = a^3/4 = 2sqrt(2)/4 = sqrt(2)/2 ~ 0.707.
    # So R ~ (3 * n * 0.707 / 4 * pi)^(1/3) ~ 0.5 * n^(1/3) * 2 ?
    # Let's use R = 1.5 * n^(1/3) as a safe upper bound, then cut.
    radius = 1.5 * (n ** (1.0/3.0))
    points = []
    
    basis = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    
    max_coord = int(radius) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in basis:
                    p = np.array([i + dx, j + dy, k + dz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
    
    points.sort(key=lambda p: np.linalg.norm(p))
    
    if len(points) >= n:
        jitter = rng.uniform(-0.02, 0.02, (n, 3))
        return np.array(points[:n]) + jitter
    else:
        # Fill remaining with random points inside the sphere
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                points.append(p)
        return np.array(points)

def _generate_bcc_chunk(n, seed=42):
    """Generate a BCC-like cluster."""
    rng = np.random.default_rng(seed)
    
    radius = 1.5 * (n ** (1.0/3.0))
    points = []
    
    basis = [(0, 0, 0), (0.5, 0.5, 0.5)]
    
    max_coord = int(radius) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in basis:
                    p = np.array([i + dx, j + dy, k + dz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
    
    points.sort(key=lambda p: np.linalg.norm(p))
    
    if len(points) >= n:
        jitter = rng.uniform(-0.02, 0.02, (n, 3))
        return np.array(points[:n]) + jitter
    else:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
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
        energy, relaxed = relax(positions, potential, rho)
        if _min_distance(relaxed) >= 0.5:
            return energy, relaxed
        else:
            return float('inf'), None
    except Exception:
        return float('inf'), None

def _polish(positions, rho, max_iter=200):
    """
    Perform a high-precision local minimization using L-BFGS-B.
    """
    n = positions.shape[0]
    x_flat = positions.flatten()
    
    def objective(x):
        pos = x.reshape(-1, 3)
        if _min_distance(pos) < 0.5:
            return 1e10
        return _morse_energy(pos, rho)
    
    def gradient(x):
        pos = x.reshape(-1, 3)
        if _min_distance(pos) < 0.5:
            return np.zeros_like(x)
        grad_3d = morse_energy_gradient(pos, rho)
        return grad_3d.flatten()
    
    try:
        res = minimize(objective, x_flat, jac=gradient, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.success or res.fun < 1e10:
            pos_res = res.x.reshape(-1, 3)
            if _min_distance(pos_res) >= 0.5:
                return res.fun, pos_res
    except Exception:
        pass
    
    return _morse_energy(positions, rho), positions

def _basin_hop(start_positions, potential, rho, max_iterations=100, initial_step=0.5, initial_temp=2.0):
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
    
    # Adaptive step size based on n
    # Larger n needs smaller steps to avoid collisions
    step_size *= (10.0 / n) ** 0.5
    
    for it in range(max_iterations):
        if _get_remaining_time() < 5.0:
            break
        
        noise = np.random.normal(0, step_size, current_positions.shape)
        new_positions = current_positions + noise
        
        new_energy, new_relaxed = _relax_and_validate(new_positions, potential, rho)
        
        if new_relaxed is None:
            # Try a smaller step
            noise = np.random.normal(0, step_size * 0.5, current_positions.shape)
            new_positions = current_positions + noise
            new_energy, new_relaxed = _relax_and_validate(new_positions, potential, rho)
        
        if new_relaxed is None:
            continue
        
        delta_e = new_energy - current_energy
        # Metropolis acceptance
        if delta_e < 0 or (temp > 1e-6 and np.random.rand() < math.exp(-delta_e / temp)):
            current_energy = new_energy
            current_positions = new_relaxed
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_positions.copy()
                
            # Occasionally increase step size to escape local minima
            if it % 10 == 0 and it > 0:
                step_size *= 1.1
        else:
            # If rejected, slightly decrease step size? Or keep it.
            pass
            
        temp *= 0.95
        step_size *= 0.99
        
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
    
    # Generate diverse initial guesses
    initial_guesses = []
    
    # 1. Icosahedral shells (good for small/medium n, low/medium rho)
    for seed in [42, 100, 200]:
        try:
            initial_guesses.append(_generate_icosahedral_shell(n, seed=seed))
        except Exception:
            pass
            
    # 2. FCC chunks (good for high rho or large n)
    try:
        initial_guesses.append(_generate_fcc_chunk(n, seed=42))
        initial_guesses.append(_generate_fcc_chunk(n, seed=100))
    except Exception:
        pass
        
    # 3. BCC chunks
    try:
        initial_guesses.append(_generate_bcc_chunk(n, seed=42))
    except Exception:
        pass
        
    # 4. Random clusters
    for seed in [42, 100, 200]:
        try:
            initial_guesses.append(_generate_random_cluster(n, seed=seed, min_dist=0.6))
        except Exception:
            pass

    # Relax all initial guesses
    relaxed_results = []
    for guess in initial_guesses:
        if _get_remaining_time() < 10.0:
            break
            
        # Ensure shape is correct
        if guess.shape[0] != n:
            continue
            
        energy, positions = _relax_and_validate(guess, potential, rho)
        if positions is not None and energy < float('inf'):
            relaxed_results.append((energy, positions))
    
    relaxed_results.sort(key=lambda x: x[0])
    
    if relaxed_results:
        best_energy = relaxed_results[0][0]
        best_positions = relaxed_results[0][1].copy()
    
    # Perform basin hopping from the best few relaxed results
    if best_positions is not None and _get_remaining_time() > 30.0:
        num_bh_starts = min(len(relaxed_results), 5)
        
        for idx in range(num_bh_starts):
            if _get_remaining_time() < 20.0:
                break
            
            start_energy, start_pos = relaxed_results[idx]
            
            # Adjust parameters based on rho and n
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
                max_iter = 150
            elif n <= 60:
                max_iter = 100
            else:
                max_iter = 60
            
            bh_energy, bh_positions = _basin_hop(
                start_pos, potential, rho,
                max_iterations=max_iter,
                initial_step=initial_step,
                initial_temp=initial_temp
            )
            
            if bh_positions is not None and bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_positions.copy()
    
    # Final polish
    if best_positions is not None and _get_remaining_time() > 5.0:
        polished_energy, polished_pos = _polish(best_positions, rho)
        if polished_pos is not None and polished_energy < best_energy:
            best_energy = polished_energy
            best_positions = polished_pos
    
    if best_positions is None:
        # Fallback: simple shell
        best_positions = _generate_icosahedral_shell(n, seed=0)
        if not _is_valid(best_positions):
            best_positions = best_positions * 1.5
    
    return best_positions
# EVOLVE-BLOCK-END
