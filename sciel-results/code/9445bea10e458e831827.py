# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 320.0  # 320s wall clock limit (evaluator kills at 330s)

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
    return float(energy)

def _min_distance(x):
    """Compute minimum distance between any pair of atoms (vectorized)."""
    n = x.shape[0]
    if n < 2:
        return np.inf
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    return float(np.min(dist))

def _is_valid(x):
    """Check if the configuration is valid (atoms at least 0.5 apart)."""
    return _min_distance(x) >= 0.5

def _generate_icosahedral_cluster(n, seed=42):
    """
    Generate an icosahedral cluster with concentric shells.
    Icosahedral shells: 1, 12, 42, 120, ...
    This is optimal for short-range Morse potentials (high rho).
    """
    rng = np.random.default_rng(seed)
    
    # Icosahedron vertices (12 vertices)
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    norm = np.linalg.norm(vertices[0])
    vertices = vertices / norm
    
    positions = [np.array([0.0, 0.0, 0.0])]
    remaining = n - 1
    
    # Shell 1: 12 atoms on icosahedron
    shell1_radius = 1.0
    if remaining >= 12:
        for v in vertices:
            positions.append(v * shell1_radius)
        remaining -= 12
    
    # Shell 2: 42 atoms (truncated icosahedron / fullerenes)
    if remaining >= 1:
        # Generate points on a sphere for second shell
        shell2_radius = 2.0
        # Use golden spiral for uniform distribution
        n_in_shell = min(remaining, 42)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n_in_shell):
            z = 1.0 - 2.0 * i / (n_in_shell - 1) if n_in_shell > 1 else 0.0
            r = np.sqrt(1.0 - z*z)
            theta = golden_angle * i
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * shell2_radius
            positions.append(p)
        remaining -= n_in_shell
    
    # Add more shells if needed
    shell_idx = 3
    while remaining > 0:
        shell_radius = shell_idx * 1.0
        max_in_shell = min(remaining, 4 * shell_idx * (shell_idx + 1) + 2)
        for _ in range(max_in_shell):
            u = rng.uniform(-1, 1)
            theta = rng.uniform(0, 2 * np.pi)
            x = shell_radius * np.sqrt(1 - u**2) * np.cos(theta)
            y = shell_radius * np.sqrt(1 - u**2) * np.sin(theta)
            z = shell_radius * u
            positions.append(np.array([x, y, z]))
        remaining -= max_in_shell
        shell_idx += 1
    
    return np.array(positions[:n])

def _generate_fcc_cluster(n, seed=42):
    """
    Generate an FCC-like cluster.
    Optimal for long-range Morse potentials (low rho).
    """
    rng = np.random.default_rng(seed)
    
    radius = 1.2 * (n ** (1.0/3.0)) + 1.0
    points = []
    
    # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) in units of lattice constant
    # For pair distance 1, lattice constant a = 1/sqrt(2)
    a = 1.0 / np.sqrt(2.0)
    basis = [(0, 0, 0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)]
    
    max_coord = int(radius / a) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in basis:
                    p = a * np.array([i + dx, j + dy, k + dz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
    
    # Sort by distance from center
    points.sort(key=lambda p: np.linalg.norm(p))
    
    if len(points) >= n:
        return np.array(points[:n])
    else:
        # Fill in with random points
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                points.append(p)
            else:
                points.append(p * 0.5)
        return np.array(points)

def _generate_bcc_cluster(n, seed=42):
    """Generate a BCC-like cluster."""
    rng = np.random.default_rng(seed)
    
    radius = 1.3 * (n ** (1.0/3.0)) + 1.0
    points = []
    
    # BCC basis: (0,0,0), (0.5,0.5,0.5)
    a = 1.0 / np.sqrt(3.0)  # for pair distance 1
    basis = [(0, 0, 0), (0.5, 0.5, 0.5)]
    
    max_coord = int(radius / a) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in basis:
                    p = a * np.array([i + dx, j + dy, k + dz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
    
    points.sort(key=lambda p: np.linalg.norm(p))
    
    if len(points) >= n:
        return np.array(points[:n])
    else:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
        return np.array(points)

def _generate_hcp_cluster(n, seed=42):
    """Generate an HCP-like cluster."""
    rng = np.random.default_rng(seed)
    
    radius = 1.3 * (n ** (1.0/3.0)) + 1.0
    points = []
    
    # HCP: ABAB stacking
    a = 1.0  # in-plane spacing
    c = 1.633 * a  # c/a ratio for HCP
    
    max_coord = int(radius / a) + 2
    for layer in range(-int(radius/c), int(radius/c) + 1):
        z = layer * c
        offset = 0.5 if layer % 2 != 0 else 0.0
        for i in range(-max_coord, max_coord + 1):
            for j in range(-max_coord, max_coord + 1):
                x = a * (i + offset + 0.5 * (j % 2))
                y = a * j * np.sqrt(3.0)/2
                p = np.array([x, y, z])
                if np.linalg.norm(p) <= radius:
                    points.append(p)
    
    points.sort(key=lambda p: np.linalg.norm(p))
    
    if len(points) >= n:
        return np.array(points[:n])
    else:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
        return np.array(points)

def _generate_random_cluster(n, seed=42, min_dist=0.6):
    """Generate a random cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    
    radius = 1.8 * (n ** (1.0/3.0)) + 0.5
    points = []
    attempts = 0
    max_attempts = 100000
    
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

def _fast_relax(positions, potential, rho):
    """Fast local relaxation using scipy minimize with limited iterations."""
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
        return np.asarray(grad_3d).flatten()
    
    try:
        res = minimize(objective, x_flat, jac=gradient, method='L-BFGS-B',
                       options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < 1e10:
            pos_res = res.x.reshape(-1, 3)
            if _min_distance(pos_res) >= 0.5:
                return float(res.fun), pos_res
    except Exception:
        pass
    
    return _morse_energy(positions, rho), positions

def _polish(positions, rho, max_iter=300):
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
        return np.asarray(grad_3d).flatten()
    
    try:
        res = minimize(objective, x_flat, jac=gradient, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
        if res.success or res.fun < 1e10:
            pos_res = res.x.reshape(-1, 3)
            if _min_distance(pos_res) >= 0.5:
                return float(res.fun), pos_res
    except Exception:
        pass
    
    return _morse_energy(positions, rho), positions

def _simulated_annealing(start_positions, potential, rho, 
                         t_start=50.0, t_end=0.1, 
                         max_iterations=2000, 
                         cooling_rate=None):
    """
    Simulated annealing with L-BFGS local minimization after each move.
    This is structurally different from basin hopping: SA uses a temperature
    schedule and Metropolis acceptance, which provides a more systematic
    exploration of the energy landscape.
    """
    n = start_positions.shape[0]
    rng = np.random.default_rng()
    
    # Initial relaxation
    current_energy, current_positions = _fast_relax(start_positions, potential, rho)
    if current_energy >= 1e9:
        return float('inf'), None
    
    best_energy = current_energy
    best_positions = current_positions.copy()
    
    # Set cooling rate if not provided
    if cooling_rate is None:
        cooling_rate = (t_end / t_start) ** (1.0 / max_iterations)
    
    # Adaptive step size based on n and rho
    base_step = 0.3 + 0.2 * math.sqrt(n / 40.0)
    if rho <= 4:
        base_step *= 1.3  # longer range: larger steps
    elif rho >= 10:
        base_step *= 0.8  # shorter range: smaller steps
    
    for it in range(max_iterations):
        if _get_remaining_time() < 5.0:
            break
        
        # Temperature for this iteration
        temp = t_start * (cooling_rate ** it)
        
        # Generate perturbation: Gaussian noise with adaptive step size
        step_size = base_step * (0.5 + 0.5 * temp / t_start)
        noise = rng.normal(0, step_size, current_positions.shape)
        new_positions = current_positions + noise
        
        # Local relaxation after perturbation
        new_energy, relaxed_positions = _fast_relax(new_positions, potential, rho)
        
        if relaxed_positions is None or new_energy >= 1e9:
            continue
        
        delta_e = new_energy - current_energy
        
        # Metropolis acceptance criterion
        if delta_e < 0:
            accept = True
        else:
            accept = (temp > 1e-10 and rng.random() < math.exp(-delta_e / temp))
        
        if accept:
            current_energy = new_energy
            current_positions = relaxed_positions
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_positions.copy()
        
        # Occasionally reheat to escape deep local minima
        if it > 0 and it % 500 == 0:
            # Reheat to a fraction of initial temperature
            pass  # cooling_rate handles this implicitly
    
    return best_energy, best_positions

def construct(instance):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 6))
    potential = instance["potential"]
    
    if rho is None:
        rho = 6
    
    best_energy = float('inf')
    best_positions = None
    
    initial_guesses = []
    
    # Choose initial guesses based on rho
    if rho <= 4:
        # Long range: FCC-like structures
        for seed in [42, 1, 2, 3]:
            try:
                initial_guesses.append(_generate_fcc_cluster(n, seed=seed))
            except Exception:
                pass
        for seed in [4, 5]:
            try:
                initial_guesses.append(_generate_hcp_cluster(n, seed=seed))
            except Exception:
                pass
        for seed in [10, 11, 12]:
            try:
                initial_guesses.append(_generate_random_cluster(n, seed=seed))
            except Exception:
                pass
    elif rho <= 7:
        # Medium range: mix of FCC and icosahedral
        for seed in [42, 1, 2]:
            try:
                initial_guesses.append(_generate_fcc_cluster(n, seed=seed))
            except Exception:
                pass
        for seed in [42, 1, 2]:
            try:
                initial_guesses.append(_generate_icosahedral_cluster(n, seed=seed))
            except Exception:
                pass
        for seed in [3, 4, 5]:
            try:
                initial_guesses.append(_generate_bcc_cluster(n, seed=seed))
            except Exception:
                pass
        for seed in [10, 11, 12, 13]:
            try:
                initial_guesses.append(_generate_random_cluster(n, seed=seed))
            except Exception:
                pass
    else:
        # Short range: icosahedral structures
        for seed in [42, 1, 2, 3]:
            try:
                initial_guesses.append(_generate_icosahedral_cluster(n, seed=seed))
            except Exception:
                pass
        for seed in [4, 5]:
            try:
                initial_guesses.append(_generate_bcc_cluster(n, seed=seed))
            except Exception:
                pass
        for seed in [10, 11, 12, 13, 14]:
            try:
                initial_guesses.append(_generate_random_cluster(n, seed=seed))
            except Exception:
                pass
    
    # Fast relax all initial guesses
    relaxed_results = []
    for guess in initial_guesses:
        if _get_remaining_time() < 10.0:
            break
        
        energy, positions = _fast_relax(guess, potential, rho)
        if positions is not None and energy < float('inf'):
            relaxed_results.append((energy, positions))
    
    relaxed_results.sort(key=lambda x: x[0])
    
    if relaxed_results:
        best_energy = relaxed_results[0][0]
        best_positions = relaxed_results[0][1].copy()
    
    # Simulated annealing from multiple starting points
    # Allocate time proportionally: more iterations for smaller n
    if n <= 40:
        sa_max_iter = 3000
        num_sa_starts = min(len(relaxed_results), 6)
    elif n <= 60:
        sa_max_iter = 2000
        num_sa_starts = min(len(relaxed_results), 5)
    else:
        sa_max_iter = 1500
        num_sa_starts = min(len(relaxed_results), 4)
    
    # Temperature schedule parameters based on rho
    if rho <= 4:
        t_start = 80.0
        t_end = 0.05
    elif rho <= 7:
        t_start = 50.0
        t_end = 0.1
    else:
        t_start = 30.0
        t_end = 0.2
    
    for idx in range(num_sa_starts):
        if _get_remaining_time() < 8.0:
            break
        
        start_energy, start_pos = relaxed_results[idx]
        
        # Use different temperature schedules for different starts
        # to diversify the search
        t_start_i = t_start * (1.0 + 0.1 * idx)
        t_end_i = t_end * (1.0 + 0.05 * idx)
        
        sa_energy, sa_positions = _simulated_annealing(
            start_pos, potential, rho,
            t_start=t_start_i,
            t_end=t_end_i,
            max_iterations=sa_max_iter
        )
        
        if sa_positions is not None and sa_energy < best_energy:
            best_energy = sa_energy
            best_positions = sa_positions.copy()
    
    # Final polish with high precision
    if best_positions is not None and _get_remaining_time() > 5.0:
        polished_energy, polished_pos = _polish(best_positions, rho, max_iter=300)
        if polished_pos is not None and polished_energy < best_energy:
            best_energy = polished_energy
            best_positions = polished_pos
    
    # Safety: if no valid positions found, generate a fallback
    if best_positions is None or not _is_valid(best_positions):
        best_positions = _generate_icosahedral_cluster(n, seed=0)
        if not _is_valid(best_positions):
            best_positions = _generate_fcc_cluster(n, seed=0)
        if not _is_valid(best_positions):
            best_positions = _generate_random_cluster(n, seed=99)
    
    return best_positions
# EVOLVE-BLOCK-END
