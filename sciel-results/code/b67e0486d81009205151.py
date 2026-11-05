# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
from concurrent.futures import ThreadPoolExecutor, as_completed

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 320.0  # 320s wall clock limit (evaluator kills at 330s)

def _get_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return _DEADLINE - (time.time() - _GLOBAL_START_TIME)

def _morse_energy(x, rho):
    """Compute Morse energy for a set of positions (vectorized, upper triangle only)."""
    n = x.shape[0]
    if n < 2:
        return 0.0
    # Compute pairwise distances using upper triangle only
    i, j = np.triu_indices(n, k=1)
    diff = x[i] - x[j]
    dist = np.sqrt(np.sum(diff * diff, axis=1))
    er = rho * (1.0 - dist)
    er = np.clip(er, -500, 500)
    exp_er = np.exp(er)
    energy = np.sum(exp_er * (exp_er - 2.0))
    return energy

def _min_distance(x):
    """Compute minimum distance between any pair of atoms (vectorized)."""
    n = x.shape[0]
    if n < 2:
        return np.inf
    i, j = np.triu_indices(n, k=1)
    diff = x[i] - x[j]
    dist = np.sqrt(np.sum(diff * diff, axis=1))
    return np.min(dist)

def _is_valid(x):
    """Check if the configuration is valid (atoms at least 0.5 apart)."""
    return _min_distance(x) >= 0.5

def _rotation_matrix(angle, axis):
    """Create a rotation matrix."""
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-10:
        return np.eye(3)
    axis = axis / norm
    c = np.cos(angle)
    s = np.sin(angle)
    x, y, z = axis
    R = np.array([
        [c + x*x*(1-c),   x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s, c + y*y*(1-c),   y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)]
    ])
    return R

def _generate_fibonacci_sphere(n, radius=1.0, seed=42):
    """Generate points on a sphere using Fibonacci spiral for even distribution."""
    rng = np.random.default_rng(seed)
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Fibonacci spiral on sphere
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    positions = []
    for i in range(n):
        t = i / max(1, n - 1)
        z = 1.0 - 2.0 * t  # z goes from 1 to -1
        r = np.sqrt(max(0.0, 1.0 - z * z))
        theta = golden_angle * i
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        positions.append(np.array([x, y, z]))
    
    positions = np.array(positions) * radius
    # Center the cluster
    positions -= positions.mean(axis=0)
    return positions

def _generate_icosahedral_shell(n, seed=42):
    """Generate cluster using icosahedral symmetry shells."""
    rng = np.random.default_rng(seed)
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices (12)
    ico_verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])
    
    positions = [np.zeros(3)]  # Center atom
    
    # Add shells
    shell_radii = [1.0, 1.8, 2.5, 3.2, 3.8, 4.5]
    shell_counts = [12, 24, 42, 60, 80, 110]
    
    remaining = n - 1
    for radius, max_count in zip(shell_radii, shell_counts):
        if remaining <= 0:
            break
        count = min(remaining, max_count)
        # Use icosahedron vertices scaled, plus rotations
        for i in range(count):
            v = ico_verts[i % 12].copy()
            # Apply random rotation for diversity
            if i >= 12:
                angle = rng.uniform(0, 2 * math.pi)
                R = _rotation_matrix(angle, [0, 0, 1])
                v = R @ v
            positions.append(v * radius)
        remaining -= count
    
    # If still need more, add random points on outer shell
    while len(positions) < n:
        theta = rng.uniform(0, 2 * math.pi)
        phi_angle = rng.uniform(0, math.pi)
        r = shell_radii[-1] + 0.5
        p = np.array([
            r * np.sin(phi_angle) * np.cos(theta),
            r * np.sin(phi_angle) * np.sin(theta),
            r * np.cos(phi_angle)
        ])
        positions.append(p)
    
    return np.array(positions[:n])

def _generate_fcc_chunk(n, seed=42):
    """Generate FCC-like chunk."""
    rng = np.random.default_rng(seed)
    basis = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    
    # Estimate radius needed for n atoms in FCC
    # FCC density: ~1 atom per unit cell of volume 1, so radius ~ (n/4)^(1/3)
    radius = max(2.0, (n / 4.0) ** (1.0/3.0) + 1.0)
    max_coord = int(radius) + 2
    
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in basis:
                    p = np.array([i + dx, j + dy, k + dz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
    
    points.sort(key=lambda p: np.linalg.norm(p))
    
    if len(points) >= n:
        selected = np.array(points[:n])
    else:
        selected = np.array(points)
        while len(selected) < n:
            p = rng.uniform(-radius, radius, 3)
            selected = np.vstack([selected, [p]])
    
    return selected[:n]

def _generate_hcp_chunk(n, seed=42):
    """Generate HCP-like chunk."""
    rng = np.random.default_rng(seed)
    a = 1.0  # lattice parameter
    c = 1.633 * a  # ideal HCP c/a ratio
    
    # HCP basis
    basis = [
        (0, 0, 0),
        (a, 0, 0),
        (a/2, a*math.sqrt(3)/2, 0),
        (a/2, a*math.sqrt(3)/2, c/2),
        (a, 0, c/2),
        (0, 0, c)
    ]
    
    radius = max(2.0, (n / 2.0) ** (1.0/3.0) + 1.0)
    max_coord = int(radius / a) + 2
    
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for bx, by, bz in basis:
                    p = np.array([i*a + bx, j*a + by, k*c + bz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
    
    points.sort(key=lambda p: np.linalg.norm(p))
    
    if len(points) >= n:
        selected = np.array(points[:n])
    else:
        selected = np.array(points)
        while len(selected) < n:
            p = rng.uniform(-radius, radius, 3)
            selected = np.vstack([selected, [p]])
    
    return selected[:n]

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

def _polish(positions, rho, max_iter=500):
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
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
        if res.success or res.fun < 1e10:
            pos_res = res.x.reshape(-1, 3)
            if _min_distance(pos_res) >= 0.5:
                return res.fun, pos_res
    except Exception:
        pass
    
    return _morse_energy(positions, rho), positions

def _parallel_tempering_bh(start_positions_list, potential, rho, n_replicas=4, max_iterations=100):
    """
    Perform Parallel Tempering Basin Hopping.
    start_positions_list: list of initial positions for each replica.
    """
    n = start_positions_list[0].shape[0]
    
    # Initialize replicas
    replicas = []
    for i, start_pos in enumerate(start_positions_list):
        energy, pos = _relax_and_validate(start_pos, potential, rho)
        if pos is None:
            # Fallback if relaxation fails
            energy = _morse_energy(start_pos, rho)
            pos = start_pos
        replicas.append({
            'pos': pos,
            'energy': energy,
            'step_size': 0.5 * (1.0 + 0.5 * i), # Increasing step size for "hotter" replicas
            'temp': 2.0 * (1.0 + 0.5 * i)       # Increasing temperature for "hotter" replicas
        })
    
    best_energy = min(r['energy'] for r in replicas)
    best_pos = replicas[np.argmin([r['energy'] for r in replicas])]['pos'].copy()
    
    for it in range(max_iterations):
        if _get_remaining_time() < 10.0:
            break
            
        # 1. Move each replica
        for i, rep in enumerate(replicas):
            # Perturb
            noise = np.random.normal(0, rep['step_size'], rep['pos'].shape)
            new_pos = rep['pos'] + noise
            
            # Relax
            new_energy, new_relaxed = _relax_and_validate(new_pos, potential, rho)
            
            if new_relaxed is None:
                # If relaxation fails, try smaller step
                noise_small = np.random.normal(0, rep['step_size'] * 0.5, rep['pos'].shape)
                new_pos_small = rep['pos'] + noise_small
                new_energy, new_relaxed = _relax_and_validate(new_pos_small, potential, rho)
                
                if new_relaxed is None:
                    # Keep current
                    continue
            
            # Metropolis Accept/Reject
            delta_e = new_energy - rep['energy']
            if delta_e < 0 or np.random.rand() < math.exp(-delta_e / rep['temp']):
                rep['pos'] = new_relaxed
                rep['energy'] = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pos = new_relaxed.copy()
        
        # 2. Attempt swaps between adjacent replicas (i, i+1)
        for i in range(len(replicas) - 1):
            rep1 = replicas[i]
            rep2 = replicas[i+1]
            
            # Calculate swap probability
            # P_swap = min(1, exp(-(E1/T2 - E2/T1)))
            # Note: In PT, we usually define "temperature" as a parameter that controls step size and acceptance.
            # Here we use the 'temp' field as the effective temperature for the Metropolis swap.
            delta_swap = (rep1['energy'] / rep2['temp']) - (rep2['energy'] / rep1['temp'])
            if delta_swap < 0 or np.random.rand() < math.exp(-delta_swap):
                # Swap positions and energies
                rep1['pos'], rep2['pos'] = rep2['pos'], rep1['pos']
                rep1['energy'], rep2['energy'] = rep2['energy'], rep1['energy']
                # Swap step sizes and temps? Usually yes, to maintain the "ladder"
                rep1['step_size'], rep2['step_size'] = rep2['step_size'], rep1['step_size']
                rep1['temp'], rep2['temp'] = rep2['temp'], rep1['temp']
                
                # Update best if needed (energies didn't change, just labels)
                # No need to update best_pos/energy here as they are global bests

    return best_energy, best_pos

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
    
    # 1. Fibonacci sphere
    try:
        fib = _generate_fibonacci_sphere(n, radius=2.0, seed=42)
        initial_guesses.append(fib)
    except Exception:
        pass
    
    # 2. Icosahedral shells
    try:
        ico = _generate_icosahedral_shell(n, seed=42)
        initial_guesses.append(ico)
    except Exception:
        pass
    
    # 3. FCC chunk
    try:
        fcc = _generate_fcc_chunk(n, seed=42)
        initial_guesses.append(fcc)
    except Exception:
        pass
    
    # 4. HCP chunk
    try:
        hcp = _generate_hcp_chunk(n, seed=42)
        initial_guesses.append(hcp)
    except Exception:
        pass
    
    # 5. Random compact cluster
    try:
        rng = np.random.default_rng(123)
        radius = 2.5
        positions = []
        for _ in range(n):
            p = rng.uniform(-radius, radius, 3)
            positions.append(p)
        initial_guesses.append(np.array(positions))
    except Exception:
        pass
        
    # 6. Add more random starts if time permits
    if len(initial_guesses) < 6 and _get_remaining_time() > 200:
        for seed in [7, 13, 23, 37]:
            try:
                rng = np.random.default_rng(seed)
                radius = 2.0 + 0.5 * seed
                positions = []
                for _ in range(n):
                    p = rng.uniform(-radius, radius, 3)
                    positions.append(p)
                initial_guesses.append(np.array(positions))
            except Exception:
                pass

    # Relax all initial guesses to get valid starting points for PT
    relaxed_guesses = []
    for guess in initial_guesses:
        if _get_remaining_time() < 30.0:
            break
        energy, pos = _relax_and_validate(guess, potential, rho)
        if pos is not None:
            relaxed_guesses.append((energy, pos))
    
    if not relaxed_guesses:
        # Fallback if all relaxations failed
        return initial_guesses[0]

    # Sort by energy
    relaxed_guesses.sort(key=lambda x: x[0])
    
    # Select top N replicas for Parallel Tempering
    n_replicas = min(4, len(relaxed_guesses))
    pt_starts = [pos for e, pos in relaxed_guesses[:n_replicas]]
    
    # Determine max iterations based on remaining time
    # Assume ~0.1s per iteration per replica (optimistic)
    # Total time for PT = max_iter * n_replicas * time_per_iter
    # We want to use ~80% of remaining time
    remaining = _get_remaining_time()
    # Rough estimate: 1s per iteration for N=80
    max_iter = int(remaining * 0.7 / (n_replicas * 1.0))
    max_iter = max(20, min(max_iter, 200)) # Clamp between 20 and 200
    
    pt_energy, pt_pos = _parallel_tempering_bh(pt_starts, potential, rho, n_replicas=n_replicas, max_iterations=max_iter)
    
    if pt_pos is not None and pt_energy < best_energy:
        best_energy = pt_energy
        best_positions = pt_pos
    
    # Final polish
    if best_positions is not None and _get_remaining_time() > 5.0:
        polished_energy, polished_pos = _polish(best_positions, rho, max_iter=200)
        if polished_pos is not None and polished_energy < best_energy:
            best_energy = polished_energy
            best_positions = polished_pos
    elif best_positions is None:
        # If PT failed completely, use best relaxed guess
        best_positions = relaxed_guesses[0][1]

    if best_positions is None:
        # Fallback: simple shell
        rng = np.random.default_rng(0)
        positions = [np.array([0.0, 0.0, 0.0])]
        remaining = n - 1
        shell_index = 1
        while remaining > 0:
            shell_radius = shell_index * 1.1
            max_in_shell = min(remaining, int(4 * shell_index * (shell_index + 1)))
            if max_in_shell <= 0:
                max_in_shell = 1
            for _ in range(max_in_shell):
                u = rng.uniform(-1, 1)
                theta = rng.uniform(0, 2 * np.pi)
                r = shell_radius
                x = r * np.sqrt(1 - u**2) * np.cos(theta)
                y = r * np.sqrt(1 - u**2) * np.sin(theta)
                z = r * u
                positions.append(np.array([x, y, z]))
            remaining -= max_in_shell
            shell_index += 1
        best_positions = np.array(positions[:n])
        if not _is_valid(best_positions):
            best_positions = best_positions * 1.5

    return best_positions
# EVOLVE-BLOCK-END
