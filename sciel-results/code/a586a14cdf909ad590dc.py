"""
Strategy: Basin hopping with vectorized Morse energy/gradient and diverse crystalline starts.
Uses vectorized energy/gradient computation for speed, multiple diverse initializations
(FCC, BCC, HCP, shell, random), and basin hopping to escape local minima.
"""
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy_and_grad(positions, rho):
    """Compute Morse energy and gradient using vectorized operations."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Compute all pair differences: shape (n, n, 3)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # Mask diagonal (self-interactions)
    mask = ~np.eye(n, dtype=bool)
    
    # Compute exp(rho*(1-r)) for all pairs
    with np.errstate(over='ignore'):
        exp_term = np.exp(rho * (1.0 - dist))
    
    # Pair energy: exp_term * (exp_term - 2)
    pair_e = exp_term * (exp_term - 2.0)
    pair_e[~mask] = 0.0
    
    energy = 0.5 * np.sum(pair_e)
    
    # Gradient: dE/dx_i = sum_j dE/dr_ij * (x_i - x_j)/r_ij
    # dE/dr = -rho * exp_term * (2*exp_term - 2)
    dE_dr = -rho * exp_term * (2.0 * exp_term - 2.0)
    dE_dr[~mask] = 0.0
    
    # factor = dE_dr / r
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = dE_dr / dist
    factor[~mask] = 0.0
    factor[np.isnan(factor)] = 0.0
    
    # grad[i] = sum_j factor[i,j] * (x_i - x_j)
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad


def _morse_energy_only(positions, rho):
    """Compute only Morse energy (vectorized)."""
    e, _ = _morse_energy_and_grad(positions, rho)
    return e


def _is_valid(positions, min_dist=0.5):
    """Check if all pairs are at least min_dist apart."""
    n = len(positions)
    if n < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    mask = ~np.eye(n, dtype=bool)
    return np.min(dist[mask]) >= min_dist


def _generate_fcc(n, spacing=1.0):
    """Generate FCC-like cluster."""
    lattice = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                lattice.append([i * spacing, j * spacing, k * spacing])
                lattice.append([(i + 0.5) * spacing, (j + 0.5) * spacing, k * spacing])
                lattice.append([(i + 0.5) * spacing, j * spacing, (k + 0.5) * spacing])
                lattice.append([i * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing])
    
    unique = []
    seen = set()
    for p in lattice:
        key = (round(p[0], 3), round(p[1], 3), round(p[2], 3))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    unique.sort(key=lambda p: np.linalg.norm(p))
    return np.array(unique[:n], dtype=float)


def _generate_bcc(n, spacing=1.0):
    """Generate BCC-like cluster."""
    lattice = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                lattice.append([i * spacing, j * spacing, k * spacing])
                lattice.append([(i + 0.5) * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing])
    
    unique = []
    seen = set()
    for p in lattice:
        key = (round(p[0], 3), round(p[1], 3), round(p[2], 3))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    unique.sort(key=lambda p: np.linalg.norm(p))
    return np.array(unique[:n], dtype=float)


def _generate_hcp(n, spacing=1.0):
    """Generate HCP-like cluster."""
    a = spacing
    c = spacing * np.sqrt(8.0 / 3.0)
    lattice = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-4, 5):
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0) / 2.0 * j
                z = k * c
                lattice.append([x, y, z])
                x2 = a * (i + 0.5 * ((j + 1) % 2)) + a / 2.0
                y2 = a * np.sqrt(3.0) / 2.0 * j + a * np.sqrt(3.0) / 6.0
                z2 = (k + 0.5) * c
                lattice.append([x2, y2, z2])
    
    unique = []
    seen = set()
    for p in lattice:
        key = (round(p[0], 3), round(p[1], 3), round(p[2], 3))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    unique.sort(key=lambda p: np.linalg.norm(p))
    return np.array(unique[:n], dtype=float)


def _generate_shell(n, spacing=1.0):
    """Generate spherical shell-like cluster."""
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    radius = spacing * n ** (1.0 / 3.0) * 0.5
    points = []
    for i in range(n):
        z = 1.0 - 2.0 * i / max(n - 1, 1)
        radius_xy = np.sqrt(max(0, 1.0 - z * z))
        theta = 2.0 * np.pi * i / golden_ratio
        x = radius_xy * np.cos(theta)
        y = radius_xy * np.sin(theta)
        points.append([x * radius, y * radius, z * radius])
    return np.array(points, dtype=float)


def _generate_icosahedral(n, spacing=1.0):
    """Generate icosahedral-like cluster for short-range potentials."""
    # Start with icosahedron vertices and subdivide
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # 12 vertices of regular icosahedron
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico[0])
    
    points = list(ico)
    
    # Add more points by placing on faces or expanding
    rng = np.random.default_rng(42)
    while len(points) < n:
        # Add point near existing points but offset
        idx = rng.integers(0, len(points))
        base = points[idx].copy()
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        new_pt = base + direction * 0.8
        # Check minimum distance
        if all(np.linalg.norm(new_pt - p) > 0.6 for p in points):
            points.append(new_pt)
    
    return np.array(points[:n], dtype=float) * spacing


def _generate_random_compact(n, spacing=1.0, seed=42):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 200:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if points:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= 0.6 * spacing:
                    points.append(p)
            else:
                points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points, dtype=float)


def _local_minimize(positions, rho, maxiter=100):
    """Relax using L-BFGS-B with vectorized Morse energy and gradient."""
    def objective_and_grad(x):
        pos = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(pos, rho)
        return e, g.flatten()
    
    x0 = positions.flatten()
    result = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B',
                      options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8})
    return result.fun, result.x.reshape(-1, 3)


def _basin_hop(positions, rho, n_hops=5, step_size=0.5, rng=None):
    """Perform basin hopping: perturb, relax, accept if lower energy."""
    if rng is None:
        rng = np.random.default_rng()
    
    current_energy, current_pos = _local_minimize(positions, rho)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    for _ in range(n_hops):
        # Random perturbation
        perturbation = rng.normal(0, step_size, size=current_pos.shape)
        perturbed = current_pos + perturbation
        
        # Relax
        hop_energy, hop_pos = _local_minimize(perturbed, rho)
        
        # Metropolis-like acceptance (always accept if lower, sometimes accept if higher)
        if hop_energy < current_energy:
            current_energy = hop_energy
            current_pos = hop_pos
        else:
            # Accept with some probability to escape local minima
            delta_e = hop_energy - current_energy
            if delta_e < 5.0 and rng.random() < np.exp(-delta_e / 2.0):
                current_energy = hop_energy
                current_pos = hop_pos
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
    
    return best_energy, best_pos


def construct(instance):
    """Construct lowest-energy Morse cluster using basin hopping with diverse starts."""
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s limit
    
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(12345)
    
    # Generate diverse starting points
    generators = [
        lambda: _generate_fcc(size),
        lambda: _generate_bcc(size),
        lambda: _generate_hcp(size),
        lambda: _generate_shell(size),
        lambda: _generate_icosahedral(size),
        lambda: _generate_random_compact(size, seed=42),
        lambda: _generate_random_compact(size, seed=123),
        lambda: _generate_random_compact(size, seed=789),
        lambda: _generate_fcc(size, spacing=0.95),
        lambda: _generate_bcc(size, spacing=1.05),
    ]
    
    best_energy = np.inf
    best_positions = None
    
    # Phase 1: Local minimize from each diverse start
    for gen_idx, gen_func in enumerate(generators):
        if time.time() > deadline - 60:  # Reserve time for basin hopping
            break
        
        try:
            init_pos = gen_func()
            if len(init_pos) < size:
                continue
            
            energy, relaxed_pos = _local_minimize(init_pos, rho, maxiter=150)
            
            if _is_valid(relaxed_pos) and energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos.copy()
                
        except Exception:
            continue
    
    # Phase 2: Basin hopping from the best positions found
    if best_positions is not None and time.time() < deadline - 30:
        try:
            hop_energy, hop_pos = _basin_hop(best_positions, rho, n_hops=8, 
                                             step_size=0.4, rng=rng)
            if _is_valid(hop_pos) and hop_energy < best_energy:
                best_energy = hop_energy
                best_positions = hop_pos.copy()
        except Exception:
            pass
    
    # Phase 3: Try basin hopping from other good starting points if time permits
    if time.time() < deadline - 20:
        # Try from a few other diverse starts
        for gen_idx in [0, 1, 2]:  # FCC, BCC, HCP
            if time.time() > deadline - 20:
                break
            try:
                init_pos = generators[gen_idx]()
                if len(init_pos) < size:
                    continue
                energy, relaxed_pos = _local_minimize(init_pos, rho, maxiter=100)
                if _is_valid(relaxed_pos):
                    hop_energy, hop_pos = _basin_hop(relaxed_pos, rho, n_hops=4,
                                                     step_size=0.5, rng=rng)
                    if _is_valid(hop_pos) and hop_energy < best_energy:
                        best_energy = hop_energy
                        best_positions = hop_pos.copy()
            except Exception:
                continue
    
    # Fallback if no valid result
    if best_positions is None:
        init_pos = _generate_random_compact(size, seed=42)
        energy, best_positions = relax(init_pos, potential, rho)
    
    return best_positions
