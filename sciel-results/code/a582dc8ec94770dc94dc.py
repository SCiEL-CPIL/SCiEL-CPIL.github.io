"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Parameters for each kind
KIND_PARAMS = {
    '12-6': {'n_exp': 12, 'm_exp': 6, 'c': 144.41},
    '9-6': {'n_exp': 9, 'm_exp': 6, 'c': 39.432},
    '10-8': {'n_exp': 10, 'm_exp': 8, 'c': 34.408},
}

# Total time budget in seconds
TOTAL_BUDGET = 300.0
SAFETY_MARGIN = 15.0  # seconds


def _compute_energy(positions, n_exp, m_exp, c):
    """Compute Sutton-Chen energy for a set of positions."""
    n = len(positions)
    if n < 2:
        return 0.0
    # Compute all pair distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    
    # Pair terms
    r_n = dist ** (-n_exp)
    r_m = dist ** (-m_exp)
    
    # Sum over j for each i
    sum_rn = np.sum(r_n, axis=1)
    sum_rm = np.sum(r_m, axis=1)
    
    # Sutton-Chen energy
    e_atom = 0.5 * sum_rn - c * np.sqrt(np.maximum(sum_rm, 0))
    return np.sum(e_atom)


def _generate_fcc(n, seed=0):
    """Generate positions on an FCC lattice."""
    rng = np.random.default_rng(seed)
    # FCC lattice vectors
    a = 1.0
    # Generate a cube of FCC points
    max_coord = int(np.ceil(n ** (1/3) * 1.5))
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC positions: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in units of a
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    points.append(p)
    points = np.array(points)
    if len(points) < n:
        # Add random points
        while len(points) < n:
            p = rng.uniform(-max_coord, max_coord, 3)
            points = np.vstack([points, p])
    else:
        # Randomly select n points
        indices = rng.choice(len(points), n, replace=False)
        points = points[indices]
    return points


def _generate_hcp(n, seed=0):
    """Generate positions on an HCP lattice."""
    rng = np.random.default_rng(seed)
    # HCP lattice
    a = 1.0
    c = np.sqrt(8/3) * a
    # Generate hexagonal layers
    max_coord = int(np.ceil(n ** (1/3) * 1.5))
    points = []
    for layer in range(-max_coord, max_coord + 1):
        z = layer * c / 2
        if layer % 2 == 0:
            # A layer
            for i in range(-max_coord, max_coord + 1):
                for j in range(-max_coord, max_coord + 1):
                    x = a * (i + 0.5 * (j % 2))
                    y = a * np.sqrt(3) / 2 * j
                    points.append(np.array([x, y, z]))
        else:
            # B layer
            for i in range(-max_coord, max_coord + 1):
                for j in range(-max_coord, max_coord + 1):
                    x = a * (i + 0.5)
                    y = a * np.sqrt(3) / 2 * (j + 0.5)
                    points.append(np.array([x, y, z]))
    points = np.array(points)
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-max_coord * a, max_coord * a, 3)
            points = np.vstack([points, p])
    else:
        indices = rng.choice(len(points), n, replace=False)
        points = points[indices]
    return points


def _generate_icosahedral(n, seed=0):
    """Generate positions based on icosahedral shells."""
    rng = np.random.default_rng(seed)
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico_vertices = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ])
    ico_vertices = ico_vertices / np.linalg.norm(ico_vertices, axis=1, keepdims=True)
    
    points = [np.zeros(3)]  # Center
    # Shell 1: 12 vertices
    for v in ico_vertices:
        points.append(v * 1.0)
    
    # Shell 2: 30 points (midpoints of edges)
    if len(points) < n:
        edge_midpoints = []
        for i in range(12):
            for j in range(i + 1, 12):
                if np.linalg.norm(ico_vertices[i] - ico_vertices[j]) < 0.5:
                    mid = (ico_vertices[i] + ico_vertices[j]) / 2
                    mid = mid / np.linalg.norm(mid)
                    edge_midpoints.append(mid * 1.5)
        for p in edge_midpoints[:n - len(points)]:
            points.append(p)
    
    # Add more shells if needed
    while len(points) < n:
        # Fibonacci sphere for next shell
        num_pts = min(n - len(points), 30)
        radius = 2.0 + 0.5 * (len(points) // 30)
        for i in range(num_pts):
            z = 1 - 2 * (i + 0.5) / num_pts
            r = np.sqrt(1 - z ** 2)
            theta = 2 * np.pi * i / num_pts
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * radius
            points.append(p)
    
    return np.array(points[:n])


def _relax_with_deadline(positions, potential, kind, deadline):
    """Relax positions with a deadline check."""
    try:
        energy, relaxed = relax(positions, potential, kind)
        if time.time() > deadline:
            return energy, positions
        return energy, relaxed
    except Exception:
        return np.inf, positions


def construct(instance):
    """Search for global minimum of Sutton-Chen cluster using multiple initializations."""
    start_time = time.time()
    deadline = start_time + TOTAL_BUDGET - SAFETY_MARGIN
    
    n = instance['n']
    potential = instance['potential']
    kind = instance['kind']
    params = KIND_PARAMS[kind]
    n_exp = params['n_exp']
    m_exp = params['m_exp']
    c = params['c']
    
    best_energy = np.inf
    best_positions = None
    
    # Generate various initial configurations
    initializers = [
        ('fcc', _generate_fcc),
        ('hcp', _generate_hcp),
        ('ico', _generate_icosahedral),
    ]
    
    # Try multiple seeds for each initializer
    seeds = range(8)
    
    for init_name, init_func in initializers:
        for seed in seeds:
            if time.time() > deadline:
                break
            
            try:
                positions = init_func(n, seed=seed)
            except Exception:
                continue
            
            # Ensure valid positions
            if not np.all(np.isfinite(positions)):
                continue
            
            # Check minimum distance
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            min_dist = np.sqrt(np.min(dist_sq))
            if min_dist < 0.35:
                continue
            
            # Relax
            energy, relaxed = _relax_with_deadline(positions, potential, kind, deadline)
            
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed.copy()
        
        if time.time() > deadline:
            break
    
    # If we haven't found a valid solution, try random initialization
    if best_positions is None:
        rng = np.random.default_rng(42)
        for _ in range(10):
            if time.time() > deadline:
                break
            positions = rng.uniform(-1, 1, (n, 3))
            # Scale to appropriate size
            positions *= 0.7 * n ** (1/3)
            
            # Check validity
            if not np.all(np.isfinite(positions)):
                continue
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            min_dist = np.sqrt(np.min(dist_sq))
            if min_dist < 0.35:
                continue
            
            energy, relaxed = _relax_with_deadline(positions, potential, kind, deadline)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed.copy()
    
    # Final polish with tighter tolerance if time allows
    if best_positions is not None and time.time() < deadline - 5:
        try:
            # Use scipy minimize with tighter settings
            def energy_func(x):
                return _compute_energy(x.reshape(-1, 3), n_exp, m_exp, c)
            
            result = minimize(
                energy_func,
                best_positions.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < best_energy:
                best_positions = result.x.reshape(-1, 3)
                best_energy = result.fun
        except Exception:
            pass
    
    # Ensure final validity
    if best_positions is None:
        # Fallback: simple random
        rng = np.random.default_rng(0)
        best_positions = rng.uniform(-1, 1, (n, 3))
    
    # Final validity check
    if not np.all(np.isfinite(best_positions)):
        rng = np.random.default_rng(0)
        best_positions = rng.uniform(-1, 1, (n, 3))
    
    diff = best_positions[:, np.newaxis, :] - best_positions[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist = np.sqrt(np.min(dist_sq))
    if min_dist < 0.35:
        # Spread out points
        scale = 0.36 / min_dist
        best_positions *= scale
    
    return best_positions
# EVOLVE-BLOCK-END
