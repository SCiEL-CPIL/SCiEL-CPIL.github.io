"""Fragment-merger strategy for Dzugutov cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _dzugutov_energy(pos):
    """Compute Dzugutov potential energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(r, np.inf)
    E = 0.0
    # r < a term: A*(r^-m - B)*exp(c/(r-a))
    mask_a = r < a
    if np.any(mask_a):
        ra = r[mask_a]
        E += np.sum(A * (ra**(-m) - B) * np.exp(c / (ra - a)))
    # a <= r < b term: B*exp(d/(r-b))
    mask_b = (r >= a) & (r < b)
    if np.any(mask_b):
        rb = r[mask_b]
        E += np.sum(B * np.exp(d / (rb - b)))
    return E

def _get_icosahedron_vertices(spacing=1.1):
    """12 vertices of regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit radius then scale
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    return verts * spacing

def _get_decahedron_vertices(spacing=1.1):
    """14 vertices of a regular decahedron (cuboctahedron-like) centered at origin."""
    # 12 cube corners + 2 face centers on z-axis... actually decahedron = 12 + 2
    # Use cuboctahedron: 12 vertices
    # Better: use icosahedron truncated or just use a 14-atom cluster
    # Mackay decahedron: 13 atoms (1 core + 12 shell)
    # For 14: add one more
    verts = _get_icosahedron_vertices(spacing)
    # Add center
    verts = np.vstack([verts, np.zeros(3)])
    return verts[:14]  # 14 atoms

def _pad_to_n(points, n, min_dist=0.55):
    """Ensure the array has exactly n rows, padding with random points if necessary."""
    points = np.array(points)
    if len(points) > n:
        return points[:n]
    while len(points) < n:
        # Try to add a point that is far enough from existing ones
        for _ in range(100):
            # Pick a random position near the existing cluster
            if len(points) > 0:
                center = np.mean(points, axis=0)
                radius = np.linalg.norm(points - center, axis=1).max() + 0.5
            else:
                center = np.zeros(3)
                radius = 1.0
            
            new_p = center + np.random.uniform(-radius, radius, 3)
            
            # Check distance to all existing points
            if len(points) == 0:
                break
            dists = np.linalg.norm(points - new_p, axis=1)
            if np.min(dists) >= min_dist:
                break
            # If we can't find a valid spot quickly, just accept a random one
            # The relaxer should handle small overlaps, but 0.5 is a hard constraint
            # So we force a valid position by trying many times
        points = np.vstack([points, new_p])
    return points[:n]

def _get_core_shell(n, spacing=1.1):
    """Generate core-shell icosahedral structure with n atoms."""
    if n <= 1:
        return np.zeros((n, 3))
    # Core
    points = [np.zeros(3)]
    # Shell 1: 12 icosahedron vertices
    shell1 = _get_icosahedron_vertices(spacing)
    for v in shell1:
        if len(points) < n:
            points.append(v)
    # Shell 2: 42 atoms (icosahedral second shell)
    if n > 13:
        # Generate icosahedral second shell points
        phi = (1 + np.sqrt(5)) / 2
        # Face centers and edge midpoints of icosahedron
        # Simplified: use fibonacci sphere for shell 2
        n_shell2 = min(n - len(points), 42)
        if n_shell2 > 0:
            # Fibonacci sphere
            i = np.arange(n_shell2)
            golden = (1 + np.sqrt(5)) / 2
            z = 1 - 2 * (i + 0.5) / n_shell2
            r = np.sqrt(1 - z**2)
            theta = 2 * np.pi * i / golden
            shell2 = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
            # Scale to appropriate radius (about 1.7 * spacing for icosahedral shell 2)
            shell2 = shell2 * spacing * 1.7
            for v in shell2:
                if len(points) < n:
                    points.append(v)
    
    # Pad if necessary
    if len(points) < n:
        points = _pad_to_n(points, n, min_dist=spacing * 0.8)
    else:
        points = np.array(points[:n])
        
    return points

def _get_random_compact(n, spacing=1.1, seed=42):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 200:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 * spacing for q in points):
                points.append(p)
        attempts += 1
    # Fill remaining with random positions
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return _pad_to_n(points, n, min_dist=spacing * 0.8)

def _fragment_merge(n, spacing=1.1, seed=42):
    """Build cluster from icosahedral fragments."""
    rng = np.random.default_rng(seed)
    
    # Strategy: build from 13-atom icosahedral units
    # For n atoms, use floor(n/13) icosahedra + remainder
    n_ico = min(n // 13, 3)  # max 3 icosahedra
    n_remainder = n - n_ico * 13
    
    all_points = []
    # Place icosahedra at different positions
    ico_positions = []
    if n_ico >= 1:
        ico_positions.append(np.zeros(3))
    if n_ico >= 2:
        ico_positions.append(np.array([spacing * 2.5, 0, 0]))
    if n_ico >= 3:
        ico_positions.append(np.array([0, spacing * 2.5, 0]))
    
    for pos in ico_positions:
        ico = _get_core_shell(13, spacing) + pos
        all_points.extend(ico.tolist())
    
    # Add remainder atoms
    if n_remainder > 0:
        # Place remainder near existing clusters
        if all_points:
            center = np.mean(all_points, axis=0)
        else:
            center = np.zeros(3)
        remainder = _get_core_shell(n_remainder, spacing) + center
        all_points.extend(remainder.tolist())
    
    # If we still need more atoms (n_remainder > 13), add more
    while len(all_points) < n:
        if all_points:
            center = np.mean(all_points, axis=0)
        else:
            center = np.zeros(3)
        p = center + rng.normal(0, spacing * 0.5, 3)
        all_points.append(p)
        
    # Ensure exact size and validity
    return _pad_to_n(all_points, n, min_dist=spacing * 0.8)

def _local_minimize(pos, potential='dzugutov', parameter=None, max_iter=200):
    """Local minimization using gradient-based method."""
    try:
        result = minimize(
            lambda x: _dzugutov_energy(x.reshape(-1, 3)),
            pos.flatten(),
            jac=lambda x: dzugutov_energy_gradient(x.reshape(-1, 3)).flatten(),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        return result.x.reshape(-1, 3), _dzugutov_energy(result.x.reshape(-1, 3))
    except:
        return pos, _dzugutov_energy(pos)

def _basin_hopping(pos, potential='dzugutov', parameter=None, n_iter=50, step_size=0.5, seed=42):
    """Basin hopping with annealing."""
    rng = np.random.default_rng(seed)
    current_pos = pos.copy()
    current_energy = _dzugutov_energy(current_pos)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    for i in range(n_iter):
        # Annealed step size
        step = step_size * (1.0 - i / n_iter * 0.5)
        # Random kick
        kick = rng.normal(0, step, size=current_pos.shape)
        trial_pos = current_pos + kick
        
        # Local minimize
        trial_pos, trial_energy = _local_minimize(trial_pos, potential, parameter, max_iter=100)
        
        # Metropolis accept
        if trial_energy < current_energy:
            current_pos = trial_pos
            current_energy = trial_energy
            if trial_energy < best_energy:
                best_pos = trial_pos.copy()
                best_energy = trial_energy
        else:
            # Accept with probability
            delta_E = trial_energy - current_energy
            T = max(0.01, 1.0 - i / n_iter)
            if rng.random() < np.exp(-delta_E / T):
                current_pos = trial_pos
                current_energy = trial_energy
    
    return best_pos, best_energy

def _anneal(pos, potential='dzugutov', parameter=None, n_iter=30, seed=42):
    """Repulsive annealing: add continuous repulsive term to escape local minima."""
    rng = np.random.default_rng(seed)
    current_pos = pos.copy()
    current_energy = _dzugutov_energy(current_pos)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    for i in range(n_iter):
        # Annealing temperature
        T = max(0.05, 1.0 * (1 - i / n_iter))
        # Add small random perturbation scaled by T
        kick = rng.normal(0, T * 0.3, size=current_pos.shape)
        trial_pos = current_pos + kick
        trial_pos, trial_energy = _local_minimize(trial_pos, potential, parameter, max_iter=80)
        
        if trial_energy < current_energy:
            current_pos = trial_pos
            current_energy = trial_energy
            if trial_energy < best_energy:
                best_pos = trial_pos.copy()
                best_energy = trial_energy
        else:
            delta_E = trial_energy - current_energy
            if rng.random() < np.exp(-delta_E / T):
                current_pos = trial_pos
                current_energy = trial_energy
    
    return best_pos, best_energy

def construct(instance):
    """Find lowest-energy Dzugutov cluster using fragment-merger + basin hopping."""
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Generate diverse initial structures
    initial_structures = []
    
    # 1. Core-shell icosahedral (Relaxed Mackay Template)
    try:
        s1 = _get_core_shell(size, spacing=1.1)
        # Immediate local minimization to act as the "relaxed template"
        s1_relaxed, _ = _local_minimize(s1, potential, parameter, max_iter=100)
        initial_structures.append(s1_relaxed)
    except:
        pass
    
    # 2. Fragment merge
    for seed in [42, 123, 456]:
        try:
            s = _fragment_merge(size, spacing=1.1, seed=seed)
            s_relaxed, _ = _local_minimize(s, potential, parameter, max_iter=100)
            initial_structures.append(s_relaxed)
        except:
            pass
    
    # 3. Random compact
    for seed in [7, 8, 9]:
        try:
            s = _get_random_compact(size, spacing=1.1, seed=seed)
            s_relaxed, _ = _local_minimize(s, potential, parameter, max_iter=100)
            initial_structures.append(s_relaxed)
        except:
            pass
    
    # 4. Mackay-like with defects
    try:
        s = _get_core_shell(size, spacing=1.05)  # slightly different spacing
        s_relaxed, _ = _local_minimize(s, potential, parameter, max_iter=100)
        initial_structures.append(s_relaxed)
    except:
        pass
    
    if not initial_structures:
        # Fallback: simple grid
        x = np.linspace(-2, 2, int(np.cbrt(size)) + 1)
        grid = np.array([[xx, yy, zz] for xx in x for yy in x for zz in x])
        if len(grid) < size:
            grid = _pad_to_n(grid, size)
        else:
            grid = grid[:size]
        initial_structures.append(grid)
    
    # Evaluate and refine each initial structure
    best_pos = None
    best_energy = np.inf
    
    for i, init_pos in enumerate(initial_structures):
        if time.time() > deadline - 60:
            break
        
        # The initial structures are already locally minimized in the generation step.
        # We just need to track the best one and then perform global search from it.
        try:
            current_energy = _dzugutov_energy(init_pos)
        except:
            current_energy = np.inf
            
        if current_energy < best_energy:
            best_pos = init_pos.copy()
            best_energy = current_energy
            
        # Basin hopping from this starting point
        if time.time() > deadline - 30:
            break
        
        try:
            bh_pos, bh_energy = _basin_hopping(
                init_pos, potential, parameter,
                n_iter=30, step_size=0.4, seed=i * 100 + 42
            )
        except:
            bh_pos, bh_energy = init_pos, current_energy
        
        if bh_energy < best_energy:
            best_pos = bh_pos.copy()
            best_energy = bh_energy
        
        # Annealing
        if time.time() > deadline - 15:
            break
        
        try:
            ann_pos, ann_energy = _anneal(
                init_pos, potential, parameter,
                n_iter=20, seed=i * 200 + 7
            )
        except:
            ann_pos, ann_energy = init_pos, current_energy
        
        if ann_energy < best_energy:
            best_pos = ann_pos.copy()
            best_energy = ann_energy
    
    # Final polish if time permits
    if best_pos is not None and time.time() < deadline - 10:
        try:
            polished_pos, polished_energy = _local_minimize(
                best_pos, potential, parameter, max_iter=300
            )
            if polished_energy < best_energy:
                best_pos = polished_pos
                best_energy = polished_energy
        except:
            pass
    
    # Use the library's relax as a final check
    if best_pos is not None and time.time() < deadline - 5:
        try:
            final_energy, final_pos = relax(best_pos, potential, parameter)
            if final_energy < best_energy:
                best_pos = final_pos
                best_energy = final_energy
        except:
            pass
    
    if best_pos is None:
        best_pos = initial_structures[0]
        
    # Final check: ensure shape is correct
    if best_pos.shape[0] != size:
        best_pos = _pad_to_n(best_pos, size)
        
    return best_pos
# EVOLVE-BLOCK-END
