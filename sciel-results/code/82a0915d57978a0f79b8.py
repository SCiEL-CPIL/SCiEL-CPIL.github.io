import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
_DEADLINE = _START_TIME + 285.0  # More conservative deadline
_LOCK = threading.Lock()
_best_energy = float('inf')
_best_pos = None

def _time_left():
    return _DEADLINE - time.time()

def _get_icosahedron_verts():
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms

def _generate_mackay(n):
    """
    Generate n atoms for a truncated Mackay icosahedron with proper shell structure.
    """
    verts = _get_icosahedron_verts()
    atoms = [np.zeros(3)]
    
    # Shell 1: 12 atoms at radius r1
    r1 = 1.0
    shell1 = verts * r1
    atoms.extend(shell1)
    
    if n <= 13:
        return np.array(atoms[:n])
    
    # Shell 2: 42 atoms
    # Compute edge and face positions properly
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            dists[i,j] = np.linalg.norm(verts[i] - verts[j])
    min_dist = np.min(dists[dists > 1e-6])
    
    edges = []
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            if np.isclose(dists[i,j], min_dist, rtol=1e-3):
                edges.append((i, j))
                
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                if (np.isclose(dists[i,j], min_dist, rtol=1e-3) and
                    np.isclose(dists[j,k], min_dist, rtol=1e-3) and
                    np.isclose(dists[i,k], min_dist, rtol=1e-3)):
                    faces.append((i, j, k))
    
    # Shell 2 radius: distance from center to edge midpoints
    # For regular icosahedron with circumradius 1, inradius is ~0.795
    # Edge midpoint distance from center
    if edges:
        i, j = edges[0]
        mid = (verts[i] + verts[j]) / 2.0
        r2 = np.linalg.norm(mid)
    else:
        r2 = 1.55
    
    # Add edge midpoints (30)
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid_norm = np.linalg.norm(mid)
        mid_dir = mid / mid_norm
        atoms.append(mid_dir * r2)
        
    # Add face centers (12)
    for i, j, k in faces:
        center = (verts[i] + verts[j] + verts[k]) / 3.0
        center_norm = np.linalg.norm(center)
        center_dir = center / center_norm
        atoms.append(center_dir * r2)
        
    return np.array(atoms[:n])

def _generate_dodecahedron_seed(n, seed=45):
    """Generate seed based on dodecahedral symmetry."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
        [1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0],
        [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]
    ])
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    rng = np.random.default_rng(seed)
    atoms = [np.zeros(3)]
    atoms.extend(verts * 1.0)
    
    # Add more atoms if needed
    while len(atoms) < n:
        # Add atoms at random positions near existing ones
        base = atoms[rng.integers(0, len(atoms))]
        new_pos = base + rng.normal(0, 0.3, 3)
        if all(np.linalg.norm(new_pos - q) >= 0.5 for q in atoms):
            atoms.append(new_pos)
        else:
            atoms.append(new_pos + rng.normal(0, 0.5, 3))
    
    return np.array(atoms[:n])

def _generate_cuboctahedron_seed(n, seed=46):
    """Generate seed based on cuboctahedral symmetry."""
    verts = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]
    ])
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    rng = np.random.default_rng(seed)
    atoms = [np.zeros(3)]
    atoms.extend(verts * 1.0)
    
    while len(atoms) < n:
        base = atoms[rng.integers(0, len(atoms))]
        new_pos = base + rng.normal(0, 0.3, 3)
        if all(np.linalg.norm(new_pos - q) >= 0.5 for q in atoms):
            atoms.append(new_pos)
        else:
            atoms.append(new_pos + rng.normal(0, 0.5, 3))
    
    return np.array(atoms[:n])

def _random_seed(n, scale, seed):
    rng = np.random.default_rng(seed)
    radius = scale
    points = []
    min_dist = 0.6
    for _ in range(n * 100):
        if len(points) >= n: break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) < radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
    while len(points) < n:
        points.append(rng.uniform(-radius, radius, 3))
    return np.array(points)

def _relax_and_evaluate(pos, n, potential, parameter):
    try:
        energy, relaxed_pos = relax(pos, potential, parameter)
        # Check validity
        for i in range(n):
            for j in range(i+1, n):
                if np.linalg.norm(relaxed_pos[i] - relaxed_pos[j]) < 0.5:
                    return energy, pos
        return energy, relaxed_pos
    except Exception:
        return float('inf'), pos

def _basin_hopping(pos, n, potential, parameter, n_steps=20, step_size=0.2, seed=0):
    global _best_energy, _best_pos
    
    rng = np.random.default_rng(seed)
    current_energy, current_pos = _relax_and_evaluate(pos, n, potential, parameter)
    local_best_energy = current_energy
    local_best_pos = current_pos.copy()
    
    # Adaptive step size
    current_step = step_size
    no_improve_count = 0
    
    for i in range(n_steps):
        if _time_left() < 5.0:
            break
            
        perturbed = current_pos + rng.normal(0, current_step, current_pos.shape)
        new_energy, new_pos = _relax_and_evaluate(perturbed, n, potential, parameter)
        
        # Metropolis
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            no_improve_count = 0
            # Slightly increase step size on improvement
            current_step = min(current_step * 1.1, 0.5)
            if new_energy < local_best_energy:
                local_best_energy = new_energy
                local_best_pos = new_pos.copy()
        else:
            delta = new_energy - current_energy
            if delta < 1.0:
                accept_prob = np.exp(-delta / 0.2)
                if rng.random() < accept_prob:
                    current_energy = new_energy
                    current_pos = new_pos
            no_improve_count += 1
            # Decrease step size if no improvement
            if no_improve_count % 5 == 0:
                current_step = max(current_step * 0.9, 0.05)
    
    with _LOCK:
        if local_best_energy < _best_energy:
            _best_energy = local_best_energy
            _best_pos = local_best_pos.copy()
            
    return local_best_energy, local_best_pos

def construct(instance):
    global _best_energy, _best_pos
    _best_energy = float('inf')
    _best_pos = None
    
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Generate diverse seeds
    seeds = []
    
    # 1. Mackay Truncated (primary icosahedral seed)
    mackay_pos = _generate_mackay(n)
    seeds.append(mackay_pos)
    
    # 2. Compressed Mackay
    seeds.append(mackay_pos * 0.9)
    
    # 3. Expanded Mackay
    seeds.append(mackay_pos * 1.1)
    
    # 4. Dodecahedral seed
    seeds.append(_generate_dodecahedron_seed(n, seed=45))
    
    # 5. Cuboctahedral seed
    seeds.append(_generate_cuboctahedron_seed(n, seed=46))
    
    # 6. Random seed
    est_r = (3 * n / (4 * np.pi)) ** (1/3) * 0.8
    seeds.append(_random_seed(n, est_r, 42))
    
    # 7. Perturbed Mackay
    rng = np.random.default_rng(44)
    seeds.append(mackay_pos + rng.normal(0, 0.15, mackay_pos.shape))
    
    # Distribute seeds to 4 threads (each thread gets multiple seeds)
    n_threads = 4
    seeds_per_thread = len(seeds) // n_threads
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for t in range(n_threads):
            thread_seeds = seeds[t*seeds_per_thread:(t+1)*seeds_per_thread]
            if t == n_threads - 1:
                thread_seeds.extend(seeds[(t+1)*seeds_per_thread:])
            
            for seed_idx, seed in enumerate(thread_seeds):
                future = executor.submit(_basin_hopping, seed, n, potential, parameter, 
                                         n_steps=40, step_size=0.2, seed=100+t*10+seed_idx)
                futures.append(future)
            
        for f in as_completed(futures):
            try:
                f.result()
            except Exception:
                pass
            
    # Final high-precision polish
    if _best_pos is not None and _time_left() > 15.0:
        # Multiple refinement passes with decreasing step size
        for polish_step in [0.1, 0.05, 0.02]:
            if _time_left() < 5.0:
                break
            _basin_hopping(_best_pos, n, potential, parameter, 
                           n_steps=15, step_size=polish_step, seed=999)
                       
    if _best_pos is None:
        _best_pos = seeds[0]
        
    return _best_pos
