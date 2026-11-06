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
    
    # Shell 2: 42 atoms (30 edge midpoints + 12 face centers)
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
    
    # Shell 2 radius: for icosahedral packing, r2/r1 ≈ 1.618 (golden ratio)
    r2 = r1 * 1.618
    
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

def _random_seed(n, scale, seed):
    rng = np.random.default_rng(seed)
    radius = scale
    points = []
    min_dist = 0.55
    max_attempts = n * 200
    for _ in range(max_attempts):
        if len(points) >= n: break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) < radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
    while len(points) < n:
        points.append(rng.uniform(-radius, radius, 3))
    return np.array(points)

def _generate_diverse_seeds(n, base_pos):
    """Generate multiple diverse seeds from a base icosahedral structure."""
    seeds = [base_pos.copy()]
    rng = np.random.default_rng(12345)
    
    # Compressed version
    seeds.append(base_pos * 0.85)
    # Expanded version
    seeds.append(base_pos * 1.15)
    # Deformed: random anisotropic scaling
    scale_vec = rng.uniform(0.8, 1.2, 3)
    seeds.append(base_pos * scale_vec)
    # Small random perturbation
    seeds.append(base_pos + rng.normal(0, 0.08, base_pos.shape))
    # Larger random perturbation
    seeds.append(base_pos + rng.normal(0, 0.15, base_pos.shape))
    
    return seeds

def _relax_and_evaluate(pos, n, potential, parameter):
    try:
        energy, relaxed_pos = relax(pos, potential, parameter)
        # Check validity
        if relaxed_pos is not None and len(relaxed_pos) == n:
            min_pair_dist = np.inf
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(relaxed_pos[i] - relaxed_pos[j])
                    if d < min_pair_dist:
                        min_pair_dist = d
            if min_pair_dist >= 0.5:
                return energy, relaxed_pos
        return energy, pos
    except Exception:
        return float('inf'), pos

def _basin_hopping(pos, n, potential, parameter, n_steps=30, step_size=0.25, seed=0):
    global _best_energy, _best_pos
    
    rng = np.random.default_rng(seed)
    current_energy, current_pos = _relax_and_evaluate(pos, n, potential, parameter)
    local_best_energy = current_energy
    local_best_pos = current_pos.copy()
    
    # Adaptive step size: start larger, shrink as we progress
    current_step = step_size
    
    for i in range(n_steps):
        if _time_left() < 8.0:
            break
            
        # Shrink step size gradually
        current_step = step_size * (0.95 ** i)
        
        perturbed = current_pos + rng.normal(0, current_step, current_pos.shape)
        new_energy, new_pos = _relax_and_evaluate(perturbed, n, potential, parameter)
        
        # Metropolis criterion with adaptive temperature
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            if new_energy < local_best_energy:
                local_best_energy = new_energy
                local_best_pos = new_pos.copy()
        else:
            delta = new_energy - current_energy
            if delta < 2.0:
                # Temperature decreases as we progress
                temp = 0.3 * (0.97 ** i)
                accept_prob = np.exp(-delta / temp)
                if rng.random() < accept_prob:
                    current_energy = new_energy
                    current_pos = new_pos
    
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
    mackay_pos = _generate_mackay(n)
    
    # Create diverse icosahedral-based seeds
    ico_seeds = _generate_diverse_seeds(n, mackay_pos)
    
    # Add random seeds
    est_r = (3 * n / (4 * np.pi)) ** (1/3) * 0.75
    random_seed1 = _random_seed(n, est_r, 42)
    random_seed2 = _random_seed(n, est_r * 1.1, 43)
    
    # Combine all seeds
    all_seeds = ico_seeds + [random_seed1, random_seed2]
    
    # Assign seeds to 4 threads
    # Thread 0: Best icosahedral seed (original Mackay)
    # Thread 1: Compressed icosahedral
    # Thread 2: Expanded icosahedral  
    # Thread 3: Random seed
    thread_seeds = [
        all_seeds[0],  # Original Mackay
        all_seeds[1],  # Compressed
        all_seeds[2],  # Expanded
        random_seed1   # Random
    ]
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for idx, seed in enumerate(thread_seeds):
            # Use different step sizes for different seed types
            if idx < 3:  # Icosahedral-based
                step = 0.25
            else:  # Random
                step = 0.3
            future = executor.submit(_basin_hopping, seed, n, potential, parameter, 
                                     n_steps=40, step_size=step, seed=100+idx)
            futures.append(future)
            
        for f in as_completed(futures):
            try:
                f.result()
            except Exception:
                pass
    
    # Final refinement pass on best found
    if _best_pos is not None and _time_left() > 15.0:
        _basin_hopping(_best_pos, n, potential, parameter, 
                       n_steps=15, step_size=0.12, seed=999)
    
    # Additional polish: try to improve with smaller steps
    if _best_pos is not None and _time_left() > 8.0:
        _basin_hopping(_best_pos, n, potential, parameter, 
                       n_steps=10, step_size=0.08, seed=1000)
    
    if _best_pos is None:
        _best_pos = mackay_pos
        
    return _best_pos
