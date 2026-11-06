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
    # Find edges and faces of the icosahedron
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
    
    # Add edge midpoints (30) and face centers (12)
    r2 = 1.6
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid_norm = np.linalg.norm(mid)
        mid_dir = mid / mid_norm
        atoms.append(mid_dir * r2)
        
    for i, j, k in faces:
        center = (verts[i] + verts[j] + verts[k]) / 3.0
        center_norm = np.linalg.norm(center)
        center_dir = center / center_norm
        atoms.append(center_dir * r2)
        
    return np.array(atoms[:n])

def _random_seed(n, scale, seed):
    """Generate random seed with minimum distance constraint."""
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

def _generate_dodecahedron_seed(n, scale, seed):
    """Generate seed based on dodecahedral structure."""
    rng = np.random.default_rng(seed)
    phi = (1 + np.sqrt(5)) / 2
    # Dodecahedron vertices (20 vertices)
    verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
        [1/phi, phi, 0], [-1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, -phi, 0],
        [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]
    ])
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms * scale
    
    atoms = [np.zeros(3)]
    atoms.extend(verts.tolist())
    
    # Add more atoms if needed by perturbing
    while len(atoms) < n:
        base = rng.choice(len(atoms))
        p = atoms[base] + rng.normal(0, scale * 0.3, 3)
        if all(np.linalg.norm(p - q) >= 0.55 for q in atoms):
            atoms.append(p)
        else:
            atoms.append(p)
    
    return np.array(atoms[:n])

def _generate_cuboctahedron_seed(n, scale, seed):
    """Generate seed based on cuboctahedral structure."""
    rng = np.random.default_rng(seed)
    # Cuboctahedron vertices (12 vertices)
    verts = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]
    ])
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms * scale
    
    atoms = [np.zeros(3)]
    atoms.extend(verts.tolist())
    
    while len(atoms) < n:
        base = rng.choice(len(atoms))
        p = atoms[base] + rng.normal(0, scale * 0.3, 3)
        if all(np.linalg.norm(p - q) >= 0.55 for q in atoms):
            atoms.append(p)
        else:
            atoms.append(p)
    
    return np.array(atoms[:n])

def _relax_and_evaluate(pos, n, potential, parameter):
    """Relax structure and evaluate energy with validity check."""
    try:
        energy, relaxed_pos = relax(pos, potential, parameter)
        # Check validity - minimum distance constraint
        min_dist = 0.5
        valid = True
        for i in range(n):
            for j in range(i+1, n):
                if np.linalg.norm(relaxed_pos[i] - relaxed_pos[j]) < min_dist:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Try to fix by moving atoms apart
            fixed_pos = relaxed_pos.copy()
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.linalg.norm(fixed_pos[i] - fixed_pos[j])
                    if dist < min_dist:
                        direction = (fixed_pos[i] - fixed_pos[j]) / max(dist, 1e-10)
                        shift = (min_dist - dist) / 2 + 0.01
                        fixed_pos[i] += direction * shift
                        fixed_pos[j] -= direction * shift
            
            # Re-evaluate after fixing
            try:
                energy, fixed_pos = relax(fixed_pos, potential, parameter)
                return energy, fixed_pos
            except:
                return float('inf'), pos
        
        return energy, relaxed_pos
    except Exception:
        return float('inf'), pos

def _basin_hopping(pos, n, potential, parameter, n_steps=30, step_size=0.2, seed=0, temperature=0.1):
    """Basin hopping with adaptive step size and temperature."""
    global _best_energy, _best_pos
    
    rng = np.random.default_rng(seed)
    current_energy, current_pos = _relax_and_evaluate(pos, n, potential, parameter)
    local_best_energy = current_energy
    local_best_pos = current_pos.copy()
    
    current_step = step_size
    current_temp = temperature
    
    for i in range(n_steps):
        if _time_left() < 5.0:
            break
        
        # Adaptive step size: decrease if improving, increase if stuck
        if current_energy < local_best_energy * 0.99:
            current_step = max(0.05, current_step * 0.9)
        else:
            current_step = min(0.5, current_step * 1.1)
        
        # Adaptive temperature: cool down over time
        current_temp = max(0.01, temperature * (1 - i / (2 * n_steps)))
        
        perturbed = current_pos + rng.normal(0, current_step, current_pos.shape)
        new_energy, new_pos = _relax_and_evaluate(perturbed, n, potential, parameter)
        
        # Metropolis acceptance
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            if new_energy < local_best_energy:
                local_best_energy = new_energy
                local_best_pos = new_pos.copy()
        else:
            delta = new_energy - current_energy
            if delta < 2.0:  # Only consider small energy increases
                accept_prob = np.exp(-delta / current_temp)
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
    seeds = []
    
    # 1. Mackay icosahedron
    mackay_pos = _generate_mackay(n)
    seeds.append(mackay_pos)
    
    # 2. Dodecahedron-based
    est_r = (3 * n / (4 * np.pi)) ** (1/3) * 0.8
    seeds.append(_generate_dodecahedron_seed(n, est_r, 42))
    
    # 3. Cuboctahedron-based
    seeds.append(_generate_cuboctahedron_seed(n, est_r, 43))
    
    # 4. Random seed
    seeds.append(_random_seed(n, est_r, 44))
    
    # 5. Perturbed Mackay
    rng = np.random.default_rng(45)
    seeds.append(mackay_pos + rng.normal(0, 0.15, mackay_pos.shape))
    
    # 6. Another random with different scale
    seeds.append(_random_seed(n, est_r * 1.2, 46))
    
    # Distribute seeds to 4 threads with multiple runs each
    # Thread 1: Seeds 0, 1
    # Thread 2: Seeds 2, 3
    # Thread 3: Seeds 4, 5
    # Thread 4: Best of all + refinement
    
    def run_basin_hopping_sequence(seed_list, base_seed):
        """Run basin hopping sequentially on multiple seeds."""
        best_e = float('inf')
        best_p = None
        for idx, seed in enumerate(seed_list):
            if _time_left() < 15.0:
                break
            e, p = _basin_hopping(seed, n, potential, parameter, 
                                   n_steps=40, step_size=0.2, 
                                   seed=base_seed + idx, temperature=0.15)
            if e < best_e:
                best_e = e
                best_p = p
        return best_e, best_p
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        futures.append(executor.submit(run_basin_hopping_sequence, [seeds[0], seeds[1]], 100))
        futures.append(executor.submit(run_basin_hopping_sequence, [seeds[2], seeds[3]], 200))
        futures.append(executor.submit(run_basin_hopping_sequence, [seeds[4], seeds[5]], 300))
        
        # Thread 4: Wait and then refine the best found so far
        def refine_best():
            # Wait a bit for other threads to make progress
            time.sleep(5)
            with _LOCK:
                if _best_pos is not None:
                    best_pos = _best_pos.copy()
                else:
                    best_pos = seeds[0]
            
            # Do intensive refinement
            e, p = _basin_hopping(best_pos, n, potential, parameter, 
                                   n_steps=50, step_size=0.1, 
                                   seed=999, temperature=0.05)
            return e, p
        
        futures.append(executor.submit(refine_best))
        
        for f in as_completed(futures):
            try:
                f.result()
            except Exception:
                pass
            
    # Final high-precision polish
    if _best_pos is not None and _time_left() > 15.0:
        # Do a final refinement with smaller steps
        _basin_hopping(_best_pos, n, potential, parameter, 
                       n_steps=30, step_size=0.05, seed=1234, temperature=0.02)
                       
    if _best_pos is None:
        # Fallback to first seed
        _best_pos = seeds[0]
        
    return _best_pos
