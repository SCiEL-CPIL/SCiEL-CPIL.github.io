import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0
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

def _get_mackay_icosahedron(n, r1=1.0, r2=1.55):
    """Generate a Mackay icosahedron seed with proper shell structure."""
    verts = _get_icosahedron_verts()
    atoms = [np.zeros(3)]
    
    # Shell 1: 12 vertices
    atoms.extend((verts * r1).tolist())
    
    if n <= 13:
        return np.array(atoms[:n])
    
    # Find edges and faces
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
    
    # Shell 2: edge midpoints (30) and face centers (20)
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid_norm = np.linalg.norm(mid)
        if mid_norm > 1e-10:
            mid_dir = mid / mid_norm
            atoms.append(mid_dir * r2)
    
    for i, j, k in faces:
        center = (verts[i] + verts[j] + verts[k]) / 3.0
        center_norm = np.linalg.norm(center)
        if center_norm > 1e-10:
            center_dir = center / center_norm
            atoms.append(center_dir * r2)
    
    # If we still need more atoms, add shell 3
    if len(atoms) < n:
        # Generate more positions on a sphere
        rng = np.random.default_rng(42)
        while len(atoms) < n:
            theta = rng.uniform(0, 2 * np.pi)
            phi_val = rng.uniform(-np.pi/2, np.pi/2)
            r = r2 * 1.3
            p = np.array([
                r * np.cos(phi_val) * np.cos(theta),
                r * np.cos(phi_val) * np.sin(theta),
                r * np.sin(phi_val)
            ])
            # Check minimum distance
            if all(np.linalg.norm(p - np.array(q)) >= 0.5 for q in atoms):
                atoms.append(p)
    
    return np.array(atoms[:n])

def _generate_random_seed(n, scale, seed):
    """Generate random seed with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = scale
    points = []
    min_dist = 0.55
    max_attempts = n * 300
    for _ in range(max_attempts):
        if len(points) >= n: break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) < radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
    while len(points) < n:
        points.append(rng.uniform(-radius, radius, 3))
    return np.array(points)

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

def _kick_relax_worker(seed_pos, n, potential, parameter, n_kicks, base_seed, kick_scale):
    """Perform kick-relax iterations and return best result."""
    global _best_energy, _best_pos
    
    rng = np.random.default_rng(base_seed)
    current_energy, current_pos = _relax_and_evaluate(seed_pos, n, potential, parameter)
    local_best_energy = current_energy
    local_best_pos = current_pos.copy()
    
    for i in range(n_kicks):
        if _time_left() < 10.0:
            break
        
        # Adaptive kick size
        kick_size = kick_scale * (0.5 + 0.5 * rng.random())
        perturbed = current_pos + rng.normal(0, kick_size, current_pos.shape)
        new_energy, new_pos = _relax_and_evaluate(perturbed, n, potential, parameter)
        
        if new_energy < local_best_energy:
            local_best_energy = new_energy
            local_best_pos = new_pos.copy()
            current_energy = new_energy
            current_pos = new_pos
        elif new_energy < current_energy + 0.5:
            # Accept small increases to escape local minima
            if rng.random() < 0.3:
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
    
    # Estimate cluster radius
    est_r = (3 * n / (4 * np.pi)) ** (1/3) * 0.85
    
    # Generate diverse seeds
    seeds = []
    
    # 1. Mackay icosahedron
    mackay_pos = _get_mackay_icosahedron(n)
    seeds.append(mackay_pos)
    
    # 2. Perturbed Mackay
    rng = np.random.default_rng(100)
    seeds.append(mackay_pos + rng.normal(0, 0.1, mackay_pos.shape))
    
    # 3. Random seed
    seeds.append(_generate_random_seed(n, est_r, 200))
    
    # 4. Another random with different scale
    seeds.append(_generate_random_seed(n, est_r * 1.2, 300))
    
    # 5. Tighter random
    seeds.append(_generate_random_seed(n, est_r * 0.7, 400))
    
    # Distribute seeds to 4 threads
    def run_worker(seed_idx, base_seed, n_kicks, kick_scale):
        if seed_idx < len(seeds):
            _kick_relax_worker(seeds[seed_idx], n, potential, parameter, 
                              n_kicks, base_seed, kick_scale)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        # Thread 1: Mackay with more kicks
        futures.append(executor.submit(run_worker, 0, 1000, 60, 0.25))
        # Thread 2: Perturbed Mackay
        futures.append(executor.submit(run_worker, 1, 2000, 50, 0.2))
        # Thread 3: Random
        futures.append(executor.submit(run_worker, 2, 3000, 45, 0.3))
        # Thread 4: Another random
        futures.append(executor.submit(run_worker, 3, 4000, 45, 0.35))
        
        for f in as_completed(futures):
            try:
                f.result()
            except Exception:
                pass
    
    # Final refinement if time allows
    if _best_pos is not None and _time_left() > 20.0:
        # Do intensive refinement on the best found
        _kick_relax_worker(_best_pos, n, potential, parameter, 
                          30, 5000, 0.08)
    
    if _best_pos is None:
        # Fallback to first seed
        _best_pos = seeds[0]
    
    return _best_pos
