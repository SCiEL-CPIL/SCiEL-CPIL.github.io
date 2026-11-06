"""
Strategy: Multi-Start Parallel Basin Hopping in Full Cartesian Space.
The Dzugutov potential favors polytetrahedral order, which is not strictly icosahedral.
The previous SCVO approach constrained the search to a low-dimensional parameter space
(shell radii/twists), which may miss the true global minima.
This strategy uses a high-dimensional, multi-start Basin Hopping approach:
1. Generate diverse initial structures: Mackay icosahedra, BCC chunks, and random gas.
2. Run parallel Basin Hopping threads in the full 3N Cartesian space.
3. Use adaptive step sizes and Metropolis acceptance to escape local minima.
4. Enforce strict time limits to ensure all threads finish within the budget.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
TIME_LIMIT = 290.0  # Safety margin before 300s limit

def _check_time():
    return time.time() - _START_TIME < TIME_LIMIT

def _get_time_remaining():
    return max(0, TIME_LIMIT - (time.time() - _START_TIME))

def _generate_mackay_icosahedron(n):
    """
    Generate a Mackay icosahedral cluster of n atoms.
    Shells: 1 (center), 12 (1st), 42 (2nd), 120 (3rd).
    For n > 175, we truncate the 3rd shell or add random points.
    """
    if n <= 0:
        return np.zeros((0, 3))
    
    phi = (1 + np.sqrt(5)) / 2
    
    # Shell 1: Center
    points = [np.zeros(3)]
    
    # Shell 2: 12 vertices of icosahedron
    v1 = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    v1 = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
    r1 = 1.0
    points.extend((v1 * r1).tolist())
    
    if n > 13:
        # Shell 3: 42 points (20 face centers + 30 edge midpoints)
        # Compute adjacency
        dists = np.linalg.norm(v1[:, np.newaxis, :] - v1[np.newaxis, :, :], axis=2)
        min_dist = np.min(dists[dists > 1e-6])
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                if abs(dists[i,j] - min_dist) < 1e-6:
                    edges.append((i,j))
        
        mids = []
        for i, j in edges:
            mid = (v1[i] + v1[j]) / 2
            mids.append(mid / np.linalg.norm(mid))
        mids = np.array(mids)
        
        faces = []
        for i in range(12):
            for j in range(i+1, 12):
                if abs(dists[i,j] - min_dist) < 1e-6:
                    for k in range(j+1, 12):
                        if abs(dists[i,k] - min_dist) < 1e-6 and abs(dists[j,k] - min_dist) < 1e-6:
                            faces.append((i,j,k))
        
        centers = []
        for i, j, k in faces:
            c = (v1[i] + v1[j] + v1[k]) / 3
            centers.append(c / np.linalg.norm(c))
        centers = np.array(centers)
        
        shell2 = np.vstack([mids, centers])
        shell2 = shell2 / np.linalg.norm(shell2, axis=1, keepdims=True)
        r2 = 1.9
        # Add as many as needed
        num_needed = min(42, n - len(points))
        points.extend((shell2[:num_needed] * r2).tolist())
        
    if len(points) < n:
        # Shell 4: 120 points (3rd Mackay shell)
        # Approximate by rotating shell 2 points? Or just add random points on a sphere?
        # For robustness, let's add points on a sphere of radius r3 = 2.8
        # Using a Fibonacci sphere for better distribution
        num_needed = n - len(points)
        golden_angle = np.pi * (3 - np.sqrt(5))
        r3 = 2.8
        for i in range(num_needed):
            z = 1 - 2 * (i + 0.5) / num_needed
            r = np.sqrt(1 - z*z)
            theta = golden_angle * i
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * r3
            points.append(p)
            
    pos_array = np.array(points[:n])
    
    # Ensure minimum distance 0.5
    # Simple iterative nudge
    for _ in range(10):
        moved = False
        for i in range(len(pos_array)):
            for j in range(i+1, len(pos_array)):
                d = np.linalg.norm(pos_array[i] - pos_array[j])
                if d < 0.5 and d > 1e-10:
                    direction = (pos_array[i] - pos_array[j]) / d
                    pos_array[i] += 0.01 * direction
                    pos_array[j] -= 0.01 * direction
                    moved = True
        if not moved:
            break
            
    return pos_array

def _generate_bcc_cluster(n):
    """
    Generate a BCC chunk of n atoms.
    """
    # BCC lattice points
    # Unit cell: (0,0,0) to (1,1,1) with center at (0.5,0.5,0.5)
    # We generate a cubic chunk and trim
    side = int(np.ceil(np.cbrt(n))) + 2
    pts = []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                # Corner
                pts.append([i, j, k])
                # Center
                pts.append([i+0.5, j+0.5, k+0.5])
                
    pts = np.array(pts)
    # Center the cluster
    pts -= np.mean(pts, axis=0)
    # Sort by distance from center
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)
    pts = pts[idx[:n]]
    
    # Scale to reasonable size
    # Typical Dzugutov nearest neighbor is ~1.0-1.2
    # BCC nearest neighbor in this grid is sqrt(0.5^2+0.5^2+0.5^2) = sqrt(0.75) ~ 0.866
    # Let's scale so that the 1st shell is around 1.0
    scale = 1.0 / np.sqrt(0.75)
    pts *= scale
    
    # Ensure minimum distance
    for _ in range(10):
        moved = False
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 0.5 and d > 1e-10:
                    direction = (pts[i] - pts[j]) / d
                    pts[i] += 0.01 * direction
                    pts[j] -= 0.01 * direction
                    moved = True
        if not moved:
            break
            
    return pts

def _generate_random_cluster(n, seed=0):
    """
    Generate a random cluster of n atoms in a box.
    """
    rng = np.random.default_rng(seed)
    # Box size such that density is reasonable
    # N ~ 50, volume ~ N * 2 => side ~ 4-5
    side = 4.0
    pts = rng.uniform(-side/2, side/2, (n, 3))
    
    # Ensure minimum distance
    for _ in range(50):
        moved = False
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 0.5 and d > 1e-10:
                    direction = (pts[i] - pts[j]) / d
                    pts[i] += 0.05 * direction
                    pts[j] -= 0.05 * direction
                    moved = True
        if not moved:
            break
            
    return pts

def _relax_with_timeout(positions, potential, parameter, timeout=10.0):
    """Relax with a timeout to prevent hanging."""
    result = {'energy': None, 'pos': None, 'done': False}
    
    def _do_relax():
        try:
            e, p = relax(positions, potential, parameter)
            result['energy'] = e
            result['pos'] = p
        except Exception:
            pass
        result['done'] = True
        
    t = threading.Thread(target=_do_relax)
    t.start()
    t.join(timeout=timeout)
    
    if result['done'] and result['pos'] is not None:
        return result['energy'], result['pos']
    return None, None

def _basin_hopping_worker(initial_pos, n, potential, parameter, max_iter, time_limit, result_holder, lock, seed=0):
    """
    Perform Basin Hopping in Cartesian space.
    """
    if not _check_time():
        return
        
    # Initial relaxation
    energy, pos = _relax_with_timeout(initial_pos, potential, parameter, timeout=15.0)
    if energy is None:
        return
        
    best_energy = energy
    best_pos = pos.copy()
    current_energy = energy
    current_pos = pos.copy()
    
    rng = np.random.default_rng(seed)
    step_size = 0.8
    
    for i in range(max_iter):
        if not _check_time():
            break
        if time.time() > time_limit:
            break
            
        # Perturb a random subset of atoms
        # More atoms for larger clusters
        n_perturb = max(2, min(8, n // 5))
        indices = rng.choice(n, size=n_perturb, replace=False)
        perturbed = current_pos.copy()
        perturbed[indices] += rng.normal(0, step_size, (n_perturb, 3))
        
        new_energy, new_pos = _relax_with_timeout(perturbed, potential, parameter, timeout=10.0)
        if new_energy is None:
            continue
            
        # Metropolis acceptance
        # T starts at 0.5 and cools to 0.01
        T = 0.5 * (1.0 - i / max(1, max_iter)) + 0.01
        if new_energy < current_energy or rng.random() < np.exp(-(new_energy - current_energy) / T):
            current_energy = new_energy
            current_pos = new_pos
            
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos.copy()
            
            with lock:
                if result_holder['energy'] is None or best_energy < result_holder['energy']:
                    result_holder['energy'] = best_energy
                    result_holder['pos'] = best_pos.copy()
            
        # Adaptive step size
        if new_energy < current_energy:
            step_size = max(0.3, step_size * 0.95)
        else:
            step_size = min(1.5, step_size * 1.05)
            
    with lock:
        if result_holder['energy'] is None or best_energy < result_holder['energy']:
            result_holder['energy'] = best_energy
            result_holder['pos'] = best_pos.copy()

def construct(instance):
    """
    Strategy:
    1. Generate diverse initial structures (Icosahedral, BCC, Random).
    2. Run parallel Basin Hopping threads.
    3. Return the best result found.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind", "dzugutov"))
    
    time_limit = _START_TIME + TIME_LIMIT - 5
    
    result_holder = {'energy': None, 'pos': None}
    lock = threading.Lock()
    
    # Generate initial structures
    ico_pos = _generate_mackay_icosahedron(size)
    bcc_pos = _generate_bcc_cluster(size)
    rand_pos1 = _generate_random_cluster(size, seed=1)
    rand_pos2 = _generate_random_cluster(size, seed=2)
    
    # Disturb the ico and bcc slightly to create more diversity
    rng = np.random.default_rng(42)
    ico_pos_dist = ico_pos + rng.normal(0, 0.2, ico_pos.shape)
    bcc_pos_dist = bcc_pos + rng.normal(0, 0.2, bcc_pos.shape)
    
    # Define workers
    # We have 4 threads available.
    # 2 for Ico-based, 1 for BCC-based, 1 for Random-based
    
    def worker_ico1():
        _basin_hopping_worker(ico_pos, size, potential, parameter, 200, time_limit, result_holder, lock, seed=1)
        
    def worker_ico2():
        _basin_hopping_worker(ico_pos_dist, size, potential, parameter, 200, time_limit, result_holder, lock, seed=2)
        
    def worker_bcc():
        _basin_hopping_worker(bcc_pos_dist, size, potential, parameter, 200, time_limit, result_holder, lock, seed=3)
        
    def worker_rand():
        _basin_hopping_worker(rand_pos1, size, potential, parameter, 200, time_limit, result_holder, lock, seed=4)
        
    threads = [
        threading.Thread(target=worker_ico1),
        threading.Thread(target=worker_ico2),
        threading.Thread(target=worker_bcc),
        threading.Thread(target=worker_rand),
    ]
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
    if result_holder['pos'] is not None:
        return result_holder['pos']
    else:
        # Fallback
        return ico_pos

# EVOLVE-BLOCK-END
