"""
Strategy: Symmetry-Constrained Variational Optimization (SCVO) + Basin Hopping.
The Dzugutov potential favors icosahedral/polytetrahedral order.
Instead of searching 3n dimensions, we parametrize the structure by icosahedral
shells with variable radii and twist angles. This reduces the search space to ~6-10
dimensions, allowing for more effective global optimization via Basin Hopping on
these parameters.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
TIME_LIMIT = 280.0  # Safety margin before 300s limit

def _check_time():
    return time.time() - _START_TIME < TIME_LIMIT

def _get_time_remaining():
    return max(0, TIME_LIMIT - (time.time() - _START_TIME))

def _get_icosahedral_directions():
    """
    Generate the 120 directions for the 3rd shell (Mackay icosahedron) 
    and 42 for the 2nd shell, and 12 for the 1st.
    Returns a list of arrays of unit vectors.
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # 1st shell: 12 vertices of icosahedron
    v1 = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    v1 = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
    
    # 2nd shell: 42 points
    # 20 face centers + 30 edge midpoints
    # Face centers are normal to faces.
    # Let's use a known set or generate.
    # A simpler way: The 42 points are the vertices of a rhombic triacontahedron? No.
    # Let's use the explicit coordinates for the 42 shell.
    # The 42 points are at:
    # (0, 1, phi) type rotated? No, those are the 12.
    # The 42 points are:
    # 20 face centers: distance sqrt(3) * r1?
    # Let's compute them from the 12 vertices.
    
    # Define 20 faces of the icosahedron.
    # This is tedious to hardcode. 
    # Alternative: Use the fact that the 42 points are the 12 vertices + 30 edge midpoints? 
    # No, 12+30=42. But the 12 are already in shell 1.
    # So shell 2 is the 30 edge midpoints + 20 face centers? 
    # Actually, the Mackay icosahedron shell 2 (42 atoms) consists of:
    # - 12 atoms at the vertices of a larger icosahedron? No.
    # - The standard Mackay icosahedron shells are:
    #   1: 1 atom (center)
    #   12: 12 atoms (1st shell, icosahedron vertices)
    #   42: 42 atoms (2nd shell)
    #   120: 120 atoms (3rd shell)
    
    # Let's use a robust method: generate points on a sphere with icosahedral symmetry.
    # For the 42 points, they form a "cuboctahedron" like structure?
    # Actually, the 42 points are the vertices of a "great rhombicuboctahedron" projected?
    
    # Let's use a simpler approximation for the seed:
    # We will generate the 12 icosahedron vertices.
    # For shell 2, we can take the 30 edge midpoints and 20 face centers of the icosahedron defined by the 12 vertices.
    
    # Compute edge midpoints
    # We need the adjacency list of the icosahedron.
    # Distance between adjacent vertices is d_edge.
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
    
    # Compute face centers
    # Faces are triangles of vertices with distance min_dist.
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
    
    # For shell 3 (120 points), we can approximate by rotating the shell 2 points 
    # or using a denser icosahedral grid.
    # A 120-point shell is the 3rd Mackay shell.
    # It consists of 12 "caps" of 10 points? 
    # Let's generate 120 points using a Fibonacci sphere or similar, 
    # but constrained to be near the icosahedral symmetry.
    # For simplicity in the seed, we can use a denser set.
    # Actually, for n=38-60, we might not need the full 120 shell.
    # n=38: 1 + 12 + 25 (part of shell 2)
    # n=55: 1 + 12 + 42 (full shell 2)
    # n=60: 1 + 12 + 42 + 5 (part of shell 3)
    
    # So we only need shell 2 fully, and maybe a few from shell 3.
    # Let's generate a "shell 3" candidate set of 120 points.
    # The 120 points are the vertices of the "great dodecahedral icosahedron"?
    # Let's use a simple heuristic: rotate the 12 icosahedron vertices by small angles?
    # No, let's use the 120 points of the 3rd shell of the Mackay icosahedron.
    # They are located at:
    # 12 * (0, 1, phi) rotated?
    # Actually, the 120 points are the 12 vertices of the icosahedron, 
    # rotated by the icosahedral group and shifted?
    
    # Let's just generate 120 points on a sphere with icosahedral symmetry using a library or manual construction.
    # Since we don't have a library, let's use a "golden spiral" on a sphere for the remaining points,
    # but that breaks symmetry.
    
    # Better: Use the 42 points from shell 2, and generate more points by applying 
    # icosahedral rotations to a single new point.
    # The icosahedral group has 60 elements.
    # If we pick a point not on a symmetry axis, its orbit has 60 points.
    # Two orbits = 120 points.
    
    # Let's define the icosahedral rotation matrices.
    # This is complex. 
    # Alternative: For the purpose of the seed, we can just use a dense random set 
    # for the outer shells and rely on relaxation. 
    # But SCVO needs a good parametric form.
    
    # Let's stick to Shell 1 (12) and Shell 2 (42). 
    # For n > 55, we add random points on a sphere of radius r3.
    
    return v1, shell2

def _generate_icosahedral_cluster(n, r1=1.0, r2=1.9, r3=2.8, twist1=0.0, twist2=0.0, seed=42):
    """
    Generate a cluster with icosahedral symmetry.
    r1, r2, r3: radii of shells 1, 2, 3.
    twist1, twist2: twist angles for shells 2 and 3.
    """
    rng = np.random.default_rng(seed)
    v1, shell2 = _get_icosahedral_directions()
    
    positions = [np.zeros(3)]  # Center
    
    # Shell 1: 12 atoms
    if n > 1:
        # Apply twist1? Usually shell 1 is fixed.
        pos1 = v1 * r1
        positions.extend(pos1.tolist())
    
    # Shell 2: 42 atoms
    if n > 13:
        # Apply twist2
        # Rotation around z-axis? Or around a symmetry axis?
        # Let's rotate around the z-axis for simplicity, or use a general rotation.
        # A twist around the 5-fold axis is better.
        # Let's just rotate the whole shell2 around the z-axis by twist2.
        theta = twist2
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
        pos2 = (shell2 @ R.T) * r2
        # Add only as many as needed
        num_needed = min(42, n - len(positions))
        positions.extend(pos2[:num_needed].tolist())
    
    # Shell 3: remaining atoms
    if len(positions) < n:
        # Generate random points on sphere of radius r3
        num_needed = n - len(positions)
        # Use Fibonacci sphere for better distribution
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(num_needed):
            z = 1 - 2 * i / max(1, num_needed - 1)
            r = np.sqrt(1 - z*z)
            theta = golden_angle * i + twist1 # Use twist1 as phase
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * r3
            positions.append(p)
            
    pos_array = np.array(positions[:n])
    
    # Ensure minimum distance
    # Simple relaxation-like nudge if too close
    for i in range(len(pos_array)):
        for j in range(i+1, len(pos_array)):
            d = np.linalg.norm(pos_array[i] - pos_array[j])
            if d < 0.5 and d > 1e-10:
                direction = (pos_array[i] - pos_array[j]) / d
                pos_array[i] += 0.01 * direction
                pos_array[j] -= 0.01 * direction
                
    return pos_array

def _relax_with_timeout(positions, potential, parameter, timeout=5.0):
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

def _scvo_worker(initial_params, n, potential, parameter, max_iter, time_limit, result_holder, lock):
    """
    Perform Basin Hopping on the parameter space (r1, r2, r3, twist1, twist2).
    """
    if not _check_time():
        return
        
    # Initial evaluation
    pos = _generate_icosahedral_cluster(n, *initial_params)
    energy, pos = _relax_with_timeout(pos, potential, parameter, timeout=10.0)
    if energy is None:
        return
        
    best_energy = energy
    best_pos = pos.copy()
    current_energy = energy
    current_params = initial_params.copy()
    
    rng = np.random.default_rng(hash(id(initial_params)) % 2**32)
    
    # Step size in parameter space
    step_sizes = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    
    for i in range(max_iter):
        if not _check_time():
            break
        if time.time() > time_limit:
            break
            
        # Perturb parameters
        new_params = current_params + rng.normal(0, step_sizes)
        # Constrain radii to be positive and increasing
        new_params[0] = max(0.5, new_params[0])
        new_params[1] = max(new_params[0] + 0.5, new_params[1])
        new_params[2] = max(new_params[1] + 0.5, new_params[2])
        # Twists can be anything
        
        pos = _generate_icosahedral_cluster(n, *new_params)
        new_energy, new_pos = _relax_with_timeout(pos, potential, parameter, timeout=8.0)
        if new_energy is None:
            continue
            
        # Metropolis acceptance
        T = 0.5 * (1.0 - i / max(1, max_iter))
        if new_energy < current_energy or rng.random() < np.exp(-(new_energy - current_energy) / max(T, 0.01)):
            current_energy = new_energy
            current_params = new_params
            
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos.copy()
            
            with lock:
                if result_holder['energy'] is None or best_energy < result_holder['energy']:
                    result_holder['energy'] = best_energy
                    result_holder['pos'] = best_pos.copy()
            
        # Adaptive step size
        if new_energy < current_energy:
            step_sizes *= 0.95
        else:
            step_sizes *= 1.05
            step_sizes = np.clip(step_sizes, 0.01, 0.5)
            
    with lock:
        if result_holder['energy'] is None or best_energy < result_holder['energy']:
            result_holder['energy'] = best_energy
            result_holder['pos'] = best_pos.copy()

def _bh_cartesian_worker(initial_pos, n, potential, parameter, max_iter, time_limit, result_holder, lock):
    """
    Standard Basin Hopping in Cartesian space as a fallback/diversity.
    """
    if not _check_time():
        return
        
    energy, pos = _relax_with_timeout(initial_pos, potential, parameter, timeout=10.0)
    if energy is None:
        return
        
    best_energy = energy
    best_pos = pos.copy()
    current_energy = energy
    current_pos = pos.copy()
    
    rng = np.random.default_rng(hash(id(initial_pos)) % 2**32)
    step_size = 0.6
    
    for i in range(max_iter):
        if not _check_time():
            break
        if time.time() > time_limit:
            break
            
        n_perturb = max(1, min(5, len(current_pos) // 10))
        indices = rng.choice(len(current_pos), size=n_perturb, replace=False)
        perturbed = current_pos.copy()
        perturbed[indices] += rng.normal(0, step_size, (n_perturb, 3))
        
        new_energy, new_pos = _relax_with_timeout(perturbed, potential, parameter, timeout=8.0)
        if new_energy is None:
            continue
            
        T = 0.5 * (1.0 - i / max(1, max_iter))
        if new_energy < current_energy or rng.random() < np.exp(-(new_energy - current_energy) / max(T, 0.01)):
            current_energy = new_energy
            current_pos = new_pos
            
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos.copy()
            
            with lock:
                if result_holder['energy'] is None or best_energy < result_holder['energy']:
                    result_holder['energy'] = best_energy
                    result_holder['pos'] = best_pos.copy()
            
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
    1. Generate initial guesses using SCVO parameters.
    2. Run parallel SCVO basin hopping and Cartesian basin hopping.
    3. Return the best result found.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind", "dzugutov"))
    
    time_limit = _START_TIME + TIME_LIMIT - 10
    
    result_holder = {'energy': None, 'pos': None}
    lock = threading.Lock()
    
    # Initial parameters for SCVO
    # r1 ~ 1.0, r2 ~ 1.9, r3 ~ 2.8
    # Twists ~ 0.0
    base_params = [1.0, 1.9, 2.8, 0.0, 0.0]
    
    # Generate diverse initial parameters
    rng = np.random.default_rng(size)
    seeds_params = []
    for i in range(4):
        p = base_params + rng.normal(0, 0.2, 5)
        p[0] = max(0.5, p[0])
        p[1] = max(p[0] + 0.5, p[1])
        p[2] = max(p[1] + 0.5, p[2])
        seeds_params.append(p)
        
    # Generate Cartesian seeds
    ico_pos = _generate_icosahedral_cluster(size, *base_params)
    rand_pos = ico_pos + np.random.default_rng(999).normal(0, 0.5, ico_pos.shape)
    
    # Define workers
    def worker_scvo(i):
        _scvo_worker(seeds_params[i], size, potential, parameter, 100, time_limit, result_holder, lock)
        
    def worker_bh_ico():
        _bh_cartesian_worker(ico_pos, size, potential, parameter, 100, time_limit, result_holder, lock)
        
    def worker_bh_rand():
        _bh_cartesian_worker(rand_pos, size, potential, parameter, 100, time_limit, result_holder, lock)
        
    # Run 4 workers in parallel
    threads = [
        threading.Thread(target=worker_scvo, args=(0,)),
        threading.Thread(target=worker_scvo, args=(1,)),
        threading.Thread(target=worker_bh_ico),
        threading.Thread(target=worker_bh_rand),
    ]
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
    if result_holder['pos'] is not None:
        return result_holder['pos']
    else:
        return ico_pos

# EVOLVE-BLOCK-END
