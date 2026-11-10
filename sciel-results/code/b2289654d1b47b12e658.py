"""
Strategy: Deterministic Parallel Multi-Start Global Optimization
- Uses a pool of 4 threads to relax diverse geometric seeds in parallel.
- Seeds include Icosahedral, Decahedral, FCC, HCP, and Random Compact structures.
- After initial relaxation, the best candidates are further refined (polished) in parallel.
- This avoids the serial bottleneck of Basin Hopping and leverages CPU parallelism.
- Deadline-guarded to ensure valid output within 330s.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Deadline guard
_T0 = time.time()
_DEADLINE = 325.0  # Safety margin below 330s

def _time_left():
    return _DEADLINE - (time.time() - _T0)

def _icosahedron_vertices():
    """Generate vertices of a regular icosahedron normalized to unit radius."""
    phi = (1 + 5**0.5) / 2
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def _decahedron_vertices():
    """
    Generate vertices of a regular dodecahedron (approximation for decahedral clusters).
    For simplicity, we use a set of points that approximates the decahedral symmetry.
    A true regular dodecahedron has 20 vertices.
    """
    phi = (1 + 5**0.5) / 2
    # Vertices of a regular dodecahedron
    v = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
        [1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0],
        [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def _generate_icosahedral_seed(n):
    """
    Generate a seed based on Mackay icosahedra.
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    v12 = _icosahedron_vertices()
    base13 = np.vstack([np.array([[0.0, 0.0, 0.0]]), v12])
    
    if n <= 13:
        return base13[:n]
        
    rng = np.random.default_rng(12345)
    extra = []
    # Place extra atoms on the surface, slightly outside the icosahedron
    for _ in range(n - 13):
        dir = rng.normal(size=3)
        dir /= np.linalg.norm(dir)
        p = dir * 1.15
        extra.append(p)
        
    if extra:
        base13 = np.vstack([base13, np.array(extra)])
        
    return base13

def _generate_decahedral_seed(n):
    """
    Generate a seed based on decahedral symmetry.
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    v20 = _decahedron_vertices()
    base21 = np.vstack([np.array([[0.0, 0.0, 0.0]]), v20])
    
    if n <= 21:
        return base21[:n]
        
    rng = np.random.default_rng(54321)
    extra = []
    for _ in range(n - 21):
        dir = rng.normal(size=3)
        dir /= np.linalg.norm(dir)
        p = dir * 1.2
        extra.append(p)
        
    if extra:
        base21 = np.vstack([base21, np.array(extra)])
        
    return base21

def _generate_fcc_seed(n):
    """
    Generate a seed based on FCC (Face-Centered Cubic) cluster.
    """
    points = []
    # FCC lattice vectors
    a = 1.0
    # Generate points in a cubic box around the origin
    # For a cluster, we just need a compact set of points with FCC-like local environment
    # Simple approach: take points from an FCC lattice within a radius
    r_max = (n / 2.0) ** (1.0 / 3.0) * 0.8
    rng = np.random.default_rng(42)
    
    # Create a grid of FCC points
    # FCC points: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in units of a/sqrt(2)
    # Let's use a = sqrt(2) for nearest neighbor distance 1
    a = np.sqrt(2)
    grid_range = int(r_max / a) + 2
    for i in range(-grid_range, grid_range + 1):
        for j in range(-grid_range, grid_range + 1):
            for k in range(-grid_range, grid_range + 1):
                # Add all 4 FCC basis points
                for dx, dy, dz in [(0,0,0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)]:
                    p = np.array([i*a + dx, j*a + dy, k*a + dz])
                    if np.linalg.norm(p) <= r_max:
                        points.append(p)
    
    if len(points) < n:
        # Fill with random points if not enough
        rng = np.random.default_rng(123)
        while len(points) < n:
            p = rng.uniform(-r_max, r_max, 3)
            if np.linalg.norm(p) <= r_max:
                points.append(p)
    else:
        # Randomly select n points
        idx = rng.choice(len(points), n, replace=False)
        points = [points[i] for i in idx]
        
    return np.array(points)

def _generate_hcp_seed(n):
    """
    Generate a seed based on HCP (Hexagonal Close-Packed) cluster.
    """
    points = []
    # HCP lattice: A-B-A stacking
    a = 1.0 # nearest neighbor distance
    c = np.sqrt(8.0/3.0) * a
    # Basis: (0,0,0) and (1/3, 1/sqrt(3), c/3)
    # Let's just generate points in a hexagonal cylinder
    r_max = (n / 2.0) ** (1.0 / 3.0) * 0.8
    rng = np.random.default_rng(99)
    
    # Generate points in a hexagonal grid
    # Layer z = 0, c/3, 2c/3, c, ...
    num_layers = int(2 * r_max / c) + 2
    for l in range(num_layers):
        z = l * c / 2
        if z > r_max:
            break
        # Offset for B layers
        if l % 2 == 0:
            offset = 0
        else:
            offset = np.array([1/3, 1/(3*np.sqrt(3))])
            
        grid_range = int(r_max / a) + 2
        for i in range(-grid_range, grid_range + 1):
            for j in range(-grid_range, grid_range + 1):
                # Hexagonal coordinates to Cartesian
                x = a * (i + j/2.0)
                y = a * (j * np.sqrt(3) / 2.0)
                p = np.array([x, y, z]) + offset * a
                if np.linalg.norm(p) <= r_max:
                    points.append(p)
    
    if len(points) < n:
        rng = np.random.default_rng(456)
        while len(points) < n:
            p = rng.uniform(-r_max, r_max, 3)
            if np.linalg.norm(p) <= r_max:
                points.append(p)
    else:
        idx = rng.choice(len(points), n, replace=False)
        points = [points[i] for i in idx]
        
    return np.array(points)

def _generate_random_compact(n, seed_val):
    """
    Generate a random compact cluster using rejection sampling.
    """
    rng = np.random.default_rng(seed_val)
    points = []
    radius = (n / 2.0) ** (1.0 / 3.0) * 0.8 
    min_dist = 0.6
    
    attempts = 0
    while len(points) < n and attempts < n * 200:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if points:
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) < min_dist:
                continue
        points.append(p)
        
    if len(points) < n:
        # Fallback
        points = []
        for _ in range(n):
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
            
    return np.array(points)

def _worker(args):
    """
    Worker function for parallel relaxation.
    """
    pos, potential, kind = args
    try:
        e, pos_out = relax(pos, potential, kind)
        return e, pos_out
    except:
        return float('inf'), pos

def construct(instance):
    """
    Strategy:
    1. Generate diverse seeds (Icosahedral, Decahedral, FCC, HCP, Random).
    2. Relax all seeds in parallel using 4 threads.
    3. Pick the best few and polish them in parallel.
    """
    n = instance["n"]
    potential = instance["potential"]
    kind = instance["kind"]
    
    # Generate seeds
    seeds = []
    try:
        seeds.append(_generate_icosahedral_seed(n))
    except:
        pass
    try:
        seeds.append(_generate_decahedral_seed(n))
    except:
        pass
    try:
        seeds.append(_generate_fcc_seed(n))
    except:
        pass
    try:
        seeds.append(_generate_hcp_seed(n))
    except:
        pass
    
    # Add 4 random seeds
    for i in range(4):
        try:
            seeds.append(_generate_random_compact(n, 1000 + i))
        except:
            pass
            
    if not seeds:
        seeds.append(np.random.default_rng(1).normal(0, 1, (n, 3)))
        
    # Filter valid seeds (non-finite, min dist check)
    valid_seeds = []
    for s in seeds:
        if np.all(np.isfinite(s)):
            # Check min dist
            min_d = 100.0
            for i in range(len(s)):
                for j in range(i+1, len(s)):
                    d = np.linalg.norm(s[i] - s[j])
                    if d < min_d:
                        min_d = d
            if min_d > 0.35:
                valid_seeds.append(s)
            else:
                # If invalid, try to fix by randomizing slightly? 
                # Or just skip. For safety, skip.
                pass
                
    if not valid_seeds:
        # Fallback to a simple linear chain (high energy but valid)
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos
        
    # Parallel Relaxation Phase 1
    tasks = [(s, potential, kind) for s in valid_seeds]
    
    best_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_worker, t): i for i, t in enumerate(tasks)}
        for future in concurrent.futures.as_completed(futures):
            if _time_left() < 10.0:
                break
            try:
                e, pos = future.result()
                if e < float('inf'):
                    best_results.append((e, pos))
            except:
                pass
                
    if not best_results:
        # Fallback
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos
        
    # Sort by energy
    best_results.sort(key=lambda x: x[0])
    
    # Take top 4 for polishing
    top_candidates = [pos for e, pos in best_results[:4]]
    
    # Parallel Relaxation Phase 2 (Polishing)
    # We can try to perturb them slightly and relax again to escape shallow minima
    polish_tasks = []
    rng = np.random.default_rng(777)
    for pos in top_candidates:
        # Create a slightly perturbed version
        perturbed = pos + rng.normal(0, 0.05, pos.shape)
        polish_tasks.append((perturbed, potential, kind))
        
    polished_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_worker, t): i for i, t in enumerate(polish_tasks)}
        for future in concurrent.futures.as_completed(futures):
            if _time_left() < 5.0:
                break
            try:
                e, pos = future.result()
                if e < float('inf'):
                    polished_results.append((e, pos))
            except:
                pass
                
    # Combine results
    all_results = best_results + polished_results
    if all_results:
        all_results.sort(key=lambda x: x[0])
        best_e, best_pos = all_results[0]
        return best_pos
    else:
        # Fallback
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos
# EVOLVE-BLOCK-END
