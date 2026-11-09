"""
Strategy: Polytetrahedral (PT) motif seeding for Dzugutov clusters.
The Dzugutov potential favors PT order (icosahedral shells).
We generate candidate structures based on:
1. Icosahedral shells (Mackay-like).
2. FCC-like structures.
3. Random structures.
We relax these in parallel using 4 threads.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax

def _relax_worker(args):
    """Worker function for parallel relaxation."""
    positions, potential = args
    try:
        energy, relaxed_pos = relax(positions, potential, None)
        return energy, relaxed_pos
    except Exception:
        return 1e10, positions

def _generate_icosahedron(n):
    """Generate an icosahedral-like structure for n atoms."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    phi = (1 + np.sqrt(5)) / 2
    # Vertices of a regular icosahedron
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts[0]) # Normalize to unit sphere
    
    # Scale to appropriate radius
    # Approximate optimal density for Dzugutov is high.
    # Let's assume volume per atom ~ 1.0.
    # R ~ (3n/4pi)^(1/3) ~ 0.62 n^(1/3)
    # But icosahedra are not spheres.
    # Let's use R = 0.8 * n^(1/3) as a starting guess.
    
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 1: 12 atoms
    if n >= 13:
        R1 = 1.0
        shell1 = verts * R1
        points.extend(shell1)
        
    # Shell 2: 42 atoms (12 vertices + 30 edge midpoints of a larger icosahedron)
    if n >= 55:
        R2 = R1 * 1.8
        shell2_verts = verts * R2
        points.extend(shell2_verts)
        
        # Add 30 edge midpoints
        # Find edges: pairs of vertices with minimum distance
        min_dist = np.inf
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(shell2_verts[i] - shell2_verts[j])
                if d < min_dist:
                    min_dist = d
                    
        # Add midpoints for all pairs with distance close to min_dist
        added = 0
        for i in range(12):
            for j in range(i+1, 12):
                if added >= 30: break
                d = np.linalg.norm(shell2_verts[i] - shell2_verts[j])
                if np.isclose(d, min_dist, rtol=1e-5):
                    mid = (shell2_verts[i] + shell2_verts[j]) / 2
                    points.append(mid)
                    added += 1
            if added >= 30: break

    # If n < 55, we might have too many points if we added shell 1.
    # We need to truncate to n.
    if len(points) > n:
        points = np.array(points)
        dists = np.linalg.norm(points, axis=1)
        # Keep the closest n points
        keep_indices = np.argsort(dists)[:n]
        points = points[keep_indices]
    else:
        # If we have fewer points than n, add random points on the outer shell
        if len(points) < n:
            points = np.array(points)
            max_r = np.max(np.linalg.norm(points, axis=1))
            needed = n - len(points)
            for _ in range(needed):
                vec = np.random.randn(3)
                vec = vec / np.linalg.norm(vec)
                points = np.vstack([points, vec * max_r])
                
    # Center the structure
    points -= points.mean(axis=0)
    
    return points

def _generate_fcc(n):
    """Generate an FCC-like structure."""
    points = []
    size = int(np.ceil(n ** (1/3))) + 1
    for x in range(size):
        for y in range(size):
            for z in range(size):
                if len(points) < n:
                    points.append([x, y, z])
    points = np.array(points)
    points -= points.mean(axis=0)
    R = 0.8 * n ** (1/3)
    max_r = np.max(np.linalg.norm(points, axis=1))
    if max_r > 0:
        points *= R / max_r
    return points

def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    # 300s total for 6 instances. 50s per instance.
    # Keep 5s margin.
    deadline = start_time + 45.0
    
    starts = []
    
    # 1. Icosahedral
    try:
        ico = _generate_icosahedron(n)
        starts.append(ico)
        # Perturbed icosahedral
        for _ in range(5):
            pert = ico + np.random.randn(n, 3) * 0.1
            starts.append(pert)
    except Exception:
        pass
        
    # 2. FCC
    try:
        fcc = _generate_fcc(n)
        starts.append(fcc)
        for _ in range(5):
            pert = fcc + np.random.randn(n, 3) * 0.1
            starts.append(pert)
    except Exception:
        pass
        
    # 3. Random
    for _ in range(10):
        R = 1.0 * n ** (1/3)
        pts = np.random.randn(n, 3) * R
        starts.append(pts)
        
    best_energy = 1e10
    best_pos = None
    
    # Use thread pool
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_relax_worker, (s, potential)): s for s in starts}
        
        for future in as_completed(futures):
            if time.time() > deadline:
                break
            try:
                energy, pos = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos
            except Exception:
                pass
                
    # If we have time, do basin hopping on the best found
    if best_pos is not None and time.time() < deadline - 5.0:
        current_best = best_pos.copy()
        current_energy = best_energy
        rng = np.random.default_rng(42)
        
        while time.time() < deadline - 2.0:
            # Perturb
            kick = 0.2
            perturbed = current_best + rng.normal(0, kick, (n, 3))
            
            # Relax
            try:
                energy, pos = relax(perturbed, potential, None)
                if energy < current_energy:
                    current_energy = energy
                    current_best = pos
            except Exception:
                pass
                
    return current_best if current_best is not None else np.random.randn(n, 3)
# EVOLVE-BLOCK-END
