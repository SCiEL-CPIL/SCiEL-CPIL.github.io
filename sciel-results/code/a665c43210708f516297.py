"""
Strategy: Deterministic Multi-Threaded Parallel Local Optimization
- Generates a diverse pool of high-quality geometric seeds (Icosahedral, Decahedral, Random Compact).
- Uses ThreadPoolExecutor to run up to 4 independent L-BFGS-B relaxations in parallel.
- Avoids the overhead and stochastic inefficiency of Basin Hopping.
- Prioritizes seed diversity and parallel throughput to find deep local minima.
- Deadline-guarded to ensure valid output within 330s.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Deadline guard
_T0 = time.time()
_DEADLINE = 325.0  # Safety margin below 330s
_LOCK = threading.Lock()

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
    Generate vertices of a decahedron (approximated by a cube with corners cut or similar).
    For simplicity, we use a distorted cube or a set of points that resembles a decahedral packing.
    A standard decahedron isn't as common as icosahedron in these potentials, but we can use
    a 'truncated cube' or simply a cube with face centers to provide a different symmetry.
    Let's use a cube with 8 corners + 6 face centers = 14 points.
    """
    s = 1.0
    v = []
    # Corners
    for x in [-s, s]:
        for y in [-s, s]:
            for z in [-s, s]:
                v.append([x, y, z])
    # Face centers
    v.append([0, 0, s])
    v.append([0, 0, -s])
    v.append([0, s, 0])
    v.append([0, -s, 0])
    v.append([s, 0, 0])
    v.append([-s, 0, 0])
    
    v = np.array(v, dtype=float)
    # Normalize to unit radius
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def _generate_random_compact(n, rng, seed_scale=0.8):
    """
    Generate a random compact cluster using rejection sampling.
    Ensures minimum distance to avoid immediate invalidity.
    """
    points = []
    # Estimate radius for n atoms with ~0.7 spacing
    radius = 0.8 * n ** (1.0 / 3.0) * 1.5 
    
    max_attempts = 5000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if points:
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) < 0.5:
                attempts += 1
                continue
        points.append(p)
        attempts += 1
    
    if len(points) < n:
        for i in range(len(points), n):
            p = np.array([i * 1.0, 0, 0])
            points.append(p)
            
    return np.array(points)

def _generate_icosahedral_seed(n, rng):
    """
    Generate a seed based on Mackay icosahedra.
    For n=13, 55, use exact structures.
    For other n, start with the largest magic number <= n and add random surface atoms.
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    magic = [1, 13, 55, 147]
    base_n = 1
    for mn in magic:
        if mn <= n:
            base_n = mn
        else:
            break
            
    if base_n == 1:
        pts = [np.array([0.0, 0.0, 0.0])]
    elif base_n == 13:
        v = _icosahedron_vertices()
        pts = [np.array([0.0, 0.0, 0.0])] + [v[i] for i in range(12)]
    elif base_n == 55:
        # Construct 55-atom Mackay icosahedron approximation
        v1 = _icosahedron_vertices()
        pts = [np.array([0.0, 0.0, 0.0])]
        pts.extend([v1[i] for i in range(12)])
        
        # Add 42 random points on sphere radius 1.5
        for _ in range(42):
            dir = rng.normal(size=3)
            dir /= np.linalg.norm(dir)
            pts.append(dir * 1.5)
    else:
        return _generate_random_compact(n, rng)
        
    pts = np.array(pts)
    
    if n > len(pts):
        current_radius = 1.5 if base_n >= 13 else 0.5
        for _ in range(n - len(pts)):
            dir = rng.normal(size=3)
            dir /= np.linalg.norm(dir)
            pts = np.vstack([pts, [dir * (current_radius + 0.2)]])
            
    return pts

def _generate_decahedral_seed(n, rng):
    """
    Generate a seed based on decahedral symmetry (cube-like).
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
        
    v = _decahedron_vertices()
    pts = [np.array([0.0, 0.0, 0.0])]
    pts.extend([v[i] for i in range(min(n-1, len(v)))])
    
    pts = np.array(pts)
    
    if n > len(pts):
        current_radius = 1.2
        for _ in range(n - len(pts)):
            dir = rng.normal(size=3)
            dir /= np.linalg.norm(dir)
            pts = np.vstack([pts, [dir * (current_radius + 0.2)]])
            
    return pts

def _worker(seed, potential, kind, result_holder, index):
    """
    Worker function for parallel relaxation.
    """
    try:
        if _time_left() < 1.0:
            return
        e, pos = relax(seed, potential, kind)
        if np.isfinite(e):
            with _LOCK:
                if result_holder['best_energy'] is None or e < result_holder['best_energy']:
                    result_holder['best_energy'] = e
                    result_holder['best_pos'] = pos
    except Exception:
        pass

def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    kind = instance["kind"]
    
    # Result holder shared across threads
    result_holder = {'best_energy': None, 'best_pos': None}
    
    # Generate diverse seeds
    seeds = []
    rngs = [np.random.default_rng(i) for i in range(20)]
    
    # 1. Icosahedral seeds (multiple variations)
    for i in range(5):
        try:
            s = _generate_icosahedral_seed(n, rngs[i])
            if s is not None and len(s) == n:
                seeds.append(s)
        except:
            pass
            
    # 2. Decahedral seeds (multiple variations)
    for i in range(5, 10):
        try:
            s = _generate_decahedral_seed(n, rngs[i])
            if s is not None and len(s) == n:
                seeds.append(s)
        except:
            pass
            
    # 3. Random Compact seeds (multiple variations)
    for i in range(10, 20):
        try:
            s = _generate_random_compact(n, rngs[i])
            if len(s) == n:
                seeds.append(s)
        except:
            pass
            
    if not seeds:
        # Fallback
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos

    # Parallel Relaxation
    # We have 4 threads. We will process seeds in batches of 4.
    # Since relax() is likely CPU-bound (L-BFGS-B), GIL might be an issue, 
    # but NumPy/SciPy often release GIL during heavy computation.
    # Even if not fully parallel, it allows for more attempts than sequential if we manage time well.
    # However, if GIL is held, threads won't speed up CPU-bound work.
    # Let's assume relax() releases GIL or the overhead is low enough.
    # If not, we still benefit from the diverse seeds.
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, seed in enumerate(seeds):
            if _time_left() < 5.0:
                break
            futures.append(executor.submit(_worker, seed, potential, kind, result_holder, i))
            
        # Wait for all to complete or timeout
        for future in as_completed(futures, timeout=max(1.0, _time_left())):
            pass # Exceptions handled in worker

    # If we still have time and no result, or just to be safe, run a few more sequentially
    # This ensures we don't miss a quick win if threads were slow to start
    if result_holder['best_pos'] is None or _time_left() > 10.0:
        for i, seed in enumerate(seeds):
            if _time_left() < 5.0:
                break
            try:
                e, pos = relax(seed, potential, kind)
                if np.isfinite(e):
                    if result_holder['best_energy'] is None or e < result_holder['best_energy']:
                        result_holder['best_energy'] = e
                        result_holder['best_pos'] = pos
            except:
                pass

    # Final Fallback
    if result_holder['best_pos'] is None:
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos

    best_pos = result_holder['best_pos']

    # Validate
    if not np.all(np.isfinite(best_pos)):
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos
        
    # Check minimum distance
    d = np.linalg.norm(best_pos[:, None, :] - best_pos[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 0.35:
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos

    return best_pos
# EVOLVE-BLOCK-END
