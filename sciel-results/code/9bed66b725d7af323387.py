"""Dzugutov cluster global minima: Mackay icosahedral seeds + parallel basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global deadline tracking
_START_TIME = None
_DEADLINE = 290.0  # seconds, with safety margin
_THREAD_LOCK = threading.Lock()


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _energy(pos):
    """Compute Dzugutov energy vectorized."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(pos)
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    
    # Avoid division by zero
    dist_safe = np.where(dist > 1e-10, dist, 1e10)
    
    # Dzugutov potential
    e1 = np.where(dist_safe < a, A * (dist_safe**(-m) - B) * np.exp(c / (dist_safe - a)), 0.0)
    e2 = np.where(dist_safe < b, B * np.exp(d / (dist_safe - b)), 0.0)
    # Only count each pair once (upper triangle)
    total = 0.5 * np.sum(e1 + e2)
    return total


def _icosahedron_vertices():
    """Generate 12 vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    return verts


def _mackay_icosahedron(n):
    """Generate a Mackay icosahedron-like cluster with n atoms."""
    if n <= 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Mackay icosahedra: 13, 55, 147...
    # For general n, build shells
    ico = _icosahedron_vertices()
    positions = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 1: 12 vertices at radius ~1.1
    r1 = 1.1
    for v in ico:
        positions.append(r1 * v)
    
    # Shell 2: 30 edge midpoints at radius ~1.8
    if n > 13:
        # Generate edge midpoints
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(ico[i] - ico[j])
                if 1.4 < d < 1.6:  # close to edge length
                    edges.append((i, j))
        
        r2 = 1.8
        for i, j in edges:
            mid = 0.5 * (ico[i] + ico[j])
            mid = mid / np.linalg.norm(mid)
            positions.append(r2 * mid)
    
    # Shell 3: 20 face centers at radius ~2.3
    if n > 43:
        # Generate face centers
        faces = []
        for i in range(12):
            for j in range(i+1, 12):
                for k in range(j+1, 12):
                    d1 = np.linalg.norm(ico[i] - ico[j])
                    d2 = np.linalg.norm(ico[j] - ico[k])
                    d3 = np.linalg.norm(ico[i] - ico[k])
                    if 1.4 < d1 < 1.6 and 1.4 < d2 < 1.6 and 1.4 < d3 < 1.6:
                        faces.append((i, j, k))
        
        r3 = 2.3
        for i, j, k in faces:
            center = (ico[i] + ico[j] + ico[k]) / 3.0
            center = center / np.linalg.norm(center)
            positions.append(r3 * center)
    
    # Trim or pad to get exactly n atoms
    positions = np.array(positions)
    if len(positions) > n:
        positions = positions[:n]
    elif len(positions) < n:
        # Add atoms in a dense packing
        rng = np.random.default_rng(42)
        while len(positions) < n:
            p = rng.uniform(-2.5, 2.5, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in positions):
                positions = np.vstack([positions, p])
    
    return positions


def _basin_hopping(pos, potential, parameter, max_iter=50, step_size=0.1):
    """Simple basin hopping: perturb, relax, accept if lower energy."""
    best_pos = pos.copy()
    best_energy = _energy(pos)
    rng = np.random.default_rng(hash(tuple(pos[0].round(6))) % (2**32))
    
    for _ in range(max_iter):
        if _remaining() < 5.0:
            break
        # Perturb
        perturbation = rng.normal(0, step_size, pos.shape)
        new_pos = pos + perturbation
        
        # Relax
        try:
            energy, relaxed = relax(new_pos, potential, parameter)
        except:
            continue
        
        # Accept if better
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed
            pos = relaxed
            # Reduce step size on acceptance
            step_size *= 0.9
        else:
            # Occasionally accept worse states (simulated annealing style)
            if rng.random() < 0.1:
                pos = relaxed
            step_size *= 1.05
    
    return best_energy, best_pos


def _worker(task_id, n, potential, parameter, seed_offset=0):
    """Worker function for parallel basin hopping."""
    rng = np.random.default_rng(task_id * 1000 + seed_offset)
    
    # Generate multiple starting configurations
    configs = []
    
    # Mackay icosahedron
    configs.append(_mackay_icosahedron(n))
    
    # Random compact clusters
    radius = 1.1 * n ** (1.0 / 3.0) * 0.75
    for _ in range(3):
        points = []
        attempts = 0
        while len(points) < n and attempts < 1000:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                    points.append(p)
            attempts += 1
        if len(points) == n:
            configs.append(np.array(points))
    
    # Icosahedral shell with random twists
    ico = _icosahedron_vertices()
    for twist in [0.0, 0.1, 0.2, 0.3]:
        positions = [np.array([0.0, 0.0, 0.0])]
        r1 = 1.1
        for v in ico:
            # Apply twist
            angle = twist
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            v_rot = np.array([v[0]*cos_a - v[1]*sin_a, v[0]*sin_a + v[1]*cos_a, v[2]])
            positions.append(r1 * v_rot)
        
        if n > 13:
            r2 = 1.8
            for i in range(12):
                for j in range(i+1, 12):
                    d = np.linalg.norm(ico[i] - ico[j])
                    if 1.4 < d < 1.6:
                        mid = 0.5 * (ico[i] + ico[j])
                        mid = mid / np.linalg.norm(mid)
                        # Apply twist
                        mid_rot = np.array([mid[0]*cos_a - mid[1]*sin_a, mid[0]*sin_a + mid[1]*cos_a, mid[2]])
                        positions.append(r2 * mid_rot)
        
        if len(positions) > n:
            positions = positions[:n]
        elif len(positions) < n:
            while len(positions) < n:
                p = rng.uniform(-2.5, 2.5, 3)
                if all(np.linalg.norm(p - q) >= 0.5 for q in positions):
                    positions.append(p)
        
        if len(positions) == n:
            configs.append(np.array(positions))
    
    # Run basin hopping on each config
    best_energy = np.inf
    best_pos = None
    
    for cfg in configs:
        if _remaining() < 10.0:
            break
        try:
            energy, relaxed = _basin_hopping(cfg, potential, parameter, max_iter=30, step_size=0.08)
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed
        except:
            continue
    
    if best_pos is None:
        # Fallback
        best_pos = configs[0]
        best_energy = _energy(best_pos)
    
    return task_id, best_energy, best_pos


def construct(instance):
    """Find low-energy Dzugutov cluster configurations using parallel basin hopping."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Use 4 threads
    n_threads = min(4, max(1, int(_remaining() / 50)))
    
    results = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for i in range(n_threads):
            future = executor.submit(_worker, i, n, potential, parameter, seed_offset=i*7)
            futures.append(future)
        
        for future in as_completed(futures):
            if _remaining() < 5.0:
                break
            try:
                task_id, energy, pos = future.result(timeout=5.0)
                results.append((energy, pos))
            except:
                continue
    
    if not results:
        # Fallback: return a simple configuration
        return _mackay_icosahedron(n)
    
    # Return the best result
    best = min(results, key=lambda x: x[0])
    return best[1]
# EVOLVE-BLOCK-END
