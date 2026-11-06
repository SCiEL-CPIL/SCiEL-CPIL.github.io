"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

def _icosahedral_shell(n, rng=None):
    """Generate n points in icosahedral shells (Mackay icosahedra)."""
    if rng is None:
        rng = np.random.default_rng()
    
    phi = (1 + np.sqrt(5)) / 2
    # 12 unique icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [1, phi, 0], [-1, phi, 0],
        [phi, 0, 1], [phi, 0, -1], [1, phi, 0], [-1, phi, 0],
        [0, 1, -phi], [0, -1, -phi], [phi, 0, 1], [phi, 0, -1]
    ])
    # Deduplicate
    unique_verts = []
    seen = set()
    for v in verts:
        key = tuple(np.round(v, 6))
        if key not in seen:
            seen.add(key)
            unique_verts.append(v)
    unique_verts = np.array(unique_verts[:12])
    unique_verts /= np.linalg.norm(unique_verts, axis=1, keepdims=True)
    
    points = []
    
    # Shell 0: center
    if n > 0:
        points.append(np.array([0.0, 0.0, 0.0]))
        n -= 1
    
    # Shell 1: 12 vertices at r ~ 1.1
    if n > 0:
        count = min(12, n)
        for i in range(count):
            points.append(unique_verts[i] * 1.1)
        n -= count
    
    # Shell 2: 42 vertices at r ~ 2.2
    # Approximate by taking midpoints of edges and face centers of icosahedron
    if n > 0:
        # Generate points on icosahedron surface using golden ratio spiral
        shell_r = 2.2
        count = min(42, n)
        golden = (1 + np.sqrt(5)) / 2
        for i in range(count):
            t = (i + 0.5) / count
            z = 1 - 2 * t
            r = np.sqrt(1 - z**2)
            theta = 2 * np.pi * i / golden
            p = np.array([r * np.cos(theta), r * np.sin(theta), z])
            p /= np.linalg.norm(p)
            points.append(p * shell_r)
        n -= count
    
    # Shell 3: 92 vertices at r ~ 3.3 (for larger clusters)
    if n > 0:
        shell_r = 3.3
        count = min(92, n)
        golden = (1 + np.sqrt(5)) / 2
        for i in range(count):
            t = (i + 0.5) / count
            z = 1 - 2 * t
            r = np.sqrt(1 - z**2)
            theta = 2 * np.pi * i / golden
            p = np.array([r * np.cos(theta), r * np.sin(theta), z])
            p /= np.linalg.norm(p)
            points.append(p * shell_r)
        n -= count
    
    # Fill remaining with random points in outer shell
    if n > 0:
        min_dist = 0.7
        max_attempts = 5000
        added = 0
        while added < n and max_attempts > 0:
            r = rng.uniform(2.5, 4.0)
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            p = v * r
            if points:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) < min_dist:
                    max_attempts -= 1
                    continue
            points.append(p)
            added += 1
            max_attempts -= 1
        if added < n:
            for _ in range(n - added):
                r = rng.uniform(1.0, 4.0)
                v = rng.normal(size=3)
                v /= np.linalg.norm(v)
                points.append(v * r)
    
    return np.array(points)

def _random_dense_packing(n, rng=None):
    """Generate n points in a dense random packing."""
    if rng is None:
        rng = np.random.default_rng()
    
    radius = 1.3 * n ** (1.0/3.0)
    pts = []
    min_dist = 0.6
    max_attempts = 10000
    added = 0
    while added < n and max_attempts > 0:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            max_attempts -= 1
            continue
        if pts:
            dists = np.linalg.norm(np.array(pts) - p, axis=1)
            if np.min(dists) < min_dist:
                max_attempts -= 1
                continue
        pts.append(p)
        added += 1
        max_attempts -= 1
    
    if added < n:
        # Fill with random points
        for _ in range(n - added):
            p = rng.uniform(-radius, radius, 3)
            pts.append(p)
    
    return np.array(pts)

def _basin_hopping(pos, potential, parameter, n_iter=30, kick_scale=0.4, rng=None):
    """Basin hopping with adaptive kick size."""
    if rng is None:
        rng = np.random.default_rng()
    
    try:
        energy, pos = relax(pos, potential, parameter)
    except Exception:
        return float('inf'), pos
    
    best_energy = energy
    best_pos = pos.copy()
    
    for i in range(n_iter):
        # Adaptive kick: decrease over time
        kick = kick_scale * (1.0 - i / (2.0 * n_iter))
        new_pos = pos + rng.normal(0, kick, pos.shape)
        
        try:
            new_energy, new_pos_relaxed = relax(new_pos, potential, parameter)
        except Exception:
            continue
        
        if new_energy < energy:
            pos = new_pos_relaxed
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
    
    return best_energy, best_pos

def construct(instance):
    """
    Finds the lowest-energy structure for a Dzugutov cluster of size n.
    Uses multiple initialization strategies and basin hopping.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    # Total time budget for all 6 instances is 300s, so ~50s per instance
    # But we need to be safe, so use 45s per instance
    time_limit = 45.0
    
    rng = np.random.default_rng(42 + size)
    
    best_energy = float('inf')
    best_positions = None
    
    # Strategy 1: Icosahedral initialization + Basin Hopping (multiple restarts)
    for i in range(8):
        if time.time() - start_time > time_limit * 0.7:
            break
        
        pos = _icosahedral_shell(size, rng)
        # Add small noise
        pos += rng.normal(0, 0.05, pos.shape)
        
        try:
            e, p = _basin_hopping(pos, potential, parameter, n_iter=25, kick_scale=0.35, rng=rng)
            if e < best_energy:
                best_energy = e
                best_positions = p
        except Exception:
            pass
    
    # Strategy 2: Random dense packing + Basin Hopping
    if time.time() - start_time < time_limit * 0.85:
        for i in range(4):
            if time.time() - start_time > time_limit * 0.85:
                break
            
            pos = _random_dense_packing(size, rng)
            pos += rng.normal(0, 0.1, pos.shape)
            
            try:
                e, p = _basin_hopping(pos, potential, parameter, n_iter=20, kick_scale=0.4, rng=rng)
                if e < best_energy:
                    best_energy = e
                    best_positions = p
            except Exception:
                pass
    
    # Strategy 3: Final relaxation of best position with more iterations
    if best_positions is not None and time.time() - start_time < time_limit:
        try:
            e, p = _basin_hopping(best_positions, potential, parameter, n_iter=15, kick_scale=0.2, rng=rng)
            if e < best_energy:
                best_energy = e
                best_positions = p
        except Exception:
            pass
    
    if best_positions is None:
        # Fallback
        rng2 = np.random.default_rng(size)
        radius = 2.0
        points = rng2.uniform(-radius, radius, (size, 3))
        try:
            e, best_positions = relax(points, potential, parameter)
        except Exception:
            best_positions = points
    
    return best_positions

# EVOLVE-BLOCK-END
