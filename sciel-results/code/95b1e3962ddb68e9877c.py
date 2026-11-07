"""
Sutton-Chen transition-metal cluster global minima search.
Uses hierarchical structure prediction (icosahedral for small n, FCC/twinned for large n)
with parallel basin-hopping optimization.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_TOTAL_TIME_LIMIT = 290.0  # Safety margin before 300s kill
_START_TIME = None

def _time_left():
    if _START_TIME is None:
        return _TOTAL_TIME_LIMIT
    return _TOTAL_TIME_LIMIT - (time.time() - _START_TIME)

def _validate(positions):
    """Check if positions are valid: finite and min distance >= 0.35."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    # Vectorized distance check
    i_idx, j_idx = np.triu_indices(n, k=1)
    d = np.linalg.norm(positions[i_idx] - positions[j_idx])
    if np.any(d < 0.35):
        return False
    return True

def _get_icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron centered at origin, normalized to unit radius."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

def _generate_icosahedral_cluster(n, seed=0):
    """
    Generate an icosahedral cluster.
    Shell 0: 1 atom at origin.
    Shell 1: 12 atoms at radius R1 = 1.0.
    Shell 2: 30 atoms (edge midpoints) at radius R2.
    Shell 3: 20 atoms (face centers) at radius R3.
    Total: 1 + 12 + 30 + 20 = 63 atoms.
    If n > 63, add atoms on a larger shell.
    If n < 63, truncate from the outside in.
    """
    rng = np.random.default_rng(seed)
    
    ico_verts = _get_icosahedron_vertices()
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 1: 12 vertices
    R1 = 1.0
    shell1 = ico_verts * R1
    for v in shell1:
        points.append(v)
        
    # Calculate R2 (edge midpoints)
    v0 = ico_verts[0]
    v1 = ico_verts[1]
    edge_len = np.linalg.norm(v0 - v1)
    R2 = np.sqrt(R1**2 - (edge_len/2.0)**2)
    
    # Find edges
    edges = set()
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            if np.isclose(d, edge_len, atol=1e-6):
                edges.add((i, j))
                
    shell2 = []
    for (i, j) in edges:
        mid = (ico_verts[i] + ico_verts[j]) / 2.0
        mid_dir = mid / np.linalg.norm(mid)
        shell2.append(mid_dir * R2)
    for v in shell2:
        points.append(v)
        
    # Calculate R3 (face centers)
    faces = set()
    for i in range(12):
        for j in range(i+1, 12):
            if (i, j) not in edges: continue
            for k in range(j+1, 12):
                if (j, k) not in edges: continue
                if (i, k) not in edges: continue
                faces.add(tuple(sorted([i, j, k])))
                
    shell3 = []
    R3 = 0.0
    for (i, j, k) in faces:
        center = (ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3.0
        center_dir = center / np.linalg.norm(center)
        R3 = np.linalg.norm(center)
        shell3.append(center_dir * R3)
    for v in shell3:
        points.append(v)
        
    points = np.array(points)
    
    if len(points) >= n:
        # If we need fewer, we should remove outer shells first.
        # Order: Center (1), Shell1 (12), Shell2 (30), Shell3 (20).
        # To keep the core intact, we should truncate from the end.
        return points[:n]
    else:
        # Need more atoms than 63.
        # Add atoms on a shell outside R3.
        # A good place is the vertices of a larger icosahedron or random points on a sphere.
        # Let's use random points on a sphere of radius R3 * 1.2 to start, then relax.
        remaining = n - len(points)
        for _ in range(remaining):
            # Random direction
            u = rng.normal(0, 1, 3)
            u /= np.linalg.norm(u)
            r = R3 * (1.1 + rng.uniform(0, 0.2))
            new_p = u * r
            points = np.vstack([points, [new_p]])
            
        return points

def _generate_fcc_cluster(n, seed=0):
    """Generate points from an FCC lattice inside a sphere."""
    rng = np.random.default_rng(seed)
    a = 1.0
    # Approximate radius for n atoms in FCC
    # Density ~ 4 atoms per a^3
    R = (3.0 * n / (4.0 / a**3) / (4.0 * np.pi)) ** (1.0/3.0) * 1.1
    
    max_coord = int(np.ceil(R / a)) + 2
    points = []
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                # FCC basis
                points.append(np.array([ix, iy, iz]) * a)
                points.append(np.array([ix + 0.5, iy + 0.5, iz]) * a)
                points.append(np.array([ix + 0.5, iy, iz + 0.5]) * a)
                points.append(np.array([ix, iy + 0.5, iz + 0.5]) * a)
                
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    sorted_pts = points[order]
    
    if len(sorted_pts) >= n:
        return sorted_pts[:n]
    else:
        # If not enough, add random points
        pts = list(sorted_pts)
        while len(pts) < n:
            p = rng.uniform(-R, R, 3)
            pts.append(p)
        return np.array(pts)

def _generate_twinned_fcc(n, seed=0):
    """
    Generate a twinned FCC structure.
    Two FCC spheres sharing a common face, joined at the interface.
    This is a common low-energy structure for medium-large clusters.
    """
    rng = np.random.default_rng(seed)
    
    # Generate two FCC clusters of size n/2
    n1 = n // 2
    n2 = n - n1
    
    c1 = _generate_fcc_cluster(n1, seed=seed)
    c2 = _generate_fcc_cluster(n2, seed=seed+1)
    
    # Translate c2 so that it touches c1.
    # Find the "front" face of c1 and "back" face of c2.
    # For simplicity, we just translate c2 along the x-axis by the sum of their radii.
    r1 = np.max(np.linalg.norm(c1, axis=1))
    r2 = np.max(np.linalg.norm(c2, axis=1))
    
    # Move c2 to x = r1 + r2 - overlap
    # Overlap of ~0.5a to ensure connection
    offset = np.array([r1 + r2 - 0.5, 0, 0])
    c2 = c2 + offset
    
    combined = np.vstack([c1, c2])
    return combined[:n]

def _generate_random_compact(n, seed=0):
    """Generate a random compact cluster using rejection sampling."""
    rng = np.random.default_rng(seed)
    radius = n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    max_attempts = n * 1000
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if points:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= 0.5:
                    points.append(p)
            else:
                points.append(p)
        attempts += 1
        
    while len(points) < n:
        points.append(rng.uniform(-radius, radius, 3))
        
    return np.array(points)

def _perturb(positions, magnitude, seed):
    """Perturb positions by adding random noise."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, magnitude, positions.shape)
    return positions + noise

def _basin_hopping_worker(start_pos, kind, max_iterations, seed):
    """
    Perform basin hopping optimization starting from start_pos.
    Returns (best_energy, best_positions).
    """
    try:
        current_energy, current_pos = sutton_chen_energy_gradient(start_pos, kind)
        if not _validate(current_pos):
            return None, None
            
        best_energy = current_energy
        best_pos = current_pos.copy()
        
        step_size = 0.1
        seed_counter = seed
        
        for i in range(max_iterations):
            if _time_left() < 5.0:
                break
                
            # Perturb
            new_pos = _perturb(current_pos, step_size, seed_counter)
            seed_counter += 1
            
            # Relax
            try:
                new_energy, new_pos_relaxed = relax(new_pos, 'sutton_chen', kind)
            except:
                continue
                
            if not _validate(new_pos_relaxed):
                continue
                
            # Metropolis criterion
            delta_e = new_energy - current_energy
            if delta_e < 0:
                # Accept
                current_energy = new_energy
                current_pos = new_pos_relaxed
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pos = new_pos_relaxed.copy()
                # Reduce step size slightly if improving
                step_size = max(0.05, step_size * 0.95)
            else:
                # Metropolis acceptance
                # T is fixed or adaptive? Let's use a simple fixed T relative to energy scale
                # E_pub is around -1000 to -2000 for these sizes.
                # Let's use T = 0.5 * abs(best_energy) / n
                T = 0.5 * abs(best_energy) / len(start_pos)
                if np.random.rand() < np.exp(-delta_e / T):
                    current_energy = new_energy
                    current_pos = new_pos_relaxed
                    # Increase step size if stuck
                    step_size = min(0.5, step_size * 1.05)
                else:
                    # Rejected, maybe increase step size to escape local min
                    step_size = min(0.5, step_size * 1.02)
                    
        return best_energy, best_pos
    except Exception:
        return None, None

def construct(instance):
    """Find near-global minimum for Sutton-Chen transition-metal clusters."""
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    
    # Generate diverse starting configurations
    starts = []
    
    # 1. Icosahedral starts (Good for n < 50, and as core for larger)
    for i in range(5):
        try:
            ico = _generate_icosahedral_cluster(n, seed=i)
            if len(ico) == n and _validate(ico):
                starts.append(ico)
        except:
            pass
            
    # 2. FCC starts (Good for larger n)
    for i in range(5):
        try:
            fcc = _generate_fcc_cluster(n, seed=100+i)
            if len(fcc) == n and _validate(fcc):
                starts.append(fcc)
        except:
            pass

    # 3. Twinned FCC starts (Good for 50 < n < 100)
    for i in range(5):
        try:
            twf = _generate_twinned_fcc(n, seed=200+i)
            if len(twf) == n and _validate(twf):
                starts.append(twf)
        except:
            pass

    # 4. Random Compact starts (Diversity)
    for i in range(5):
        try:
            rand = _generate_random_compact(n, seed=300+i)
            if len(rand) == n and _validate(rand):
                starts.append(rand)
        except:
            pass

    if not starts:
        starts = [_generate_random_compact(n, seed=0)]
        
    best_energy = np.inf
    best_positions = starts[0]
    
    # Determine number of iterations per worker
    # Larger n takes longer to relax.
    # n=38: ~100 iters, n=80: ~50 iters
    max_iters = max(20, int(5000 / n))
    
    # Use all 4 threads
    # We have 20 starts. We can run 5 starts per thread? Or just 4 starts in parallel?
    # Let's run 4 starts in parallel, each doing max_iters basin hopping steps.
    # If we have more starts than threads, we can queue them.
    # But basin hopping is sequential per start.
    # Let's just pick the best 4 diverse starts and run them in parallel.
    # Or, run all 20 starts, 4 at a time.
    
    # To maximize coverage, let's run 4 workers, each processing 5 starts sequentially?
    # No, that's slow.
    # Let's run 4 workers, each doing basin hopping on 1 start.
    # We can increase the number of starts to 16, and run 4 batches of 4.
    
    # Strategy: Run 4 workers in parallel. Each worker processes 1 start.
    # We will run 4 rounds of 4 starts each, if time permits.
    
    num_rounds = 4
    starts_per_round = 4
    total_starts_to_use = min(len(starts), num_rounds * starts_per_round)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        for r in range(num_rounds):
            if _time_left() < 10.0:
                break
                
            # Select starts for this round
            round_starts = starts[r*starts_per_round : (r+1)*starts_per_round]
            if not round_starts:
                break
                
            futures = []
            for i, start in enumerate(round_starts):
                seed = r * 100 + i
                futures.append(executor.submit(_basin_hopping_worker, start, kind, max_iters, seed))
                
            for future in as_completed(futures):
                if _time_left() < 5.0:
                    break
                try:
                    energy, positions = future.result()
                    if energy is not None and energy < best_energy:
                        best_energy = energy
                        best_positions = positions.copy()
                except:
                    pass

    # Final validation
    if not _validate(best_positions):
        # Fallback to a valid random structure
        best_positions = _generate_random_compact(n, seed=999)
        for i in range(len(best_positions)):
            for j in range(i + 1, len(best_positions)):
                d = np.linalg.norm(best_positions[i] - best_positions[j])
                if d < 0.35:
                    direction = best_positions[i] - best_positions[j]
                    if np.linalg.norm(direction) > 1e-10:
                        direction = direction / np.linalg.norm(direction)
                    else:
                        direction = np.array([1.0, 0, 0])
                    best_positions[j] = best_positions[i] + direction * 0.36
                    
    return best_positions
# EVOLVE-BLOCK-END
