"""
Rewritten program for Dzugutov cluster global minima.
Focuses on icosahedral/polytetrahedral structures with diverse initial conditions,
relaxed in parallel with time management, basin hopping, and a final polish pass.
"""
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global deadline tracking
_START_TIME = None
_DEADLINE_MARGIN = 8.0  # seconds before the 330s limit to stop

def _get_elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME

def _remaining_time():
    return 330.0 - _DEADLINE_MARGIN - _get_elapsed()

def _generate_fcc_chunk(n, rng):
    """Generate an FCC lattice chunk roughly spherical."""
    R = (3.0 * n / (4.0 * np.pi) * 10.0) ** (1.0 / 3.0) * 1.15
    a = 2.1  # Lattice constant approx equilibrium distance
    
    points = []
    max_idx = int(R / a) + 2
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                for dx, dy, dz in [(0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    if np.linalg.norm(p) <= R:
                        points.append(p)
    
    points = np.array(points)
    if len(points) == 0:
        return None
    
    if len(points) > n:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n]
        points = points[idx]
    elif len(points) < n:
        while len(points) < n:
            p = rng.uniform(-R, R, 3)
            if np.linalg.norm(p) <= R:
                if len(points) > 0:
                    d = np.linalg.norm(points - p, axis=1)
                    if np.min(d) >= 0.5:
                        points = np.vstack([points, [p]])
                else:
                    points = np.array([[p]])
    
    points -= np.mean(points, axis=0)
    return points

def _generate_random_cluster(n, rng):
    """Generate a random compact cluster."""
    spacing = 1.0
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    max_attempts = 5000 * n
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(points) == 0 or np.min(np.linalg.norm(p - points, axis=1)) >= 0.5:
                points.append(p)
        attempts += 1
    
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    return points

def _generate_mackay_icosahedron(n, rng, seed_offset=0):
    """Generate a Mackay icosahedral cluster with proper shell structure."""
    phi = (1 + np.sqrt(5)) / 2
    r1 = 2.0  # First shell radius
    
    rng_local = np.random.default_rng(seed_offset + 42)
    
    # Center atom
    points = [np.array([0.0, 0.0, 0.0])]
    
    # First shell: 12 icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * r1
    points.extend(verts)
    
    # Second shell: 42 positions
    r2 = r1 * 1.5
    
    second_shell = []
    attempts = 0
    max_attempts = 8000
    while len(second_shell) < 42 and attempts < max_attempts:
        u = rng_local.uniform(-1, 1)
        v = rng_local.uniform(0, 2 * np.pi)
        p = np.array([
            np.sqrt(1 - u**2) * np.cos(v),
            np.sqrt(1 - u**2) * np.sin(v),
            u
        ]) * r2
        all_pts = np.array(points + second_shell)
        if len(all_pts) == 0 or np.min(np.linalg.norm(p - all_pts, axis=1)) >= 0.5:
            second_shell.append(p)
        attempts += 1
    
    points.extend(second_shell)
    
    # Third shell and beyond
    current_shell = 2
    while len(points) < n:
        current_shell += 1
        r_shell = r1 * (current_shell * 0.5 + 0.5)
        num_to_add = min(n - len(points), 42)
        
        shell_points = []
        attempts = 0
        max_attempts = 15000
        while len(shell_points) < num_to_add and attempts < max_attempts:
            u = rng_local.uniform(-1, 1)
            v = rng_local.uniform(0, 2 * np.pi)
            p = np.array([
                np.sqrt(1 - u**2) * np.cos(v),
                np.sqrt(1 - u**2) * np.sin(v),
                u
            ]) * r_shell
            all_pts = np.array(points + shell_points)
            if len(all_pts) == 0 or np.min(np.linalg.norm(p - all_pts, axis=1)) >= 0.5:
                shell_points.append(p)
            attempts += 1
        
        points.extend(shell_points)
    
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points

def _generate_deformed_icosahedron(n, rng, seed_offset=0):
    """Generate a slightly deformed icosahedral cluster for diversity."""
    phi = (1 + np.sqrt(5)) / 2
    r1 = 2.0
    
    rng_local = np.random.default_rng(seed_offset + 99)
    
    points = [np.array([0.0, 0.0, 0.0])]
    
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * r1
    
    # Add small random deformation
    deformation = rng_local.normal(0, 0.15, (12, 3))
    verts = verts + deformation
    
    points.extend(verts)
    
    r2 = r1 * 1.5
    second_shell = []
    attempts = 0
    max_attempts = 8000
    while len(second_shell) < 42 and attempts < max_attempts:
        u = rng_local.uniform(-1, 1)
        v = rng_local.uniform(0, 2 * np.pi)
        p = np.array([
            np.sqrt(1 - u**2) * np.cos(v),
            np.sqrt(1 - u**2) * np.sin(v),
            u
        ]) * r2
        all_pts = np.array(points + second_shell)
        if len(all_pts) == 0 or np.min(np.linalg.norm(p - all_pts, axis=1)) >= 0.5:
            second_shell.append(p)
        attempts += 1
    
    points.extend(second_shell)
    
    current_shell = 2
    while len(points) < n:
        current_shell += 1
        r_shell = r1 * (current_shell * 0.5 + 0.5)
        num_to_add = min(n - len(points), 42)
        
        shell_points = []
        attempts = 0
        max_attempts = 15000
        while len(shell_points) < num_to_add and attempts < max_attempts:
            u = rng_local.uniform(-1, 1)
            v = rng_local.uniform(0, 2 * np.pi)
            p = np.array([
                np.sqrt(1 - u**2) * np.cos(v),
                np.sqrt(1 - u**2) * np.sin(v),
                u
            ]) * r_shell
            all_pts = np.array(points + shell_points)
            if len(all_pts) == 0 or np.min(np.linalg.norm(p - all_pts, axis=1)) >= 0.5:
                shell_points.append(p)
            attempts += 1
        
        points.extend(shell_points)
    
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points

def _relax_worker(args):
    """Worker function for parallel relaxation."""
    positions, potential = args
    try:
        energy, relaxed_positions = relax(positions, potential)
        return energy, relaxed_positions
    except Exception:
        return float('inf'), positions

def _perturb(positions, rng, magnitude=0.3):
    """Perturb positions for basin hopping."""
    n = len(positions)
    num_move = max(1, n // 5)
    idx = rng.choice(n, size=num_move, replace=False)
    new_positions = positions.copy()
    new_positions[idx] += rng.normal(0, magnitude, (num_move, 3))
    return new_positions

def construct(instance):
    """Best of multiple diverse starting clusters, each relaxed locally,
    with basin hopping and a final polish pass."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    
    # Check deadline
    if _remaining_time() < 5.0:
        rng = np.random.default_rng(size)
        radius = 1.0 * size ** (1.0 / 3.0) * 0.8
        points = rng.uniform(-radius, radius, (size, 3))
        return points
    
    rng = np.random.default_rng(size)
    
    initial_clusters = []
    
    # 1. Mackay icosahedral (primary - Dzugutov favors polytetrahedral)
    ico = _generate_mackay_icosahedron(size, rng, seed_offset=size)
    initial_clusters.append(ico)
    
    # 2. Deformed icosahedral for diversity
    ico_def = _generate_deformed_icosahedron(size, rng, seed_offset=size + 50)
    initial_clusters.append(ico_def)
    
    # 3. FCC-like cluster
    fcc = _generate_fcc_chunk(size, rng)
    if fcc is not None:
        initial_clusters.append(fcc)
    
    # 4. Random clusters (diverse)
    for i in range(3):
        random_cluster = _generate_random_cluster(size, rng)
        initial_clusters.append(random_cluster)
    
    # Relax all clusters in parallel
    args_list = [(cluster, potential) for cluster in initial_clusters]
    
    best_energy = float('inf')
    best_positions = initial_clusters[0]
    
    # Use a thread pool for parallel relaxation
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_relax_worker, args) for args in args_list]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                energy, positions = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass
            
            # Check deadline
            if _remaining_time() < 5.0:
                break
    
    # Basin hopping: perturb best positions and relax again
    if _remaining_time() > 10:
        bh_rng = np.random.default_rng(size + 1000)
        max_bh_steps = min(8, int(_remaining_time() / 4))
        
        for step in range(max_bh_steps):
            if _remaining_time() < 8.0:
                break
            
            # Adaptive magnitude: start larger, decrease
            magnitude = 0.3 * (1.0 - 0.1 * step)
            perturbed = _perturb(best_positions, bh_rng, magnitude=magnitude)
            
            try:
                energy, positions = relax(perturbed, potential)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass
    
    # Final polish: one more relaxation with the best positions
    if _remaining_time() > 3.0:
        try:
            energy, positions = relax(best_positions, potential)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            pass
    
    return best_positions
