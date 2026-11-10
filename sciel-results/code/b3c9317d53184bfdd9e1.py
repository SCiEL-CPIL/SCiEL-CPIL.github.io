"""Dzugutov cluster global minima: polytetrahedral seed + basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
import threading
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global deadline management
_START_TIME = None
_DEADLINE = 315.0  # safety margin before 330s kill


def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)


def _generate_polytetrahedral_seed(n, rng):
    """
    Generate a seed structure with polytetrahedral character.
    Uses Mackay icosahedral shells for better polytetrahedral order.
    """
    phi = (1 + np.sqrt(5)) / 2
    icosahedron_verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1],
        [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    icosahedron_verts /= np.linalg.norm(icosahedron_verts[0])
    
    points = [np.zeros(3)]  # center atom (shell 0)
    
    # Shell 1: 12 atoms at icosahedron vertices
    shell1_scale = 1.0
    for v in icosahedron_verts:
        points.append(v * shell1_scale)
    
    # Shell 2: 42 atoms (30 edge midpoints + 12 face centers)
    if n > 13:
        # Edge midpoints of icosahedron
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                dist = np.linalg.norm(icosahedron_verts[i] - icosahedron_verts[j])
                if 1.3 < dist < 1.7:  # edge length threshold
                    mid = (icosahedron_verts[i] + icosahedron_verts[j]) / 2
                    edges.append(mid)
        
        # Face centers: use cross products of adjacent vertices
        face_centers = []
        for i in range(12):
            for j in range(i + 1, 12):
                for k in range(j + 1, 12):
                    v1, v2, v3 = icosahedron_verts[i], icosahedron_verts[j], icosahedron_verts[k]
                    # Check if these form a face (all pairwise distances similar)
                    d12 = np.linalg.norm(v1 - v2)
                    d23 = np.linalg.norm(v2 - v3)
                    d13 = np.linalg.norm(v1 - v3)
                    if abs(d12 - d23) < 0.1 and abs(d23 - d13) < 0.1:
                        fc = (v1 + v2 + v3) / 3
                        # Check if this face center is new
                        is_new = True
                        for existing in face_centers:
                            if np.linalg.norm(fc - existing) < 0.1:
                                is_new = False
                                break
                        if is_new:
                            face_centers.append(fc)
        
        shell2_scale = 1.55
        for e in edges:
            if len(points) < n:
                points.append(e * shell2_scale)
        
        for fc in face_centers:
            if len(points) < n:
                points.append(fc * shell2_scale)
    
    # Shell 3: Generate additional atoms using icosahedral symmetry directions
    if n > 55:
        # Use more icosahedral directions for shell 3
        # Generate points on icosahedron faces more systematically
        shell3_scale = 2.3
        # Create more points by subdividing the icosahedron
        all_directions = []
        for v in icosahedron_verts:
            all_directions.append(v)
        for i in range(12):
            for j in range(i + 1, 12):
                dist = np.linalg.norm(icosahedron_verts[i] - icosahedron_verts[j])
                if 1.3 < dist < 1.7:
                    mid = (icosahedron_verts[i] + icosahedron_verts[j]) / 2
                    mid /= np.linalg.norm(mid)
                    all_directions.append(mid)
        
        # Add more directions by combining
        for i in range(len(all_directions)):
            for j in range(i + 1, min(i + 5, len(all_directions))):
                combined = all_directions[i] + all_directions[j]
                norm = np.linalg.norm(combined)
                if norm > 0.1:
                    combined /= norm
                    all_directions.append(combined)
        
        # Shuffle and take needed number
        rng.shuffle(all_directions)
        for d in all_directions:
            if len(points) < n:
                points.append(d * shell3_scale)
    
    # If we still need more atoms, add them in a random spherical shell
    while len(points) < n:
        theta = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-1, 1)
        r = np.sqrt(1 - z * z)
        p = np.array([r * np.cos(theta), r * np.sin(theta), z])
        shell_radius = 2.8 + rng.uniform(0, 0.5)
        points.append(p * shell_radius)
    
    # If we have too many, truncate
    points = points[:n]
    
    # Apply small random perturbations to break symmetry and create diversity
    points = np.array(points)
    points += rng.normal(0, 0.06, points.shape)
    
    # Ensure minimum distance constraint
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = np.linalg.norm(points[i] - points[j])
            if d < 0.5:
                diff = points[i] - points[j]
                if np.linalg.norm(diff) < 1e-10:
                    diff = np.array([1, 0, 0])
                diff /= np.linalg.norm(diff)
                points[i] += diff * 0.26
                points[j] -= diff * 0.26
    
    return np.array(points)


def _generate_fcc_seed(n, rng):
    """Generate an FCC-like seed structure."""
    points = []
    # FCC unit cell basis
    basis = np.array([
        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]
    ])
    
    # Generate atoms in a cubic grid
    side = int(np.ceil(n ** (1/3)))
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                for b in basis:
                    if len(points) >= n:
                        break
                    p = np.array([ix, iy, iz]) * 1.05 + b * 1.05
                    points.append(p)
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    
    # If not enough, add more
    while len(points) < n:
        theta = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-1, 1)
        r = np.sqrt(1 - z * z)
        p = np.array([r * np.cos(theta), r * np.sin(theta), z])
        points.append(p * (side * 0.75))
    
    points = np.array(points[:n])
    points += rng.normal(0, 0.04, points.shape)
    return points


def _generate_random_compact(n, rng):
    """Generate a random compact cluster."""
    radius = 1.0 * n ** (1.0 / 3.0) * 0.65
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 150:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.52:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    # If not enough, add more without distance check
    while len(points) < n:
        theta = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-1, 1)
        r = np.sqrt(1 - z * z)
        p = np.array([r * np.cos(theta), r * np.sin(theta), z])
        points.append(p * radius)
    
    return np.array(points[:n])


def _basin_hopping(positions, potential, rng, max_iters=50, step_scale=0.35):
    """
    Perform basin hopping: random displacement followed by local relaxation.
    Returns the best structure found.
    """
    best_positions = positions.copy()
    best_energy, _ = relax(positions, potential)
    
    for i in range(max_iters):
        # Random displacement with adaptive step size (geometric decay)
        progress = i / max(max_iters, 1)
        current_step = step_scale * (0.4 + 0.6 * (1.0 - progress))
        step = current_step * rng.normal(0, 1, positions.shape)
        new_positions = positions + step
        
        # Relax
        energy, relaxed_positions = relax(new_positions, potential)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = relaxed_positions.copy()
            positions = relaxed_positions.copy()
        else:
            # Accept with Metropolis-like criterion for diversity
            delta_e = energy - best_energy
            if delta_e < 30.0:  # avoid overflow
                accept_prob = np.exp(-0.25 * delta_e)
            else:
                accept_prob = 0.0
            if rng.random() < accept_prob:
                positions = relaxed_positions.copy()
            else:
                positions = positions  # keep previous
    
    return best_positions, best_energy


def _relax_and_record(seed, potential, results, lock):
    """Worker function for parallel relaxation."""
    energy, positions = relax(seed, potential)
    with lock:
        results.append((energy, positions))


def _parallel_basin_hopping(seed, potential, rng, max_iters, step_scale, results, lock):
    """Worker function for parallel basin hopping."""
    best_positions, best_energy = _basin_hopping(seed, potential, rng, max_iters, step_scale)
    with lock:
        results.append((best_energy, best_positions))

def construct(instance):
    """
    Find low-energy Dzugutov clusters using polytetrahedral seeds and basin hopping.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n = instance["n"]
    potential = instance["potential"]
    rng = np.random.default_rng(n * 7919)  # deterministic seed per n
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse initial structures
    seeds = []
    seeds.append(_generate_polytetrahedral_seed(n, rng))
    seeds.append(_generate_fcc_seed(n, rng))
    seeds.append(_generate_random_compact(n, rng))
    for _ in range(2):
        seeds.append(_generate_random_compact(n, rng))
    
    # Relax all seeds in parallel
    results = []
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_relax_and_record, seed, potential, results, lock) for seed in seeds]
        for f in concurrent.futures.as_completed(futures):
            f.result()
    
    for energy, positions in results:
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()
    
    # Parallel basin hopping from top seeds
    time_remaining = _get_time_remaining()
    if time_remaining > 60 and best_positions is not None:
        # Sort results by energy and take top 3 for basin hopping
        results_sorted = sorted(results, key=lambda x: x[0])
        bh_seeds = [pos for _, pos in results_sorted[:3]]
        
        # Determine iterations per chain based on time
        time_per_chain = time_remaining / 4  # 4 chains total (3 BH + 1 reserve)
        max_iters_per_chain = max(10, min(60, int(time_per_chain / 2.5)))
        
        bh_results = []
        bh_lock = threading.Lock()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i, seed in enumerate(bh_seeds):
                chain_rng = np.random.default_rng(n * 7919 + i * 1000)
                futures.append(executor.submit(
                    _parallel_basin_hopping, seed, potential, chain_rng, 
                    max_iters_per_chain, 0.35, bh_results, bh_lock
                ))
            for f in concurrent.futures.as_completed(futures):
                f.result()
        
        for energy, positions in bh_results:
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
    
    # Final high-precision polish: gradient-based optimization
    if best_positions is not None and _get_time_remaining() > 20:
        # Try gradient-based refinement
        def _energy_func(pos):
            flat = pos.flatten()
            e, _ = relax(flat.reshape(-1, 3), potential)
            return e
        
        def _gradient_func(pos):
            flat = pos.flatten()
            _, grad = dzugutov_energy_gradient(flat.reshape(-1, 3))
            return grad.flatten()
        
        # Multiple gradient-based refinements from perturbed best
        for trial in range(3):
            if _get_time_remaining() < 15:
                break
            if trial == 0:
                start_pos = best_positions.copy()
            else:
                start_pos = best_positions + 0.03 * rng.normal(0, 1, best_positions.shape)
            
            try:
                res = minimize(
                    _energy_func, start_pos.flatten(), 
                    jac=_gradient_func, method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.success and res.fun < best_energy:
                    best_energy = res.fun
                    best_positions = res.x.reshape(-1, 3)
            except Exception:
                pass
        
        # Final relaxation
        final_energy, final_positions = relax(best_positions, potential)
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_positions
    
    if best_positions is None:
        # Fallback: return a simple linear arrangement
        best_positions = np.array([[i * 1.1, 0, 0] for i in range(n)])
    
    return best_positions
# EVOLVE-BLOCK-END
