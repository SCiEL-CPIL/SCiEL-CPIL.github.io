# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 300.0  # 300s total budget, return before kill


def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)


def _generate_icosahedral(n, rng):
    """Generate an icosahedral-like cluster for polytetrahedral packing."""
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0])
    
    points = [np.array([0.0, 0.0, 0.0])]  # center
    
    # Build shells using icosahedral directions
    directions = list(ico_verts)
    
    # Add edge midpoints (normalized)
    for i in range(12):
        for j in range(i+1, 12):
            d = ico_verts[i] + ico_verts[j]
            d = d / np.linalg.norm(d)
            directions.append(d)
    
    # Add face centers (approximate by summing 3 vertices)
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d = ico_verts[i] + ico_verts[j] + ico_verts[k]
                norm = np.linalg.norm(d)
                if norm > 0.1:
                    directions.append(d / norm)
    
    # Deduplicate directions
    unique_dirs = [directions[0]]
    for d in directions[1:]:
        if all(np.linalg.norm(d - ud) > 0.1 for ud in unique_dirs):
            unique_dirs.append(d)
    directions = unique_dirs
    
    # Build shells
    shell_radii = [1.0, 1.8, 2.6, 3.4, 4.2, 5.0]
    dir_idx = 0
    for shell_r in shell_radii:
        for d in directions:
            if len(points) >= n:
                break
            noise = rng.normal(0, 0.05, 3)
            p = shell_r * d + noise
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        if len(points) >= n:
            break
    
    # If we still need more, add random points
    while len(points) < n:
        p = rng.uniform(-6, 6, 3)
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
    
    return np.array(points[:n])


def _generate_fcc(n, rng):
    """Generate an FCC-like cluster."""
    a = 2.0
    points = []
    max_coord = int((n / 4.0) ** (1.0/3.0) * a) + 3
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for basis in [(0, 0, 0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)]:
                    p = np.array([i*a, j*a, k*a]) + np.array(basis)
                    if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                        points.append(p)
                    if len(points) >= n:
                        break
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    
    while len(points) < n:
        p = rng.uniform(-6, 6, 3)
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
    
    return np.array(points[:n])


def _generate_random(n, rng):
    """Generate a random compact cluster."""
    radius = 1.2 * n ** (1.0/3.0) * 0.7
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
        attempts += 1
    
    # Fallback: fill with slightly perturbed positions
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Nudge to satisfy minimum distance
            for _ in range(100):
                if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                    points.append(p)
                    break
                p = p + rng.normal(0, 0.1, 3)
            else:
                p = np.array([len(points)*0.5, 0, 0])
                points.append(p)
    
    return np.array(points[:n])


def _basin_hopping(positions, potential, rng, time_limit=40.0):
    """Basin Hopping algorithm with local relaxation and adaptive step size."""
    current_energy, current_pos = relax(positions, potential)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    start_time = time.time()
    n_atoms = len(current_pos)
    
    # Adaptive step size based on cluster size
    # Larger clusters need smaller relative steps
    base_step = 0.5 + 0.3 * (38.0 / n_atoms)
    min_step = 0.2
    max_step = 1.2
    step_size = base_step
    consecutive_rejects = 0
    
    while (time.time() - start_time) < time_limit:
        # Perturbation: move a random subset of atoms
        n_move = max(3, n_atoms // 4)
        move_idx = rng.choice(n_atoms, n_move, replace=False)
        
        delta = rng.normal(0, step_size, (n_move, 3))
        perturbed = current_pos.copy()
        perturbed[move_idx] += delta
        
        # Check minimum distance and fix if necessary
        diffs = perturbed[:, np.newaxis, :] - perturbed[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        if np.min(dists) < 0.5:
            # Simple fix: move close pairs apart
            fixed = False
            for _ in range(20):
                if np.min(dists) >= 0.5:
                    fixed = True
                    break
                min_idx = np.unravel_index(np.argmin(dists), dists.shape)
                i_idx, j_idx = min_idx
                if i_idx == j_idx:
                    continue
                diff = perturbed[i_idx] - perturbed[j_idx]
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    diff = rng.normal(0, 0.1, 3)
                    dist = np.linalg.norm(diff)
                push = 0.5 * (diff / dist)
                perturbed[i_idx] += push
                perturbed[j_idx] -= push
                diffs = perturbed[:, np.newaxis, :] - perturbed[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diffs**2, axis=2))
                np.fill_diagonal(dists, np.inf)
            
            if not fixed:
                continue

        new_energy, new_pos = relax(perturbed, potential)
        
        # Acceptance criterion: Metropolis with adaptive temperature
        T = 0.5 + 0.5 * (time_limit - (time.time() - start_time)) / time_limit
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            consecutive_rejects = 0
            # Increase step size slightly on success
            step_size = min(max_step, step_size * 1.05)
        else:
            delta_e = new_energy - current_energy
            if rng.random() < np.exp(-delta_e / T):
                current_energy = new_energy
                current_pos = new_pos
                consecutive_rejects = 0
            else:
                consecutive_rejects += 1
                # Decrease step size after rejections
                if consecutive_rejects > 5:
                    step_size = max(min_step, step_size * 0.95)
                    consecutive_rejects = 0
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos.copy()
            
    return best_energy, best_pos


def _generate_icosahedral_v2(n, rng):
    """Improved icosahedral cluster with better polytetrahedral packing."""
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0])
    
    points = [np.array([0.0, 0.0, 0.0])]  # center
    
    # Generate all unique directions from icosahedral symmetry
    directions = list(ico_verts)
    
    # Add edge midpoints
    for i in range(12):
        for j in range(i+1, 12):
            d = ico_verts[i] + ico_verts[j]
            norm = np.linalg.norm(d)
            if norm > 0.01:
                directions.append(d / norm)
    
    # Add face centers
    ico_faces = [
        [0,1,8],[0,8,4],[0,4,2],[0,2,6],[0,6,1],
        [1,6,9],[1,9,3],[1,3,5],[1,5,0],[2,4,10],[2,10,7],[2,7,6],
        [3,9,11],[3,11,5],[3,5,0],[4,8,10],[4,10,7],[4,7,2],
        [5,11,9],[5,9,1],[5,0,3],[6,7,10]
    ]
    for face in ico_faces:
        d = ico_verts[face[0]] + ico_verts[face[1]] + ico_verts[face[2]]
        norm = np.linalg.norm(d)
        if norm > 0.01:
            directions.append(d / norm)
    
    # Deduplicate
    unique_dirs = [directions[0]]
    for d in directions[1:]:
        if all(np.linalg.norm(d - ud) > 0.05 for ud in unique_dirs):
            unique_dirs.append(d)
    directions = unique_dirs
    
    # Build shells with appropriate radii for polytetrahedral packing
    # Optimal nearest-neighbor distance ~1.9 for Dzugutov
    shell_radii = [1.9, 3.4, 4.8, 6.2, 7.6, 9.0]
    dir_idx = 0
    for shell_r in shell_radii:
        for d in directions:
            if len(points) >= n:
                break
            noise = rng.normal(0, 0.03, 3)
            p = shell_r * d + noise
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        if len(points) >= n:
            break
    
    while len(points) < n:
        p = rng.uniform(-8, 8, 3)
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
    
    return np.array(points[:n])


def construct(instance):
    """Find low-energy structures using Basin Hopping with parallel seeds."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    
    rng = np.random.default_rng(size)
    
    # Time budget: 300s total for 6 instances = 50s per instance
    # Use 46s for search, 4s for safety
    time_per_instance = 46.0
    
    # Generate diverse initial structures
    generators = [
        lambda: _generate_icosahedral_v2(size, rng),
        lambda: _generate_icosahedral(size, rng),
        lambda: _generate_fcc(size, rng),
        lambda: _generate_random(size, rng),
    ]
    
    best_energy = np.inf
    best_positions = None
    
    # Run Basin Hopping from each seed with adaptive time allocation
    # Give more time to the first (best structured) seed
    n_seeds = len(generators)
    # Weighted time: icosahedral_v2 gets more time
    weights = [0.35, 0.25, 0.20, 0.20]
    for i, gen in enumerate(generators):
        if time.time() - _START_TIME > _DEADLINE - 4:
            break
        
        time_for_seed = time_per_instance * weights[i] if i < len(weights) else time_per_instance / n_seeds
        
        try:
            pos = gen()
            if pos.shape[0] == size:
                energy, pos_result = _basin_hopping(pos, potential, rng, time_limit=time_for_seed)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = pos_result
        except Exception:
            pass
    
    # Final relaxation
    if best_positions is not None:
        try:
            _, best_positions = relax(best_positions, potential)
        except Exception:
            pass
    
    if best_positions is None:
        best_positions = _generate_random(size, rng)
        try:
            _, best_positions = relax(best_positions, potential)
        except Exception:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
