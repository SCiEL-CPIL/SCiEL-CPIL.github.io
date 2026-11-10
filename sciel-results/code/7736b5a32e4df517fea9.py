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


def _generate_icosahedral_shell(n, rng, radius=1.9):
    """Generate atoms on an icosahedral shell."""
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0])
    
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Generate directions
    directions = list(ico_verts)
    for i in range(12):
        for j in range(i+1, 12):
            d = ico_verts[i] + ico_verts[j]
            norm = np.linalg.norm(d)
            if norm > 0.01:
                directions.append(d / norm)
                
    # Sort directions by angle or just use them
    # Place atoms on shell
    num_needed = n - 1
    if num_needed <= len(directions):
        # Shuffle directions
        idx = list(range(len(directions)))
        rng.shuffle(idx)
        for i in range(num_needed):
            d = directions[idx[i]]
            p = radius * d
            # Add small noise
            p += rng.normal(0, 0.05, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
            else:
                # Try to find a valid position
                for _ in range(10):
                    p = radius * d + rng.normal(0, 0.1, 3)
                    if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                        points.append(p)
                        break
    else:
        # Need more atoms, add second shell
        for i in range(len(directions)):
            d = directions[i]
            p = radius * d
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        # Add remaining randomly
        while len(points) < n:
            p = rng.uniform(-radius*2, radius*2, 3)
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


def _hybrid_search(positions, potential, rng, time_limit=40.0):
    """Hybrid Langevin-Annealing + Basin Hopping."""
    start_time = time.time()
    pos = positions.copy()
    n_atoms = len(pos)
    
    # Phase 1: Langevin Dynamics (50% of time)
    t_langevin = time_limit * 0.5
    t_bh = time_limit * 0.5
    
    best_energy = np.inf
    best_pos = pos.copy()
    
    # Initial relax
    try:
        E, pos = relax(pos, potential)
        best_energy = E
        best_pos = pos.copy()
    except:
        pass
        
    # Langevin
    dt = 0.005
    gamma = 1.0
    T_start = 1.0
    T_end = 0.2
    
    velocity = np.zeros_like(pos)
    
    while (time.time() - start_time) < t_langevin:
        progress = (time.time() - start_time) / t_langevin
        T = T_start * np.exp(-5 * progress) + T_end
        
        try:
            E, grad = dzugutov_energy_gradient(pos)
        except:
            break
            
        force = -grad
        random_force = rng.normal(0, np.sqrt(2 * T / dt), pos.shape)
        
        v = velocity * np.exp(-gamma * dt)
        v += (force * dt)
        v += rng.normal(0, np.sqrt(2 * T * gamma * dt), pos.shape)
        
        pos += v * dt
        velocity = v
        
        # Check min distance
        diffs = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 0.5:
            for _ in range(5):
                if np.min(dists) >= 0.5: break
                min_idx = np.unravel_index(np.argmin(dists), dists.shape)
                i_idx, j_idx = min_idx
                diff = pos[i_idx] - pos[j_idx]
                dist = np.linalg.norm(diff)
                if dist < 1e-6: continue
                push = 0.5 * (diff / dist)
                pos[i_idx] += push
                pos[j_idx] -= push
                diffs = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diffs**2, axis=2))
                np.fill_diagonal(dists, np.inf)
                
        # Periodic relax
        if (time.time() - start_time) % 0.05 < dt:
            try:
                E_new, pos_relaxed = relax(pos, potential)
                if E_new < best_energy:
                    best_energy = E_new
                    best_pos = pos_relaxed.copy()
                pos = pos_relaxed
                velocity = np.zeros_like(pos)
            except:
                pass
                
    # Phase 2: Basin Hopping (50% of time)
    # Use the best position found so far
    current_pos = best_pos.copy()
    current_energy = best_energy
    
    base_step = 0.5
    step_size = base_step
    
    while (time.time() - start_time) < time_limit:
        n_move = max(3, n_atoms // 4)
        move_idx = rng.choice(n_atoms, n_move, replace=False)
        delta = rng.normal(0, step_size, (n_move, 3))
        perturbed = current_pos.copy()
        perturbed[move_idx] += delta
        
        diffs = perturbed[:, np.newaxis, :] - perturbed[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        if np.min(dists) < 0.5:
            for _ in range(10):
                if np.min(dists) >= 0.5: break
                min_idx = np.unravel_index(np.argmin(dists), dists.shape)
                i_idx, j_idx = min_idx
                diff = perturbed[i_idx] - perturbed[j_idx]
                dist = np.linalg.norm(diff)
                if dist < 1e-6: continue
                push = 0.5 * (diff / dist)
                perturbed[i_idx] += push
                perturbed[j_idx] -= push
                diffs = perturbed[:, np.newaxis, :] - perturbed[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diffs**2, axis=2))
                np.fill_diagonal(dists, np.inf)
            else:
                continue
                
        try:
            new_energy, new_pos = relax(perturbed, potential)
        except:
            continue
            
        T = 0.5
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            step_size = min(1.0, step_size * 1.05)
        else:
            delta_e = new_energy - current_energy
            if rng.random() < np.exp(-delta_e / T):
                current_energy = new_energy
                current_pos = new_pos
            else:
                step_size = max(0.2, step_size * 0.95)
                
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
        lambda: _generate_icosahedral_shell(size, rng),
        lambda: _generate_fcc(size, rng),
        lambda: _generate_random(size, rng),
    ]
    
    best_energy = np.inf
    best_positions = None
    
    # Run Hybrid Search from each seed with adaptive time allocation
    n_seeds = len(generators)
    weights = [0.35, 0.30, 0.15, 0.20]
    for i, gen in enumerate(generators):
        if time.time() - _START_TIME > _DEADLINE - 4:
            break
        
        time_for_seed = time_per_instance * weights[i] if i < len(weights) else time_per_instance / n_seeds
        
        try:
            pos = gen()
            if pos.shape[0] == size:
                energy, pos_result = _hybrid_search(pos, potential, rng, time_limit=time_for_seed)
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
