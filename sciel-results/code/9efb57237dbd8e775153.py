"""Morse cluster global minima: Adaptive Basin Hopping with diverse seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 290.0  # 10s safety margin for 300s limit
_INSTANCE_COUNT = 6
_INSTANCE_INDEX = [0]

def _generate_fcc_lattice(n, spacing=1.0):
    """Generate n points from an FCC lattice centered at origin."""
    r_max = spacing * (n ** (1.0/3.0) + 2)
    limit = int(r_max / spacing) + 2
    
    candidates = []
    for i in range(-2*limit, 2*limit + 1):
        for j in range(-2*limit, 2*limit + 1):
            for k in range(-2*limit, 2*limit + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * (spacing / 2.0)
                    candidates.append(p)
    
    if not candidates:
        return np.random.uniform(-r_max, r_max, (n, 3))
        
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    sorted_idx = np.argsort(dists)
    selected = candidates[sorted_idx[:n]]
    
    selected -= np.mean(selected, axis=0)
    return selected

def _generate_bcc_lattice(n, spacing=1.0):
    """Generate n points from a BCC lattice centered at origin."""
    r_max = spacing * (n ** (1.0/3.0) + 2)
    limit = int(r_max / spacing) + 2
    
    candidates = []
    for i in range(-2*limit, 2*limit + 1):
        for j in range(-2*limit, 2*limit + 1):
            for k in range(-2*limit, 2*limit + 1):
                if (i % 2) == (j % 2) and (j % 2) == (k % 2):
                    p = np.array([i, j, k]) * (spacing / 2.0)
                    candidates.append(p)
    
    if not candidates:
        return np.random.uniform(-r_max, r_max, (n, 3))
        
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    sorted_idx = np.argsort(dists)
    selected = candidates[sorted_idx[:n]]
    
    selected -= np.mean(selected, axis=0)
    return selected

def _generate_hcp_lattice(n, spacing=1.0):
    """Generate n points from an HCP-like lattice centered at origin."""
    r_max = spacing * (n ** (1.0/3.0) + 2)
    limit = int(r_max / spacing) + 2
    
    candidates = []
    a = spacing
    c = spacing * np.sqrt(8.0/3.0)
    
    for i in range(-limit, limit + 1):
        for j in range(-limit, limit + 1):
            for k in range(-limit, limit + 1):
                # HCP: layers at z = k*c/2, alternating AB stacking
                z = k * c / 2.0
                if k % 2 == 0:
                    x = i * a + j * a * 0.5
                    y = j * a * np.sqrt(3.0) / 2.0
                else:
                    x = i * a + j * a * 0.5 + a / 3.0
                    y = j * a * np.sqrt(3.0) / 2.0 + a * np.sqrt(3.0) / 6.0
                p = np.array([x, y, z])
                candidates.append(p)
    
    if not candidates:
        return np.random.uniform(-r_max, r_max, (n, 3))
        
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    sorted_idx = np.argsort(dists)
    selected = candidates[sorted_idx[:n]]
    
    selected -= np.mean(selected, axis=0)
    return selected

def _generate_random(n, spacing=1.0, seed=None):
    """Generate n random points."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = rng.uniform(-radius, radius, (n, 3))
    return points

def _enforce_min_distance(positions, min_dist=0.5):
    """Ensure all pairwise distances are at least min_dist."""
    n = len(positions)
    if n < 2:
        return positions.copy()
        
    for _ in range(30):
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        
        mask = (dist < min_dist) & (~np.eye(n, dtype=bool))
        if not np.any(mask):
            break
            
        i_idx, j_idx = np.where(mask)
        
        moved = False
        for i, j in zip(i_idx, j_idx):
            if i >= j: continue
            d = dist[i, j]
            if d < 1e-8:
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            else:
                direction = (positions[i] - positions[j]) / d
                
            push = (min_dist - d) / 2.0 + 0.01
            positions[i] += push * direction
            positions[j] -= push * direction
            moved = True
            
        if not moved:
            break
            
    return positions

def construct(instance):
    """Construct low-energy Morse cluster configuration using adaptive basin hopping."""
    _INSTANCE_INDEX[0] += 1
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    # Time management
    remaining_total = _HARD_DEADLINE - time.time()
    instances_left = _INSTANCE_COUNT - _INSTANCE_INDEX[0] + 1
    time_budget = max(5.0, min(45.0, remaining_total / instances_left))
    
    start_time = time.time()
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse seed structures
    seeds = []
    
    # 1. FCC Lattice
    try:
        seeds.append(_generate_fcc_lattice(size, spacing=1.0))
    except Exception:
        pass
        
    # 2. BCC Lattice
    try:
        seeds.append(_generate_bcc_lattice(size, spacing=1.0))
    except Exception:
        pass
        
    # 3. HCP Lattice
    try:
        seeds.append(_generate_hcp_lattice(size, spacing=1.0))
    except Exception:
        pass

    # 4. Multiple random structures with different seeds
    for s in [42, size, size * 7, 123, 456, 789]:
        try:
            seeds.append(_generate_random(size, spacing=1.0, seed=s))
        except Exception:
            pass

    if not seeds:
        seeds.append(_generate_random(size, 1.0, 123))

    # Phase 1: Relax all seeds to find initial good configurations
    for seed_idx, seed_pos in enumerate(seeds):
        if time.time() - start_time > time_budget * 0.3:
            break
            
        jittered = seed_pos + np.random.randn(*seed_pos.shape) * 0.05
        jittered = _enforce_min_distance(jittered, 0.5)
        
        try:
            energy, positions = relax(jittered, potential, rho)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue

    if best_positions is None:
        best_positions = seeds[0].copy()
        try:
            best_energy, best_positions = relax(best_positions, potential, rho)
        except Exception:
            best_energy = float('inf')

    # Phase 2: Adaptive Basin Hopping
    time_elapsed = time.time() - start_time
    remaining_time = time_budget - time_elapsed
    
    if remaining_time > 3.0 and best_positions is not None:
        # Measure average relax time
        t0 = time.time()
        try:
            test_kick = best_positions + np.random.randn(*best_positions.shape) * 0.3
            test_kick = _enforce_min_distance(test_kick, 0.5)
            relax(test_kick, potential, rho)
            avg_relax_time = time.time() - t0
        except Exception:
            avg_relax_time = 0.5
            
        # Estimate max number of iterations
        if avg_relax_time < 0.05:
            max_total_kicks = int(remaining_time * 0.8 / (avg_relax_time + 0.02))
        else:
            max_total_kicks = int(remaining_time * 0.8 / (avg_relax_time * 1.2))
            
        max_total_kicks = max(10, min(max_total_kicks, 500))
        
        # Number of restarts
        n_restarts = min(6, max(2, max_total_kicks // 15))
        kicks_per_restart = max(5, max_total_kicks // n_restarts)
        
        for restart in range(n_restarts):
            if time.time() - start_time > time_budget - 2.0:
                break
            
            # For each restart, start from a different seed
            if restart < len(seeds):
                current_pos = seeds[restart].copy()
            else:
                current_pos = best_positions.copy()
            
            # Jitter and relax
            current_pos = current_pos + np.random.randn(*current_pos.shape) * 0.1
            current_pos = _enforce_min_distance(current_pos, 0.5)
            
            try:
                current_energy, current_pos = relax(current_pos, potential, rho)
            except Exception:
                continue
            
            # Adaptive Basin Hopping Loop
            kick_scale = 0.3
            accept_count = 0
            total_count = 0
            window_size = 10
            
            for kick_iter in range(kicks_per_restart):
                if time.time() - start_time > time_budget - 1.5:
                    break
                
                kicked = current_pos + np.random.randn(*current_pos.shape) * kick_scale
                kicked = _enforce_min_distance(kicked, 0.5)
                
                try:
                    new_energy, new_pos = relax(kicked, potential, rho)
                    total_count += 1
                    
                    # Accept if better
                    if new_energy < current_energy:
                        current_energy = new_energy
                        current_pos = new_pos
                        accept_count += 1
                        
                        # Update global best
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_positions = current_pos.copy()
                    else:
                        # Occasionally accept worse to escape local minima
                        # Probability decreases as we go deeper
                        temp = 0.1 * (1.0 + kick_iter / max(1, kicks_per_restart))
                        if np.random.rand() < np.exp(-(new_energy - current_energy) / temp):
                            current_energy = new_energy
                            current_pos = new_pos
                            accept_count += 1
                            
                except Exception:
                    continue
                
                # Adaptive step size adjustment
                if total_count >= window_size:
                    accept_ratio = accept_count / total_count
                    if accept_ratio < 0.3:
                        kick_scale *= 1.2  # Increase step if stuck
                    elif accept_ratio > 0.7:
                        kick_scale *= 0.8  # Decrease step if refining
                    kick_scale = np.clip(kick_scale, 0.05, 1.0)
                    accept_count = 0
                    total_count = 0

    # Final polish
    if best_positions is not None:
        try:
            final_energy, final_pos = relax(best_positions, potential, rho)
            if final_energy < best_energy:
                best_positions = final_pos
                best_energy = final_energy
        except Exception:
            pass

    # Emergency time check
    if time.time() > _HARD_DEADLINE - 5.0:
        if best_positions is None:
            best_positions = _generate_random(size, 1.0, 999)
        return _enforce_min_distance(best_positions, 0.5)

    best_positions = _enforce_min_distance(best_positions, 0.5)
    return best_positions
# EVOLVE-BLOCK-END
