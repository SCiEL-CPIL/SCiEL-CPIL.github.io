"""Force-directed circle packing in unit square with simulated annealing."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time


def _pairwise_dists(pos):
    """Compute pairwise distances between all circles."""
    dx = pos[:, 0:1] - pos[:, 0:1].T
    dy = pos[:, 1:2] - pos[:, 1:2].T
    return np.sqrt(dx * dx + dy * dy)


def _compute_energy(pos, radii, hardness):
    """
    Compute total energy: repulsion between overlapping circles + wall repulsion.
    Returns (energy, overlap_count, max_wall_violation).
    """
    n = len(radii)
    D = _pairwise_dists(pos)
    min_dist = radii[:, None] + radii[None, :]
    # Overlap amount (positive means overlap)
    overlap = np.maximum(0, min_dist - D)
    np.fill_diagonal(overlap, 0)
    repulsion_energy = 0.5 * np.sum(overlap ** 2) * hardness
    overlap_count = np.sum(overlap > 1e-10)
    
    # Wall violations
    wall_overlap = np.maximum(0, radii - pos[:, 0]) + np.maximum(0, radii - (1.0 - pos[:, 0]))
    wall_overlap += np.maximum(0, radii - pos[:, 1]) + np.maximum(0, radii - (1.0 - pos[:, 1]))
    wall_energy = 0.5 * np.sum(wall_overlap ** 2) * hardness
    max_wall_violation = np.max(wall_overlap)
    
    return repulsion_energy + wall_energy, overlap_count, max_wall_violation


def _grow_radii(pos, target_sum, max_iter=50000, deadline=None):
    """
    Grow all radii uniformly while maintaining no overlaps using force-directed relaxation.
    Returns final radii array and sum.
    """
    n = len(pos)
    # Start with small radii
    radii = np.full(n, 0.01)
    best_sum = np.sum(radii)
    best_radii = radii.copy()
    best_pos = pos.copy()
    
    # Adaptive step size for radius growth
    step = 0.005
    failed_count = 0
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try to grow radii
        radii_new = radii + step
        pos_new = pos.copy()
        
        # Force-directed relaxation to resolve overlaps
        converged = False
        for relax_iter in range(200):
            if deadline is not None and time.time() > deadline:
                break
            
            D = _pairwise_dists(pos_new)
            min_dist = radii_new[:, None] + radii_new[None, :]
            
            # Compute forces
            forces = np.zeros_like(pos_new)
            for i in range(n):
                for j in range(i + 1, n):
                    diff = pos_new[j] - pos_new[i]
                    d = np.linalg.norm(diff)
                    if d < 1e-12:
                        diff = np.array([1e-6, 1e-6])
                        d = np.linalg.norm(diff)
                    overlap = min_dist[i, j] - d
                    if overlap > 0:
                        force_mag = overlap * 0.5
                        forces[i] -= force_mag * diff / d
                        forces[j] += force_mag * diff / d
            
            # Wall forces
            for i in range(n):
                r = radii_new[i]
                if pos_new[i, 0] < r:
                    forces[i, 0] += (r - pos_new[i, 0]) * 2.0
                if pos_new[i, 0] > 1.0 - r:
                    forces[i, 0] -= (pos_new[i, 0] - (1.0 - r)) * 2.0
                if pos_new[i, 1] < r:
                    forces[i, 1] += (r - pos_new[i, 1]) * 2.0
                if pos_new[i, 1] > 1.0 - r:
                    forces[i, 1] -= (pos_new[i, 1] - (1.0 - r)) * 2.0
            
            # Update positions
            pos_new += 0.5 * forces
            
            # Clamp to valid range
            for i in range(n):
                r = radii_new[i]
                pos_new[i, 0] = np.clip(pos_new[i, 0], r, 1.0 - r)
                pos_new[i, 1] = np.clip(pos_new[i, 1], r, 1.0 - r)
            
            # Check convergence
            if np.max(np.abs(forces)) < 1e-6:
                converged = True
                break
        
        # Check if configuration is valid
        D = _pairwise_dists(pos_new)
        min_dist = radii_new[:, None] + radii_new[None, :]
        overlaps = np.maximum(0, min_dist - D)
        np.fill_diagonal(overlaps, 0)
        wall_viol = np.max(
            np.maximum(0, radii_new - pos_new[:, 0]) +
            np.maximum(0, radii_new - (1.0 - pos_new[:, 0])) +
            np.maximum(0, radii_new - pos_new[:, 1]) +
            np.maximum(0, radii_new - (1.0 - pos_new[:, 1]))
        )
        
        if np.max(overlaps) < 1e-6 and wall_viol < 1e-6:
            # Success: update best
            current_sum = np.sum(radii_new)
            if current_sum > best_sum:
                best_sum = current_sum
                best_radii = radii_new.copy()
                best_pos = pos_new.copy()
            radii = radii_new
            pos = pos_new
            # Increase step slightly
            step = min(step * 1.02, 0.01)
            failed_count = 0
        else:
            # Failed: reduce step
            failed_count += 1
            step = max(step * 0.7, 0.001)
            if failed_count > 50:
                break
        
        # Check if we've reached target
        if np.sum(radii) >= target_sum * 0.999:
            break
    
    return best_radii, best_sum, best_pos


def _simulated_annealing_init(n, seed=42):
    """Generate initial random configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.rand(n, 2) * 0.8 + 0.1
    return pos


def _anneal(pos, radii, T_start=1.0, T_end=0.001, n_iter=10000, deadline=None, seed=42):
    """
    Simulated annealing to optimize packing.
    """
    n = len(radii)
    rng = np.random.RandomState(seed)
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Compute initial energy
    energy, _, _ = _compute_energy(pos, radii, hardness=10.0)
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Temperature schedule
        t = it / max(n_iter - 1, 1)
        T = T_start * (T_end / T_start) ** t
        
        # Propose perturbation
        pos_new = pos.copy()
        # Move a random subset of circles
        k = rng.randint(1, max(2, n // 4) + 1)
        idx = rng.choice(n, size=k, replace=False)
        perturb_scale = 0.05 * (1.0 - t * 0.5)
        pos_new[idx] += rng.randn(k, 2) * perturb_scale
        
        # Clamp to boundaries
        for i in range(n):
            r = radii[i]
            pos_new[i, 0] = np.clip(pos_new[i, 0], r, 1.0 - r)
            pos_new[i, 1] = np.clip(pos_new[i, 1], r, 1.0 - r)
        
        # Compute new energy
        energy_new, overlap_new, wall_new = _compute_energy(pos_new, radii, hardness=10.0)
        
        # Accept if better or with probability exp(-delta_E / T)
        delta_E = energy_new - energy
        if delta_E < 0 or rng.random() < np.exp(-delta_E / max(T, 1e-10)):
            pos = pos_new
            energy = energy_new
        
        # Try to grow radii slightly
        if it % 10 == 0 and overlap_new < 5 and wall_new < 0.01:
            radii_try = radii * (1.0 + 0.002)
            # Quick check if still valid
            D = _pairwise_dists(pos)
            min_dist = radii_try[:, None] + radii_try[None, :]
            overlaps = np.maximum(0, min_dist - D)
            np.fill_diagonal(overlaps, 0)
            if np.max(overlaps) < 1e-4:
                radii = radii_try
                current_score = np.sum(radii)
                if current_score > best_score:
                    best_score = current_score
                    best_pos = pos.copy()
                    best_radii = radii.copy()
    
    return best_pos, best_radii, best_score


def construct():
    """
    Pack 26 circles in unit square, maximizing sum of radii.
    Returns array of shape (26, 3) with rows (x, y, radius).
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    best_score = 0.0
    best_result = None
    
    # Multiple restarts with different initial configurations
    n_restarts = 5
    rng = np.random.RandomState(123)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # Initial configuration
        if restart == 0:
            # Grid-like initial placement
            side = int(np.ceil(np.sqrt(N)))
            pos = np.array([
                [(i + 0.5) / side, (j + 0.5) / side]
                for i in range(side) for j in range(side)
            ][:N])
        else:
            # Random initial placement
            pos = rng.rand(N, 2) * 0.6 + 0.2
        
        # Phase 1: Grow radii from small to large
        radii, sum_r, pos_grown = _grow_radii(pos, target_sum=1.0, max_iter=20000, deadline=deadline)
        
        if time.time() > deadline - 10.0:
            if sum_r > best_score:
                best_score = sum_r
                best_result = np.column_stack([pos_grown, radii])
            continue
        
        # Phase 2: Simulated annealing refinement
        pos_annealed, radii_annealed, score_annealed = _anneal(
            pos_grown, radii, T_start=0.5, T_end=0.01, 
            n_iter=5000, deadline=deadline, seed=1000 + restart
        )
        
        if score_annealed > best_score:
            best_score = score_annealed
            best_result = np.column_stack([pos_annealed, radii_annealed])
        
        # Phase 3: Final radius growth on annealed configuration
        if time.time() > deadline - 5.0:
            continue
        
        radii_final, sum_final, pos_final = _grow_radii(
            pos_annealed, target_sum=best_score * 1.1, max_iter=10000, deadline=deadline
        )
        
        if sum_final > best_score:
            best_score = sum_final
            best_result = np.column_stack([pos_final, radii_final])
    
    # Fallback to seed if no valid result
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    # Final validation and cleanup
    pos = best_result[:, 0:2]
    radii = best_result[:, 2]
    
    # Ensure all circles are fully inside [0,1]^2
    for i in range(N):
        r = radii[i]
        pos[i, 0] = np.clip(pos[i, 0], r, 1.0 - r)
        pos[i, 1] = np.clip(pos[i, 1], r, 1.0 - r)
    
    # Final overlap resolution: shrink if needed
    D = _pairwise_dists(pos)
    for i in range(N):
        for j in range(i + 1, N):
            d = D[i, j]
            min_d = radii[i] + radii[j]
            if d < min_d:
                # Shrink both circles to resolve overlap
                excess = min_d - d
                radii[i] -= excess / 2
                radii[j] -= excess / 2
    
    # Ensure radii are positive and circles fit in square
    for i in range(N):
        radii[i] = max(radii[i], 0.001)
        pos[i, 0] = np.clip(pos[i, 0], radii[i], 1.0 - radii[i])
        pos[i, 1] = np.clip(pos[i, 1], radii[i], 1.0 - radii[i])
    
    result = np.column_stack([pos, radii])
    return result
# EVOLVE-BLOCK-END
