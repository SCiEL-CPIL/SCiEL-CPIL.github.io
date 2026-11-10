"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time


def _score(positions):
    """Compute the largest common radius for a set of positions in [-0.5, 0.5]^3."""
    n = positions.shape[0]
    if n == 0:
        return 0.0
    # Wall distances
    wall_dists = np.min(np.min(0.5 - np.abs(positions), axis=1), axis=0)
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(wall_dists, min_pair_dist / 2.0)


def _initial_grid():
    """Generate initial positions on a grid."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _force_directed_relax(positions, max_iter=5000, dt=0.001, deadline=None):
    """
    Force-directed relaxation to maximize minimum distance.
    Uses repulsive forces between spheres and walls.
    """
    positions = positions.copy()
    n = positions.shape[0]
    best_pos = positions.copy()
    best_score = _score(positions)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(positions)
        
        # Pairwise repulsion
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1e-10)
        
        # Force magnitude: 1/d^2 (repulsive)
        # Only consider pairs closer than some cutoff for efficiency
        mask = dists < 0.5
        for i in range(n):
            for j in range(i + 1, n):
                if mask[i, j]:
                    d = dists[i, j]
                    if d > 1e-10:
                        f = 1.0 / (d * d)
                        direction = diff[i, j] / d
                        forces[i] += f * direction
                        forces[j] -= f * direction
        
        # Wall repulsion
        for i in range(n):
            for k in range(3):
                # Distance to each wall
                d_neg = positions[i, k] + 0.5  # distance to -0.5 wall
                d_pos = 0.5 - positions[i, k]  # distance to +0.5 wall
                if d_neg < 0.2:
                    forces[i, k] += 5.0 / (d_neg * d_neg + 1e-10)
                if d_pos < 0.2:
                    forces[i, k] -= 5.0 / (d_pos * d_pos + 1e-10)
        
        # Update positions
        positions += dt * forces
        
        # Clamp to cube
        positions = np.clip(positions, -0.4999, 0.4999)
        
        # Compute score
        current_score = _score(positions)
        if current_score > best_score:
            best_score = current_score
            best_pos = positions.copy()
        
        # Decay step size
        dt *= 0.999
    
    return best_pos, best_score


def _pairwise_hill_climb(positions, max_iter=10000, deadline=None):
    """
    Hill climbing: move one sphere at a time to improve the minimum distance.
    """
    positions = positions.copy()
    n = positions.shape[0]
    best_score = _score(positions)
    best_pos = positions.copy()
    
    step_size = 0.01
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            for axis in range(3):
                for direction in [-1, 1]:
                    new_pos = positions.copy()
                    new_pos[i, axis] += direction * step_size
                    # Clamp
                    new_pos[i, axis] = np.clip(new_pos[i, axis], -0.4999, 0.4999)
                    
                    new_score = _score(new_pos)
                    if new_score > best_score + 1e-12:
                        positions = new_pos
                        best_score = new_score
                        best_pos = positions.copy()
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try smaller step
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return best_pos, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with grid
    positions = _initial_grid()
    best_pos = positions.copy()
    best_score = _score(positions)
    
    # Force-directed relaxation
    if time.time() < deadline:
        pos, score = _force_directed_relax(positions, max_iter=3000, dt=0.002, deadline=deadline)
        if score > best_score:
            best_pos = pos
            best_score = score
    
    # Hill climbing refinement
    if time.time() < deadline:
        pos, score = _pairwise_hill_climb(best_pos, max_iter=5000, deadline=deadline)
        if score > best_score:
            best_pos = pos
            best_score = score
    
    # Final check
    final_score = _score(best_pos)
    if final_score < best_score:
        # Something went wrong, use the original
        best_pos = positions
    
    return best_pos
# EVOLVE-BLOCK-END
