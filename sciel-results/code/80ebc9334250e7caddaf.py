import numpy as np
import time

N = 50

def _calculate_radius(pos):
    if pos.shape != (N, 2) or not np.all(np.isfinite(pos)):
        return 0.0
    if np.any(np.abs(pos) > 0.5 + 1e-9):
        return 0.0
        
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    wall_dists = np.min(0.5 - np.abs(pos), axis=1)
    min_wall_dist = np.min(wall_dists)
    
    radius = min(min_pair_dist / 2.0, min_wall_dist)
    return radius

def _init_grid(n):
    # 7x7 grid
    ticks = (np.arange(7) + 0.5) / 7.0 - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    # Add center
    pos = np.vstack([grid, [[0.0, 0.0]]])
    # Perturb
    np.random.seed(42)
    pos += np.random.normal(0, 0.01, pos.shape)
    pos = np.clip(pos, -0.49, 0.49)
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    pos = _init_grid(N)
    
    # Relaxation
    for _ in range(2000):
        if time.time() > deadline:
            break
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        min_pair = np.min(dist) / 2.0
        wall_dists = np.min(0.5 - np.abs(pos), axis=1)
        min_wall = np.min(wall_dists)
        r = min(min_pair, min_wall)
        
        if r < 1e-6:
             # If radius is tiny, just break and let SA handle
             pass

        mask = dist < 2 * r + 1e-9
        mask.fill_diagonal(False)
        
        forces = np.zeros_like(pos)
        norms = dist.copy()
        norms[norms < 1e-9] = 1.0
        unit_diff = diff / norms[:, :, np.newaxis]
        
        strength = 0.001
        forces += np.sum(mask[:, :, np.newaxis] * unit_diff, axis=1) * strength
        
        # Wall forces
        # Vectorized wall force
        # If pos > 0.5 - r, force -x
        # If pos < -0.5 + r, force +x
        # Same for y
        margin = 0.5 - r
        # x-axis
        forces[:, 0] -= np.where(pos[:, 0] > margin, strength, 0)
        forces[:, 0] += np.where(pos[:, 0] < -margin, strength, 0)
        # y-axis
        forces[:, 1] -= np.where(pos[:, 1] > margin, strength, 0)
        forces[:, 1] += np.where(pos[:, 1] < -margin, strength, 0)
        
        pos += forces
        pos = np.clip(pos, -0.499, 0.499)
        
    best_pos = pos.copy()
    best_score = _calculate_radius(best_pos)
    
    # Simulated Annealing
    T = 0.01
    T_min = 1e-7
    cooling = 0.9995
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        i = rng.integers(0, N)
        angle = rng.uniform(0, 2 * np.pi)
        step_size = 0.005
        
        dx = np.array([np.cos(angle), np.sin(angle)]) * step_size
        new_pos = pos.copy()
        new_pos[i] += dx
        new_pos[i] = np.clip(new_pos[i], -0.499, 0.499)
        
        new_score = _calculate_radius(new_pos)
        
        delta = new_score - best_score
        
        if delta > 0:
            pos = new_pos
            if new_score > best_score:
                best_score = new_score
                best_pos = pos.copy()
        else:
            if rng.random() < np.exp(delta / T):
                pos = new_pos
        
        T *= cooling
        if T < T_min:
            T = T_min
            if rng.random() < 0.01:
                T = 0.005

    return best_pos
