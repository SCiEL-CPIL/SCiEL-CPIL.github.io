"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

def _score(positions):
    """Compute the largest common radius for a set of positions in [-0.5, 0.5]^3."""
    n = positions.shape[0]
    if n == 0:
        return 0.0
    
    # Wall distances
    # positions shape (N, 3)
    # dist to x=-0.5 is positions[:, 0] + 0.5
    # dist to x=+0.5 is 0.5 - positions[:, 0]
    wall_dists = np.min(np.stack([
        positions[:, 0] + 0.5,
        0.5 - positions[:, 0],
        positions[:, 1] + 0.5,
        0.5 - positions[:, 1],
        positions[:, 2] + 0.5,
        0.5 - positions[:, 2]
    ]), axis=0)
    min_wall_dist = np.min(wall_dists)
    
    # Pairwise distances
    dists = cdist(positions, positions)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _initial_grid():
    """Generate initial positions on a jittered grid."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:N].copy()
    jitter = np.random.uniform(-0.02, 0.02, grid.shape)
    grid += jitter
    grid = np.clip(grid, -0.49, 0.49)
    return grid

def _initial_random():
    """Generate random initial positions."""
    positions = np.random.uniform(-0.49, 0.49, (N, 3))
    return positions

def _force_directed_relax(positions, max_iter=500, dt=0.002, deadline=None):
    """
    Fast force-directed relaxation to polish the solution.
    """
    positions = positions.copy()
    n = positions.shape[0]
    best_pos = positions.copy()
    best_score = _score(positions)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        forces = np.zeros_like(positions)
        
        # Pairwise repulsion
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, 1e-10)
        
        # Use a soft potential
        # Only apply force if distance is small
        mask = dists < 0.2
        for i in range(n):
            for j in range(i + 1, n):
                if mask[i, j]:
                    d = dists[i, j]
                    if d > 1e-10:
                        f = 1.0 / (d * d)
                        direction = (positions[i] - positions[j]) / d
                        forces[i] += f * direction
                        forces[j] -= f * direction
        
        # Wall repulsion
        for i in range(n):
            for k in range(3):
                d_neg = positions[i, k] + 0.5
                d_pos = 0.5 - positions[i, k]
                if d_neg < 0.1:
                    forces[i, k] += 10.0 / (d_neg * d_neg + 1e-10)
                if d_pos < 0.1:
                    forces[i, k] -= 10.0 / (d_pos * d_pos + 1e-10)
        
        positions += dt * forces
        positions = np.clip(positions, -0.4999, 0.4999)
        
        current_score = _score(positions)
        if current_score > best_score:
            best_score = current_score
            best_pos = positions.copy()
            
        dt *= 0.95
        
    return best_pos, best_score

def _run_sa(positions, deadline):
    """Run Simulated Annealing starting from given positions."""
    curr_pos = positions.copy()
    curr_score = _score(curr_pos)
    best_pos = curr_pos.copy()
    best_score = curr_score
    
    T = 0.05
    alpha = 0.99995
    min_T = 1e-6
    step_scale = 0.02
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        i = np.random.randint(N)
        step_size = step_scale * (1.0 + 10.0 * T)
        delta = np.random.normal(0, step_size, 3)
        
        new_pos = curr_pos.copy()
        new_pos[i] += delta
        new_pos[i] = np.clip(new_pos[i], -0.4999, 0.4999)
        
        new_score = _score(new_pos)
        
        if new_score >= curr_score:
            curr_pos = new_pos
            curr_score = new_score
        else:
            delta_e = curr_score - new_score
            prob = np.exp(-delta_e / T)
            if np.random.random() < prob:
                curr_pos = new_pos
                curr_score = new_score
        
        if curr_score > best_score:
            best_score = curr_score
            best_pos = curr_pos.copy()
            
        T *= alpha
        
    return best_pos, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_pos = None
    best_score = 0.0
    
    # Run 1: Jittered Grid
    pos1 = _initial_grid()
    deadline1 = start_time + 60.0
    pos1, score1 = _run_sa(pos1, deadline1)
    if score1 > best_score:
        best_score = score1
        best_pos = pos1
        
    # Run 2: Random (if time permits)
    if time.time() < start_time + 100.0:
        pos2 = _initial_random()
        deadline2 = start_time + 120.0
        pos2, score2 = _run_sa(pos2, deadline2)
        if score2 > best_score:
            best_score = score2
            best_pos = pos2
            
    # Final Polish
    if best_pos is not None and time.time() < deadline - 5.0:
        pos_p, score_p = _force_directed_relax(best_pos, max_iter=1000, dt=0.001, deadline=deadline)
        if score_p > best_score:
            best_score = score_p
            best_pos = pos_p
            
    if best_pos is None:
        # Fallback
        best_pos = _initial_grid()
        
    return best_pos
# EVOLVE-BLOCK-END
