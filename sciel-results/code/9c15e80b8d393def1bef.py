# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

N = 50

def _min_pair_dist(centers):
    """Return minimum pairwise distance using KDTree."""
    if len(centers) < 2:
        return float('inf')
    tree = cKDTree(centers)
    dists, idxs = tree.query(centers, k=2)
    return np.min(dists[:, 1])

def _min_wall_dist(centers):
    """Return minimum distance from any center to any wall."""
    return np.min(0.5 - np.max(np.abs(centers), axis=1))

def _score(centers):
    """Score is the radius r such that spheres of radius r fit."""
    min_pair = _min_pair_dist(centers)
    min_wall = _min_wall_dist(centers)
    return min(min_pair / 2.0, min_wall)

def _is_valid(centers):
    """Check if centers are valid (inside cube, finite, correct shape)."""
    if centers.shape != (N, 3):
        return False
    if not np.all(np.isfinite(centers)):
        return False
    if np.any(centers < -0.5 - 1e-10) or np.any(centers > 0.5 + 1e-10):
        return False
    return True

def _ffd_packing():
    """
    First Fit Decreasing style initialization.
    Place spheres greedily, trying to maximize minimum distance.
    """
    # Use a 4x4x4 grid (64 points) and pick 50
    m = 4
    spacing = 1.0 / m
    grid_points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = -0.5 + (i + 0.5) * spacing
                y = -0.5 + (j + 0.5) * spacing
                z = -0.5 + (k + 0.5) * spacing
                grid_points.append([x, y, z])
    
    grid_points = np.array(grid_points)
    
    # Shuffle and take first 50
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(grid_points))
    selected = grid_points[perm[:N]]
    
    # Add small perturbation to break symmetry
    selected += rng.normal(0, 0.01, size=(N, 3))
    selected = np.clip(selected, -0.45, 0.45)
    
    return selected

def _local_search(centers, time_limit, rng=None):
    """
    Local search: try moving each point in multiple directions.
    Accept moves that improve the score.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    current = centers.copy()
    best = current.copy()
    best_score = _score(best)
    
    start_time = time.time()
    step_sizes = [0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    while time.time() - start_time < time_limit:
        improved = False
        current_score = _score(current)
        
        if current_score > best_score:
            best_score = current_score
            best = current.copy()
        
        # Try moving each point
        for i in range(N):
            if time.time() - start_time >= time_limit:
                break
            
            orig_pos = current[i].copy()
            orig_score = current_score
            
            # Try multiple directions and step sizes
            directions = [
                [1, 0, 0], [-1, 0, 0],
                [0, 1, 0], [0, -1, 0],
                [0, 0, 1], [0, 0, -1],
                [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
                [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
                [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
                [1, 1, 1], [-1, -1, 1], [1, -1, 1], [-1, 1, 1],
                [1, 1, -1], [-1, -1, -1], [1, -1, -1], [-1, 1, -1]
            ]
            
            for step in step_sizes:
                if time.time() - start_time >= time_limit:
                    break
                
                for d in directions:
                    d = np.array(d, dtype=float)
                    d = d / np.linalg.norm(d)
                    
                    new_pos = orig_pos + d * step
                    new_pos = np.clip(new_pos, -0.5, 0.5)
                    
                    # Quick check: compute min distance to other points
                    mask = np.ones(N, dtype=bool)
                    mask[i] = False
                    other_dists = np.linalg.norm(current[mask] - new_pos, axis=1)
                    min_other = np.min(other_dists) if len(other_dists) > 0 else float('inf')
                    min_wall = 0.5 - np.max(np.abs(new_pos))
                    
                    # Estimate new score
                    est_score = min(min_other / 2.0, min_wall)
                    
                    if est_score > orig_score + 1e-10:
                        # Accept the move
                        current[i] = new_pos
                        current_score = est_score
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # No improvement, try random perturbation
            idx = rng.randint(N)
            current[idx] += rng.normal(0, 0.01, size=3)
            current[idx] = np.clip(current[idx], -0.5, 0.5)
    
    return best, best_score

def _optimize_with_sa(centers, time_limit, rng=None):
    """
    Simulated annealing optimization.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    current = centers.copy()
    best = current.copy()
    best_score = _score(best)
    
    start_time = time.time()
    T = 0.05  # Initial temperature
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit:
        T *= cooling_rate
        if T < 1e-8:
            break
        
        # Pick a random point to move
        idx = rng.randint(N)
        orig_pos = current[idx].copy()
        
        # Random perturbation
        step = T * 0.5
        new_pos = orig_pos + rng.normal(0, step, size=3)
        new_pos = np.clip(new_pos, -0.5, 0.5)
        
        # Compute new score
        current[idx] = new_pos
        new_score = _score(current)
        
        # Metropolis criterion
        delta = new_score - best_score
        if delta > 0 or rng.rand() < np.exp(delta / (T + 1e-10)):
            # Accept
            if new_score > best_score:
                best_score = new_score
                best = current.copy()
        else:
            # Reject
            current[idx] = orig_pos
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Step 1: FFD initialization
    centers = _ffd_packing()
    best_centers = centers.copy()
    best_score = _score(best_centers)
    
    # Step 2: Local search
    time_remaining = deadline - time.time()
    if time_remaining > 10:
        ls_centers, ls_score = _local_search(centers, time_remaining - 10)
        if ls_score > best_score:
            best_centers = ls_centers
            best_score = ls_score
    
    # Step 3: Simulated annealing
    time_remaining = deadline - time.time()
    if time_remaining > 10:
        sa_centers, sa_score = _optimize_with_sa(best_centers, time_remaining - 5)
        if sa_score > best_score:
            best_centers = sa_centers
            best_score = sa_score
    
    # Step 4: Final local search polish
    time_remaining = deadline - time.time()
    if time_remaining > 5:
        polish_centers, polish_score = _local_search(best_centers, time_remaining - 2)
        if polish_score > best_score:
            best_centers = polish_centers
            best_score = polish_score
    
    return best_centers
# EVOLVE-BLOCK-END
