"""Optimized packing of 50 equal spheres in a unit cube using simulated annealing + polish."""
import numpy as np
import time
import math
from scipy.spatial.distance import cdist

N = 50

# EVOLVE-BLOCK-START

def _score(centers):
    """Compute the radius score: min of pairwise distances/2 and wall distances."""
    if len(centers) < 2:
        return 0.5
    coords = np.abs(centers)
    wall_dists = 0.5 - coords
    min_wall = np.min(wall_dists)
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(min_pair / 2.0, min_wall)

def _min_pair_dist(centers):
    """Return the minimum pairwise distance."""
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists)

def _min_wall_dist(centers):
    """Return the minimum wall distance."""
    coords = np.abs(centers)
    wall_dists = 0.5 - coords
    return np.min(wall_dists)

def _soft_min_score(centers, beta=50.0):
    """
    Differentiable soft-minimum of pairwise distances/2 and wall distances.
    Lower beta = smoother, higher beta = closer to true min.
    """
    coords = np.abs(centers)
    wall_dists = 0.5 - coords
    
    # For pairwise, use a subset approach: consider all pairs but use soft min
    # Actually for efficiency, let's compute the true min but use a smooth potential
    # for the gradient-based part.
    
    # Compute all pairwise distances
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    
    # Soft min of pairwise distances / 2
    # Use log-sum-exp trick for stability
    pair_vals = dists / 2.0
    # Only consider values that are relatively close to the min
    min_pair = np.min(pair_vals)
    # Mask: only consider pairs within 2*min_pair of the min
    mask = pair_vals < min_pair + 0.05
    if np.any(mask):
        masked_vals = pair_vals[mask]
        # Soft min
        max_val = np.max(masked_vals)
        exp_vals = np.exp(-beta * (masked_vals - max_val))
        soft_min_pair = max_val - np.log(np.sum(exp_vals)) / beta
    else:
        soft_min_pair = min_pair
    
    # Soft min of wall distances
    min_wall = np.min(wall_dists)
    mask_w = wall_dists < min_wall + 0.05
    if np.any(mask_w):
        masked_w = wall_dists[mask_w]
        max_w = np.max(masked_w)
        exp_w = np.exp(-beta * (masked_w - max_w))
        soft_min_wall = max_w - np.log(np.sum(exp_w)) / beta
    else:
        soft_min_wall = min_wall
    
    # Overall soft min
    all_vals = np.array([soft_min_pair, soft_min_wall])
    max_all = np.max(all_vals)
    exp_all = np.exp(-beta * (all_vals - max_all))
    result = max_all - np.log(np.sum(exp_all)) / beta
    
    return result

def _repulsion_energy(centers, r_target=0.136):
    """
    Repulsion energy: sum of (2*r_target - d)^2 for pairs closer than 2*r_target,
    plus wall penalty.
    """
    energy = 0.0
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    
    min_d = np.min(dists)
    if min_d < 2.0 * r_target:
        # Sum squared overlaps
        overlaps = np.maximum(0, 2.0 * r_target - dists)
        energy += np.sum(overlaps ** 2)
    
    # Wall penalty
    coords = np.abs(centers)
    wall_dists = 0.5 - coords
    wall_overlaps = np.maximum(0, r_target - wall_dists)
    energy += np.sum(wall_overlaps ** 2)
    
    return energy

def _initial_grid(m=4):
    """Initialize with a regular grid, taking first N points."""
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    centers += np.random.uniform(-0.002, 0.002, (N, 3))
    centers = np.clip(centers, -0.49, 0.49)
    return centers

def _initial_bcc(m=3):
    """Initialize with BCC lattice points."""
    points = []
    for z in range(m):
        for y in range(m):
            for x in range(m):
                # Corner-like points
                points.append(((x + 0.5) / m - 0.5, (y + 0.5) / m - 0.5, (z + 0.5) / m - 0.5))
                # Body center
                points.append(((x + 1.0) / m - 0.5, (y + 1.0) / m - 0.5, (z + 1.0) / m - 0.5))
    points = np.array(points)
    # Take first N
    centers = points[:N].copy()
    centers += np.random.uniform(-0.002, 0.002, (N, 3))
    centers = np.clip(centers, -0.49, 0.49)
    return centers

def _initial_random():
    """Random initialization."""
    centers = np.random.uniform(-0.45, 0.45, (N, 3))
    return centers

def _simulated_annealing(centers, time_limit=60.0):
    """
    Simulated annealing to maximize the minimum distance.
    Uses a repulsive force model with temperature-controlled moves.
    """
    current = centers.copy()
    best = current.copy()
    best_score = _score(current)
    
    start_time = time.time()
    t_start = 1.0
    t_end = 0.001
    n_steps = 50000
    
    # Precompute for efficiency
    for step in range(n_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Temperature schedule
        progress = step / n_steps
        temperature = t_start * (t_end / t_start) ** progress
        
        # Pick a random sphere to move
        idx = np.random.randint(N)
        
        # Generate a random displacement
        # Scale displacement by temperature
        scale = 0.02 * (1.0 + temperature * 10)
        displacement = np.random.normal(0, scale, 3)
        
        new_pos = current[idx].copy()
        new_pos += displacement
        new_pos = np.clip(new_pos, -0.49, 0.49)
        
        # Compute old and new scores (use a fast approximation)
        # For speed, only compute the affected distances
        old_dists = cdist(current[idx:idx+1], current)
        old_min_d = np.min(old_dists[0, np.arange(N) != idx])
        old_wall = np.min(0.5 - np.abs(current[idx]))
        old_score = min(old_min_d / 2.0, old_wall)
        
        new_dists = cdist(new_pos.reshape(1, 3), current)
        new_dists[0, idx] = np.inf
        new_min_d = np.min(new_dists[0, np.arange(N) != idx])
        new_wall = np.min(0.5 - np.abs(new_pos))
        new_score = min(new_min_d / 2.0, new_wall)
        
        # Acceptance criterion
        delta = new_score - old_score
        if delta > 0 or np.random.random() < math.exp(delta / (temperature + 1e-10)):
            current[idx] = new_pos
            
            # Check if this is the new best
            # Only do full score check every few steps
            if step % 100 == 0:
                full_score = _score(current)
                if full_score > best_score:
                    best_score = full_score
                    best = current.copy()
    
    return best, best_score

def _polish(centers, time_limit=20.0):
    """
    Local polish: iteratively resolve the tightest constraint by moving spheres.
    """
    current = centers.copy()
    best = current.copy()
    best_score = _score(current)
    start_time = time.time()
    
    for iteration in range(100000):
        if time.time() - start_time > time_limit:
            break
        
        # Find the tightest constraint
        dists = cdist(current, current)
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx[0], min_idx[1]
        
        wall_dists = 0.5 - np.abs(current)
        min_wall = np.min(wall_dists)
        wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        wi, waxis = wall_idx[0], wall_idx[1]
        
        current_score = min(min_d / 2.0, min_wall)
        
        if current_score > best_score:
            best_score = current_score
            best = current.copy()
        
        # Determine which constraint is tightest
        if min_d / 2.0 < min_wall:
            # Pair constraint is tightest: move i and j apart
            diff = current[i] - current[j]
            d = np.linalg.norm(diff)
            if d > 1e-10:
                target_d = 2.0 * (best_score + 0.001)
                if d < target_d:
                    push = (diff / d) * (target_d - d) * 0.5
                    current[i] += push
                    current[j] -= push
        else:
            # Wall constraint is tightest: move sphere away from wall
            pos = current[wi, waxis]
            if pos > 0:
                current[wi, waxis] = min(pos + 0.002, 0.49)
            else:
                current[wi, waxis] = max(pos - 0.002, -0.49)
        
        # Clip to cube
        current = np.clip(current, -0.49, 0.49)
        
        # Add small random perturbation periodically
        if iteration % 500 == 0:
            idx = np.random.randint(N)
            current[idx] += np.random.normal(0, 0.005, 3)
            current = np.clip(current, -0.49, 0.49)
    
    return best, best_score

def _kick_search(centers, time_limit=15.0):
    """
    Kick-based local search: repeatedly find the closest pair and kick them apart,
    also handling wall collisions.
    """
    current = centers.copy()
    best = current.copy()
    best_score = _score(current)
    start_time = time.time()
    
    for iteration in range(50000):
        if time.time() - start_time > time_limit:
            break
        
        dists = cdist(current, current)
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx[0], min_idx[1]
        
        wall_dists = 0.5 - np.abs(current)
        min_wall = np.min(wall_dists)
        
        current_score = min(min_d / 2.0, min_wall)
        
        if current_score > best_score:
            best_score = current_score
            best = current.copy()
        
        # Kick the closest pair apart
        if min_d < 2.0 * (best_score + 0.005):
            diff = current[i] - current[j]
            d = np.linalg.norm(diff)
            if d > 1e-10:
                target = 2.0 * (best_score + 0.005)
                push = (diff / d) * (target - min_d) * 0.5
                current[i] += push
                current[j] -= push
        
        # Handle wall collisions
        for k in range(N):
            for axis in range(3):
                if current[k, axis] > 0.5 - (best_score + 0.005):
                    current[k, axis] = 0.5 - (best_score + 0.005)
                elif current[k, axis] < -0.5 + (best_score + 0.005):
                    current[k, axis] = -0.5 + (best_score + 0.005)
        
        current = np.clip(current, -0.49, 0.49)
        
        # Random perturbation
        if iteration % 200 == 0:
            idx = np.random.randint(N)
            current[idx] += np.random.normal(0, 0.008, 3)
            current = np.clip(current, -0.49, 0.49)
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    best_centers = None
    best_score = 0.0
    
    # Multiple initializations
    initializations = [
        _initial_grid(4),
        _initial_bcc(3),
        _initial_random(),
    ]
    
    for init_idx, init in enumerate(initializations):
        if time.time() > deadline - 40.0:
            break
        
        centers = init.copy()
        score = _score(centers)
        
        # Phase 1: Simulated annealing
        if time.time() < deadline - 35.0:
            sa_time = min(25.0, deadline - time.time() - 30.0)
            sa_centers, sa_score = _simulated_annealing(centers, time_limit=sa_time)
            if sa_score > score:
                score = sa_score
                centers = sa_centers
        
        # Phase 2: Kick search
        if time.time() < deadline - 15.0:
            kick_time = min(12.0, deadline - time.time() - 10.0)
            kick_centers, kick_score = _kick_search(centers, time_limit=kick_time)
            if kick_score > score:
                score = kick_score
                centers = kick_centers
        
        # Phase 3: Polish
        if time.time() < deadline - 5.0:
            polish_time = min(10.0, deadline - time.time() - 3.0)
            polish_centers, polish_score = _polish(centers, time_limit=polish_time)
            if polish_score > score:
                score = polish_score
                centers = polish_centers
        
        if best_centers is None or score > best_score:
            best_score = score
            best_centers = centers
    
    # Final polish on the best result
    if best_centers is not None and time.time() < deadline - 3.0:
        final_time = min(8.0, deadline - time.time() - 2.0)
        final_centers, final_score = _polish(best_centers, time_limit=final_time)
        if final_score > best_score:
            best_score = final_score
            best_centers = final_centers
    
    if best_centers is None:
        best_centers = _initial_grid(4)
    
    return best_centers

# EVOLVE-BLOCK-END
