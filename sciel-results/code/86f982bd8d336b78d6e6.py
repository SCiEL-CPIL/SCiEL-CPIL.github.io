"""
Combined program for packing 200 equal circles in a unit square.
Strategy:
1. Global repulsion to find a stable configuration.
2. Bottleneck-driven local optimization to maximize the minimum distance.
3. Final polish for precision.
"""
import numpy as np
import time
from scipy.spatial.distance import pdist

N = 200
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0


def _remaining_time():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)


def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    if len(centers) == 0:
        return 0.0
    # Distance to walls
    dw = np.minimum(
        np.minimum(0.5 + centers[:, 0], 0.5 - centers[:, 0]),
        np.minimum(0.5 + centers[:, 1], 0.5 - centers[:, 1])
    )
    wall_term = np.min(dw)
    
    n = len(centers)
    if n < 2:
        return float(wall_term)
    
    # Pairwise distances
    d = pdist(centers)
    min_pair = np.min(d)
    pair_term = min_pair / 2.0
    
    return min(float(wall_term), float(pair_term))


def _pdist_index_to_pair(k, n):
    """Convert pdist condensed index to (i, j) with i < j."""
    start = 0
    for i in range(n - 1):
        block_size = n - i - 1
        if k < start + block_size:
            j = i + 1 + (k - start)
            return i, j
        start += block_size
    return 0, 1


def _initial_grid():
    """Create initial grid arrangement with slight jitter."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    # Add small random jitter to break symmetry
    centers += np.random.normal(0, 0.001, centers.shape)
    centers = np.clip(centers, -0.5 + 1e-6, 0.5 - 1e-6)
    return centers


def _optimize_repulsion(centers, time_limit=10.0):
    """Global repulsion-based optimization."""
    start_time = time.time()
    centers = centers.copy().astype(np.float64)
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    learning_rate = 0.001
    decay = 0.999
    repulsion_power = 4
    
    for step in range(100000):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, 1e-10)
        
        repulsion = 1.0 / (dist ** repulsion_power)
        np.fill_diagonal(repulsion, 0)
        
        force_mag = repulsion * learning_rate
        forces += np.sum(force_mag[:, :, np.newaxis] * diff, axis=1)
        
        # Wall repulsion
        dist_left = centers[:, 0] + 0.5
        dist_right = 0.5 - centers[:, 0]
        dist_bottom = centers[:, 1] + 0.5
        dist_top = 0.5 - centers[:, 1]
        
        wall_strength = 0.5
        forces[:, 0] += wall_strength * learning_rate / (dist_left ** 2 + 1e-10)
        forces[:, 0] -= wall_strength * learning_rate / (dist_right ** 2 + 1e-10)
        forces[:, 1] += wall_strength * learning_rate / (dist_bottom ** 2 + 1e-10)
        forces[:, 1] -= wall_strength * learning_rate / (dist_top ** 2 + 1e-10)
        
        centers += forces
        centers[:, 0] = np.clip(centers[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
        centers[:, 1] = np.clip(centers[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
        
        current_r = _compute_radius(centers)
        if current_r > best_r:
            best_r = current_r
            best_centers = centers.copy()
        
        learning_rate *= decay
    
    return best_centers, best_r


def _bottleneck_local_opt(centers, time_limit=15.0):
    """
    Bottleneck-driven local optimization.
    Identifies the constraint limiting the radius (closest pair or wall)
    and moves the involved points to increase that distance.
    """
    start_time = time.time()
    centers = centers.copy().astype(np.float64)
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    step_size = 0.002
    no_improve_count = 0
    
    for it in range(500000):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Check remaining time globally
        if _remaining_time() < 5.0:
            break

        dw = np.minimum(
            np.minimum(0.5 + centers[:, 0], 0.5 - centers[:, 0]),
            np.minimum(0.5 + centers[:, 1], 0.5 - centers[:, 1])
        )
        min_wall_idx = np.argmin(dw)
        min_wall_dist = dw[min_wall_idx]
        
        d = pdist(centers)
        min_pair_idx = np.argmin(d)
        min_pair_dist = d[min_pair_idx]
        
        # Determine if the bottleneck is a pair or a wall
        if min_pair_dist / 2.0 <= min_wall_dist:
            # Bottleneck is a pair of circles
            i, j = _pdist_index_to_pair(min_pair_idx, N)
            vec = centers[i] - centers[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-9:
                # If coincident, move randomly
                vec = np.random.randn(2)
                norm = np.linalg.norm(vec)
            direction = vec / norm
            move = step_size * direction
            centers[i] += move
            centers[j] -= move
        else:
            # Bottleneck is a wall
            idx = min_wall_idx
            x, y = centers[idx]
            dl = x + 0.5
            dr = 0.5 - x
            db = y + 0.5
            dt = 0.5 - y
            min_d = min(dl, dr, db, dt)
            
            if min_d == dl:
                move = np.array([step_size, 0])
            elif min_d == dr:
                move = np.array([-step_size, 0])
            elif min_d == db:
                move = np.array([0, step_size])
            else:
                move = np.array([0, -step_size])
            
            centers[idx] += move
        
        centers = np.clip(centers, -0.5 + 1e-10, 0.5 - 1e-10)
        
        new_r = _compute_radius(centers)
        if new_r > best_r:
            best_r = new_r
            best_centers = centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 50:
                # Perturb a random point to escape local minimum
                idx = np.random.randint(N)
                centers[idx] += np.random.randn(2) * 0.002
                centers[idx] = np.clip(centers[idx], -0.5 + 1e-10, 0.5 - 1e-10)
                no_improve_count = 0
                step_size = max(0.0005, step_size * 0.9)
            elif no_improve_count > 20:
                step_size = max(0.0005, step_size * 0.95)
    
    return best_centers, best_r


def _polish(centers, time_limit=5.0):
    """
    High-precision polish using very small steps on bottleneck constraints.
    """
    start_time = time.time()
    centers = centers.copy().astype(np.float64)
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    step_size = 0.0001
    
    for it in range(500000):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        if _remaining_time() < 2.0:
            break

        dw = np.minimum(
            np.minimum(0.5 + centers[:, 0], 0.5 - centers[:, 0]),
            np.minimum(0.5 + centers[:, 1], 0.5 - centers[:, 1])
        )
        min_wall_idx = np.argmin(dw)
        min_wall_dist = dw[min_wall_idx]
        
        d = pdist(centers)
        min_pair_idx = np.argmin(d)
        min_pair_dist = d[min_pair_idx]
        
        if min_pair_dist / 2.0 <= min_wall_dist:
            i, j = _pdist_index_to_pair(min_pair_idx, N)
            vec = centers[i] - centers[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-9:
                continue
            direction = vec / norm
            move = step_size * direction
            centers[i] += move
            centers[j] -= move
        else:
            idx = min_wall_idx
            x, y = centers[idx]
            dl = x + 0.5
            dr = 0.5 - x
            db = y + 0.5
            dt = 0.5 - y
            min_d = min(dl, dr, db, dt)
            if min_d == dl:
                move = np.array([step_size, 0])
            elif min_d == dr:
                move = np.array([-step_size, 0])
            elif min_d == db:
                move = np.array([0, step_size])
            else:
                move = np.array([0, -step_size])
            centers[idx] += move
        
        centers = np.clip(centers, -0.5 + 1e-10, 0.5 - 1e-10)
        
        new_r = _compute_radius(centers)
        if new_r > best_r:
            best_r = new_r
            best_centers = centers.copy()
    
    return best_centers, best_r


def construct():
    """Construct the optimal packing of 200 equal circles in a unit square."""
    np.random.seed(42)
    
    best_centres = None
    best_r = 0.0
    
    # Initial grid
    initial_centres = _initial_grid()
    
    # Strategy:
    # 1. Run global repulsion to get a good starting point
    # 2. Run bottleneck local optimization
    # 3. Run polish
    # Repeat with different random seeds/perturbations if time allows
    
    num_restarts = 3
    
    for restart in range(num_restarts):
        if _remaining_time() < 20.0:
            break
        
        if restart == 0:
            centres = initial_centres.copy()
        else:
            # Perturb the previous best or initial grid
            if best_centres is not None:
                centres = best_centres + np.random.randn(N, 2) * 0.01
            else:
                centres = initial_centres + np.random.randn(N, 2) * 0.01
            
            centres = np.clip(centres, -0.5 + 1e-6, 0.5 - 1e-6)
        
        # Step 1: Global Repulsion (10s)
        t1 = _remaining_time()
        time_limit_repulsion = min(10.0, t1 - 20.0)
        if time_limit_repulsion > 1.0:
            centres, r_rep = _optimize_repulsion(centres, time_limit=time_limit_repulsion)
        else:
            r_rep = _compute_radius(centres)
        
        # Step 2: Bottleneck Local Optimization (15s)
        t2 = _remaining_time()
        time_limit_bottleneck = min(15.0, t2 - 10.0)
        if time_limit_bottleneck > 1.0:
            centres, r_bottleneck = _bottleneck_local_opt(centres, time_limit=time_limit_bottleneck)
        else:
            r_bottleneck = _compute_radius(centres)
        
        # Step 3: Polish (5s)
        t3 = _remaining_time()
        time_limit_polish = min(5.0, t3 - 2.0)
        if time_limit_polish > 0.5:
            centres, r_polish = _polish(centres, time_limit=time_limit_polish)
        else:
            r_polish = _compute_radius(centres)
        
        # Update best
        if r_polish > best_r:
            best_r = r_polish
            best_centres = centres.copy()
    
    # Final safety check and return
    if best_centres is None:
        return initial_centres
    
    # Ensure validity
    best_centres = np.clip(best_centres, -0.5, 0.5)
    return best_centres
