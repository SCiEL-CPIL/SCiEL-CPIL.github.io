"""Rewritten program for packing 200 equal circles in a unit square."""
import numpy as np
import time
from scipy.spatial.distance import pdist, squareform

N = 200
START_TIME = time.time()
TIME_LIMIT = 148.0  # Safety margin before 150s kill

def _remaining_time():
    return TIME_LIMIT - (time.time() - START_TIME)

def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    # Distance to walls: min over all points of min distance to any wall
    dw = np.minimum(
        np.minimum(0.5 + centers[:, 0], 0.5 - centers[:, 0]),
        np.minimum(0.5 + centers[:, 1], 0.5 - centers[:, 1])
    )
    wall_term = np.min(dw)
    
    n = len(centers)
    if n < 2:
        return float(wall_term)
    
    d = pdist(centers)
    min_pair = np.min(d)
    pair_term = min_pair / 2.0
    
    return min(float(wall_term), float(pair_term))

def _initial_grid():
    """Create initial grid arrangement with slight jitter to break symmetry."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    # Add small random jitter
    centers += np.random.normal(0, 0.001, centers.shape)
    # Clip to ensure inside
    centers = np.clip(centers, -0.5 + 1e-6, 0.5 - 1e-6)
    return centers

def _initial_hexagonal():
    """Create hexagonal packing arrangement."""
    # For hexagonal packing, we can try to fit a hexagonal grid
    # with appropriate spacing
    centers = []
    # Try different row spacings
    for attempt in range(20):
        # Randomly perturb a hexagonal grid
        m = int(np.ceil(np.sqrt(N * 2 / np.sqrt(3))))
        # Hexagonal lattice points
        for row in range(int(m * 1.2)):
            for col in range(int(m * 1.2)):
                x = col * 1.0 + (0.5 if row % 2 else 0)
                y = row * np.sqrt(3) / 2
                # Normalize to fit in [-0.5, 0.5]
                x = x - m * 0.6
                y = y - m * np.sqrt(3) / 4
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    centers.append([x, y])
        if len(centers) >= N:
            break
    centers = np.array(centers[:N])
    centers += np.random.normal(0, 0.001, centers.shape)
    centers = np.clip(centers, -0.5 + 1e-6, 0.5 - 1e-6)
    return centers

def _pdist_index_to_pair(k, n):
    """Convert pdist condensed index to (i, j) with i < j."""
    start = 0
    for i in range(n - 1):
        block_size = n - i - 1
        if k < start + block_size:
            j = i + 1 + (k - start)
            return i, j
        start += block_size
    return 0, 1  # Fallback

def _optimize_simultaneous(centers, max_iters=30000):
    """
    Optimize by simultaneously pushing apart all bottleneck pairs and wall contacts.
    This helps escape local minima better than sequential pair optimization.
    """
    centers = centers.copy()
    r = _compute_radius(centers)
    best_r = r
    best_centers = centers.copy()
    
    step_size = 0.005
    no_improve_count = 0
    
    for it in range(max_iters):
        if _remaining_time() < 3.0:
            break
            
        # Compute all pairwise distances
        d = pdist(centers)
        min_pair_dist = np.min(d)
        pair_term = min_pair_dist / 2.0
        
        # Wall distances
        dw = np.minimum(
            np.minimum(0.5 + centers[:, 0], 0.5 - centers[:, 0]),
            np.minimum(0.5 + centers[:, 1], 0.5 - centers[:, 1])
        )
        min_wall_dist = np.min(dw)
        wall_term = min_wall_dist
        
        current_r = min(pair_term, wall_term)
        
        # Find all pairs that are close to the minimum (within 10% of min)
        threshold_pair = min_pair_dist * 1.1
        close_pairs_mask = d < threshold_pair
        close_pairs = np.where(close_pairs_mask)[0]
        
        # Find all points close to walls
        wall_threshold = min_wall_dist + 0.005
        close_wall_idx = np.where(dw < wall_threshold)[0]
        
        # Apply forces
        forces = np.zeros_like(centers)
        
        # Forces from close pairs
        for k in close_pairs:
            i, j = _pdist_index_to_pair(k, N)
            vec = centers[i] - centers[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                direction = vec / norm
                # Force proportional to how close they are
                strength = step_size * (1.0 - (d[k] - min_pair_dist) / max(threshold_pair - min_pair_dist, 1e-9))
                forces[i] += direction * strength
                forces[j] -= direction * strength
        
        # Forces from walls
        for idx in close_wall_idx:
            x, y = centers[idx]
            dl = x + 0.5
            dr = 0.5 - x
            db = y + 0.5
            dt = 0.5 - y
            
            min_d = min(dl, dr, db, dt)
            strength = step_size * (1.0 - (dw[idx] - min_wall_dist) / max(wall_threshold - min_wall_dist, 1e-9))
            
            if min_d == dl:
                forces[idx] += np.array([strength, 0])
            elif min_d == dr:
                forces[idx] += np.array([-strength, 0])
            elif min_d == db:
                forces[idx] += np.array([0, strength])
            else:
                forces[idx] += np.array([0, -strength])
        
        # Apply forces
        centers += forces
        centers = np.clip(centers, -0.5, 0.5)
        
        new_r = _compute_radius(centers)
        
        if new_r > best_r:
            best_r = new_r
            best_centers = centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 50:
                # Random kick
                for _ in range(3):
                    idx = np.random.randint(N)
                    centers[idx] += np.random.randn(2) * 0.008
                    centers[idx] = np.clip(centers[idx], -0.5, 0.5)
                no_improve_count = 0
            elif no_improve_count > 20:
                step_size = max(0.001, step_size * 0.95)
    
    return best_centers, best_r

def _optimize_hill_climbing(centers, max_iters=20000):
    """
    Hill-climbing optimization that tries random perturbations and keeps improvements.
    """
    centers = centers.copy()
    r = _compute_radius(centers)
    best_r = r
    best_centers = centers.copy()
    
    perturbation_scale = 0.01
    no_improve_count = 0
    
    for it in range(max_iters):
        if _remaining_time() < 3.0:
            break
        
        # Try a random perturbation
        n_perturbed = np.random.randint(1, 5)
        idxs = np.random.choice(N, n_perturbed, replace=False)
        old_positions = centers[idxs].copy()
        
        centers[idxs] += np.random.randn(n_perturbed, 2) * perturbation_scale
        centers = np.clip(centers, -0.5, 0.5)
        
        new_r = _compute_radius(centers)
        
        if new_r > best_r:
            best_r = new_r
            best_centers = centers.copy()
            no_improve_count = 0
        else:
            # Revert
            centers[idxs] = old_positions
            no_improve_count += 1
            if no_improve_count > 30:
                perturbation_scale *= 0.9
                no_improve_count = 0
    
    return best_centers, best_r

def construct():
    """Construct optimal circle packing."""
    # Try multiple initial configurations
    best_radius = 0.0
    best_centers = None
    
    # Start with grid
    centers = _initial_grid()
    r = _compute_radius(centers)
    best_radius = r
    best_centers = centers.copy()
    
    # Try hexagonal
    if _remaining_time() > 10:
        centers_hex = _initial_hexagonal()
        r_hex = _compute_radius(centers_hex)
        if r_hex > best_radius:
            best_radius = r_hex
            best_centers = centers_hex.copy()
    
    # Optimize with simultaneous repulsion
    if _remaining_time() > 20:
        try:
            optimized, opt_radius = _optimize_simultaneous(best_centers)
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = optimized
        except Exception:
            pass
    
    # Further optimize with hill climbing
    if _remaining_time() > 10:
        try:
            optimized, opt_radius = _optimize_hill_climbing(best_centers)
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = optimized
        except Exception:
            pass
    
    # Final simultaneous optimization
    if _remaining_time() > 15:
        try:
            optimized, opt_radius = _optimize_simultaneous(best_centers, max_iters=10000)
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = optimized
        except Exception:
            pass
    
    # Ensure all centers are within bounds
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
