"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.spatial import cKDTree

N_CIRCLES = 200
HALF_SIDE = 0.5


def compute_radius(centers):
    """Compute the maximum valid radius for the given centers."""
    if len(centers) != N_CIRCLES:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > HALF_SIDE + 1e-12):
        return 0.0
    
    # Distance to boundary
    min_dist_to_wall = np.min(HALF_SIDE - np.abs(centers), axis=None)
    if min_dist_to_wall <= 0:
        return 0.0
    
    # Min pairwise distance
    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=0.1)  # Only need to check close pairs
    if len(pairs) == 0:
        min_pair_dist = 1e9
    else:
        min_pair_dist = min(np.linalg.norm(centers[i] - centers[j]) for i, j in pairs)
    
    r = min(min_dist_to_wall, min_pair_dist / 2)
    return r


def score_config(centers):
    """Return the radius if valid, else 0."""
    if centers.shape != (N_CIRCLES, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > HALF_SIDE + 1e-10):
        return 0.0
    
    # Check for coincident centers
    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=1e-9)
    if len(pairs) > 0:
        return 0.0
    
    return compute_radius(centers)


def hex_lattice_init(n_points=200):
    """Generate a hexagonal lattice initialization clipped to the unit square."""
    # Hexagonal packing: rows offset by half spacing
    # Target spacing ~ 1/sqrt(200) ≈ 0.07
    spacing = 0.072
    row_height = spacing * np.sqrt(3) / 2
    
    points = []
    # Generate a generous grid of hex points
    n_x = int(1.0 / spacing) + 4
    n_y = int(1.0 / row_height) + 4
    
    for row in range(n_y):
        y = -0.5 + (row + 0.5) * row_height
        offset = 0.0
        if row % 2 == 1:
            offset = spacing / 2
        for col in range(n_x):
            x = -0.5 + (col + 0.5) * spacing + offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have more points than needed, randomly select
    if len(points) > n_points:
        idx = np.random.permutation(len(points))
        points = points[idx[:n_points]]
    elif len(points) < n_points:
        # Pad with random points
        extra = np.random.uniform(-0.5, 0.5, size=(n_points - len(points), 2))
        points = np.vstack([points, extra])
    
    return points


def repulsion_relaxation(centers, r_min, n_steps=2000, cooling=0.995):
    """
    Physics-based repulsion relaxation.
    Each circle repels others with force ~ 1/d^2 when d < 2*r_min,
    and is repelled from walls.
    Uses geometric cooling to reduce step size.
    """
    current = centers.copy()
    n = len(current)
    
    # Initial step size proportional to r_min
    step_scale = r_min * 0.5
    
    for step in range(n_steps):
        if step % 10 == 0:
            # Rebuild tree every 10 steps for efficiency
            tree = cKDTree(current)
            # Find pairs within 2*r_min + margin
            search_r = 2 * r_min * 1.5
            pairs = tree.query_pairs(r=search_r)
        else:
            # Reuse pairs (approximate, but faster)
            pass
        
        # Compute forces
        forces = np.zeros_like(current)
        
        for i, j in pairs:
            diff = current[i] - current[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                # Random small push
                diff = np.random.normal(0, 0.001, size=2)
                dist = np.linalg.norm(diff)
            
            # Repulsion force: strong when close
            if dist < 2 * r_min:
                # Force magnitude ~ (2*r_min - dist)^2 / dist^2
                force_mag = (2 * r_min - dist) ** 2 / (dist ** 2 + 1e-12)
                force_vec = force_mag * diff / dist
            else:
                # Weak long-range repulsion to prevent clustering
                force_mag = 0.1 * (2 * r_min / dist) ** 4
                force_vec = force_mag * diff / dist
            
            forces[i] += force_vec
            forces[j] -= force_vec
        
        # Wall repulsion
        # Distance to each wall
        dist_left = current[:, 0] + HALF_SIDE
        dist_right = HALF_SIDE - current[:, 0]
        dist_bottom = current[:, 1] + HALF_SIDE
        dist_top = HALF_SIDE - current[:, 1]
        
        wall_margin = r_min * 0.8
        
        # Left wall
        mask = dist_left < wall_margin
        if np.any(mask):
            f = (wall_margin - dist_left[mask]) ** 2 / (wall_margin ** 2 + 1e-12)
            forces[mask, 0] += f
        
        # Right wall
        mask = dist_right < wall_margin
        if np.any(mask):
            f = (wall_margin - dist_right[mask]) ** 2 / (wall_margin ** 2 + 1e-12)
            forces[mask, 0] -= f
        
        # Bottom wall
        mask = dist_bottom < wall_margin
        if np.any(mask):
            f = (wall_margin - dist_bottom[mask]) ** 2 / (wall_margin ** 2 + 1e-12)
            forces[mask, 1] += f
        
        # Top wall
        mask = dist_top < wall_margin
        if np.any(mask):
            f = (wall_margin - dist_top[mask]) ** 2 / (wall_margin ** 2 + 1e-12)
            forces[mask, 1] -= f
        
        # Normalize forces to prevent explosion
        force_norms = np.linalg.norm(forces, axis=1)
        max_force = np.max(force_norms) if len(force_norms) > 0 else 1.0
        if max_force > 0:
            forces = forces / max_force
        
        # Update positions
        current += step_scale * forces
        
        # Clamp to boundary
        current = np.clip(current, -HALF_SIDE, HALF_SIDE)
        
        # Cool down
        step_scale *= cooling
    
    return current


def adaptive_r_increase(centers, r_start, r_max, n_trials=50):
    """
    Binary-search-like adaptive radius increase.
    Try to relax with increasing r_min until it fails.
    """
    best_config = centers.copy()
    best_r = r_start
    
    r_low = r_start
    r_high = r_max
    
    for trial in range(n_trials):
        r_mid = (r_low + r_high) / 2
        
        # Try to relax with this r_min
        relaxed = repulsion_relaxation(centers, r_mid, n_steps=500, cooling=0.998)
        
        # Check if valid: all pairs >= 2*r_mid and all walls >= r_mid
        tree = cKDTree(relaxed)
        pairs = tree.query_pairs(r=2 * r_mid + 1e-9)
        valid = True
        for i, j in pairs:
            if np.linalg.norm(relaxed[i] - relaxed[j]) < 2 * r_mid - 1e-10:
                valid = False
                break
        
        if valid and np.all(np.abs(relaxed) <= HALF_SIDE - r_mid + 1e-10):
            # Success: can increase r
            r_low = r_mid
            best_config = relaxed.copy()
            best_r = r_mid
        else:
            # Failure: decrease r
            r_high = r_mid
        
        if r_high - r_low < 1e-7:
            break
    
    return best_config, best_r


def local_optimization(centers, n_rounds=10):
    """Iteratively push circles apart to improve packing."""
    current = centers.copy()
    current_r = compute_radius(current)
    
    for _ in range(n_rounds):
        tree = cKDTree(current)
        
        # Find close pairs
        close_pairs = tree.query_pairs(r=2 * (current_r + 0.005))
        
        if len(close_pairs) == 0:
            break
        
        # Move apart close pairs
        moved = False
        for i, j in close_pairs:
            diff = current[i] - current[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                continue
            diff_norm = diff / dist
            push = (2 * current_r - dist) / 2 + 0.0005
            if push > 0:
                current[i] += push * diff_norm
                current[j] -= push * diff_norm
                moved = True
        
        # Clamp to boundary
        current = np.clip(current, -HALF_SIDE, HALF_SIDE)
        
        if not moved:
            break
        
        new_r = compute_radius(current)
        if new_r < current_r:
            break
        current_r = new_r
    
    return current, current_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin from 150s
    
    np.random.seed(42)
    
    # Generate multiple hex lattice initializations with different parameters
    best_config = None
    best_r = 0.0
    
    # Try different hex spacings
    for spacing in [0.070, 0.072, 0.074, 0.076]:
        if time.time() > deadline - 20.0:
            break
        
        # Generate hex lattice
        points = []
        row_height = spacing * np.sqrt(3) / 2
        n_x = int(1.0 / spacing) + 4
        n_y = int(1.0 / row_height) + 4
        
        for row in range(n_y):
            y = -0.5 + (row + 0.5) * row_height
            offset = 0.0
            if row % 2 == 1:
                offset = spacing / 2
            for col in range(n_x):
                x = -0.5 + (col + 0.5) * spacing + offset
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
        
        points = np.array(points)
        
        if len(points) >= N_CIRCLES:
            # Randomly select N points
            idx = np.random.permutation(len(points))
            init = points[idx[:N_CIRCLES]]
        else:
            init = points
        
        # Add small random jitter
        init = init + np.random.uniform(-0.005, 0.005, size=init.shape)
        init = np.clip(init, -HALF_SIDE, HALF_SIDE)
        
        # Run repulsion relaxation with adaptive r increase
        r_start = compute_radius(init)
        
        # First, do a coarse relaxation
        relaxed = repulsion_relaxation(init, r_start, n_steps=1000, cooling=0.997)
        r_coarse = compute_radius(relaxed)
        
        # Then adaptive r increase
        relaxed2, r_final = adaptive_r_increase(relaxed, r_coarse, 0.04, n_trials=30)
        
        # Final local optimization
        if time.time() < deadline - 5.0:
            relaxed2, r_final = local_optimization(relaxed2, n_rounds=5)
        
        if r_final > best_r:
            best_r = r_final
            best_config = relaxed2.copy()
    
    # If no good config found, fallback to simple grid
    if best_config is None or best_r < 0.03:
        m = int(np.ceil(np.sqrt(N_CIRCLES)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_config = grid[:N_CIRCLES]
        best_r = compute_radius(best_config)
    
    return best_config
# EVOLVE-BLOCK-END
