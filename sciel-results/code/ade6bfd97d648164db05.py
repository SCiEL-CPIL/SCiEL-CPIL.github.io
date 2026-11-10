"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

N = 200
TIME_LIMIT = 145.0  # seconds, leaving margin before 150s kill


def _compute_radius(centres):
    """Compute the packing radius for a set of centres."""
    if len(centres) != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check bounds: centres must be inside [-0.5, 0.5]^2
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0
    # Distance to walls
    wall_dists = np.min(np.stack([
        0.5 - centres[:, 0],
        0.5 + centres[:, 0],
        0.5 - centres[:, 1],
        0.5 + centres[:, 1]
    ], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    if min_wall <= 0:
        return 0.0
    # Minimum pairwise distance
    # Use scipy if available for speed, else fallback
    try:
        from scipy.spatial.distance import cdist
        dists = cdist(centres, centres)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
    except ImportError:
        # Fallback: compute all pairwise distances
        min_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centres[i] - centres[j])
                if d < min_dist:
                    min_dist = d
    return min(min_dist / 2.0, min_wall)


def _grid_packing():
    """Create a square grid packing with optimal dimensions for N points."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _hexagonal_packing():
    """Create a hexagonal packing pattern, optimized for N points."""
    best_centres = None
    best_r = -1
    
    # Try various numbers of rows
    for n_rows in range(10, 25):
        # For hexagonal packing, alternating rows have offset
        # Row 0: n_cols points, Row 1: n_cols - 1 points (or vice versa)
        # Let's use a pattern where we can fit N points
        
        # Try both orientations: start with full row or offset row
        for start_offset in [0, 1]:
            # Estimate columns needed
            # If start_offset=0: rows 0,2,4,... have n_cols, rows 1,3,5,... have n_cols-1
            # If start_offset=1: rows 0,2,4,... have n_cols-1, rows 1,3,5,... have n_cols
            
            # Let's try to find n_cols such that total points >= N
            # For simplicity, let's try a range of n_cols
            for n_cols in range(10, 25):
                # Calculate how many points fit in this configuration
                total_points = 0
                for row in range(n_rows):
                    if (row + start_offset) % 2 == 0:
                        total_points += n_cols
                    else:
                        total_points += n_cols - 1
                
                if total_points < N:
                    continue
                
                # Calculate spacing s
                # Horizontal: n_cols * s <= 1 (for full rows)
                # Vertical: (n_rows - 1) * s * sqrt(3)/2 + s <= 1 (approx)
                # Actually, vertical extent is (n_rows - 1) * s * sqrt(3)/2
                # And we need to center it
                
                # Width needed: n_cols * s for full rows
                # But offset rows are shifted by s/2, so they fit in same width if n_cols-1 points
                # Let's say s = 1.0 / n_cols (horizontal constraint)
                # Vertical: (n_rows - 1) * s * sqrt(3)/2 <= 1
                # So s <= 2 / ((n_rows - 1) * sqrt(3)) if n_rows > 1
                
                s_horiz = 1.0 / n_cols
                if n_rows > 1:
                    s_vert = 2.0 / ((n_rows - 1) * np.sqrt(3))
                else:
                    s_vert = 1.0
                
                s = min(s_horiz, s_vert)
                r = s / 2.0
                
                if r <= best_r:
                    continue
                
                # Create the grid
                centres = []
                y_start = -0.5 + s * np.sqrt(3) / 2  # Center vertically
                for row in range(n_rows):
                    y = y_start + row * s * np.sqrt(3) / 2
                    if (row + start_offset) % 2 == 0:
                        # Full row: n_cols points
                        x_start = -0.5 + s / 2
                        for col in range(n_cols):
                            x = x_start + col * s
                            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                                centres.append((x, y))
                    else:
                        # Offset row: n_cols - 1 points
                        x_start = -0.5 + s  # Shift by s/2 from full row start
                        for col in range(n_cols - 1):
                            x = x_start + col * s
                            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                                centres.append((x, y))
                
                centres = np.array(centres)
                if len(centres) < N:
                    continue
                
                centres = centres[:N]
                actual_r = _compute_radius(centres)
                if actual_r > best_r:
                    best_r = actual_r
                    best_centres = centres
    
    if best_centres is not None:
        return best_centres
    
    # Fallback to grid
    return _grid_packing()


def _local_optimization(centres, max_iters=500, step_size=0.02):
    """
    Structural Change: Replace simple greedy random perturbation with a 
    Simulated Annealing (SA) approach. This allows escaping local minima 
    by accepting worse moves probabilistically, which is crucial for 
    dense packing problems where the landscape is rugged.
    """
    best_centres = centres.copy()
    current_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    current_r = _compute_radius(current_centres)
    
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 5)  # Leave 5s buffer
    
    # SA Parameters
    T_start = 0.05
    T_end = 0.001
    cooling_rate = (T_end / T_start) ** (1.0 / max_iters)
    
    try:
        from scipy.spatial.distance import cdist
    except ImportError:
        cdist = None

    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        T = T_start * (cooling_rate ** it)
        
        # Select a random point to move, or the most constrained one
        # Mixing strategies: 80% random, 20% most constrained
        if np.random.rand() < 0.8:
            idx = np.random.randint(N)
        else:
            # Calculate constraints for most constrained point selection
            wall_dists = np.min(np.stack([
                0.5 - current_centres[:, 0],
                0.5 + current_centres[:, 0],
                0.5 - current_centres[:, 1],
                0.5 + current_centres[:, 1]
            ], axis=1), axis=1)
            
            if cdist is not None:
                dists = cdist(current_centres, current_centres)
                np.fill_diagonal(dists, np.inf)
                min_pair_dists = np.min(dists, axis=1)
            else:
                min_pair_dists = np.full(N, np.inf)
                for i in range(N):
                    for j in range(i + 1, N):
                        d = np.linalg.norm(current_centres[i] - current_centres[j])
                        min_pair_dists[i] = min(min_pair_dists[i], d)
                        min_pair_dists[j] = min(min_pair_dists[j], d)
            
            constraints = np.minimum(min_pair_dists / 2.0, wall_dists)
            idx = np.argmin(constraints)
        
        # Generate a perturbation
        # Adaptive step size: larger when T is high, smaller when T is low
        current_step = step_size * (1.0 + T / T_start)
        angle = np.random.uniform(0, 2 * np.pi)
        delta = current_step * np.array([np.cos(angle), np.sin(angle)])
        
        new_centres = current_centres.copy()
        new_centres[idx] += delta
        
        # Check bounds
        if np.any(new_centres < -0.5) or np.any(new_centres > 0.5):
            continue
        
        new_r = _compute_radius(new_centres)
        
        # Acceptance criterion
        if new_r > current_r:
            # Always accept improvement
            current_centres = new_centres
            current_r = new_r
            if new_r > best_r:
                best_centres = new_centres
                best_r = new_r
        else:
            # Accept worse move with probability exp(-delta_r / T)
            # Note: delta_r is negative here (new_r - current_r)
            delta_r = new_r - current_r
            if delta_r < 0:
                prob = np.exp(delta_r / T)
                if np.random.rand() < prob:
                    current_centres = new_centres
                    current_r = new_r
    
    return best_centres, best_r


def construct():
    start_time = time.time()
    
    # Start with a good initial configuration
    # Try hexagonal packing
    centres = _hexagonal_packing()
    current_r = _compute_radius(centres)
    
    # Local optimization
    if time.time() - start_time < TIME_LIMIT - 15:
        centres, current_r = _local_optimization(centres)
    
    # Final check
    final_r = _compute_radius(centres)
    if final_r <= 0:
        # Fallback to a valid grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return centres
# EVOLVE-BLOCK-END
