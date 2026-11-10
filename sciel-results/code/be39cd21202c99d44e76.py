"""Program for 200 equal circles in a unit square."""
import numpy as np
import time

N = 200
TIME_LIMIT = 145.0  # seconds, leaving margin before 150s kill


def _fibonacci_lattice(n):
    """Create a Fibonacci (sunflower) lattice of n points in the unit square."""
    # Golden angle
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    
    # Fibonacci lattice: points at (sqrt(i/n)*cos(i*golden_angle), sqrt(i/n)*sin(i*golden_angle))
    # This gives a nice uniform distribution
    i = np.arange(1, n + 1, dtype=np.float64)
    r = np.sqrt(i / n)
    theta = i * golden_angle
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Scale to fit in [-0.5, 0.5]^2
    # The max radius is 1.0, so we need to scale by 0.5
    # But we also need to ensure points are well inside the square
    # The Fibonacci lattice has points near the boundary, so let's scale slightly less
    scale = 0.48  # Slightly less than 0.5 to keep points well inside
    
    centres = np.column_stack([x * scale, y * scale])
    
    return centres


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


def _force_based_refinement(centres, max_iters=2000, time_budget=100.0):
    """Force-based refinement treating circle boundaries as rigid constraints."""
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    try:
        from scipy.spatial.distance import cdist
    except ImportError:
        cdist = None
    
    # Force-based simulation with repulsive forces
    # Each circle repels others when too close, and is attracted to center when near walls
    
    # Start with a reasonable force scale
    force_scale = 1.0
    damping = 0.95
    velocities = np.zeros_like(best_centres)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(best_centres)
        
        if cdist is not None:
            dists = cdist(best_centres, best_centres)
            np.fill_diagonal(dists, np.inf)
        else:
            dists = np.full((N, N), np.inf)
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(best_centres[i] - best_centres[j])
                    dists[i, j] = d
                    dists[j, i] = d
        
        # Repulsive forces between circles
        # Use a smooth approximation: force ~ 1/d^2 when d < threshold
        # We want to push apart circles that are too close
        # The target minimum distance is 2*r where r is the current packing radius
        current_min_r = _compute_radius(best_centres)
        target_dist = 2.0 * current_min_r
        
        # For efficiency, only compute forces for pairs that are close
        # Use a cutoff: only pairs within 3*target_dist
        cutoff = 3.0 * target_dist
        close_mask = (dists < cutoff) & (dists > 1e-10)
        
        for i in range(N):
            close_j = np.where(close_mask[i])[0]
            for j in close_j:
                d = dists[i, j]
                if d < 1e-10:
                    continue
                # Repulsive force: stronger when closer
                # Force magnitude ~ (target_dist/d)^2 - 1 when d < target_dist
                if d < target_dist:
                    f_mag = force_scale * ((target_dist / d) ** 2 - 1.0)
                else:
                    f_mag = 0.0
                
                # Direction from j to i
                diff = best_centres[i] - best_centres[j]
                f_dir = diff / d
                forces[i] += f_mag * f_dir
        
        # Wall forces: push circles away from walls
        wall_margin = 0.02  # Start pushing when within this distance of wall
        for i in range(N):
            for axis in range(2):
                pos = best_centres[i, axis]
                # Distance to nearest wall on this axis
                if pos < 0:
                    wall_dist = 0.5 + pos  # Distance to left/bottom wall
                else:
                    wall_dist = 0.5 - pos  # Distance to right/top wall
                
                if wall_dist < wall_margin:
                    # Force pushing away from wall
                    f_mag = force_scale * ((wall_margin / max(wall_dist, 1e-10)) ** 2 - 1.0)
                    if pos < 0:
                        forces[i, axis] += f_mag  # Push right/up
                    else:
                        forces[i, axis] -= f_mag  # Push left/down
        
        # Update velocities with damping
        velocities = damping * velocities + forces * 0.001
        velocities = np.clip(velocities, -0.01, 0.01)
        
        # Update positions
        new_centres = best_centres + velocities
        
        # Clamp to square
        new_centres = np.clip(new_centres, -0.49, 0.49)
        
        # Check if this is an improvement
        new_r = _compute_radius(new_centres)
        if new_r > best_r + 1e-12:
            best_r = new_r
            best_centres = new_centres
            # Reduce force scale slightly on improvement
            force_scale *= 0.98
        else:
            # No improvement, reduce force scale more aggressively
            force_scale *= 0.95
        
        # If force scale is too small, break
        if force_scale < 1e-6:
            break
    
    return best_centres, best_r


def _local_optimization(centres, max_iters=500, step_size=0.01):
    """Local optimization using gradient-free method with constraint handling."""
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 5)  # Leave 5s buffer
    
    try:
        from scipy.spatial.distance import cdist
    except ImportError:
        cdist = None
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Calculate constraints
        wall_dists = np.min(np.stack([
            0.5 - best_centres[:, 0],
            0.5 + best_centres[:, 0],
            0.5 - best_centres[:, 1],
            0.5 + best_centres[:, 1]
        ], axis=1), axis=1)
        
        if cdist is not None:
            dists = cdist(best_centres, best_centres)
            np.fill_diagonal(dists, np.inf)
            min_pair_dists = np.min(dists, axis=1)
        else:
            min_pair_dists = np.full(N, np.inf)
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(best_centres[i] - best_centres[j])
                    min_pair_dists[i] = min(min_pair_dists[i], d)
                    min_pair_dists[j] = min(min_pair_dists[j], d)
        
        constraints = np.minimum(min_pair_dists / 2.0, wall_dists)
        idx = np.argmin(constraints)
        
        # Try perturbations on the most constrained point
        improved = False
        n_tries = 10
        for _ in range(n_tries):
            if time.time() > deadline:
                break
            
            # Random direction
            angle = np.random.uniform(0, 2 * np.pi)
            delta = step_size * np.array([np.cos(angle), np.sin(angle)])
            new_centres = best_centres.copy()
            new_centres[idx] += delta
            
            # Check bounds
            if np.any(new_centres < -0.5) or np.any(new_centres > 0.5):
                continue
            
            new_r = _compute_radius(new_centres)
            if new_r > best_r + 1e-12:
                best_r = new_r
                best_centres = new_centres
                improved = True
                # Reduce step size slightly on improvement
                step_size *= 0.95
                break
        
        if not improved:
            # No improvement, try smaller step
            step_size *= 0.9
            if step_size < 1e-6:
                break
    
    return best_centres, best_r


def construct():
    start_time = time.time()
    
    # Try multiple initializations and pick the best
    initializations = []
    
    # 1. Fibonacci lattice (sunflower pattern)
    fib_centres = _fibonacci_lattice(N)
    fib_r = _compute_radius(fib_centres)
    initializations.append((fib_centres, fib_r))
    
    # 2. Hexagonal packing
    hex_centres = _hexagonal_packing()
    hex_r = _compute_radius(hex_centres)
    initializations.append((hex_centres, hex_r))
    
    # 3. Grid packing
    grid_centres = _grid_packing()
    grid_r = _compute_radius(grid_centres)
    initializations.append((grid_centres, grid_r))
    
    # Pick the best initialization
    initializations.sort(key=lambda x: x[1], reverse=True)
    centres, current_r = initializations[0]
    
    # Force-based refinement
    if time.time() - start_time < TIME_LIMIT - 15:
        centres, current_r = _force_based_refinement(centres, time_budget=TIME_LIMIT - 20)
    
    # Local optimization (fine-tuning)
    if time.time() - start_time < TIME_LIMIT - 5:
        centres, current_r = _local_optimization(centres, max_iters=200)
    
    # Final check
    final_r = _compute_radius(centres)
    if final_r <= 0:
        # Fallback to a valid grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return centres
