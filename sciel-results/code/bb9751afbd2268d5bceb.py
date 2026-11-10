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


def _fibonacci_lattice():
    """Create a Fibonacci/sunflower lattice for uniform distribution."""
    n = N
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))  # ~2.39996 rad
    # Sunflower pattern: r = sqrt(i/n) * R, theta = i * golden_angle
    # Scale to fit in [-0.5, 0.5]^2
    # The maximum radius in the square is 0.5/sqrt(2) for corner, but we want to fill the square
    # Use a scaling that fits within the square
    R = 0.5 / np.sqrt(2.0)  # radius to corner, but we'll scale differently
    
    # Better: use the fact that for n points, max r ~ sqrt(n) * spacing
    # For a square of side 1, we want points distributed with ~1/sqrt(n) spacing
    # Let's use r_i = sqrt((i + 0.5)/n) * 0.5 * sqrt(2) to fill the square better
    # Actually, let's just use r = sqrt(i/n) * 0.5 and theta = i * golden_angle
    # Then clip to square
    
    indices = np.arange(n)
    r = np.sqrt((indices + 0.5) / n) * 0.5
    theta = indices * golden_angle
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    centres = np.column_stack([x, y])
    
    # Clip to square (should already be inside if r <= 0.5)
    # But some points may be outside if r*cos(theta) > 0.5, which shouldn't happen for r<=0.5
    # Actually for r=0.5, x can be up to 0.5, so we're fine
    
    return centres


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


def _force_based_refinement(centres, max_iters=3000, dt=0.001):
    """
    Force-based simulation: treat circles as rigid bodies with repulsive forces.
    Each point experiences forces from:
    1. Repulsion from other points (inverse distance squared, capped)
    2. Repulsion from walls (when too close)
    3. Gentle attraction toward center to prevent drift
    """
    best_centres = centres.copy()
    current_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    current_r = _compute_radius(current_centres)
    
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 10)  # Leave 10s buffer for other work
    
    try:
        from scipy.spatial.distance import cdist
    except ImportError:
        cdist = None
    
    # Parameters
    repulsion_strength = 0.1
    wall_repulsion_strength = 0.05
    center_attraction = 0.001
    damping = 0.9  # Velocity damping
    
    velocities = np.zeros_like(current_centres)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(current_centres)
        
        # Pairwise repulsion
        if cdist is not None:
            dists = cdist(current_centres, current_centres)
            np.fill_diagonal(dists, np.inf)
            # Only consider close pairs for efficiency
            close_mask = dists < 0.15
            # For each point, compute force from close neighbors
            for i in range(N):
                close_idx = np.where(close_mask[i] & (dists[i] < 0.15))[0]
                if len(close_idx) > 0:
                    for j in close_idx:
                        d = dists[i, j]
                        if d > 1e-10:
                            # Repulsive force: k / d^2, direction away from j
                            force_mag = repulsion_strength / (d * d)
                            force_dir = (current_centres[i] - current_centres[j]) / d
                            forces[i] += force_mag * force_dir
        else:
            # Fallback: only check nearest neighbors via random sampling
            for i in range(N):
                # Sample 20 random other points
                for _ in range(20):
                    j = np.random.randint(N)
                    if j == i:
                        continue
                    d = np.linalg.norm(current_centres[i] - current_centres[j])
                    if d < 0.15 and d > 1e-10:
                        force_mag = repulsion_strength / (d * d)
                        force_dir = (current_centres[i] - current_centres[j]) / d
                        forces[i] += force_mag * force_dir
        
        # Wall repulsion
        for axis in range(2):
            # Left wall
            dist_left = current_centres[:, axis] + 0.5
            close_left = dist_left < 0.05
            forces[close_left, axis] += wall_repulsion_strength / (dist_left[close_left] + 1e-10)**2
            # Right wall
            dist_right = 0.5 - current_centres[:, axis]
            close_right = dist_right < 0.05
            forces[close_right, axis] -= wall_repulsion_strength / (dist_right[close_right] + 1e-10)**2
        
        # Gentle center attraction to prevent drift
        forces -= center_attraction * current_centres
        
        # Update velocities with damping
        velocities = damping * velocities + dt * forces
        velocities *= 0.95  # Extra damping to prevent oscillation
        
        # Cap velocity to prevent large jumps
        max_vel = 0.01
        vel_mags = np.linalg.norm(velocities, axis=1)
        scale = np.where(vel_mags > max_vel, max_vel / (vel_mags + 1e-10), 1.0)
        velocities *= scale[:, np.newaxis]
        
        # Update positions
        new_centres = current_centres + velocities
        
        # Enforce bounds strictly
        new_centres = np.clip(new_centres, -0.499, 0.499)
        
        new_r = _compute_radius(new_centres)
        
        if new_r >= current_r:
            current_centres = new_centres
            current_r = new_r
            if new_r > best_r:
                best_centres = new_centres.copy()
                best_r = new_r
        else:
            # Reject move, keep velocities for next step but damp more
            velocities *= 0.5
    
    return best_centres, best_r


def _local_optimization(centres, max_iters=500, step_size=0.02):
    """
    Simulated Annealing for local optimization.
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
        if np.random.rand() < 0.8:
            idx = np.random.randint(N)
        else:
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
        current_step = step_size * (1.0 + T / T_start)
        angle = np.random.uniform(0, 2 * np.pi)
        delta = current_step * np.array([np.cos(angle), np.sin(angle)])
        
        new_centres = current_centres.copy()
        new_centres[idx] += delta
        
        if np.any(new_centres < -0.5) or np.any(new_centres > 0.5):
            continue
        
        new_r = _compute_radius(new_centres)
        
        if new_r > current_r:
            current_centres = new_centres
            current_r = new_r
            if new_r > best_r:
                best_centres = new_centres
                best_r = new_r
        else:
            delta_r = new_r - current_r
            if delta_r < 0:
                prob = np.exp(delta_r / T)
                if np.random.rand() < prob:
                    current_centres = new_centres
                    current_r = new_r
    
    return best_centres, best_r


def construct():
    start_time = time.time()
    
    # Try multiple initialization strategies and keep the best
    candidates = []
    
    # 1. Fibonacci lattice
    fib_centres = _fibonacci_lattice()
    fib_r = _compute_radius(fib_centres)
    if fib_r > 0:
        candidates.append((fib_r, fib_centres))
    
    # 2. Hexagonal packing
    hex_centres = _hexagonal_packing()
    hex_r = _compute_radius(hex_centres)
    if hex_r > 0:
        candidates.append((hex_r, hex_centres))
    
    # 3. Grid packing as fallback
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid_centres = grid[:N]
    grid_r = _compute_radius(grid_centres)
    if grid_r > 0:
        candidates.append((grid_r, grid_centres))
    
    # Sort by initial radius (best first)
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_overall_centres = None
    best_overall_r = -1
    
    # For each candidate, apply force-based refinement then SA
    for init_r, init_centres in candidates:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        
        # Force-based refinement
        if time.time() - start_time < TIME_LIMIT - 30:
            refined_centres, refined_r = _force_based_refinement(init_centres)
            if refined_r > best_overall_r:
                best_overall_centres = refined_centres
                best_overall_r = refined_r
        else:
            refined_centres = init_centres
            refined_r = init_r
        
        # SA refinement
        if time.time() - start_time < TIME_LIMIT - 10:
            sa_centres, sa_r = _local_optimization(refined_centres)
            if sa_r > best_overall_r:
                best_overall_centres = sa_centres
                best_overall_r = sa_r
    
    # Final check
    if best_overall_centres is None or best_overall_r <= 0:
        return grid_centres
    
    final_r = _compute_radius(best_overall_centres)
    if final_r <= 0:
        return grid_centres
    
    return best_overall_centres
# EVOLVE-BLOCK-END
