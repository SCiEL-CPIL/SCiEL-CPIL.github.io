"""Program for 200 equal circles in a unit square."""
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay

N = 200
TIME_LIMIT = 145.0  # seconds, leaving margin before 150s kill


def _fibonacci_lattice():
    """Create a Fibonacci lattice (sunflower pattern) for uniform distribution."""
    golden_ratio = (1 + np.sqrt(5)) / 2.0
    angles = 2 * np.pi * np.arange(N) / golden_ratio
    # Radius scales with sqrt to fill the square uniformly
    radii = np.sqrt(np.arange(N) / N) * 0.5
    centres = np.column_stack([
        radii * np.cos(angles),
        radii * np.sin(angles)
    ])
    return centres


def _force_based_optimization(centres, max_iters=5000, dt_initial=0.005, dt_min=1e-8):
    """
    Force-based simulation treating circles as rigid bodies with repulsive forces.
    Uses inverse-square repulsion for close pairs and linear spring for walls.
    """
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    current_centres = centres.copy()
    current_r = best_r
    
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 10)
    
    dt = dt_initial
    forces = np.zeros_like(current_centres)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute pairwise distances
        dists = cdist(current_centres, current_centres)
        np.fill_diagonal(dists, np.inf)
        
        # Find minimum distance
        min_dist = np.min(dists)
        
        # Compute repulsive forces (inverse-square law for close pairs)
        # Only consider pairs closer than 3*r to limit computation
        force_threshold = 3.0 * min_dist
        mask = dists < force_threshold
        np.fill_diagonal(mask, False)
        
        # Compute forces from close pairs
        forces = np.zeros_like(current_centres)
        for i in range(N):
            close_indices = np.where(mask[i])[0]
            if len(close_indices) > 0:
                # Inverse-square repulsion
                dists_i = dists[i, close_indices]
                # Avoid division by zero
                dists_i = np.maximum(dists_i, 1e-10)
                force_magnitudes = 1.0 / (dists_i ** 2)
                # Force direction: away from neighbor
                deltas = current_centres[close_indices] - current_centres[i]
                force_vectors = force_magnitudes[:, np.newaxis] * deltas / dists_i[:, np.newaxis]
                forces[i] = -np.sum(force_vectors, axis=0)
        
        # Wall repulsion (linear spring)
        wall_margin = 0.01
        for axis in range(2):
            # Left wall
            left_violation = np.maximum(0, (-0.5 + wall_margin) - current_centres[:, axis])
            forces[:, axis] += left_violation * 100
            # Right wall
            right_violation = np.maximum(0, current_centres[:, axis] - (0.5 - wall_margin))
            forces[:, axis] -= right_violation * 100
        
        # Damping
        damping = 0.95
        
        # Update positions
        new_centres = current_centres + dt * forces
        new_centres = new_centres * damping + current_centres * (1 - damping)
        
        # Project back to square
        new_centres = np.clip(new_centres, -0.5, 0.5)
        
        new_r = _compute_radius(new_centres)
        
        # Accept if improved or within tolerance
        if new_r >= current_r - 1e-10:
            current_centres = new_centres
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centres = current_centres.copy()
        
        # Adaptive step size
        if new_r > current_r + 1e-12:
            dt = min(dt * 1.05, 0.01)
        else:
            dt = max(dt * 0.9, dt_min)
        
        # Early termination if step size is very small
        if dt < dt_min * 2:
            break
    
    return best_centres, best_r


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
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    return min(min_dist / 2.0, min_wall)


def _hexagonal_packing():
    """Create a hexagonal packing pattern, optimized for N points."""
    best_centres = None
    best_r = -1
    
    # Try various numbers of rows
    for n_rows in range(12, 20):
        # For hexagonal packing, alternating rows have offset
        for start_offset in [0, 1]:
            for n_cols in range(12, 20):
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
                y_start = -0.5 + s * np.sqrt(3) / 2
                for row in range(n_rows):
                    y = y_start + row * s * np.sqrt(3) / 2
                    if (row + start_offset) % 2 == 0:
                        x_start = -0.5 + s / 2
                        for col in range(n_cols):
                            x = x_start + col * s
                            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                                centres.append((x, y))
                    else:
                        x_start = -0.5 + s
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
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _local_optimization(centres, max_iters=500, initial_step=0.01):
    """Local optimization using gradient-free method with constraint handling."""
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 5)
    
    step_size = initial_step
    
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
        
        dists = cdist(best_centres, best_centres)
        np.fill_diagonal(dists, np.inf)
        min_pair_dists = np.min(dists, axis=1)
        
        constraints = np.minimum(min_pair_dists / 2.0, wall_dists)
        idx = np.argmin(constraints)
        
        # Try perturbations on the most constrained point
        improved = False
        n_tries = 30
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
                step_size *= 0.95
                break
        
        if not improved:
            step_size *= 0.85
            if step_size < 1e-7:
                break
    
    return best_centres, best_r


def _simulated_annealing(centres, max_iters=1000, t_start=0.01, t_end=1e-6):
    """Simulated annealing optimization."""
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    current_centres = centres.copy()
    current_r = best_r
    
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 5)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Temperature schedule
        t = t_start * (t_end / t_start) ** (it / max_iters)
        
        # Pick a random point to move
        idx = np.random.randint(N)
        
        # Random perturbation
        angle = np.random.uniform(0, 2 * np.pi)
        magnitude = np.random.uniform(0, t)
        delta = magnitude * np.array([np.cos(angle), np.sin(angle)])
        
        new_centres = current_centres.copy()
        new_centres[idx] += delta
        
        # Check bounds
        if np.any(new_centres < -0.5) or np.any(new_centres > 0.5):
            continue
        
        new_r = _compute_radius(new_centres)
        
        # Acceptance criterion
        if new_r > current_r:
            current_centres = new_centres
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centres = current_centres.copy()
        else:
            # Accept with probability exp((new_r - current_r) / t)
            delta_r = new_r - current_r
            if np.random.random() < np.exp(delta_r / max(t, 1e-12)):
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
    candidates.append((fib_centres, fib_r))
    
    # 2. Hexagonal packing
    hex_centres = _hexagonal_packing()
    hex_r = _compute_radius(hex_centres)
    candidates.append((hex_centres, hex_r))
    
    # Pick the best initial configuration
    candidates.sort(key=lambda x: x[1], reverse=True)
    centres, current_r = candidates[0]
    
    # Force-based optimization (new)
    if time.time() - start_time < TIME_LIMIT - 20:
        centres, current_r = _force_based_optimization(centres)
    
    # Local optimization
    if time.time() - start_time < TIME_LIMIT - 10:
        centres, current_r = _local_optimization(centres)
    
    # Simulated annealing for further improvement
    if time.time() - start_time < TIME_LIMIT - 5:
        centres, current_r = _simulated_annealing(centres)
    
    # Final local optimization
    if time.time() - start_time < TIME_LIMIT - 3:
        centres, current_r = _local_optimization(centres, max_iters=50, initial_step=0.005)
    
    # Final check
    final_r = _compute_radius(centres)
    if final_r <= 0:
        # Fallback to a valid grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return centres
