import numpy as np
import time

N = 100

def _compute_radius(centers):
    """Compute the actual radius for a set of centers."""
    # Distance to walls
    wall_dists = np.min(np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    from scipy.spatial.distance import pdist
    dists = pdist(centers, 'euclidean')
    min_pair = np.min(dists)
    
    return min(min_wall, min_pair / 2.0)

def _compute_min_pair_and_wall(centers):
    """Return (min_pair_dist, min_wall_dist) for faster evaluation."""
    wall_dists = np.min(np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    
    from scipy.spatial.distance import pdist
    dists = pdist(centers, 'euclidean')
    min_pair = np.min(dists)
    
    return min_pair, min_wall

def _initial_grid():
    """Create a slightly optimized initial grid."""
    # Use a 5x5x4 arrangement (100 points) which is more uniform
    # Actually, let's use a better packing: 5x5x4 grid
    # 5*5*4 = 100
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.5 + 1.0/(2*nx), 0.5 - 1.0/(2*nx), nx)
    ys = np.linspace(-0.5 + 1.0/(2*ny), 0.5 - 1.0/(2*ny), ny)
    zs = np.linspace(-0.5 + 1.0/(2*nz), 0.5 - 1.0/(2*nz), nz)
    
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid[:N]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configuration
    centers = _initial_grid().copy()
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    # Simulated annealing parameters
    T = 0.001  # Initial temperature
    T_min = 1e-8
    cooling_rate = 0.9995
    step_size = 0.01
    min_step = 1e-6
    
    from scipy.spatial.distance import cdist
    
    # Main SA loop
    while time.time() < deadline:
        # Generate a move: displace one or two random centers
        n_moves = 1 if np.random.random() < 0.7 else 2
        indices = np.random.choice(N, size=n_moves, replace=False)
        
        old_centers = centers[indices].copy()
        # Random displacement
        disp = np.random.normal(0, step_size, size=(n_moves, 3))
        centers[indices] = old_centers + disp
        
        # Clamp to valid range (soft)
        centers[indices] = np.clip(centers[indices], -0.499, 0.499)
        
        new_radius = _compute_radius(centers)
        
        # Metropolis acceptance
        if new_radius > best_radius:
            best_radius = new_radius
            best_centers = centers.copy()
            # Occasionally reduce step size when improving
            if np.random.random() < 0.3:
                step_size = max(min_step, step_size * 0.95)
        elif new_radius > best_radius - T:
            # Accept with probability
            delta = (best_radius - new_radius)
            if np.random.random() < np.exp(-delta / T):
                pass  # Accept
            else:
                # Reject
                centers[indices] = old_centers
        else:
            # Reject
            centers[indices] = old_centers
        
        # Cooling
        T = max(T_min, T * cooling_rate)
        # Occasionally reheat
        if np.random.random() < 0.001:
            T = max(T, 0.0005)
            step_size = max(step_size, 0.005)
        
        # Periodic local refinement on worst pairs
        if np.random.random() < 0.01:
            # Find the pair with minimum distance
            from scipy.spatial.distance import pdist
            dists = pdist(centers, 'euclidean')
            min_idx = np.argmin(dists)
            # Convert linear index to pair index
            n = N
            # Find i, j such that min_idx = i*(2n-i-3)/2 + j - i - 1
            # Simpler: just use cdist
            dist_matrix = cdist(centers, centers)
            np.fill_diagonal(dist_matrix, np.inf)
            min_pos = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            i, j = min_pos
            
            # Try to move i and j apart
            direction = centers[i] - centers[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
                # Move them apart slightly
                move = 0.002 * direction
                old_i = centers[i].copy()
                old_j = centers[j].copy()
                centers[i] = old_i + move
                centers[j] = old_j - move
                centers[i] = np.clip(centers[i], -0.499, 0.499)
                centers[j] = np.clip(centers[j], -0.499, 0.499)
                
                new_r = _compute_radius(centers)
                if new_r < best_radius - T:
                    centers[i] = old_i
                    centers[j] = old_j
    
    # Final refinement: try to improve the worst constraint
    # Use a few iterations of local search on the minimum-distance pair
    for _ in range(50):
        if time.time() > deadline:
            break
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(best_centers, best_centers)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pos = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        i, j = min_pos
        
        # Also check wall constraints
        wall_dists = np.min(np.abs(best_centers), axis=1)
        min_wall_idx = np.argmin(wall_dists)
        
        # Try small random perturbations on the critical points
        critical = [i, j, min_wall_idx]
        old_crit = best_centers[critical].copy()
        for _ in range(20):
            pert = np.random.normal(0, 0.001, size=(len(critical), 3))
            new_crit = old_crit + pert
            new_crit = np.clip(new_crit, -0.499, 0.499)
            test_centers = best_centers.copy()
            test_centers[critical] = new_crit
            test_r = _compute_radius(test_centers)
            if test_r > best_radius:
                best_radius = test_r
                best_centers = test_centers
                old_crit = new_crit.copy()
    
    return best_centers
