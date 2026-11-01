import numpy as np
import time

N = 50

def _score(points):
    """
    Calculate the packing radius for the given points.
    Returns the minimum of:
    1. Half the minimum distance between any two points.
    2. The minimum distance from any point to the boundary (1 - norm(point)).
    """
    if len(points) < 2:
        return 1.0 - np.max(np.linalg.norm(points, axis=1))
    
    # Distances between all pairs
    # Use broadcasting: (N, 1, 2) - (1, N, 2) -> (N, N, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Set diagonal to infinity to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    # Distances to boundary
    norms = np.linalg.norm(points, axis=1)
    min_boundary_dist = np.min(1.0 - norms)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)

def _optimize(points, time_limit):
    """
    Optimize the packing using a force-directed approach.
    """
    start_time = time.time()
    points = points.copy()
    n = len(points)
    
    # Initial radius estimate
    r = _score(points)
    
    # Parameters
    step_size = 0.01
    decay_rate = 0.999
    r_increase = 0.001
    max_iter = 5000
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Calculate forces
        forces = np.zeros_like(points)
        
        # Pairwise repulsion
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Overlap condition: dist < 2 * r
        overlaps = dists < 2 * r
        # Calculate force magnitude
        # Only consider overlapping pairs
        # We want to push them apart
        # Force direction is normalized diff
        # Force magnitude proportional to overlap depth
        
        # To avoid division by zero and handle only overlaps efficiently:
        # Create a mask of overlaps
        mask = overlaps & (dists > 1e-9)
        
        if np.any(mask):
            # Normalized vectors
            norm_dists = dists.copy()
            norm_dists[~mask] = 1.0 # Avoid div by zero for non-overlaps
            norm_vecs = diff / norm_dists[:, :, np.newaxis]
            
            # Overlap depth
            depth = np.where(mask, 2 * r - dists, 0.0)
            
            # Sum forces
            # Force on i from j is in direction (P_i - P_j) if P_i is pushed away from P_j?
            # Wait, diff is P_i - P_j. If P_i is close to P_j, we want to increase distance.
            # So force on i should be in direction of (P_i - P_j).
            # Force on j should be in direction of (P_j - P_i).
            # Summing diff * depth over j for a fixed i gives the net force on i.
            
            force_contrib = norm_vecs * depth[:, :, np.newaxis]
            forces += np.sum(force_contrib, axis=1)
            
        # Boundary repulsion
        norms = np.linalg.norm(points, axis=1)
        boundary_overlap = norms > (1.0 - r)
        if np.any(boundary_overlap):
            # Force towards center
            # Direction: -P_i / |P_i|
            # Magnitude: |P_i| - (1 - r)
            boundary_depth = np.where(boundary_overlap, norms - (1.0 - r), 0.0)
            boundary_norms = np.where(boundary_overlap, norms, 1.0)
            boundary_forces = -points / boundary_norms[:, np.newaxis] * boundary_depth[:, np.newaxis]
            forces += boundary_forces
            
        # Update positions
        points += step_size * forces
        
        # Decay step size
        step_size *= decay_rate
        
        # Check if we can increase r
        # If the maximum force is small, the system is stable
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force < 1e-5:
            # Try to increase r
            r += r_increase
            step_size = max(step_size, 0.005) # Reheat slightly
        else:
            # If unstable, maybe decrease r slightly to allow relaxation
            # Or just let it run with current r
            pass
            
    return points

def construct():
    start_time = time.time()
    deadline = 140.0 # Leave 10s margin for 150s limit
    
    best_points = None
    best_score = -1.0
    
    # Number of restarts depends on time available
    # Each restart takes some time. Let's estimate 2-5 seconds per good restart.
    # We can run many quick restarts or fewer long ones.
    # Let's try a strategy: 
    # 1. One long optimization from a good initial guess (hexagonal).
    # 2. Multiple random restarts with shorter optimization times.
    
    # Initial Guess 1: Hexagonal Lattice
    def hex_init():
        points = []
        # Generate hex lattice points
        # Spacing roughly 2 * r_est
        # r_est ~ 0.12, so spacing ~ 0.24
        # Let's use a grid and filter
        for i in range(-10, 11):
            for j in range(-10, 11):
                x = i * 0.25 + (j % 2) * 0.125
                y = j * 0.2165 # 0.25 * sqrt(3)/2
                if np.hypot(x, y) < 0.9:
                    points.append([x, y])
        points = np.array(points)
        # If too many, random subsample; if too few, add random
        if len(points) > N:
            indices = np.random.choice(len(points), N, replace=False)
            points = points[indices]
        elif len(points) < N:
            # Add random points
            while len(points) < N:
                p = np.random.uniform(-0.9, 0.9, 2)
                if np.hypot(p[0], p[1]) < 0.9:
                    points = np.vstack([points, p])
        return points
    
    # Initial Guess 2: Random
    def random_init():
        points = []
        while len(points) < N:
            p = np.random.uniform(-0.95, 0.95, 2)
            if np.hypot(p[0], p[1]) < 0.95:
                points.append(p)
        return np.array(points)
    
    # Run one long optimization from hex init
    init_points = hex_init()
    time_left = deadline - (time.time() - start_time)
    if time_left > 5:
        opt_points = _optimize(init_points, time_left - 2)
        score = _score(opt_points)
        if score > best_score:
            best_score = score
            best_points = opt_points
    
    # Run multiple random restarts
    # Estimate time per restart
    # Let's try to fit as many 10-second optimizations as possible
    while time.time() - start_time < deadline - 12:
        init_points = random_init()
        # Shorter optimization time for random restarts
        time_for_restart = 8.0
        opt_points = _optimize(init_points, time_for_restart)
        score = _score(opt_points)
        if score > best_score:
            best_score = score
            best_points = opt_points
            
    return best_points

# EVOLVE-BLOCK-START
import numpy as np
import time

N = 50

def _score(points):
    """
    Calculate the packing radius for the given points.
    Returns the minimum of:
    1. Half the minimum distance between any two points.
    2. The minimum distance from any point to the boundary (1 - norm(point)).
    """
    if len(points) < 2:
        return 1.0 - np.max(np.linalg.norm(points, axis=1))
    
    # Distances between all pairs
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Set diagonal to infinity to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    # Distances to boundary
    norms = np.linalg.norm(points, axis=1)
    min_boundary_dist = np.min(1.0 - norms)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)

def _optimize(points, time_limit):
    """
    Optimize the packing using a force-directed approach.
    """
    start_time = time.time()
    points = points.copy()
    n = len(points)
    
    # Initial radius estimate
    r = _score(points)
    
    # Parameters
    step_size = 0.02
    decay_rate = 0.9995
    r_increase = 0.002
    max_iter = 20000 # High cap, time limit is the real stopper
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Calculate forces
        forces = np.zeros_like(points)
        
        # Pairwise repulsion
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Overlap condition: dist < 2 * r
        overlaps = dists < 2 * r
        mask = overlaps & (dists > 1e-9)
        
        if np.any(mask):
            norm_dists = dists.copy()
            norm_dists[~mask] = 1.0
            norm_vecs = diff / norm_dists[:, :, np.newaxis]
            
            depth = np.where(mask, 2 * r - dists, 0.0)
            
            force_contrib = norm_vecs * depth[:, :, np.newaxis]
            forces += np.sum(force_contrib, axis=1)
            
        # Boundary repulsion
        norms = np.linalg.norm(points, axis=1)
        boundary_overlap = norms > (1.0 - r)
        if np.any(boundary_overlap):
            boundary_depth = np.where(boundary_overlap, norms - (1.0 - r), 0.0)
            boundary_norms = np.where(boundary_overlap, norms, 1.0)
            boundary_forces = -points / boundary_norms[:, np.newaxis] * boundary_depth[:, np.newaxis]
            forces += boundary_forces
            
        # Update positions
        points += step_size * forces
        
        # Decay step size
        step_size *= decay_rate
        
        # Check stability to increase r
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force < 1e-4:
            r += r_increase
            step_size = max(step_size, 0.005)
        else:
            # If stuck, maybe add a tiny random jitter to escape local minima
            if i % 1000 == 0:
                 points += np.random.normal(0, 0.001, points.shape)

    return points

def construct():
    start_time = time.time()
    deadline = 140.0 # Leave 10s margin for 150s limit
    
    best_points = None
    best_score = -1.0
    
    # Initial Guess 1: Hexagonal Lattice
    def hex_init():
        points = []
        # Spacing roughly 0.25
        for i in range(-10, 11):
            for j in range(-10, 11):
                x = i * 0.25 + (j % 2) * 0.125
                y = j * 0.2165
                if np.hypot(x, y) < 0.9:
                    points.append([x, y])
        points = np.array(points)
        if len(points) > N:
            indices = np.random.choice(len(points), N, replace=False)
            points = points[indices]
        elif len(points) < N:
            while len(points) < N:
                p = np.random.uniform(-0.9, 0.9, 2)
                if np.hypot(p[0], p[1]) < 0.9:
                    points = np.vstack([points, p])
        return points
    
    # Initial Guess 2: Random
    def random_init():
        points = []
        while len(points) < N:
            p = np.random.uniform(-0.95, 0.95, 2)
            if np.hypot(p[0], p[1]) < 0.95:
                points.append(p)
        return np.array(points)
    
    # Run one long optimization from hex init
    init_points = hex_init()
    time_left = deadline - (time.time() - start_time)
    if time_left > 5:
        opt_points = _optimize(init_points, time_left - 2)
        score = _score(opt_points)
        if score > best_score:
            best_score = score
            best_points = opt_points
    
    # Run multiple random restarts
    while time.time() - start_time < deadline - 10:
        init_points = random_init()
        time_for_restart = 8.0
        opt_points = _optimize(init_points, time_for_restart)
        score = _score(opt_points)
        if score > best_score:
            best_score = score
            best_points = opt_points
            
    return best_points
# EVOLVE-BLOCK-END
