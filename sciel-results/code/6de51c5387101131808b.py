import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # Initial setup: Hexagonal lattice
    # We will use multiple restarts with different random perturbations
    max_restarts = 50 # We will break early if time runs out
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # 1. Initialization
        points = hexagonal_init(N, seed=restart)
        
        # 2. Optimization
        points, score = optimize(points, deadline, restart)
        
        # 3. Update best
        if score > best_score:
            best_score = score
            best_points = points.copy()
            
            # Optional: If score is very close to theoretical max, stop?
            # Theoretical max is ~0.0531543. If we get close, we might stop.
            if best_score > 0.0531:
                # We can do a final polish or just accept
                pass

    if best_points is None:
        # Fallback: just return the initial hex lattice
        best_points = hexagonal_init(N, seed=0)
        
    return best_points

def hexagonal_init(N, seed=0):
    np.random.seed(seed)
    # Generate a hexagonal lattice covering the unit disk
    # Spacing should be roughly 2 * radius / sqrt(N)
    # Let's generate a grid and filter
    spacing = 0.05 # Initial guess, we will filter
    
    # Better: generate many points and pick the best N
    # Hex lattice basis vectors
    a = spacing
    b = a * np.sqrt(3)
    
    # Range to cover [-1, 1] x [-1, 1]
    k_max = int(1.0 / a) + 2
    
    points_list = []
    for i in range(-k_max, k_max + 1):
        for j in range(-k_max, k_max + 1):
            x = i * a + (j % 2) * a / 2.0
            y = j * b / 2.0
            if x*x + y*y <= 1.0:
                points_list.append([x, y])
                
    points = np.array(points_list)
    
    # If we have more than N points, select the N closest to the center
    # This ensures a dense packing near the center
    if len(points) >= N:
        radii = np.linalg.norm(points, axis=1)
        idx = np.argsort(radii)[:N]
        points = points[idx]
    else:
        # If not enough points, increase density or just use what we have (should not happen with small spacing)
        # For N=300, spacing 0.05 gives ~ 1000 points, so this is fine.
        # But to be safe, if len < N, we need to add random points? 
        # No, hex lattice with small spacing is dense.
        # If len(points) < N, we should reduce spacing.
        # Let's ensure we have enough.
        while len(points) < N:
             spacing *= 0.9
             points_list = []
             a = spacing
             b = a * np.sqrt(3)
             k_max = int(1.0 / a) + 2
             for i in range(-k_max, k_max + 1):
                 for j in range(-k_max, k_max + 1):
                     x = i * a + (j % 2) * a / 2.0
                     y = j * b / 2.0
                     if x*x + y*y <= 1.0:
                         points_list.append([x, y])
             points = np.array(points_list)
             if len(points) >= N:
                 radii = np.linalg.norm(points, axis=1)
                 idx = np.argsort(radii)[:N]
                 points = points[idx]
                 break
                 
    # Add small random perturbation to break symmetry and help escape local minima
    noise_scale = 0.01 * (1 + restart * 0.1) # Increase noise for later restarts
    points += np.random.normal(0, noise_scale, points.shape)
    
    # Ensure points are inside the circle
    radii = np.linalg.norm(points, axis=1)
    mask = radii > 1.0
    points[mask] /= radii[mask, None] * 1.01 # Push back inside
    
    return points

def optimize(points, deadline, restart_id):
    # Vectorized optimization
    # State: points (N, 2)
    # Objective: maximize min(pairwise_dist, boundary_dist)
    
    # Parameters
    max_iter = 2000
    step_size = 0.01
    decay = 0.99
    delta = 0.01 # Width of active constraint window
    
    current_score = calculate_score(points)
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Calculate pairwise distances
        # dists[i, j] is distance between i and j
        # Using broadcasting
        # diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        # dists = np.linalg.norm(diff, axis=2)
        
        # More efficient:
        # sq_dists = np.sum(points**2, axis=1)[:, None] + np.sum(points**2, axis=1)[None, :] - 2 * points @ points.T
        # np.maximum(sq_dists, 0) to handle numerical errors
        # dists = np.sqrt(np.maximum(sq_dists, 0))
        
        # For N=300, 300x300 matrix is fine.
        sq_dists = np.sum(points**2, axis=1)[:, None] + np.sum(points**2, axis=1)[None, :] - 2 * (points @ points.T)
        np.maximum(sq_dists, 0, out=sq_dists)
        dists = np.sqrt(sq_dists)
        
        # Boundary distances
        radii = np.linalg.norm(points, axis=1)
        b_dists = 1.0 - radii
        
        # Current min distance
        # Find min pairwise (excluding diagonal)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        min_bound_dist = np.min(b_dists)
        d_min = min(min_pair_dist, min_bound_dist)
        
        # Check for improvement
        if d_min > current_score:
            current_score = d_min
            
        # Calculate forces
        forces = np.zeros_like(points)
        
        # Pairwise repulsion for tight constraints
        # Mask: dists < d_min + delta
        # We want to push apart points that are too close.
        # Force direction: (points[i] - points[j]) / dists[i, j]
        # Magnitude: (d_min + delta - dists[i, j])
        
        # To vectorize this properly without N^2 loops in Python:
        # This is tricky because we need to sum forces from all j to i.
        
        # Let's use a simpler approximation:
        # For each pair (i,j) where dists[i,j] < d_min + delta:
        # Add force to i: + (points[i] - points[j]) * weight
        # Add force to j: - (points[i] - points[j]) * weight
        
        # We can compute the displacement vector D = points[:, None, :] - points[None, :, :]
        D = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        
        # Weights: 
        # w[i,j] = max(0, (d_min + delta - dists[i,j]))
        # But we only care about i != j.
        w = np.maximum(0, (d_min + delta - dists))
        # Zero out diagonal
        np.fill_diagonal(w, 0)
        
        # Force on i from j is w[i,j] * (points[i] - points[j]) / dists[i,j] ?
        # Actually, to maximize distance, we just need to move them apart.
        # The gradient of distance d_ij w.r.t p_i is (p_i - p_j) / d_ij.
        # So we want to move p_i in the direction of (p_i - p_j) / d_ij.
        # The magnitude of the "push" should be proportional to how much we want to increase the min distance.
        # If we treat this as a potential energy minimization where U = -sum(w * d), 
        # then force is grad U.
        
        # Let's just use the displacement vector scaled by weight.
        # F_i = sum_j w[i,j] * (p_i - p_j)
        # This is a standard repulsive force.
        
        # Calculate F_pair = w * D
        F_pair = w * D
        
        # Sum over j (axis=1) to get force on each i
        forces += np.sum(F_pair, axis=1)
        
        # Boundary forces
        # If b_dist[i] < d_min + delta, push inward.
        # Direction: -points[i] / radii[i]
        # Magnitude: (d_min + delta - b_dist[i])
        
        b_w = np.maximum(0, (d_min + delta - b_dists))
        
        # Force direction is inward: -p / |p|
        # Avoid division by zero
        safe_radii = np.where(radii > 1e-6, radii, 1.0)
        F_bound = -b_w[:, None] * (points / safe_radii[:, None])
        
        forces += F_bound
        
        # Normalize forces to prevent huge steps
        # Or just use step_size
        # Update points
        points += step_size * forces
        
        # Project back to unit circle (hard constraint)
        radii = np.linalg.norm(points, axis=1)
        mask = radii > 1.0
        if np.any(mask):
            points[mask] /= radii[mask, None]
            
        # Decay step size
        step_size *= decay
        
        # Optional: Add small noise to escape local minima
        if i % 100 == 0:
            points += np.random.normal(0, 0.001, points.shape)
            # Re-project
            radii = np.linalg.norm(points, axis=1)
            mask = radii > 1.0
            if np.any(mask):
                points[mask] /= radii[mask, None]
                
    final_score = calculate_score(points)
    return points, final_score

def calculate_score(points):
    if len(points) < 2:
        return 0.0
    # Pairwise distances
    sq_dists = np.sum(points**2, axis=1)[:, None] + np.sum(points**2, axis=1)[None, :] - 2 * (points @ points.T)
    np.maximum(sq_dists, 0, out=sq_dists)
    dists = np.sqrt(sq_dists)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Boundary
    radii = np.linalg.norm(points, axis=1)
    min_bound = np.min(1.0 - radii)
    
    return min(min_pair, min_bound) / 2.0

# EVOLVE-BLOCK-START
import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # Initial setup: Hexagonal lattice
    # We will use multiple restarts with different random perturbations
    max_restarts = 50 # We will break early if time runs out
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # 1. Initialization
        points = hexagonal_init(N, seed=restart)
        
        # 2. Optimization
        points, score = optimize(points, deadline, restart)
        
        # 3. Update best
        if score > best_score:
            best_score = score
            best_points = points.copy()
            
            # Optional: If score is very close to theoretical max, stop?
            # Theoretical max is ~0.0531543. If we get close, we might stop.
            if best_score > 0.0531:
                # We can do a final polish or just accept
                pass

    if best_points is None:
        # Fallback: just return the initial hex lattice
        best_points = hexagonal_init(N, seed=0)
        
    return best_points

def hexagonal_init(N, seed=0):
    np.random.seed(seed)
    # Generate a hexagonal lattice covering the unit disk
    # Spacing should be roughly 2 * radius / sqrt(N)
    # Let's generate a grid and filter
    spacing = 0.05 # Initial guess, we will filter
    
    # Better: generate many points and pick the best N
    # Hex lattice basis vectors
    a = spacing
    b = a * np.sqrt(3)
    
    # Range to cover [-1, 1] x [-1, 1]
    k_max = int(1.0 / a) + 2
    
    points_list = []
    for i in range(-k_max, k_max + 1):
        for j in range(-k_max, k_max + 1):
            x = i * a + (j % 2) * a / 2.0
            y = j * b / 2.0
            if x*x + y*y <= 1.0:
                points_list.append([x, y])
                
    points = np.array(points_list)
    
    # If we have more than N points, select the N closest to the center
    # This ensures a dense packing near the center
    if len(points) >= N:
        radii = np.linalg.norm(points, axis=1)
        idx = np.argsort(radii)[:N]
        points = points[idx]
    else:
        # If not enough points, increase density or just use what we have (should not happen with small spacing)
        # For N=300, spacing 0.05 gives ~ 1000 points, so this is fine.
        # But to be safe, if len < N, we need to add random points? 
        # No, hex lattice with small spacing is dense.
        # If len(points) < N, we should reduce spacing.
        # Let's ensure we have enough.
        while len(points) < N:
             spacing *= 0.9
             points_list = []
             a = spacing
             b = a * np.sqrt(3)
             k_max = int(1.0 / a) + 2
             for i in range(-k_max, k_max + 1):
                 for j in range(-k_max, k_max + 1):
                     x = i * a + (j % 2) * a / 2.0
                     y = j * b / 2.0
                     if x*x + y*y <= 1.0:
                         points_list.append([x, y])
             points = np.array(points_list)
             if len(points) >= N:
                 radii = np.linalg.norm(points, axis=1)
                 idx = np.argsort(radii)[:N]
                 points = points[idx]
                 break
                 
    # Add small random perturbation to break symmetry and help escape local minima
    noise_scale = 0.01 * (1 + restart * 0.1) # Increase noise for later restarts
    points += np.random.normal(0, noise_scale, points.shape)
    
    # Ensure points are inside the circle
    radii = np.linalg.norm(points, axis=1)
    mask = radii > 1.0
    points[mask] /= radii[mask, None] * 1.01 # Push back inside
    
    return points

def optimize(points, deadline, restart_id):
    # Vectorized optimization
    # State: points (N, 2)
    # Objective: maximize min(pairwise_dist, boundary_dist)
    
    # Parameters
    max_iter = 2000
    step_size = 0.01
    decay = 0.99
    delta = 0.01 # Width of active constraint window
    
    current_score = calculate_score(points)
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Calculate pairwise distances
        sq_dists = np.sum(points**2, axis=1)[:, None] + np.sum(points**2, axis=1)[None, :] - 2 * (points @ points.T)
        np.maximum(sq_dists, 0, out=sq_dists)
        dists = np.sqrt(sq_dists)
        
        # Boundary distances
        radii = np.linalg.norm(points, axis=1)
        b_dists = 1.0 - radii
        
        # Current min distance
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        min_bound_dist = np.min(b_dists)
        d_min = min(min_pair_dist, min_bound_dist)
        
        # Check for improvement
        if d_min > current_score:
            current_score = d_min
            
        # Calculate forces
        forces = np.zeros_like(points)
        
        # Pairwise repulsion for tight constraints
        D = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        
        # Weights: 
        w = np.maximum(0, (d_min + delta - dists))
        # Zero out diagonal
        np.fill_diagonal(w, 0)
        
        # Force on i from j is w[i,j] * (points[i] - points[j])
        F_pair = w * D
        
        # Sum over j (axis=1) to get force on each i
        forces += np.sum(F_pair, axis=1)
        
        # Boundary forces
        b_w = np.maximum(0, (d_min + delta - b_dists))
        
        # Force direction is inward: -p / |p|
        safe_radii = np.where(radii > 1e-6, radii, 1.0)
        F_bound = -b_w[:, None] * (points / safe_radii[:, None])
        
        forces += F_bound
        
        # Update points
        points += step_size * forces
        
        # Project back to unit circle (hard constraint)
        radii = np.linalg.norm(points, axis=1)
        mask = radii > 1.0
        if np.any(mask):
            points[mask] /= radii[mask, None]
            
        # Decay step size
        step_size *= decay
        
        # Optional: Add small noise to escape local minima
        if i % 100 == 0:
            points += np.random.normal(0, 0.001, points.shape)
            # Re-project
            radii = np.linalg.norm(points, axis=1)
            mask = radii > 1.0
            if np.any(mask):
                points[mask] /= radii[mask, None]
                
    final_score = calculate_score(points)
    return points, final_score

def calculate_score(points):
    if len(points) < 2:
        return 0.0
    # Pairwise distances
    sq_dists = np.sum(points**2, axis=1)[:, None] + np.sum(points**2, axis=1)[None, :] - 2 * (points @ points.T)
    np.maximum(sq_dists, 0, out=sq_dists)
    dists = np.sqrt(sq_dists)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Boundary
    radii = np.linalg.norm(points, axis=1)
    min_bound = np.min(1.0 - radii)
    
    return min(min_pair, min_bound) / 2.0
# EVOLVE-BLOCK-END
