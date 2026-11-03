import numpy as np
import time

N = 300

def compute_radius_and_bottleneck(points):
    """
    Computes the maximum common radius and identifies the bottleneck constraint.
    Returns:
    r: The radius
    b_type: 'pair' or 'bound'
    i1, i2: Indices of the bottleneck
    """
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Boundary distances
    norms = np.linalg.norm(points, axis=1)
    boundary_dists = 1.0 - norms
    
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    r_bound = np.min(boundary_dists)
    
    if r_pair < r_bound:
        # Find indices of the minimum distance pair
        # np.argmin returns the index in the flattened array
        idx = np.argmin(dists)
        i, j = np.unravel_index(idx, dists.shape)
        return r_pair, 'pair', i, j
    else:
        # Find index of the point closest to boundary
        i = np.argmin(boundary_dists)
        return r_bound, 'bound', i, -1

def project_to_unit_circle(points):
    """
    Projects points to be inside the unit circle.
    """
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # If norm > 1, scale down to 1
    # If norm <= 1, keep as is
    # Handle zero norm
    scale = np.where(norms > 1.0, 1.0 / norms, 1.0)
    return points * scale

def init_hex():
    """
    Initializes points using a hexagonal lattice.
    """
    best_points = None
    best_r = -1
    
    # Try a range of spacings
    for s in np.linspace(0.045, 0.055, 10):
        pts = []
        # Generate grid large enough to cover the circle
        for i in range(-15, 15):
            for j in range(-15, 15):
                x = i * s + (j % 2) * s * 0.5
                y = j * s * np.sqrt(3) / 2
                # Check if inside a slightly smaller circle to ensure margin
                if x*x + y*y <= (1 - s/2)**2:
                    pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) >= N:
            # Randomly select N points to break symmetry
            np.random.shuffle(pts)
            pts = pts[:N]
            r, _, _, _ = compute_radius_and_bottleneck(pts)
            if r > best_r:
                best_r = r
                best_points = pts
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Leave 5s margin
    
    np.random.seed(42)
    current_points = init_hex()
    current_radius, b_type, i1, i2 = compute_radius_and_bottleneck(current_points)
    
    best_points = current_points.copy()
    best_radius = current_radius
    
    # SA Parameters
    T = 0.005
    alpha = 0.999
    grad_step = 0.01
    random_move_prob = 0.1
    random_move_size = 0.05
    
    while time.time() < deadline:
        # 1. Generate a candidate move based on bottleneck
        r, b_type, i1, i2 = compute_radius_and_bottleneck(current_points)
        
        new_points = current_points.copy()
        
        if b_type == 'pair':
            vec = new_points[i1] - new_points[i2]
            dist = np.linalg.norm(vec)
            if dist > 1e-9:
                unit_vec = vec / dist
                new_points[i1] += grad_step * unit_vec
                new_points[i2] -= grad_step * unit_vec
        elif b_type == 'bound':
            vec = new_points[i1]
            dist = np.linalg.norm(vec)
            if dist > 1e-9:
                unit_vec = vec / dist
                new_points[i1] -= grad_step * unit_vec
        
        new_points = project_to_unit_circle(new_points)
        new_radius, _, _, _ = compute_radius_and_bottleneck(new_points)
        
        # Acceptance criterion
        if new_radius > current_radius:
            current_points = new_points
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_points = current_points.copy()
        else:
            # Accept worse solution with probability exp((new - curr) / T)
            if np.random.rand() < np.exp((new_radius - current_radius) / T):
                current_points = new_points
                current_radius = new_radius
        
        # 2. Random move to escape local minima
        if np.random.rand() < random_move_prob:
            idx = np.random.randint(0, N)
            move = np.random.normal(0, random_move_size, size=2)
            current_points[idx] += move
            current_points = project_to_unit_circle(current_points)
            current_radius, _, _, _ = compute_radius_and_bottleneck(current_points)
            
        # 3. Cooling
        T *= alpha
        if T < 1e-6:
            T = 1e-6
            # If we have time, restart from best point with perturbation
            if time.time() < deadline - 5:
                current_points = best_points + np.random.normal(0, 0.01, size=best_points.shape)
                current_points = project_to_unit_circle(current_points)
                current_radius, _, _, _ = compute_radius_and_bottleneck(current_points)
                T = 0.005
            else:
                break

    # Final projection to ensure validity
    best_points = project_to_unit_circle(best_points)
    
    return best_points
