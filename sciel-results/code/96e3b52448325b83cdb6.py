# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE_SECONDS = 115.0  # Safety margin before 150s limit

def compute_radius(pts):
    """
    Computes the maximum radius for the given centers.
    pts: (N, 2) array of centers.
    Returns r such that circles of radius r are disjoint and inside [-0.5, 0.5]^2.
    """
    # 1. Distance to boundary
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    min_bound = np.min(np.minimum(d_x, d_y))
    
    # 2. Distance between centers
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_inter = np.sqrt(np.min(dist_sq))
    
    r_inter = min_dist_inter / 2.0
    r = np.minimum(r_inter, min_bound)
    
    return r

def generate_hex_initialization():
    """
    Generates a hexagonal-like initialization by taking the N closest points
    from a larger hex grid to the center.
    """
    points = []
    s = 0.15 # approximate diameter
    y_spacing = s * np.sqrt(3) / 2.0
    x_spacing = s
    
    # Generate a 10x10 grid of hex points
    for i in range(10):
        for j in range(10):
            y = (i - 4.5) * y_spacing
            x = (j - 4.5) * x_spacing
            if i % 2 != 0:
                x += x_spacing / 2.0
            points.append([x, y])
            
    points = np.array(points)
    
    # Sort by distance to center
    dists = np.sum(points**2, axis=1)
    indices = np.argsort(dists)
    
    # Take the N closest
    points = points[indices[:N]]
    
    # Scale to fit in square
    max_coord = np.max(np.abs(points))
    if max_coord > 0:
        scale = 0.45 / max_coord
        points *= scale
        
    return points

def soft_min(values, epsilon):
    """
    Smooth approximation of the minimum of an array of values.
    """
    values = np.clip(values, -100, 100)
    v_min = np.min(values)
    shifted = values - v_min
    sum_exp = np.sum(np.exp(-epsilon * shifted))
    log_sum = np.log(sum_exp)
    soft_min_val = v_min - log_sum / epsilon
    
    return soft_min_val

def objective_soft(x, epsilon):
    """
    Objective function to minimize: -soft_min(radius_candidates).
    """
    pts = x.reshape(-1, 2)
    
    # 1. Boundary distances
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    boundary_dists = np.minimum(d_x, d_y)
    
    # 2. Inter-circle half-distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    iu = np.triu_indices(N, 1)
    inter_dists_sq = dist_sq[iu]
    inter_dists = np.sqrt(inter_dists_sq) / 2.0
    
    # Combine all candidates
    all_dists = np.concatenate([boundary_dists, inter_dists])
    
    # Soft min
    r_soft = soft_min(all_dists, epsilon)
    
    # We want to maximize r, so minimize -r
    return -r_soft

def local_refinement(pts, max_iters=2000):
    """
    Simple force-directed local refinement.
    """
    pts = pts.copy()
    for _ in range(max_iters):
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        
        # Find min distance pair
        min_idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
        i, j = min_idx
        if i == j:
            break
            
        d = pts[j] - pts[i]
        dist = np.sqrt(np.sum(d*d))
        
        if dist < 1e-9:
            d = np.array([1.0, 0.0])
            dist = 1.0
            
        direction = d / dist
        
        step = 0.002
        
        new_i = pts[i] - direction * step
        new_j = pts[j] + direction * step
        
        # Check boundaries
        if np.max(np.abs(new_i)) <= 0.5 and np.max(np.abs(new_j)) <= 0.5:
            # Check if this improves the global min radius
            old_r = compute_radius(pts)
            pts[i] = new_i
            pts[j] = new_j
            new_r = compute_radius(pts)
            
            if new_r >= old_r:
                continue
            else:
                pts[i] = pts[i] + direction * step
                pts[j] = pts[j] - direction * step
        else:
            pass
            
    return pts

def optimize_from_start(start_pts, start_time, deadline):
    """
    Runs optimization starting from start_pts.
    """
    x0 = start_pts.flatten()
    best_pts = start_pts.copy()
    best_r = compute_radius(start_pts)
    
    # Annealing schedule for epsilon
    epsilons = [10, 50, 100, 500, 1000]
    
    current_x = x0
    
    for eps in epsilons:
        if time.time() > deadline - 10:
            break
            
        try:
            result = minimize(
                objective_soft, 
                current_x, 
                args=(eps,),
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            candidate_pts = result.x.reshape(-1, 2)
            candidate_pts = np.clip(candidate_pts, -0.5, 0.5)
            
            r = compute_radius(candidate_pts)
            if r > best_r:
                best_r = r
                best_pts = candidate_pts
                current_x = candidate_pts.flatten()
                
        except Exception:
            pass
            
    # Final local refinement
    if time.time() < deadline - 5:
        best_pts = local_refinement(best_pts, max_iters=500)
        
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # List of initializations
    inits = []
    
    # 1. Hexagonal (Strongest)
    inits.append(generate_hex_initialization())
    
    # 2. Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) > N:
        grid = grid[:N]
    elif len(grid) < N:
        num_pad = N - len(grid)
        random_pts = np.random.uniform(-0.4, 0.4, size=(num_pad, 2))
        grid = np.vstack([grid, random_pts])
    inits.append(grid)
    
    # 3. Random
    inits.append(np.random.uniform(-0.4, 0.4, size=(N, 2)))
    
    # 4. Perturbed Hex
    hex_pts = generate_hex_initialization()
    hex_pts += np.random.normal(0, 0.01, size=hex_pts.shape)
    hex_pts = np.clip(hex_pts, -0.49, 0.49)
    inits.append(hex_pts)
    
    best_pts = inits[0]
    best_r = compute_radius(best_pts)
    
    for i, init in enumerate(inits):
        # Check time
        elapsed = time.time() - start_time
        if elapsed > DEADLINE_SECONDS - 15: # Leave 15s for final checks
            break
            
        # Optimize
        try:
            optimized_pts = optimize_from_start(init, start_time, deadline)
            r = compute_radius(optimized_pts)
            
            if r > best_r:
                best_r = r
                best_pts = optimized_pts
                
        except Exception:
            pass
            
    # Final safety clip
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts

# EVOLVE-BLOCK-END
