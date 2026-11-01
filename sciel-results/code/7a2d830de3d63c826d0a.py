import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 200
HALF_SIDE = 0.5

def compute_radius(centers):
    """Compute the maximum valid radius for the given centers."""
    if len(centers) != N:
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
    pairs = tree.query_pairs(r=0.1) 
    if len(pairs) == 0:
        min_pair_dist = 1e9
    else:
        min_pair_dist = min(np.linalg.norm(centers[i] - centers[j]) for i, j in pairs)
    
    r = min(min_dist_to_wall, min_pair_dist / 2)
    return r

def objective_and_grad(x):
    centers = x.reshape(-1, 2)
    
    # Boundary constraints: dist to wall
    dist_to_wall = HALF_SIDE - np.abs(centers)
    min_wall_dist = np.min(dist_to_wall)
    
    # Pairwise constraints
    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=0.1)
    
    min_pair_dist = np.inf
    min_i, min_j = -1, -1
    
    if len(pairs) > 0:
        for i, j in pairs:
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair_dist:
                min_pair_dist = d
                min_i, min_j = i, j
    
    r = min(min_wall_dist, min_pair_dist / 2)
    
    # Gradient of -r
    grad = np.zeros_like(x)
    
    if r == min_wall_dist:
        # Gradient from wall constraint
        # r = min_k (0.5 - |c_k|)
        # Let k* be the argmin
        # grad r / grad x_k* = -sign(c_k*)
        # grad (-r) / grad x_k* = sign(c_k*)
        
        # Find the indices that achieve the min wall distance
        # dist_to_wall is (N, 2)
        # min is over all elements
        flat_idx = np.argmin(dist_to_wall)
        i = flat_idx // 2
        k = flat_idx % 2
        
        # sign(centers[i, k])
        # If center is positive, moving it right increases |c|, decreases dist.
        # dist = 0.5 - c (for c>0). d(dist)/dc = -1.
        # r = dist. dr/dc = -1.
        # f = -r. df/dc = 1.
        # So grad f at x_i,k is sign(c_i,k).
        
        grad[i * 2 + k] = np.sign(centers[i, k])
        
    elif r == min_pair_dist / 2:
        # r = d(i,j) / 2
        # d = ||c_i - c_j||
        # grad d = (c_i - c_j) / d for c_i, and -(c_i - c_j) / d for c_j
        # grad r = 0.5 * grad d
        # f = -r. grad f = -0.5 * grad d
        
        diff = centers[min_i] - centers[min_j]
        d = min_pair_dist
        if d > 1e-12:
            unit_vec = diff / d
            # grad f at i: -0.5 * unit_vec
            # grad f at j: -0.5 * (-unit_vec) = 0.5 * unit_vec
            grad[min_i * 2 : min_i * 2 + 2] = -0.5 * unit_vec
            grad[min_j * 2 : min_j * 2 + 2] = 0.5 * unit_vec
        else:
            # Degenerate case, random nudge gradient
            grad[min_i * 2 : min_i * 2 + 2] = np.random.randn(2) * 0.01
            grad[min_j * 2 : min_j * 2 + 2] = -np.random.randn(2) * 0.01
            
    else:
        # Should not happen if logic is correct, but handle inf
        pass
        
    return -r, grad

def hex_grid_init():
    """Generate a hexagonal lattice initialization."""
    # Hexagonal packing has density ~0.9069.
    # For N=200, we need a grid slightly larger than sqrt(200) ~ 14.14.
    # Let's try a few grid sizes and pick the best.
    
    best_centers = None
    best_r = 0.0
    
    for rows in range(14, 17):
        for cols in range(14, 17):
            if rows * cols < N:
                continue
            
            # Generate hex points
            points = []
            for i in range(rows):
                for j in range(cols):
                    # Hex spacing
                    # x = j * sqrt(3) * s
                    # y = i * 1.5 * s + (0 if i%2==0 else sqrt(3)/2 * s)
                    # We want to fit in [-0.5, 0.5].
                    # Let's normalize after generation.
                    x = j * np.sqrt(3)
                    y = i * 1.5 + (np.sqrt(3)/2 if i % 2 else 0)
                    points.append([x, y])
            
            points = np.array(points)
            # Center and scale
            points -= points.mean(axis=0)
            max_extent = np.max(np.abs(points))
            points /= max_extent * 1.0 # Scale to fit in [-0.5, 0.5] roughly
            # Actually, we want to fill the square.
            # The aspect ratio of the hex grid might not match the square.
            # Let's just scale to fit in the square.
            points *= 0.95 # Leave some margin
            
            # Select N points
            if len(points) > N:
                idx = np.random.choice(len(points), N, replace=False)
                points = points[idx]
            
            # Center the selected points
            points -= points.mean(axis=0)
            
            # Ensure they are within bounds
            # If not, scale down
            max_abs = np.max(np.abs(points))
            if max_abs > 0.5:
                points *= (0.5 / max_abs)
            
            r = compute_radius(points)
            if r > best_r:
                best_r = r
                best_centers = points.copy()
                
    if best_centers is None:
        # Fallback to simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_config = hex_grid_init()
    best_r = compute_radius(best_config)
    
    np.random.seed(42)
    
    # Try multiple restarts with different noise levels
    for trial in range(20):
        if time.time() > deadline - 10.0:
            break
            
        # Perturb the best config
        # Noise level decreases as we get better
        noise = 0.05 * (1 - trial / 20.0) + 0.005
        x0 = best_config + np.random.uniform(-noise, noise, size=(N, 2))
        x0 = np.clip(x0, -HALF_SIDE + 0.01, HALF_SIDE - 0.01)
        
        # Optimize
        try:
            res = minimize(
                objective_and_grad, 
                x0.flatten(), 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 500, 'maxls': 20, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            x_opt = res.x.reshape(-1, 2)
            # Clip to valid region
            x_opt = np.clip(x_opt, -HALF_SIDE, HALF_SIDE)
            
            r_opt = compute_radius(x_opt)
            
            if r_opt > best_r:
                best_r = r_opt
                best_config = x_opt.copy()
                
        except Exception:
            continue
            
    return best_config
