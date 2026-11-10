import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_radius(centers):
    """
    Compute the exact radius for the given centers.
    """
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    iu = np.triu_indices(N, k=1)
    min_pair_dist = np.min(dist[iu])
    
    d_left = centers[:, 0] + 0.5
    d_right = 0.5 - centers[:, 0]
    d_bottom = centers[:, 1] + 0.5
    d_top = 0.5 - centers[:, 1]
    
    min_boundary_dist = np.min(np.concatenate([d_left, d_right, d_bottom, d_top]))
    
    r = 0.5 * min(min_pair_dist, min_boundary_dist)
    return r

def objective(x):
    """
    Objective function for Nelder-Mead.
    We want to maximize the minimum distance.
    Nelder-Mead minimizes, so we minimize -min_distance.
    """
    centers = x.reshape(N, 2)
    
    # Pairwise distances
    # Using broadcasting, this creates an N x N x 2 array, which is 50*50*2 = 5000 elements.
    # This is very fast in NumPy.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Mask diagonal (self-distance) to be large so it doesn't affect min
    # Or just take upper triangle
    iu = np.triu_indices(N, k=1)
    pair_dists = np.sqrt(dist_sq[iu])
    
    # Boundary distances
    d_left = centers[:, 0] + 0.5
    d_right = 0.5 - centers[:, 0]
    d_bottom = centers[:, 1] + 0.5
    d_top = 0.5 - centers[:, 1]
    
    b_dists = np.concatenate([d_left, d_right, d_bottom, d_top])
    
    all_dists = np.concatenate([pair_dists, b_dists])
    
    min_dist = np.min(all_dists)
    
    # Return negative to minimize
    return -min_dist

def hexagonal_init(n):
    """
    Generate a jittered hexagonal lattice initialization.
    """
    # Approximate grid size
    # Hex packing is denser, so we can fit more.
    # Let's try a grid that covers the area well.
    # For 50 circles, a 7x8 or 8x7 grid is reasonable.
    # Let's generate a hex lattice in a larger area and select the best N.
    
    # Lattice vectors for hex packing
    # a = 1, b = 1, angle = 60 deg
    # Or simply:
    # Row 0: x = 0, 1, 2, ...
    # Row 1: x = 0.5, 1.5, 2.5, ...
    # y spacing = sqrt(3)/2
    
    # We want to fit in [-0.5, 0.5]^2.
    # Let's create a grid of points.
    
    # Estimate number of rows/cols
    # Area per circle ~ 1/sqrt(3) * r^2 ? No, area of hexagon is (3*sqrt(3)/2)*s^2 where s is side length.
    # Distance between centers is s. Area = s^2 * sqrt(3)/2.
    # We want N * s^2 * sqrt(3)/2 <= 1.
    # s <= sqrt(2 / (N * sqrt(3))).
    # For N=50, s ~ sqrt(2/(50*1.732)) ~ sqrt(0.023) ~ 0.15.
    
    # Let's just generate a grid with spacing 0.15 and jitter.
    
    spacing = 0.15
    # Range -0.5 to 0.5
    # x: -0.5, -0.35, -0.2, -0.05, 0.1, 0.25, 0.4
    # This is about 7 columns.
    
    points = []
    # Generate a larger grid to ensure we have enough points
    # y from -0.5 to 0.5
    # x from -0.5 to 0.5
    
    # Let's use a structured generation
    num_rows = int(np.ceil(1.0 / (spacing * np.sqrt(3)/2))) + 2
    num_cols = int(np.ceil(1.0 / spacing)) + 2
    
    for r in range(num_rows):
        y = -0.5 + r * spacing * np.sqrt(3)/2
        if y > 0.5: break
        offset = 0
        if r % 2 == 1:
            offset = spacing / 2.0
            
        for c in range(num_cols):
            x = -0.5 + c * spacing + offset
            if x > 0.5: break
            points.append([x, y])
            
    points = np.array(points)
    
    # We have more than N points.
    # Select N points that are "best" distributed?
    # Or just take the first N?
    # Taking the first N might be biased towards top-left.
    # Let's center the grid or select randomly from the valid ones?
    # Better: Generate exactly N points in a hex pattern centered in the square.
    
    # Alternative: Generate a hex grid, keep those inside the square, 
    # if we have more than N, randomly select N?
    # Or if we have less, add random points?
    
    # Let's try a simpler robust init:
    # Jittered square grid is often very effective for Nelder-Mead.
    # But the prompt asks for hex.
    
    # Let's refine the hex generation to be centered.
    # Generate points in a hex lattice
    # Then filter to those inside [-0.49, 0.49]
    # If count > N, sample N.
    # If count < N, add random points.
    
    valid_points = points[(points[:, 0] >= -0.49) & (points[:, 0] <= 0.49) & 
                          (points[:, 1] >= -0.49) & (points[:, 1] <= 0.49)]
    
    rng = np.random.default_rng(42)
    
    if len(valid_points) >= N:
        # Randomly select N
        idx = rng.choice(len(valid_points), N, replace=False)
        centers = valid_points[idx]
    else:
        # Use all valid points and add random ones
        centers = np.vstack([valid_points, rng.uniform(-0.49, 0.49, size=(N - len(valid_points), 2))])
        
    # Add small jitter to break symmetry
    centers += rng.normal(0, 0.01, size=centers.shape)
    
    # Clip to bounds
    centers = np.clip(centers, -0.49, 0.49)
    
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_centers = None
    best_radius = -np.inf
    
    rng = np.random.default_rng(42)
    
    # Strategy:
    # 1. Start with hex init
    # 2. Run Nelder-Mead
    # 3. If time permits, do random restarts with jittered hex or random init
    
    inits = []
    
    # Init 1: Hexagonal
    inits.append(hexagonal_init(N))
    
    # Init 2: Jittered Square Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    square_init = grid[:N].copy()
    square_init += rng.normal(0, 0.02, size=square_init.shape)
    square_init = np.clip(square_init, -0.49, 0.49)
    inits.append(square_init)
    
    # Init 3: Random
    random_init = rng.uniform(-0.49, 0.49, size=(N, 2))
    inits.append(random_init)
    
    # Run Nelder-Mead on each init
    for i, init_centers in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
            
        x0 = init_centers.flatten()
        
        try:
            # Nelder-Mead options
            # maxiter: limit iterations
            # xatol, fatol: tolerance
            res = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-6,
                    'fatol': 1e-6,
                    'adaptive': True
                }
            )
            
            new_centers = res.x.reshape(N, 2)
            
            # Check validity
            if not np.all(np.isfinite(new_centers)):
                continue
                
            # Check bounds (Nelder-Mead doesn't strictly enforce bounds, but objective is -inf if outside? 
            # No, objective just returns -min_dist. If a point is outside, min_dist might be negative or small.
            # We should ensure points stay in bounds.
            # Nelder-Mead can step outside.
            # If it steps outside, the boundary distance becomes negative (e.g. x > 0.5 => 0.5 - x < 0).
            # Then min_dist is negative. Objective is -min_dist = positive.
            # This penalizes going outside.
            # However, we must return valid centers.
            
            # Clip to bounds for final check, but if it went far outside, it's bad.
            # Let's check if any point is significantly outside.
            if np.any(new_centers < -0.51) or np.any(new_centers > 0.51):
                continue
                
            new_centers = np.clip(new_centers, -0.499, 0.499)
            
            r = compute_radius(new_centers)
            
            if r > best_radius:
                best_radius = r
                best_centers = new_centers.copy()
                
        except Exception:
            pass
        
        # If we have time, do more restarts
        if time.time() > deadline - 20.0:
            break
            
        # Do 2 more restarts from the best solution found so far (if we have one)
        # Or from the current init?
        # Let's try perturbing the best solution.
        if best_centers is not None:
            for _ in range(2):
                if time.time() > deadline - 10.0:
                    break
                    
                noise = rng.normal(0, 0.05, size=best_centers.shape)
                c = best_centers + noise
                c = np.clip(c, -0.49, 0.49)
                
                x0 = c.flatten()
                try:
                    res = minimize(
                        objective,
                        x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 1000,
                            'xatol': 1e-6,
                            'fatol': 1e-6,
                            'adaptive': True
                        }
                    )
                    new_centers = res.x.reshape(N, 2)
                    if not np.all(np.isfinite(new_centers)):
                        continue
                    if np.any(new_centers < -0.51) or np.any(new_centers > 0.51):
                        continue
                    new_centers = np.clip(new_centers, -0.499, 0.499)
                    r = compute_radius(new_centers)
                    if r > best_radius:
                        best_radius = r
                        best_centers = new_centers.copy()
                except Exception:
                    pass
    
    # Fallback if no valid centers found (should not happen)
    if best_centers is None:
        best_centers = hexagonal_init(N)
        
    return best_centers
