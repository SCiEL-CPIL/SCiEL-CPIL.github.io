"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # 1. Initialization: Hexagonal lattice
    # We want to place N points.
    # A hex lattice with spacing s.
    # We generate points on a hex lattice and select the N closest to the origin that fit.
    
    def generate_hex_grid(n_points):
        # Estimate spacing
        # Area of unit circle = pi. Area per point ~ pi/n. Spacing ~ sqrt(pi/n) * sqrt(2/sqrt(3)) approx
        # Let's just try a range of spacings.
        best_pts = None
        best_r = 0
        for k in range(1, 50):
            spacing = 1.0 / k
            # Generate points in a grid that covers the unit circle
            max_coord = 1.0 + spacing
            x = np.arange(-max_coord, max_coord, spacing)
            y = np.arange(-max_coord, max_coord, spacing * np.sqrt(3) / 2)
            
            # Hex offset
            pts = []
            for i, yy in enumerate(y):
                offset = (i % 2) * spacing / 2
                for xx in x:
                    pts.append([xx + offset, yy])
            pts = np.array(pts)
            
            # Filter points inside unit circle (with margin)
            # We need margin for radius. Initially assume small radius.
            # We will compute radius later.
            # Filter: norm < 1.0
            norms = np.linalg.norm(pts, axis=1)
            pts = pts[norms < 1.0]
            
            if len(pts) < n_points:
                continue
                
            # Select N points.
            # To maximize min distance, we want points spread out.
            # A simple heuristic: sort by distance to center? No, that clumps them.
            # Actually, for a hex grid, the points are already regular.
            # We should select the N points that form the most compact cluster around the origin?
            # Or just take the first N?
            # Better: Randomly sample N points from the valid grid points?
            # No, deterministic is better for reproducibility.
            # Let's take the N points with the smallest distance to the origin?
            # This creates a circular blob.
            idx = np.argsort(np.linalg.norm(pts, axis=1))[:n_points]
            pts = pts[idx]
            
            # Calculate radius
            r = calc_radius(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
                
        if best_pts is None:
             # Fallback to random
             best_pts = np.random.rand(n_points, 2) * 2 - 1
             norms = np.linalg.norm(best_pts, axis=1)
             best_pts = best_pts / norms[:, np.newaxis] * 0.9
             
        return best_pts

    def calc_radius(pts):
        # Min distance to boundary
        d_boundary = 1.0 - np.linalg.norm(pts, axis=1)
        r_boundary = np.min(d_boundary)
        
        # Pairwise distances
        dists = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        r_pair = min_dist / 2.0
        
        return min(r_boundary, r_pair)

    # 2. Simulated Annealing
    start_time = time.time()
    time_limit = 110.0  # Leave some buffer
    
    points = generate_hex_grid(N)
    current_r = calc_radius(points)
    best_points = points.copy()
    best_r = current_r
    
    T = 0.05  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.999
    step_scale = 0.1 # Scale for random moves
    
    while T > T_min and (time.time() - start_time) < time_limit:
        # Random move: pick one point and perturb it
        idx = np.random.randint(N)
        old_pt = points[idx].copy()
        
        # Gaussian noise
        noise = np.random.normal(0, step_scale * T, 2)
        new_pt = old_pt + noise
        
        # Check if new point is inside circle (hard constraint for validity, but SA can handle soft)
        # If it goes outside, the radius becomes negative/invalid.
        # We can just calculate the new radius. If it's worse, we might reject.
        # But if it's invalid (outside), calc_radius will give a negative value or small value.
        # Actually, if norm > 1, d_boundary is negative. r will be negative.
        # This is fine, it will be rejected.
        
        points[idx] = new_pt
        new_r = calc_radius(points)
        
        # Acceptance
        if new_r >= current_r:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_points = points.copy()
        else:
            # Metropolis criterion
            delta = new_r - current_r
            if np.random.rand() < np.exp(delta / T):
                current_r = new_r
            else:
                points[idx] = old_pt
        
        T *= cooling_rate
        # Adapt step scale?
        # step_scale *= 0.9999 # Maybe not needed if T cools

    # 3. Local Hill Climbing (Polish)
    # After SA, do a local search to refine.
    # Try moving each point slightly in all directions?
    # Or just a few iterations of "move point to maximize local min dist".
    
    # A simple local polish:
    # For each point, try to move it to increase the min distance to neighbors and boundary.
    # This is expensive if done naively.
    # Let's do a simple gradient-free local search.
    
    polish_time_limit = 10.0
    polish_start = time.time()
    
    # Use the best points found
    points = best_points.copy()
    current_r = best_r
    
    # Random restarts of local search? No, just refine current.
    # We can try to "push" points apart locally.
    
    # A better polish: 
    # Identify the "bottleneck" constraints (pairs with min dist, points with min boundary dist).
    # Move points to relieve these constraints.
    
    # Let's just run a few iterations of "random perturbation and accept if better" with small steps.
    step_size = 0.01
    for i in range(1000):
        if time.time() - polish_start > polish_time_limit:
            break
            
        idx = np.random.randint(N)
        old_pt = points[idx].copy()
        # Small random move
        noise = np.random.normal(0, step_size, 2)
        new_pt = old_pt + noise
        
        points[idx] = new_pt
        new_r = calc_radius(points)
        
        if new_r > current_r:
            current_r = new_r
            best_r = new_r
            best_points = points.copy()
        else:
            points[idx] = old_pt
            
        # Decay step size
        step_size *= 0.999

    return best_points
# EVOLVE-BLOCK-END
