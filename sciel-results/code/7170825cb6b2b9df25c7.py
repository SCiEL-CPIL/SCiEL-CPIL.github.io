import numpy as np
import time
from scipy.spatial import cKDTree

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin
    
    def compute_radius(pts):
        # Min pair distance / 2
        tree = cKDTree(pts)
        pairs = tree.query_pairs(2.0)  # 2.0 is a loose bound, we just need min
        if not pairs:
            min_pair = 1.0
        else:
            # Efficiently find min distance
            # Using cdist on pairs might be heavy if many, but N=300 is small
            # However, query_pairs returns all pairs within r. 
            # Better: use pdist or just check nearest neighbors.
            # For N=300, pdist is ~45k pairs, very fast.
            from scipy.spatial.distance import pdist
            min_pair = np.min(pdist(pts))
        
        min_pair_rad = min_pair / 2.0
        min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
        return min(min_pair_rad, min_boundary)

    def hex_lattice_init(n, spacing_factor=1.0):
        # Generate hex lattice points
        # Estimate spacing to fit n points in unit circle
        # Area of unit circle = pi. Area per point approx spacing^2 * sqrt(3)/2
        # n * spacing^2 * sqrt(3)/2 ~ pi => spacing ~ sqrt(2*pi / (n*sqrt(3)))
        base_spacing = np.sqrt(2 * np.pi / (n * np.sqrt(3)))
        spacing = base_spacing * spacing_factor
        
        k = int(1.0 / spacing) + 2
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < n:
            return None
        
        # Select n points closest to center
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists)[:n]
        return points[idx].copy()

    # Multi-start strategy
    best_pts = None
    best_radius = -1.0
    
    # Try a few different initializations (hex lattice with slight perturbations or different spacings)
    num_starts = 5
    
    for start_idx in range(num_starts):
        if time.time() > deadline:
            break
            
        # Vary spacing slightly to get different initial configurations
        spacing_factor = 1.0 + 0.05 * start_idx
        pts = hex_lattice_init(N, spacing_factor)
        
        if pts is None:
            # Fallback: random points inside disk
            pts = np.random.uniform(-1, 1, (N, 2))
            r = np.hypot(pts[:, 0], pts[:, 1])
            mask = r > 1.0
            if np.any(mask):
                pts[mask] = pts[mask] / r[mask, np.newaxis] * 0.999
            # Reject if overlapping significantly? No, let optimization fix it.
        
        # Simulated Annealing / Local Search
        # Objective: maximize r = min(min_pair/2, min_boundary)
        # We want to maximize this. 
        # We can frame it as minimizing -r or minimizing sum of violations.
        # Let's use a penalty method: minimize sum of (r_target - dist)^2 for dist < r_target
        # Or simply use a greedy local search with random perturbations.
        
        # Let's use a simple hill-climbing with random kicks.
        # Step 1: Greedy relaxation to get a valid packing.
        # We want to maximize r. 
        # Let's try to increase r iteratively.
        
        current_radius = compute_radius(pts)
        current_pts = pts.copy()
        
        # Optimization loop
        max_iter = 2000
        step_size = 0.01
        temperature = 0.05
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            # Generate a random perturbation
            # Move a random subset of points
            num_moves = np.random.randint(1, 10)
            idxs = np.random.choice(N, num_moves, replace=False)
            
            new_pts = current_pts.copy()
            # Random displacement
            displacements = np.random.normal(0, step_size, (num_moves, 2))
            new_pts[idxs] += displacements
            
            # Enforce boundary
            r_new = np.hypot(new_pts[:, 0], new_pts[:, 1])
            mask = r_new > 1.0
            if np.any(mask):
                new_pts[mask] = new_pts[mask] / r_new[mask, np.newaxis] * 0.9999
            
            # Compute new radius
            new_radius = compute_radius(new_pts)
            
            # Accept if better, or with probability exp((new-old)/T)
            if new_radius > current_radius:
                current_radius = new_radius
                current_pts = new_pts
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_pts = current_pts.copy()
                # Reduce step size slowly
                step_size *= 0.999
            else:
                # Metropolis acceptance
                delta = new_radius - current_radius
                if np.random.rand() < np.exp(delta / temperature):
                    current_pts = new_pts
                    current_radius = new_radius
                    step_size *= 0.999
                else:
                    step_size *= 1.01 # Increase step size if stuck?
                    if step_size > 0.05:
                        step_size = 0.05
            
            # Cool down
            temperature *= 0.99
            
        # If we didn't beat the global best in this start, we might try a "kick" 
        # to escape local minima, but time is precious.
        
    if best_pts is None:
        # Fallback
        best_pts = hex_lattice_init(N, 1.0)
        if best_pts is None:
            best_pts = np.random.uniform(-0.5, 0.5, (N, 2))
            
    return best_pts
