"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 115s safety margin before 120s limit

    def compute_radius(pts):
        """
        Computes the maximum radius for the given points.
        r = min( min_{i<j} dist(i,j)/2, min_i (1 - dist(i, origin)) )
        """
        if len(pts) != N:
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        
        # Distance to boundary
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        
        # Pairwise distances
        # Using cKDTree for efficiency if N is large, but for N=300 direct calculation is fast enough
        # However, cKDTree.query is very fast for nearest neighbor.
        # We need the global minimum distance.
        # Let's stick to vectorized calculation for simplicity and correctness, it's O(N^2) but N=300 is small.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return min(min_pair, min_boundary)

    def generate_hex_lattice(r):
        """Generate points on a hexagonal lattice with spacing 2r that fit inside the unit circle."""
        d = 2.0 * r
        if d <= 0:
            return np.array([])
        
        # Range of indices to cover the unit circle
        k = int(1.0 / d) + 2
        
        points = []
        sqrt3 = np.sqrt(3.0)
        
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * d
                y = j * d * sqrt3 / 2.0
                dist = np.hypot(x, y)
                # Ensure the circle fits inside the unit circle
                if dist + r <= 1.0:
                    points.append([x, y])
        
        if len(points) == 0:
            return np.array([])
            
        points = np.array(points)
        
        if len(points) >= N:
            dists = np.hypot(points[:, 0], points[:, 1])
            indices = np.argsort(dists, kind="stable")[:N]
            return points[indices]
        else:
            return points

    # 1. Initialization: Find a good hexagonal lattice starting point
    # Binary search for the largest r where hex lattice yields >= N points
    r_low = 0.01
    r_high = 0.08
    
    # Ensure r_low gives enough points
    while True:
        pts = generate_hex_lattice(r_low)
        if len(pts) >= N:
            break
        r_low *= 0.8
        
    # Binary search for max r in hex lattice
    for _ in range(20):
        r_mid = (r_low + r_high) / 2.0
        pts = generate_hex_lattice(r_mid)
        if len(pts) >= N:
            r_low = r_mid
        else:
            r_high = r_mid
            
    # r_low is the largest radius for which hex lattice works directly.
    points = generate_hex_lattice(r_low)
    
    # Fallback if something went wrong
    if len(points) < N:
        angles = np.random.rand(N) * 2 * np.pi
        radii = np.sqrt(np.random.rand(N)) * 0.5
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        points = points[:N] # Ensure shape

    # Ensure we have exactly N points
    if len(points) > N:
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists, kind="stable")[:N]
        points = points[indices]
        
    best_points = points.copy()
    best_r = compute_radius(best_points)
    
    # 2. Optimization: Force-based relaxation / Local Search
    # We use a soft-min approach or simple greedy moves.
    # Given the time limit, a simple local search with random perturbations and gradient-like steps is robust.
    
    # Pre-compute some constants for speed
    current_r = best_r
    
    # Optimization loop
    # We will try to increase the radius by moving points.
    # Strategy: 
    # 1. Compute current r.
    # 2. Identify constraints: pairs with dist < 2r and points with 1-dist < r.
    # 3. Apply corrective forces.
    # 4. Alternatively, use random perturbations that increase the minimum distance.
    
    # Let's implement a simple "push apart" relaxation.
    # For each point, calculate the force vector.
    # Force from other points: if dist < 2r, push apart.
    # Force from boundary: if 1-dist < r, push in.
    
    # To make it faster and more effective, we can use a simulated annealing style or just hill climbing.
    # Given the smoothness of the objective (min of linear functions), gradient ascent on a smoothed version might work.
    # But simple local search is often sufficient for these packing problems if initialized well.
    
    max_iters = 20000
    step_size = 0.01
    
    # We will iterate, picking a random point, perturbing it, and accepting if score improves.
    # To speed up, we can only compute the score locally if possible, but global min makes it hard.
    # However, N=300 is small. Computing score takes ~1-2ms.
    # 20000 iters * 2ms = 40s. This is within budget.
    
    # Let's refine the perturbation strategy.
    # Instead of pure random, use gradient of the "soft min" or just the direction that increases the closest constraint.
    
    # A better approach: 
    # 1. Find the "bottleneck" constraints (pairs and boundaries) that define the current r.
    # 2. Move points in the direction that relaxes these constraints.
    
    # Let's implement a hybrid:
    # - Calculate all pairwise distances and boundary distances.
    # - Find the minimums.
    # - For points involved in the minimum constraints, apply a small step in the direction that increases the distance.
    
    # However, calculating all pairs every iteration is expensive if we do it naively in Python loops.
    # Vectorized numpy is fine.
    
    # Let's try a simple local search with adaptive step size.
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
            
        # Compute current score and identify active constraints
        # This is O(N^2)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        radii = np.hypot(points[:, 0], points[:, 1])
        boundary_dists = 1.0 - radii
        
        min_pair = np.min(dists)
        min_boundary = np.min(boundary_dists)
        current_r = min(min_pair / 2.0, min_boundary)
        
        # Update best if improved
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
            
        # Identify which constraints are active (within a small epsilon of the minimum)
        eps = 1e-5
        active_pairs = np.argwhere(dists < min_pair + eps)
        active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]] # Unique pairs
        
        active_boundaries = np.where(boundary_dists < min_boundary + eps)[0]
        
        # If no active constraints found (shouldn't happen), break
        if len(active_pairs) == 0 and len(active_boundaries) == 0:
            break
            
        # Select a random active constraint to resolve
        # This is a stochastic gradient descent-like approach
        
        # Generate a small random perturbation for a subset of points involved in active constraints
        # Or just pick one random point from the active set and nudge it.
        
        # Let's pick a random point from the active set
        # Create a list of candidate points
        candidates = set()
        if len(active_pairs) > 0:
            for p in active_pairs:
                candidates.add(p[0])
                candidates.add(p[1])
        if len(active_boundaries) > 0:
            for b in active_boundaries:
                candidates.add(b)
                
        if not candidates:
            candidates = set(range(N))
            
        candidate_list = list(candidates)
        idx = np.random.choice(candidate_list)
        
        # Determine direction to move
        # If the point is involved in a pair constraint, move away from the other point.
        # If involved in boundary, move towards center.
        # Combine forces.
        
        force = np.zeros(2)
        
        # Pair forces
        for p in active_pairs:
            if p[0] == idx:
                other = p[1]
                d = points[idx] - points[other]
                dist = np.linalg.norm(d)
                if dist > 0:
                    force += d / dist
            elif p[1] == idx:
                other = p[0]
                d = points[idx] - points[other]
                dist = np.linalg.norm(d)
                if dist > 0:
                    force += d / dist
                    
        # Boundary force
        if idx in active_boundaries:
            d = -points[idx] # Move towards center
            dist = np.linalg.norm(d)
            if dist > 0:
                force += d / dist
                
        # Normalize force
        norm_force = np.linalg.norm(force)
        if norm_force > 0:
            force = force / norm_force
        else:
            # Random direction if no clear force
            force = np.random.randn(2)
            force = force / np.linalg.norm(force)
            
        # Apply step
        # Adaptive step size: smaller as we get closer to optimum?
        # Or just fixed small step.
        step = 0.005
        new_points = points.copy()
        new_points[idx] += force * step
        
        # Ensure new point is inside unit circle (hard constraint for validity, though we want to maximize r)
        # If it goes outside, the score will drop, so it will be rejected.
        # But we should avoid NaNs or infinities.
        
        # Check validity of new points (basic)
        if not np.all(np.isfinite(new_points)):
            continue
            
        # Compute new score
        # To save time, we can compute the score only if the move looks promising?
        # No, we need the exact min.
        
        new_r = compute_radius(new_points)
        
        if new_r > current_r:
            points = new_points
        else:
            # If it didn't improve, maybe try a smaller step or random restart?
            # For now, just discard.
            pass
            
    return best_points
# EVOLVE-BLOCK-END
