import numpy as np
import time
import random

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a packing of N equal circles inside a unit circle.
    Uses Simulated Annealing starting from a hexagonal lattice.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Initialization: Hexagonal Lattice
    # We generate points on a hexagonal grid and select the best N that fit.
    # We need to estimate the spacing.
    # Area of unit circle = pi. Area per circle ~ pi/N.
    # Hexagonal packing density ~ 0.9069.
    # Approx radius r ~ sqrt(pi / (N * 0.9069)) / 2 ? 
    # Let's just generate a dense grid and filter.
    
    def generate_hex_initial():
        # Estimate spacing
        # If r is the radius, distance is 2r.
        # Let's try a few spacings to find one that yields >= N points inside.
        best_points = None
        best_r = 0
        
        # The theoretical max r is ~0.125. So 2r ~ 0.25.
        # Spacing in hex lattice is 2r.
        # Let's start with spacing 0.22 and go down.
        
        # To ensure we have enough points, we generate a large grid.
        for spacing in [0.22, 0.21, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15]:
            k = int(1.0 / spacing) + 2
            rows = []
            for i in range(-k, k + 1):
                for j in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    # Check if inside unit circle with margin
                    # We want points such that a circle of radius spacing/2 fits.
                    # So distance from origin must be <= 1 - spacing/2.
                    if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                        rows.append((x, y))
            
            if len(rows) >= N:
                points = np.array(rows)
                # Sort by distance to origin to keep the "center" of the packing dense?
                # Actually, for hex packing, we just want any N.
                # But to be symmetric, we can just take the first N.
                # Better: take the N points closest to the origin to ensure a compact core.
                dists = np.hypot(points[:, 0], points[:, 1])
                idx = np.argsort(dists, kind="stable")[:N]
                return points[idx], spacing / 2.0
        
        # Fallback: random points
        return np.random.rand(N, 2) * 2 - 1, 0.1

    points, current_r = generate_hex_initial()
    
    if len(points) != N:
         # Should not happen with the above logic, but just in case
         points = np.random.rand(N, 2) * 1.8 - 0.9
         current_r = 0.1

    # Calculate initial score (minimum distance / 2, min dist to boundary)
    def calc_score(pts):
        # Distance to boundary
        dists_boundary = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(dists_boundary)
        
        # Pairwise distances
        # N=50 is small, O(N^2) is fine.
        min_pair = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_pair:
                    min_pair = d
        
        return min(min_pair / 2.0, min_boundary)

    best_points = points.copy()
    best_score = calc_score(points)
    
    # Simulated Annealing Parameters
    T = 0.05  # Initial temperature (step size)
    T_min = 0.0001
    cooling_rate = 0.9995
    
    # Pre-compute indices for pairwise distance
    # For efficiency, we can use a list of pairs, but N=50 is small.
    
    while T > T_min:
        if time.time() > deadline:
            break
            
        # Generate a random move
        # Pick a random circle
        i = np.random.randint(N)
        
        # Random direction and magnitude
        angle = np.random.uniform(0, 2 * np.pi)
        step = T * np.random.uniform(0.5, 1.5)
        
        dx = step * np.cos(angle)
        dy = step * np.sin(angle)
        
        new_x = points[i, 0] + dx
        new_y = points[i, 1] + dy
        
        # Check if the new position is inside the unit circle
        # We allow it to be slightly outside in the calculation but penalize it, 
        # OR we reject it. 
        # Given the hard constraint "inside the container", it's safer to reject 
        # moves that violate the boundary constraint significantly, 
        # but SA works better with soft constraints or projection.
        # Let's project back if it goes outside, but calculate the score change.
        
        dist_origin = np.hypot(new_x, new_y)
        if dist_origin > 1.0:
            # Project back to the boundary (minus a small epsilon to be safe)
            # This ensures validity.
            scale = (1.0 - 1e-9) / dist_origin
            new_x *= scale
            new_y *= scale
            
        # Calculate score change
        # Only need to check distances involving point i
        old_min_boundary = 1.0 - np.hypot(points[i, 0], points[i, 1])
        new_min_boundary = 1.0 - np.hypot(new_x, new_y)
        
        # Find min distance to other points for old and new
        old_min_pair = np.inf
        new_min_pair = np.inf
        
        # Vectorized distance calculation for point i to all others
        # points[i] is shape (2,), points is (N, 2)
        # diff = points - points[i]
        # dists = np.linalg.norm(diff, axis=1)
        
        diff_old = points - points[i]
        diff_old[i] = 1e9 # Exclude self
        dists_old = np.linalg.norm(diff_old, axis=1)
        old_min_pair = np.min(dists_old)
        
        points_new = points.copy()
        points_new[i, 0] = new_x
        points_new[i, 1] = new_y
        
        diff_new = points_new - points_new[i]
        diff_new[i] = 1e9
        dists_new = np.linalg.norm(diff_new, axis=1)
        new_min_pair = np.min(dists_new)
        
        old_score = min(old_min_pair / 2.0, old_min_boundary)
        new_score = min(new_min_pair / 2.0, new_min_boundary)
        
        delta = new_score - old_score
        
        # Acceptance criterion
        if delta > 0:
            accept = True
        else:
            # Accept with probability exp(delta / T)
            # Note: delta is negative
            accept = np.random.rand() < np.exp(delta / T)
            
        if accept:
            points[i, 0] = new_x
            points[i, 1] = new_y
            if new_score > best_score:
                best_score = new_score
                best_points = points.copy()
        
        T *= cooling_rate
        
    # Final Polish: Local search with very small steps
    # Try to refine the best solution found
    T_polish = 0.001
    for _ in range(2000):
        if time.time() > deadline:
            break
        i = np.random.randint(N)
        angle = np.random.uniform(0, 2 * np.pi)
        step = T_polish * np.random.uniform(0.1, 0.5)
        dx = step * np.cos(angle)
        dy = step * np.sin(angle)
        
        new_x = best_points[i, 0] + dx
        new_y = best_points[i, 1] + dy
        
        dist_origin = np.hypot(new_x, new_y)
        if dist_origin > 1.0:
            scale = (1.0 - 1e-9) / dist_origin
            new_x *= scale
            new_y *= scale
            
        # Check validity and improvement
        # Simple check: is it better than current best?
        # We need to compute the full score or just the local improvement.
        # Given N=50, full score is fast.
        
        pts_temp = best_points.copy()
        pts_temp[i, 0] = new_x
        pts_temp[i, 1] = new_y
        
        s_temp = calc_score(pts_temp)
        if s_temp > best_score:
            best_score = s_temp
            best_points = pts_temp

    return best_points

# EVOLVE-BLOCK-END
