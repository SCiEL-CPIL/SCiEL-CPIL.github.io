"""Improved packing for 200 equal circles in a unit square using Hexagonal Init and Greedy Local Search."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree
import random

def _score(pts):
    """Compute the radius for a given set of points."""
    if pts.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check bounds
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
    
    # Compute min distance between any two points
    tree = cKDTree(pts)
    # We need the minimum distance. 
    # query_pairs(0) gives all pairs? No, query_pairs(r) gives pairs with dist <= r.
    # To find the global minimum distance efficiently without knowing it in advance:
    # We can use the fact that for N=200, O(N^2) is 40k ops, which is fast.
    # But cKDTree is safer if we want to be robust.
    # However, for N=200, a simple vectorized distance calculation is very fast in NumPy.
    # Let's stick to the vectorized method for simplicity and speed at this N.
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Compute min distance to boundary
    dists_to_boundary = np.minimum(
        np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
        np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
    )
    min_boundary_dist = np.min(dists_to_boundary)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r

def _hex_init(seed=42):
    """Create a hexagonal grid initialization."""
    rng = np.random.default_rng(seed)
    
    # Target density: 200 points in area 1.
    # Hexagonal packing density is ~0.9069.
    # Area per circle ~ 1/200 = 0.005.
    # Radius r ~ sqrt(0.005/pi) ~ 0.04.
    # Spacing d ~ 2*r ~ 0.08.
    
    # Let's generate a hex grid with spacing d=0.08 and see how many fit.
    d = 0.08
    # Hex grid coordinates
    # x = i * d + (j % 2) * d/2
    # y = j * d * sqrt(3)/2
    
    points = []
    # We need to cover [-0.5, 0.5] x [-0.5, 0.5]
    # i range: -6 to 6 (13 * 0.08 = 1.04)
    # j range: -6 to 6
    
    for j in range(-6, 7):
        for i in range(-6, 7):
            x = i * d + (j % 2) * d * 0.5
            y = j * d * np.sqrt(3) / 2
            # Center the grid
            # The grid is roughly centered at 0,0
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append((x, y))
    
    points = np.array(points)
    
    # We might have more or fewer than 200 points.
    if len(points) > N:
        # Randomly sample N points
        indices = rng.choice(len(points), N, replace=False)
        points = points[indices]
    elif len(points) < N:
        # If too few, add random points or increase density
        # For now, let's just use what we have and pad with random if necessary? 
        # Better to adjust d.
        # Let's try a slightly smaller d to get more points.
        d = 0.075
        points = []
        for j in range(-7, 8):
            for i in range(-7, 8):
                x = i * d + (j % 2) * d * 0.5
                y = j * d * np.sqrt(3) / 2
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append((x, y))
        points = np.array(points)
        if len(points) > N:
            indices = rng.choice(len(points), N, replace=False)
            points = points[indices]
        else:
            # Still not enough? Just pad with random points in the square
            while len(points) < N:
                p = rng.uniform(-0.45, 0.45, 2)
                points = np.vstack([points, [p]])
            points = points[:N]

    # Add small jitter to break symmetry
    points += rng.normal(0, 0.001, size=points.shape)
    points = np.clip(points, -0.49, 0.49)
    
    return points

def _local_search(pts, deadline, rng):
    """Perform greedy local search to improve the packing."""
    current_score = _score(pts)
    best_score = current_score
    best_pts = pts.copy()
    
    # Grid of possible moves for a single point
    # We try to move the point in small steps in x and y
    step_sizes = [0.001, 0.002, 0.005, 0.01]
    
    while time.time() < deadline:
        improved = False
        
        # Find the point with the smallest clearance
        # Clearance is min(dist to neighbors, dist to boundary)
        # We approximate this by finding the point that is closest to another point or boundary.
        # For efficiency, we can just pick a random point, or the one with min score contribution.
        # Let's compute the "bottleneck" point.
        
        # Calculate min distance to other points for each point
        tree = cKDTree(pts)
        # Query nearest neighbor for each point (excluding self)
        # k=2 gives self and nearest neighbor
        dists, indices = tree.query(pts, k=2)
        min_neighbor_dists = dists[:, 1]
        
        # Distance to boundary for each point
        dists_to_boundary = np.minimum(
            np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
            np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
        )
        
        # The limiting factor for each point is min(min_neighbor_dist/2, dist_to_boundary)
        # But the global radius is min over all points.
        # The "most constrained" point is likely the one with the smallest min_neighbor_dist or dist_to_boundary.
        
        # Let's pick the point with the smallest min_neighbor_dist
        # This is a heuristic.
        idx = np.argmin(min_neighbor_dists)
        
        # Try to move this point
        p = pts[idx].copy()
        best_move_score = current_score
        best_new_p = p
        
        for step in step_sizes:
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    if dx == 0 and dy == 0:
                        continue
                    new_p = p + np.array([dx, dy])
                    # Check bounds
                    if new_p[0] < -0.5 or new_p[0] > 0.5 or new_p[1] < -0.5 or new_p[1] > 0.5:
                        continue
                    # Try the move
                    pts_trial = pts.copy()
                    pts_trial[idx] = new_p
                    score_trial = _score(pts_trial)
                    if score_trial > best_move_score:
                        best_move_score = score_trial
                        best_new_p = new_p
        
        if best_move_score > current_score:
            pts[idx] = best_new_p
            current_score = best_move_score
            improved = True
            
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
                
        else:
            # If no improvement found for this point, try a random point to escape local minima?
            # Or just break if we've stuck?
            # Let's try a random point a few times before breaking.
            if rng.random() < 0.1:
                idx = rng.integers(0, N)
                # Repeat the move attempt for this random idx
                p = pts[idx].copy()
                best_move_score = current_score
                best_new_p = p
                
                for step in step_sizes:
                    for dx in [-step, 0, step]:
                        for dy in [-step, 0, step]:
                            if dx == 0 and dy == 0:
                                continue
                            new_p = p + np.array([dx, dy])
                            if new_p[0] < -0.5 or new_p[0] > 0.5 or new_p[1] < -0.5 or new_p[1] > 0.5:
                                continue
                            pts_trial = pts.copy()
                            pts_trial[idx] = new_p
                            score_trial = _score(pts_trial)
                            if score_trial > best_move_score:
                                best_move_score = score_trial
                                best_new_p = new_p
                
                if best_move_score > current_score:
                    pts[idx] = best_new_p
                    current_score = best_move_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_pts = pts.copy()
            else:
                # No improvement, break to save time for other inits?
                # Or continue with smaller steps?
                # For now, let's just break to allow multiple restarts.
                break
                
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_score = 0.0
    
    # Try multiple initializations
    for i in range(20): # Up to 20 restarts
        if time.time() > deadline:
            break
        
        # Alternate between hex and random inits
        if i % 2 == 0:
            pts = _hex_init(seed=42 + i)
        else:
            # Random init
            pts = rng.uniform(-0.45, 0.45, size=(N, 2))
            
        # Local search
        curr_pts, curr_score = _local_search(pts, deadline, rng)
        
        if curr_score > best_score:
            best_score = curr_score
            best_pts = curr_pts
    
    if best_pts is None:
        # Fallback to grid
        m = 15
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = grid[:N].copy()
        
    return best_pts
# EVOLVE-BLOCK-END
