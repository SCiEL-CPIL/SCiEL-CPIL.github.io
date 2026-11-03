"""Improved program for the 200-circle packing task using Delaunay-based optimization."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
import time

def _compute_radius(centres):
    """Compute the maximum common radius for given centres efficiently."""
    if len(centres) < 2:
        return float(np.min(0.5 - np.abs(centres)))
    
    # Use scipy.spatial.distance.cdist for efficient pairwise distances
    # However, for 200 points, cdist is fine. 
    # To be safe and fast, we can use Delaunay neighbors for min distance check in optimization loop,
    # but for final score we need exact min.
    from scipy.spatial.distance import cdist
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    min_dist_to_border = np.min(0.5 - np.abs(centres))
    
    return float(min(min_pair_dist / 2.0, min_dist_to_border))

def _get_delaunay_neighbors(tri):
    """Get a list of neighbors for each point from Delaunay triangulation."""
    neighbors = [set() for _ in range(len(tri.points))]
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(3):
                if i != j:
                    neighbors[simplex[i]].add(simplex[j])
    return [list(n) for n in neighbors]

def _optimize_delaunay(centres, max_iters=1000, learning_rate=0.01):
    """
    Optimize circle packing using Delaunay triangulation.
    We treat the problem as maximizing the minimum edge length in the Delaunay graph
    while keeping points within the square.
    """
    centres = centres.copy()
    n = len(centres)
    
    # Initial Delaunay
    try:
        tri = Delaunay(centres)
        neighbors = _get_delaunay_neighbors(tri)
    except Exception:
        return centres, _compute_radius(centres)
    
    best_radius = _compute_radius(centres)
    
    for it in range(max_iters):
        # Recompute Delaunay every few iterations or if points move significantly
        # For simplicity and robustness, recompute every iteration (N=200 is small enough)
        try:
            tri = Delaunay(centres)
            neighbors = _get_delaunay_neighbors(tri)
        except Exception:
            # If triangulation fails (e.g., collinear points), add small noise
            centres += np.random.randn(n, 2) * 1e-5
            continue

        # Compute forces based on Delaunay neighbors
        # Force is repulsive if distance < target, attractive if distance > target?
        # Actually, we want to maximize the MINIMUM distance.
        # So we should push points away from their closest neighbors.
        
        # Identify the current bottleneck radius
        # Instead of global cdist, we can check Delaunay edges for min distance
        # The min distance in a Delaunay triangulation is guaranteed to be among the edges.
        min_edge_dist = np.inf
        min_i, min_j = -1, -1
        
        # Iterate through unique edges
        visited_edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    u, v = simplex[i], simplex[j]
                    if (u, v) in visited_edges: continue
                    visited_edges.add((u, v))
                    dist = np.linalg.norm(centres[u] - centres[v])
                    if dist < min_edge_dist:
                        min_edge_dist = dist
                        min_i, min_j = u, v
        
        # Check boundary constraints
        dist_to_border = 0.5 - np.abs(centres)
        min_border_dist = np.min(dist_to_border)
        min_border_idx = np.argmin(dist_to_border)
        
        # Determine if boundary is the bottleneck
        # Radius is min(min_edge_dist/2, min_border_dist)
        
        # Strategy: Move points to increase the minimum of (edge_dist/2, border_dist)
        # We use a gradient ascent on the radius.
        # Approximate gradient: 
        # If edge (i,j) is the min, move i and j apart.
        # If border is the min, move point away from border.
        
        # We will apply small updates to all points involved in "tight" constraints.
        # Define a threshold for "tight"
        current_radius = min(min_edge_dist / 2.0, min_border_dist)
        
        # Update step
        step_size = learning_rate * (1.0 + 0.1 * np.random.rand())
        
        new_centres = centres.copy()
        
        # 1. Handle Boundary Constraints
        # Points close to boundary should move inward
        for i in range(n):
            pos = centres[i]
            d_left = pos[0] + 0.5
            d_right = 0.5 - pos[0]
            d_down = pos[1] + 0.5
            d_up = 0.5 - pos[1]
            
            min_d = min(d_left, d_right, d_down, d_up)
            
            # If this point is closer to boundary than the current radius + margin, push it in
            if min_d < current_radius + 1e-6:
                # Direction to center
                dir_x = -np.sign(pos[0])
                dir_y = -np.sign(pos[1])
                # Normalize
                norm = np.sqrt(dir_x**2 + dir_y**2)
                if norm > 1e-6:
                    dir_x /= norm
                    dir_y /= norm
                else:
                    dir_x, dir_y = np.random.randn(2)
                    norm = np.linalg.norm([dir_x, dir_y])
                    dir_x /= norm
                    dir_y /= norm
                    
                # Move inward
                new_centres[i, 0] += dir_x * step_size
                new_centres[i, 1] += dir_y * step_size

        # 2. Handle Pairwise Constraints (Delaunay Edges)
        # For each edge, if distance is small, push apart
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    u, v = simplex[i], simplex[j]
                    dist = np.linalg.norm(centres[u] - centres[v])
                    
                    # If this edge is "tight" (contributing to the minimum or close to it)
                    if dist < current_radius * 2.0 + 1e-6:
                        # Direction from v to u
                        diff = centres[u] - centres[v]
                        norm = np.linalg.norm(diff)
                        if norm < 1e-6:
                            diff = np.random.randn(2)
                            norm = np.linalg.norm(diff)
                        diff /= norm
                        
                        # Push u away from v, v away from u
                        new_centres[u] += diff * step_size * 0.5
                        new_centres[v] -= diff * step_size * 0.5

        # Clip to square
        new_centres = np.clip(new_centres, -0.5, 0.5)
        
        # Check improvement
        new_radius = _compute_radius(new_centres)
        
        if new_radius > best_radius + 1e-12:
            best_radius = new_radius
            centres = new_centres
        else:
            # No improvement, reduce learning rate or perturb
            learning_rate *= 0.95
            if learning_rate < 1e-8:
                break
            # Occasionally add noise to escape local minima
            if it % 100 == 0:
                centres += np.random.randn(n, 2) * 1e-4
                centres = np.clip(centres, -0.5, 0.5)

    return centres, best_radius

def construct():
    """
    Construct centres of 200 equal circles inside the unit square [-1/2, 1/2]^2
    to maximize the common radius.
    """
    # Initialize with a jittered grid
    # 200 points. sqrt(200) ~ 14.14. Let's use a 14x14 grid (196 points) + 4 extra?
    # Or just use a grid of size x size and take the first N.
    # Better: distribute evenly.
    
    # Use a grid that is slightly denser to allow for jitter
    m = int(np.ceil(np.sqrt(N)))
    # Generate m x m grid
    xs = np.linspace(-0.5 + 0.5/m, 0.5 - 0.5/m, m)
    ys = np.linspace(-0.5 + 0.5/m, 0.5 - 0.5/m, m)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    
    # If grid has more than N points, randomly select N? Or just take first N?
    # Taking first N might bias towards corners. 
    # Let's create a grid of exactly N by adjusting dimensions or using a spiral?
    # Simpler: Create a large grid and randomly sample N points?
    # No, random sampling might lead to bad local minima.
    # Let's stick to a structured grid but ensure we cover the area well.
    
    # For N=200, a 14x14 grid has 196 points.
    # We can add 4 points in the center or corners?
    # Let's just use a 15x15 grid (225 points) and randomly subsample 200?
    # Or better, use a grid with non-integer aspect ratio?
    
    # Let's use a grid of size 14x14 (196) and add 4 points in the middle of the sides?
    # Actually, a simple random jitter on a 14x14 grid + 4 random points is okay.
    # But let's try to make it more regular.
    
    # Alternative: Use a hexagonal lattice initialization?
    # For a square, a square grid is often a good start.
    
    # Let's use a 14x14 grid for 196 points.
    # We need 4 more. We can place them in the center of the 4 quadrants?
    # Or just expand the grid to 15x15 and drop 25 random points?
    
    # Let's go with 15x15 grid, randomly select 200 points to start.
    # This provides a good spread.
    
    m = 15
    xs = np.linspace(-0.5 + 0.5/m, 0.5 - 0.5/m, m)
    ys = np.linspace(-0.5 + 0.5/m, 0.5 - 0.5/m, m)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Randomly select N points from the grid
    indices = np.random.choice(len(grid), N, replace=False)
    centres = grid[indices].copy()
    
    # Add small jitter to break symmetry
    centres += np.random.randn(N, 2) * 0.001
    centres = np.clip(centres, -0.5, 0.5)
    
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    # Run optimization
    # The Delaunay optimization is deterministic given the start, so we do one long run
    # or a few short runs with different starts?
    # Given time limits, one robust run is better.
    
    start_time = time.time()
    time_limit = 110 # seconds
    
    # Run optimization
    optimised, radius = _optimize_delaunay(centres, max_iters=2000, learning_rate=0.05)
    
    if radius > best_radius:
        best_radius = radius
        best_centres = optimised.copy()
        
    # If time remains, try a second pass with smaller steps or different start
    if time.time() - start_time < time_limit - 5:
        # Try again with different jitter
        centres2 = grid[np.random.choice(len(grid), N, replace=False)].copy()
        centres2 += np.random.randn(N, 2) * 0.001
        centres2 = np.clip(centres2, -0.5, 0.5)
        
        optimised2, radius2 = _optimize_delaunay(centres2, max_iters=1000, learning_rate=0.05)
        
        if radius2 > best_radius:
            best_radius = radius2
            best_centres = optimised2.copy()

    return best_centres
# EVOLVE-BLOCK-END
