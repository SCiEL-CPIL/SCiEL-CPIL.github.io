import numpy as np
from scipy.optimize import linprog, minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Strategy:
    # 1. Use a hexagonal packing initialization which is dense.
    # 2. Fix the aspect ratio to 1:1 (Square) initially, as it's a strong candidate.
    # 3. Optimize the radius 'r' and positions 'x, y' to maximize the sum of radii.
    #    Since all circles will likely have similar sizes in a dense pack, 
    #    maximizing the minimum radius or the sum is closely related.
    #    However, the task asks to maximize SUM of radii.
    #    A robust approach: 
    #    a. Generate a good initial layout (Hex grid).
    #    b. Use a force-directed relaxation or local search to push circles apart.
    #    c. Scale to perimeter 4.
    
    # Let's try a different structural approach:
    # Since N=21, a 5x5 grid is 25 (too many), 4x5 is 20 (too few).
    # A hexagonal arrangement is better.
    
    # Initialize with a hexagonal lattice in a unit square [0,1]x[0,1]
    # We want to pack 21 circles.
    # Hex grid spacing s. Radius r = s / (2 * sin(60)) = s / sqrt(3) if touching.
    
    # Let's use a simple local search (simulated annealing style) to maximize the sum of radii
    # given a fixed aspect ratio, then scale.
    
    # Actually, a more powerful structural change:
    # Use `scipy.optimize.minimize` with method='Nelder-Mead' or 'Powell' on the positions
    # to maximize the minimum distance between circles (which allows larger radius).
    # Then the sum of radii is approx N * r_min.
    # But circles can have different radii.
    
    # Better: Maximize the sum of radii directly.
    # Variables: x, y, r for each circle.
    # Constraints:
    # 1. x_i >= r_i, x_i <= W - r_i
    # 2. y_i >= r_i, y_i <= H - r_i
    # 3. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    # 4. W + H = 2 (Perimeter 4)
    
    # This is a non-convex problem.
    # Let's stick to the insight from "inspiration 2": Structure-aware lattice.
    # Let's try to optimize a hexagonal lattice parameter.
    
    # 1. Define a hexagonal packing pattern for 21 circles.
    # Rows of 4, 5, 4, 4, 4? No, that's 21.
    # Hex packing usually has rows offset.
    # Let's generate a specific hex-like layout.
    
    # Layout:
    # Row 0: 4 circles
    # Row 1: 5 circles (offset)
    # Row 2: 4 circles
    # Row 3: 4 circles
    # Row 4: 4 circles
    # Total: 21.
    
    # Let's construct initial positions for this layout in a unit square.
    # Then use local optimization to adjust positions to maximize the minimum clearance,
    # which allows for larger radii.
    
    # Initial guess for hex grid in [0,1]x[0,1]
    # Spacing dx, dy.
    # For hex: dx = 2r, dy = r*sqrt(3)
    
    # We will optimize the positions (x, y) for fixed r=1 (normalized)
    # to maximize the minimum distance between any two circles and to the walls.
    # This is equivalent to maximizing the radius if we scale.
    
    # Objective: Minimize -min(distances)
    # Distances include distance to walls (x, 1-x, y, 1-y)
    
    # Let's use a simple gradient-free optimizer on the 42 position variables.
    
    def min_dist_penalty(positions):
        # positions: (21, 2) array of x, y in [0,1]
        # Calculate min distance between circles
        min_d = 1.0
        for i in range(N):
            xi, yi = positions[i]
            # Distance to walls
            d_wall = min(xi, 1-xi, yi, 1-yi)
            if d_wall < min_d:
                min_d = d_wall
            for j in range(i+1, N):
                xj, yj = positions[j]
                d = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                if d < min_d:
                    min_d = d
        return -min_d

    # Initial positions: Hex grid
    # We need to fit 21 circles.
    # Let's try a 5-row layout: 4,5,4,4,4 or similar.
    # Actually, a 5x5 grid with corners removed?
    # 5x5 = 25. Remove 4 corners -> 21.
    # This creates a "plus" shape or similar.
    
    # Let's try a simple grid 5x5 minus 4 corners (0,0), (0,4), (4,0), (4,4)
    # This leaves a central cross-like structure.
    # Or just a 5x5 grid and let the optimizer move them.
    
    # Better initialization: Hexagonal
    # Let's generate a hex grid of ~25 points and pick 21 best.
    
    # Simpler: Use a 5x5 grid, then optimize.
    # The optimizer will likely push them into a hex-like arrangement if beneficial.
    
    # Initialize 5x5 grid
    xs = np.linspace(0.1, 0.9, 5)
    ys = np.linspace(0.1, 0.9, 5)
    grid_x, grid_y = np.meshgrid(xs, ys)
    positions_init = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    # Remove 4 corners to get 21
    # Indices for corners in 5x5 (row-major):
    # 0 (0,0), 4 (0,4), 20 (4,0), 24 (4,4)
    # Keep indices not in [0, 4, 20, 24]
    keep_mask = np.ones(25, dtype=bool)
    keep_mask[[0, 4, 20, 24]] = False
    positions_init = positions_init[keep_mask]
    
    # Optimize positions to maximize min distance
    # Use Powell method as it handles non-smooth objectives (min) reasonably well
    # and doesn't require gradients.
    
    best_pos = positions_init
    best_val = min_dist_penalty(best_pos)
    
    # Run optimization
    # We have 42 variables. Powell might be slow but 120s is generous.
    # Let's limit iterations.
    
    try:
        res = minimize(min_dist_penalty, best_pos, method='Powell', 
                       options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if res.fun < best_val: # We are minimizing -min_dist, so lower is better
            best_pos = res.x
            best_val = res.fun
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 5:
        # Fallback to initial
        best_pos = positions_init
        min_d = -min_dist_penalty(best_pos)
    else:
        min_d = -best_val
        
    # Now we have positions in [0,1]x[0,1] and a "radius" r_unit = min_d / 2
    # Because distance d = 2r.
    r_unit = min_d / 2.0
    
    # The circles are in a 1x1 square.
    # Perimeter of 1x1 square is 4.
    # So the scale factor is 1.
    # The radii are r_unit.
    # Sum of radii = 21 * r_unit.
    
    # Wait, is the optimal aspect ratio 1:1?
    # The perimeter constraint is W+H=2.
    # If we use a rectangle W x H with W+H=2, the area is W*H.
    # Max area is at W=H=1.
    # Since we are packing circles, area efficiency is key.
    # So 1:1 is a very strong candidate.
    
    # However, we can try to optimize the aspect ratio too.
    # Let's assume 1:1 for now.
    
    # Construct the final array
    # x, y, r
    final_x = best_pos[:, 0]
    final_y = best_pos[:, 1]
    final_r = np.full(N, r_unit)
    
    result = np.column_stack((final_x, final_y, final_r))
    
    # Validation: Ensure no overlaps and within bounds
    # The optimization maximized min distance, so they should be disjoint.
    # But numerical errors might creep in.
    # Add a tiny epsilon shrink to radii to be safe.
    result[:, 2] *= (1 - 1e-6)
    
    return result

# EVOLVE-BLOCK-END
