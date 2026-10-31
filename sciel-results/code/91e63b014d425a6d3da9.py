"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for safety
    
    # Initial guess: Grid placement
    side = int(np.ceil(np.sqrt(N)))
    # We need exactly N points. Create a grid of side x side and take the first N.
    # To make it more symmetric, we can center the grid better.
    # For N=26, side=6 (36 points). We take 26.
    # It's better to initialize in a way that respects the "square" aspect ratio.
    
    # Let's use a specific layout initialization that is known to be decent for packing.
    # A simple grid is okay, but let's randomize slightly or use a specific pattern.
    
    # Initialize centers
    # For N=26, a 5x5 grid plus 1 is 26. Or 6x5 grid.
    # Let's just use a grid of size ceil(sqrt(N)) and pick N points.
    # Actually, a better init is to spread them out.
    
    # Create a grid of points
    grid_side = int(np.ceil(np.sqrt(N)))
    x_coords = np.linspace(0.5/grid_side, 1 - 0.5/grid_side, grid_side)
    y_coords = np.linspace(0.5/grid_side, 1 - 0.5/grid_side, grid_side)
    
    # Generate all grid points
    points = []
    for i in x_coords:
        for j in y_coords:
            points.append((i, j))
    
    # If we have more points than N, we pick the first N.
    # To make it more uniform, we might want to sort by distance from center or similar, 
    # but for N=26, taking the first N from a grid is fine as a start.
    # However, the grid might be too sparse or dense. 
    # Let's just use the first N points from the grid.
    
    init_points = points[:N]
    
    # Initial radii: small value
    init_radii = [0.05] * N
    
    # Flatten the initial variables: [x0, y0, r0, x1, y1, r1, ...]
    x0 = []
    for i in range(N):
        x0.append(init_points[i][0])
        x0.append(init_points[i][1])
        x0.append(init_radii[i])
    
    def objective(vars):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            # Return a very high penalty to stop optimization
            return 1e10
            
        # Unpack variables
        centers = []
        radii = []
        for i in range(N):
            cx = vars[3*i]
            cy = vars[3*i + 1]
            r = vars[3*i + 2]
            centers.append((cx, cy))
            radii.append(r)
        
        penalty = 0.0
        
        # Boundary constraints
        for i in range(N):
            cx, cy = centers[i]
            r = radii[i]
            # Distance to walls
            d_left = cx
            d_right = 1.0 - cx
            d_bottom = cy
            d_top = 1.0 - cy
            
            # If r > min distance to wall, there is a violation
            min_dist = min(d_left, d_right, d_bottom, d_top)
            if r > min_dist:
                # Quadratic penalty for violation
                penalty += (r - min_dist) ** 2 * 10000
            
        # Overlap constraints
        for i in range(N):
            for j in range(i + 1, N):
                cx1, cy1 = centers[i]
                cx2, cy2 = centers[j]
                r1 = radii[i]
                r2 = radii[j]
                
                dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                req_dist = r1 + r2
                
                if dist < req_dist:
                    # Quadratic penalty for overlap
                    penalty += (req_dist - dist) ** 2 * 10000
        
        # We want to maximize sum of radii.
        # Minimize -sum(radii) + penalty
        sum_radii = sum(radii)
        return -sum_radii + penalty

    # Use Nelder-Mead for robustness without gradients
    # It might be slow, so we limit iterations or use Powell which is often faster for this.
    # Let's try Powell first.
    
    try:
        res = minimize(objective, x0, method='Powell', 
                       options={'maxiter': 2000, 'disp': False})
        
        # Check if time is running out
        if time.time() - start_time > time_limit:
            # Use the initial guess or best found so far?
            # Powell returns the best point it found.
            pass
            
        best_vars = res.x
    except Exception:
        # If optimization fails, fall back to initial guess
        best_vars = x0
        
    # Extract final radii and centers
    final_circles = []
    for i in range(N):
        cx = best_vars[3*i]
        cy = best_vars[3*i + 1]
        r = best_vars[3*i + 2]
        
        # Ensure circles are inside the square by clipping radius
        min_dist = min(cx, 1.0 - cx, cy, 1.0 - cy)
        if r > min_dist:
            r = min_dist * 0.9999 # Slightly shrink to ensure valid
            
        # Ensure r is positive
        if r < 0:
            r = 1e-6
            
        final_circles.append((cx, cy, r))
        
    # Final validation and adjustment for overlaps
    # This is a simple local pass to remove any remaining overlaps by shrinking
    # We do a few iterations of shrinking overlapping circles
    
    for _ in range(50):
        changed = False
        for i in range(N):
            cx1, cy1, r1 = final_circles[i]
            for j in range(i + 1, N):
                cx2, cy2, r2 = final_circles[j]
                dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                if dist < r1 + r2:
                    # Shrink the larger one or both
                    # Simple strategy: shrink both proportionally to their radius
                    overlap = (r1 + r2) - dist
                    if overlap > 0:
                        # Reduce radii
                        # Keep centers fixed for now
                        r1 -= overlap / 2
                        r2 -= overlap / 2
                        changed = True
                        # Ensure they don't go negative or out of bounds
                        min_dist1 = min(cx1, 1.0 - cx1, cy1, 1.0 - cy1)
                        min_dist2 = min(cx2, 1.0 - cx2, cy2, 1.0 - cy2)
                        if r1 < min_dist1: r1 = min_dist1
                        if r2 < min_dist2: r2 = min_dist2
                        if r1 < 1e-6: r1 = 1e-6
                        if r2 < 1e-6: r2 = 1e-6
                        
        if not changed:
            break
            
    # Update final_circles with adjusted radii
    adjusted_circles = []
    for i in range(N):
        cx, cy, r = final_circles[i]
        # Re-clip for bounds
        min_dist = min(cx, 1.0 - cx, cy, 1.0 - cy)
        if r > min_dist:
            r = min_dist
        adjusted_circles.append((cx, cy, r))
        
    return np.array(adjusted_circles)

# EVOLVE-BLOCK-END
