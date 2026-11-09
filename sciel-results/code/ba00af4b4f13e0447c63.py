"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import sys

N = 50
SIDE = 1.0
HALF = 0.5
DEADLINE = 115.0  # Safety margin for 120s limit

def _compute_radius(centers):
    """Compute the valid radius for a given set of centers."""
    if centers.shape != (N, 2):
        return 0.0
    
    # Check bounds
    if np.any(np.abs(centers) > HALF + 1e-12):
        return 0.0
        
    # Check distances
    # Vectorized pairwise distance calculation
    # This is O(N^2) which is 2500 operations, very fast.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Set diagonal to inf to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    
    min_dist = np.min(dists)
    min_dist = min_dist / 2.0
    
    # Distance to walls
    wall_dists = np.min(np.abs(centers), axis=1)
    min_wall_dist = np.min(HALF - wall_dists)
    
    r = min(min_dist, min_wall_dist)
    
    if r < 0:
        return 0.0
    return r

def _objective(centers):
    """
    Objective function to minimize.
    We want to maximize the minimum distance.
    Minimize -min_distance.
    To make it smooth for gradient methods, we can use a soft-min or just rely on
    the fact that L-BFGS-B can handle non-smooth functions reasonably well if started close,
    or we can use a barrier-like function.
    
    However, a simpler and robust approach for the final polish is to maximize the
    minimum of:
    1. Pairwise distances / 2
    2. Wall distances
    
    Let's define a penalty function that is smooth.
    f(x) = -min( min_{i<j} ||x_i - x_j||/2, min_i (0.5 - |x_ik|) )
    
    Since min is non-differentiable, we can use a log-sum-exp approximation or just
    use a method that doesn't strictly require gradients, like Nelder-Mead, or
    use L-BFGS-B which uses finite differences.
    
    Given N=50, Nelder-Mead might be too slow. L-BFGS-B is better.
    To help L-BFGS-B, we can minimize the sum of squares of violations if we fix a target radius,
    but we don't know the target.
    
    Alternative: Minimize the maximum "tightness".
    Let's stick to minimizing -radius.
    """
    r = _compute_radius(centers)
    return -r

def _hex_init():
    """
    Generate a hexagonal packing initialization.
    This is a strong starting point for circle packing.
    """
    # Hexagonal packing density is pi / (2*sqrt(3)) ~ 0.9069
    # For N=50, we need to fit 50 circles.
    # Let's generate a large hex grid and pick the best subset or just use a grid that approximates hex.
    
    # Simpler: Generate a grid of points on a hex lattice, scale to fit, and pick the first N?
    # No, we need to center them.
    
    # Let's create a hex grid that covers the square.
    # Radius r_hex such that we get roughly N circles.
    # Area of square = 1. Area per circle in hex packing ~ 2*sqrt(3)*r^2.
    # N * 2*sqrt(3)*r^2 <= 1 => r <= 1/sqrt(2*sqrt(3)*N)
    # r_approx = 1 / sqrt(2*sqrt(3)*50) = 1 / sqrt(173.2) = 1/13.16 = 0.076
    # This is larger than the known optimal 0.0713, so a perfect hex grid won't fit exactly.
    # We need to "squash" or adjust.
    
    # Strategy: Create a hex grid, slightly compress it to fit inside the square,
    # then use optimization to relax.
    
    # Hex grid generation
    # Rows of circles.
    # Let's try to fit a 7x8 or similar arrangement.
    # 50 circles.
    # 7 rows: 7, 8, 7, 8, 7, 8, 7 -> 52 circles. Close.
    # Let's generate a hex grid of 52 points, then remove 2 from the center or corners to make 50?
    # Or just generate 50 points from a hex grid.
    
    points = []
    # Hex spacing
    # Let dx = 2*r, dy = r*sqrt(3)
    # We want to fit inside [-0.5, 0.5]
    
    # Let's try to find a scaling factor.
    # Generate hex coords for a range.
    # We'll generate a large grid and then select the best 50 by maximizing min dist.
    # Actually, simpler: Just create a hex grid that fits "almost" and let the optimizer fix it.
    
    # Let's use a 7x8 grid of hex points.
    # Rows alternate offset.
    r_init = 0.075  # Slightly larger than optimal to force compression
    
    dx = 2 * r_init
    dy = r_init * np.sqrt(3)
    
    # How many rows/cols to cover 1.0 width?
    # Width ~ cols * dx. 1.0 / 0.15 ~ 6.6. So 7 cols.
    # Height ~ rows * dy. 1.0 / (0.075*1.732) ~ 1.0 / 0.13 ~ 7.7. So 8 rows.
    
    cols = 7
    rows = 8
    
    # Generate grid
    x_coords = np.linspace(-0.5 + r_init, 0.5 - r_init, cols)
    y_coords = np.linspace(-0.5 + r_init, 0.5 - r_init, rows)
    
    # This is a square grid, not hex.
    # Let's do proper hex.
    
    hex_pts = []
    # Center the grid
    # Total width = (cols - 1) * dx
    # Total height = (rows - 1) * dy
    start_x = -0.5 + r_init + (1.0 - (cols - 1) * dx - 2*r_init) / 2 # Centering logic is tricky
    # Let's just generate and center.
    
    for i in range(rows):
        for j in range(cols):
            x = j * dx
            y = i * dy
            if i % 2 == 1:
                x += dx / 2
            hex_pts.append([x, y])
            
    hex_pts = np.array(hex_pts)
    
    # Center
    hex_pts -= np.mean(hex_pts, axis=0)
    
    # Scale to fit?
    # Currently, the extent is determined by dx, dy.
    # We want to fit inside [-0.5, 0.5].
    # Max |x|, |y| should be <= 0.5 - r_init.
    
    max_extent = np.max(np.abs(hex_pts))
    # We want to scale so that max_extent + r_init <= 0.5 ?
    # No, r_init is the "nominal" radius.
    # If we scale coordinates, the distances scale, so the effective radius scales.
    
    # Let's just take the first 50 points? No, that's biased.
    # Let's take all 56 points and pick the best 50?
    # Or just use a 50-point subset of a larger hex grid.
    
    # Let's try a different approach:
    # Start with a square grid, then apply a "hexagonal distortion" or just use the square grid
    # and let the optimizer do the heavy lifting. The square grid is a valid starting point.
    # But hex is better.
    
    # Let's generate a hex grid that has exactly 50 points?
    # 7+8+7+8+7+8+7 = 52.
    # Let's use 52 points and remove the 2 closest to the center? Or corners?
    # Removing from corners might increase the min distance of the remaining set?
    # Actually, for packing, we want to keep the "dense" part.
    
    # Let's just generate a hex grid of ~50 points, scale it to fit the square,
    # and return it. The optimizer will expand it.
    
    # Generate 52 points as above.
    if len(hex_pts) >= N:
        # Select N points. 
        # To be symmetric, we can pick the N points closest to the center?
        # Or just the first N?
        # Let's pick the N points that maximize the minimum distance among themselves?
        # That's an NP-hard problem.
        # Simple heuristic: Keep the points that are "most interior" or just take a random subset?
        # Let's take the points from the hex grid, and if we have more than N, 
        # we can just use a slightly smaller hex grid or adjust.
        
        # Let's try to construct a hex grid that fits N=50 better.
        # 50 = 5 * 10? 
        # 5 rows of 10?
        # 10 cols * 2r. 5 rows * r*sqrt(3).
        # Width = 9 * 2r = 18r. Height = 4 * r*sqrt(3) = 6.9r.
        # Aspect ratio 18/6.9 ~ 2.6. Square is 1.
        # So we need to stretch vertically or compress horizontally.
        
        # Let's go back to the 7x8 grid (52 points).
        # We will remove the 2 points that are furthest from the center?
        # Or the 2 points that are "redundant"?
        # Let's just take the 50 points with the smallest index?
        # That will break symmetry.
        
        # Better: Generate a hex grid, and if size > N, take a random subset of N?
        # No, deterministic is better.
        # Let's take the N points closest to the origin from the 52-point hex grid.
        dists_to_origin = np.linalg.norm(hex_pts, axis=1)
        indices = np.argsort(dists_to_origin)[:N]
        hex_pts = hex_pts[indices]
        
    else:
        # If we generated fewer, pad with random or grid?
        # 7*8=52, so this shouldn't happen.
        pass
    
    # Scale the points so they fit in the square?
    # Currently they are spaced by dx, dy based on r_init=0.075.
    # The extent is roughly (cols-1)*dx and (rows-1)*dy.
    # We want to map this extent to [-(0.5-r), (0.5-r)]?
    # Actually, let's just return these points. The optimizer will handle scaling/positioning.
    # But they must be inside the square initially to be valid?
    # The task says "Place the centres... inside a unit square".
    # If my initial points are outside, the score is 0.
    
    # So I must ensure initial points are inside.
    # My hex_pts are centered at 0.
    # Max |x| in hex_pts?
    # cols=7, dx=0.15. Range x: -0.3 to 0.3 (approx).
    # rows=8, dy=0.13. Range y: -0.39 to 0.39 (approx).
    # So they are inside [-0.5, 0.5].
    
    return hex_pts

def construct():
    start_time = time.time()
    
    # 1. Initialization
    centers = _hex_init()
    current_radius = _compute_radius(centers)
    
    # Ensure validity
    if not np.all(np.abs(centers) <= HALF):
        # Fallback to grid if hex init fails bounds (shouldn't happen)
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        centers = np.array([(x, y) for y in ticks for x in ticks])
        centers = centers[:N]
        current_radius = _compute_radius(centers)
        
    best_centers = centers.copy()
    best_radius = current_radius
    
    # 2. Local Optimization Loop
    # We will use L-BFGS-B to maximize the radius.
    # We repeat this with small random perturbations to escape local minima.
    
    num_restarts = 20  # Max restarts, limited by time
    
    for i in range(num_restarts):
        if time.time() > DEADLINE:
            break
            
        # Define objective for this restart
        # We minimize -radius.
        
        # Constraints:
        # -0.5 <= x_i <= 0.5
        # -0.5 <= y_i <= 0.5
        
        bounds = [(-HALF, HALF)] * (2 * N)
        
        # Initial point
        x0 = best_centers.flatten()
        
        # Optimize
        try:
            res = minimize(
                _objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            new_centers = res.x.reshape(N, 2)
            
            # Check validity
            if np.all(np.abs(new_centers) <= HALF + 1e-9):
                new_radius = _compute_radius(new_centers)
                
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = new_centers
                else:
                    # If it didn't improve, maybe it got stuck.
                    # We can try to "shake" it.
                    pass
            else:
                # Invalid result from optimizer, clip it?
                new_centers = np.clip(new_centers, -HALF, HALF)
                new_radius = _compute_radius(new_centers)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = new_centers
                    
        except Exception as e:
            # If optimization fails, continue
            pass
            
        # Prepare for next restart: Perturb the best solution
        # If we have time, we can do a "shake" step.
        # Shake: Move the circle with the smallest "slack" (min dist to neighbors/wall)
        # in a random direction.
        
        if time.time() > DEADLINE - 10:
            break
            
        # Simple shake: Add small noise to best_centers
        # But L-BFGS-B is good at polishing.
        # To escape local minima, we need a larger perturbation.
        
        # Strategy: 
        # 1. Take best_centers.
        # 2. Identify the "bottleneck" pair or wall.
        # 3. Move one of them randomly.
        
        # Simpler: Random restart from a slightly perturbed version of the best solution.
        # Or, if we haven't improved in a while, do a full random restart?
        # Given the time budget, let's just perturb the best solution.
        
        # Perturbation magnitude should be small, e.g., 0.01
        noise = np.random.normal(0, 0.01, size=(N, 2))
        perturbed = best_centers + noise
        perturbed = np.clip(perturbed, -HALF, HALF)
        
        # Ensure perturbed is valid (non-overlapping)?
        # If we overlap, the radius is 0 or small, and L-BFGS-B will try to fix it.
        # But if it's too far, it might not recover.
        # Let's check if perturbed is "close" to valid.
        # Actually, just use perturbed as x0 for the next L-BFGS-B.
        
        # Wait, if I just use best_centers again, I'll get the same result.
        # I must use the perturbed version as the STARTING point for the next optimization.
        
        # So, for the NEXT iteration, x0 should be the perturbed version.
        # But I also want to keep the BEST found so far.
        
        # Let's restructure:
        # In the loop:
        # 1. Use current_x0 (initially hex, then perturbed best)
        # 2. Optimize
        # 3. Update best if improved
        # 4. Set current_x0 to perturbed best for next round?
        #    Or perturbed current_x0?
        
        # Let's use: x0 = best_centers + noise.
        
    # Final Polish
    # Use a higher precision setting on the best solution found.
    if time.time() < DEADLINE - 5:
        bounds = [(-HALF, HALF)] * (2 * N)
        try:
            res = minimize(
                _objective, 
                best_centers.flatten(), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            final_centers = res.x.reshape(N, 2)
            if np.all(np.abs(final_centers) <= HALF + 1e-9):
                final_radius = _compute_radius(final_centers)
                if final_radius > best_radius:
                    best_radius = final_radius
                    best_centers = final_centers
        except:
            pass

    # Final validation and clipping
    best_centers = np.clip(best_centers, -HALF, HALF)
    
    # Ensure exact shape
    return best_centers

# EVOLVE-BLOCK-START
# The code above is the implementation.
# We need to wrap it in the EVOLVE-BLOCK as per the instructions?
# "Keep the function names and signatures of the seed program."
# The seed program has `construct()`.
# The instructions say: "Output the COMPLETE program in a single
