"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math

N = 50

# EVOLVE-BLOCK-START

def construct():
    # Simulated Annealing approach
    # State: positions of N circles
    # Energy: sum of overlap penalties + containment penalties
    
    # Initialization
    # Start with a random configuration or a rough hexagonal packing
    # Random is often better for SA to escape local minima, but we can try random
    np.random.seed(42) # For reproducibility during development, though evaluator might not care
    # Let's use a random start but ensure they are well inside
    pts = np.random.rand(N, 2) * 1.8 - 0.9 # Range [-0.9, 0.9]
    
    # Ensure no initial overlaps if possible? No, SA handles overlaps.
    # But we must ensure they are inside the circle?
    # The energy function will penalize being outside, so it's okay to start slightly outside or on boundary.
    # Let's project them to be strictly inside to start clean.
    r_norm = np.linalg.norm(pts, axis=1)
    mask = r_norm > 0.9
    pts[mask] = pts[mask] / r_norm[mask, None] * 0.9
    
    # Parameters
    T_start = 0.05
    T_end = 0.0001
    num_iterations = 20000
    move_size_start = 0.05
    move_size_end = 0.001
    
    # Energy function
    # We want to maximize the minimum distance between centers and distance to boundary.
    # Equivalent to minimizing the "penalty" for violations.
    # Let r be the current minimum radius.
    # Actually, it's easier to define the energy as the sum of squared overlaps for a fixed radius?
    # No, we want to find the max radius.
    
    # Alternative Objective:
    # Minimize E = sum_{i<j} max(0, d_ij - 2r)^2 + sum_i max(0, |c_i| + r - 1)^2
    # This is hard because r is a variable.
    
    # Standard approach: 
    # 1. Fix r? No.
    # 2. Use a potential that pushes circles apart and keeps them inside.
    #    If we minimize the sum of squared distances for overlapping pairs, we effectively maximize the minimum distance.
    #    Let's define Energy E = sum_{i<j} max(0, 2*r_target - dist_ij)^2 + sum_i max(0, |c_i| - (1-r_target))^2
    #    But r_target is unknown.
    
    # Better approach: 
    # The score is r = min(min_dist/2, 1 - max_dist_to_center).
    # We can simulate with a "virtual" radius that adapts, or simply minimize the overlap of circles with a fixed radius, 
    # but that doesn't maximize the radius directly if the radius is fixed.
    
    # Let's use the following heuristic which works well:
    # Treat the circles as having a fixed radius R (e.g., 0.12).
    # Minimize the overlap energy: E = sum_{i<j} max(0, 2R - dist_ij)^2 + sum_i max(0, |c_i| + R - 1)^2.
    # If we can fit them with R=0.12, we can try to increase R.
    # Or, we can just minimize the "badness" of the packing where badness is related to the minimum distance.
    
    # Let's try a simpler potential:
    # E = sum_{i<j} 1/(dist_ij^2) + sum_i 1/(1-|c_i|)^2
    # This pushes circles apart and away from the boundary.
    # Minimizing this tends to maximize the minimum distance.
    
    def calculate_energy(pts):
        E = 0.0
        # Pairwise repulsion
        # Vectorized calculation for speed
        # dist matrix
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        
        # Only consider pairs that are "close" to avoid division by zero or large numbers for far pairs?
        # 1/d^2 is fine.
        # We want to maximize min dist. Minimizing sum of 1/d^2 is a good proxy.
        # To avoid overflow, clip dist_sq to a small value.
        dist_sq_clipped = np.maximum(dist_sq, 1e-6)
        E += np.sum(1.0 / dist_sq_clipped)
        
        # Boundary penalty
        # Distance to boundary: 1 - |c_i|
        # We want this to be large. So we minimize 1/(1-|c_i|)^2?
        # Or simply penalize if |c_i| > 1 - r_min?
        # Let's use a soft constraint: Penalty for being close to boundary.
        r_norms = np.linalg.norm(pts, axis=1)
        # If r_norms > 1, it's invalid. We assume we keep them inside.
        # Let's penalize (1 - r_norms)^-2
        dist_to_boundary = 1.0 - r_norms
        dist_to_boundary = np.maximum(dist_to_boundary, 1e-6)
        E += np.sum(1.0 / (dist_to_boundary**2))
        
        return E

    # Simulated Annealing Loop
    best_pts = pts.copy()
    best_E = calculate_energy(pts)
    
    for i in range(num_iterations):
        # Annealing schedule
        progress = i / num_iterations
        T = T_start * (T_end / T_start) ** progress
        move_size = move_size_start * (move_size_end / move_size_start) ** progress
        
        # Propose a move
        idx = np.random.randint(N)
        new_pts = pts.copy()
        # Random step
        step = np.random.randn(2) * move_size
        new_pts[idx] += step
        
        # Ensure new point is not crazy outside?
        # The energy function handles it, but huge jumps might cause numerical issues.
        # Clip position to [-1.1, 1.1]
        new_pts[idx] = np.clip(new_pts[idx], -1.1, 1.1)
        
        new_E = calculate_energy(new_pts)
        delta_E = new_E - calculate_energy(pts) # Recalculating old energy is expensive, store it?
        # Let's store current energy
        # Actually, calculate_energy is O(N^2). 20000 * 50^2 = 50M ops. Should be fast in numpy.
        # Let's optimize by storing current E.
        
        # Re-doing the loop structure to be more efficient
        break # Break out to rewrite the loop properly below
        
    # Efficient SA Implementation
    current_E = calculate_energy(pts)
    for i in range(num_iterations):
        progress = i / num_iterations
        T = T_start * (T_end / T_start) ** progress
        move_size = move_size_start * (move_size_end / move_size_start) ** progress
        
        idx = np.random.randint(N)
        new_pts = pts.copy()
        step = np.random.randn(2) * move_size
        new_pts[idx] += step
        new_pts[idx] = np.clip(new_pts[idx], -1.0, 1.0) # Keep inside container roughly
        
        new_E = calculate_energy(new_pts)
        delta_E = new_E - current_E
        
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            pts = new_pts
            current_E = new_E
            
            if current_E < best_E:
                best_E = current_E
                best_pts = pts.copy()
                
    # Final Refinement: Gradient Descent on the same energy function
    # Calculate gradient numerically or analytically?
    # Numerical gradient is simpler and robust enough for N=50.
    # Or just do more SA with very small T and move size.
    # Let's do a final "cooling" phase with deterministic moves?
    # No, let's just run a few more iterations of SA with very low T.
    
    T_final = T_end
    move_size_final = move_size_end * 0.1
    for i in range(5000):
        idx = np.random.randint(N)
        new_pts = pts.copy()
        step = np.random.randn(2) * move_size_final
        new_pts[idx] += step
        new_pts[idx] = np.clip(new_pts[idx], -1.0, 1.0)
        
        new_E = calculate_energy(new_pts)
        delta_E = new_E - current_E
        
        if delta_E < 0:
            pts = new_pts
            current_E = new_E
            if current_E < best_E:
                best_E = current_E
                best_pts = pts.copy()

    # Calculate the actual radius for the best configuration
    # r = min( min_{i<j} dist_ij / 2, min_i (1 - |c_i|) )
    
    def calc_radius(pts):
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_sq = np.min(dist_sq)
        min_dist = np.sqrt(min_dist_sq)
        
        # Distance to boundary
        r_norms = np.linalg.norm(pts, axis=1)
        min_boundary_dist = 1.0 - np.max(r_norms)
        
        return min(min_dist / 2.0, min_boundary_dist)

    r_best = calc_radius(best_pts)
    
    # It's possible that a slightly worse energy configuration has a better radius?
    # The energy function is a proxy. 
    # To be safe, we can try to locally optimize the radius directly?
    # Or just trust the proxy. The proxy 1/d^2 strongly favors large minimum distances.
    
    # One more check: ensure validity
    # If any center is outside, it's invalid. We clipped to [-1,1], but norm might be >1.
    # We must project any center with |c| > 1 - r_best to the boundary?
    # Actually, the score is 0 if invalid.
    # Let's ensure all centers are strictly inside the unit circle.
    # If |c_i| >= 1, it's invalid.
    # We should project them to |c_i| = 0.999 if they are outside.
    
    r_norms = np.linalg.norm(best_pts, axis=1)
    mask = r_norms > 0.999
    if np.any(mask):
        best_pts[mask] = best_pts[mask] / r_norms[mask, None] * 0.999
        # Recalculate radius after projection
        r_best = calc_radius(best_pts)

    return best_pts

# EVOLVE-BLOCK-END
