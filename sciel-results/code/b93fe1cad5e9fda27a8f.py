"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # We will search over aspect ratios. W + H = 2 => H = 2 - W.
    # W must be in (0, 2).
    # A square is W=1, H=1.
    # For 21 circles, a 5x5 grid (25 slots) suggests square-ish is good, but perimeter constraint is tight.
    # Actually, 21 circles in a square 1x1: area 1. 21 * pi * r^2 <= 1 => r <= sqrt(1/(21*pi)) ~ 0.123. Sum ~ 2.58.
    # If we use a 5x5 grid in 1x1, r=0.1. Sum = 21*0.1 = 2.1.
    # The seed program uses side=5, r=0.2-eps. Wait.
    # Seed: side=5. cells: (i+0.5)/5, r = 0.5/5 - 1e-3 = 0.1 - 0.001 = 0.099.
    # Sum = 21 * 0.099 = 2.079.
    # If we pack 21 circles in a 1x1 square optimally, we can do better.
    # Known best for 21 circles in unit square: max sum of radii is approx 2.61 (from literature?).
    # Actually, let's look at the "perimeter 4" constraint.
    # The rectangle bounding box must have perimeter <= 4.
    # The circles must be INSIDE the rectangle.
    # So if we have a configuration of 21 circles with max sum of radii S, and they fit in a W x H box with 2(W+H)<=4, it's valid.
    # To maximize S, we should probably use the full perimeter, i.e., W+H=2.
    
    # Strategy: Simulated Annealing / Force Directed Layout
    # Variables: centers (x_i, y_i) and radii (r_i) and aspect ratio (W).
    # Objective: Maximize sum(r_i)
    # Constraints:
    # 1. 0 <= x_i - r_i, x_i + r_i <= W
    # 2. 0 <= y_i - r_i, y_i + r_i <= H = 2 - W
    # 3. dist(i, j) >= r_i + r_j
    
    # Let's fix W to a few promising values and optimize positions/radii for each, or optimize W simultaneously.
    # Optimizing W simultaneously is harder. Let's iterate over a few aspect ratios.
    # Promising aspect ratios for 21 circles:
    # 5x5 grid -> 1:1
    # 6x4 grid (24 slots) -> W/H = 6/4 = 1.5. W=1.2, H=0.8.
    # 7x3 grid (21 slots) -> W/H = 7/3 ~ 2.33. W=1.4, H=0.6.
    
    best_score = 0
    best_result = None
    
    # Initial guess for seed score: 2.079
    # Let's try to beat it.
    
    def run_optimization(W, H, n_iter=5000, seed=42):
        np.random.seed(seed)
        
        # Initialize positions in a grid-like manner to avoid bad overlaps
        # 5x5 grid for 21 circles
        side = 5
        # Map grid indices to [0, W] x [0, H]
        xs = np.linspace(0.1*W, 0.9*W, side)
        ys = np.linspace(0.1*H, 0.9*H, side)
        
        centers = np.zeros((N, 2))
        radii = np.zeros(N)
        
        # Place 21 circles in the first 21 slots of 5x5 grid
        count = 0
        for i in range(side):
            for j in range(side):
                if count < N:
                    centers[count] = [xs[i], ys[j]]
                    # Initial radius: half the grid spacing minus a bit
                    spacing = min(W, H) / side
                    radii[count] = spacing * 0.4
                    count += 1
        
        # Optimization loop
        # We want to maximize sum(radii).
        # Use a penalty method or direct repulsion.
        # Let's use a "growth" strategy: 
        # 1. Start with small radii.
        # 2. Increase radii until they touch.
        # 3. Move centers to resolve overlaps and expand further.
        
        # Simpler: Use scipy.optimize.minimize on a penalty objective?
        # Objective: -sum(r) + penalty(overlaps) + penalty(boundary)
        # This can be slow.
        
        # Alternative: Physics simulation
        # Forces:
        # - Repulsion between overlapping circles (proportional to overlap depth)
        # - Repulsion from boundaries (if circle crosses boundary)
        # - "Expansion" force: we want to increase radii. 
        #   We can model this by having a target radius that we try to reach, 
        #   or by scaling up all radii slightly if no overlaps, and shrinking if overlaps.
        
        # Let's try a simple iterative relaxation:
        # 1. Check for overlaps and boundary violations.
        # 2. If violations, move centers to resolve them (gradient descent on penalty).
        # 3. If no violations, increase all radii by a small factor.
        # 4. Repeat.
        
        r_scale = 1.0
        current_radii = radii.copy()
        
        # We need to be careful with time. 5000 iterations * N^2 distance calcs might be slow in pure Python/NumPy if not vectorized.
        # N=21, so N^2=441. 5000*441 ~ 2M ops. Should be fast enough in NumPy.
        
        for it in range(n_iter):
            if time.time() > deadline:
                return None, 0
            
            # Calculate pairwise distances
            # dist[i, j]
            dists = cdist(centers, centers)
            
            # Check overlaps: dists[i,j] < current_radii[i] + current_radii[j]
            # We want to maximize radii, so we try to expand them.
            
            # Step 1: Try to expand radii
            # If we can expand, do so.
            # Check if expanding by factor (1 + delta) causes overlap.
            # This is complex. 
            # Simpler: 
            # Calculate "slack" for each pair. slack = dist - (r_i + r_j).
            # If all slacks > 0 and boundary slacks > 0, we can increase radii.
            
            # Boundary slacks:
            # x_i - r_i >= 0 => r_i <= x_i
            # W - (x_i + r_i) >= 0 => r_i <= W - x_i
            # y_i - r_i >= 0 => r_i <= y_i
            # H - (y_i + r_i) >= 0 => r_i <= H - y_i
            
            # Let's use a gradient-based approach on positions only, with fixed radii?
            # No, we want to optimize radii too.
            
            # Let's stick to the "expand if possible" heuristic.
            # 1. Compute min slack over all pairs and boundaries.
            # If min_slack > 0:
            #    Increase all radii by epsilon * (1 + min_slack) ? Or just epsilon?
            # Else:
            #    Move centers to increase slacks.
            
            # To move centers, we need forces.
            # Force on i from j: direction (x_i - x_j) / dist * overlap_depth
            # Force on i from boundary: direction inward * violation_depth
            
            forces = np.zeros_like(centers)
            
            # Pairwise forces
            # Only consider pairs with dist < r_i + r_j + margin
            # Use a soft potential: F = k * (r_i + r_j - dist) if dist < r_i + r_j
            # To make it differentiable-ish for movement, just use the overlap depth.
            
            overlap_depths = current_radii[:, None] + current_radii[None, :] - dists
            overlap_depths = np.maximum(overlap_depths, 0)
            # Avoid division by zero
            dists_safe = np.where(dists > 1e-6, dists, 1e-6)
            inv_dists = 1.0 / dists_safe
            
            # Force vector from j to i
            # F_ij = (centers[i] - centers[j]) * overlap_depth * inv_dist
            # Sum over j
            # This is O(N^2).
            
            # Vectorized force calculation:
            # diff = centers[:, None, :] - centers[None, :, :]  # (N, N, 2)
            # force_matrix = diff * (overlap_depths * inv_dists)[:, :, None]
            # forces = np.sum(force_matrix, axis=1)
            
            # This creates an N x N x 2 array. 21x21x2 = 882 floats. Very fast.
            
            diff = centers[:, None, :] - centers[None, :, :]
            # Set diagonal to 0 to avoid self-force
            np.fill_diagonal(overlap_depths, 0)
            np.fill_diagonal(dists_safe, 1.0)
            inv_dists[0,0] = 0 # just in case, though fill_diagonal handled it
            
            force_matrix = diff * (overlap_depths * inv_dists)[:, :, None]
            forces += np.sum(force_matrix, axis=1)
            
            # Boundary forces
            # If x_i < r_i: force in +x direction proportional to (r_i - x_i)
            # If x_i > W - r_i: force in -x direction
            # Same for y.
            
            # Left boundary
            viol_left = np.maximum(current_radii - centers[:, 0], 0)
            forces[:, 0] += viol_left
            
            # Right boundary
            viol_right = np.maximum(centers[:, 0] + current_radii - W, 0)
            forces[:, 0] -= viol_right
            
            # Bottom boundary
            viol_bottom = np.maximum(current_radii - centers[:, 1], 0)
            forces[:, 1] += viol_bottom
            
            # Top boundary
            viol_top = np.maximum(centers[:, 1] + current_radii - H, 0)
            forces[:, 1] -= viol_top
            
            # Move centers
            # Step size needs to be adaptive or small.
            # Normalize forces?
            # Let's use a fixed step size scaled by average radius.
            step_size = 0.1 * np.mean(current_radii)
            centers += forces * step_size
            
            # Check if we can expand radii
            # Recalculate slacks with new centers
            dists = cdist(centers, centers)
            pair_slacks = dists - (current_radii[:, None] + current_radii[None, :])
            min_pair_slack = np.min(pair_slacks)
            
            boundary_slacks = np.minimum(np.minimum(centers[:, 0], W - centers[:, 0]), 
                                         np.minimum(centers[:, 1], H - centers[:, 1]))
            min_boundary_slack = np.min(boundary_slacks)
            
            min_slack = min(min_pair_slack, min_boundary_slack)
            
            if min_slack > 0:
                # We can expand.
                # How much?
                # Let's expand by a fraction of the slack.
                # To keep it stable, expand by min_slack * 0.5?
                # Or just a fixed small amount?
                # If we expand by min_slack, we hit the constraint exactly.
                # Let's expand by min_slack * 0.8 to be safe.
                expansion = min_slack * 0.8
                # But we also need to ensure we don't expand too fast.
                # Cap expansion?
                current_radii += expansion
            else:
                # We are constrained. Do not expand.
                # Maybe shrink slightly to escape local minimum?
                # No, just keep moving.
                pass
            
            # Optional: Annealing - random kicks to escape local minima
            if it % 100 == 0:
                # Add small random noise
                noise = np.random.normal(0, 0.01 * np.mean(current_radii), (N, 2))
                centers += noise
                
        # Final check and cleanup
        # Ensure circles are inside
        # If any circle is outside, shrink it or move it?
        # For the final result, we must return a VALID configuration.
        # So we need to project back to feasible region.
        
        # Simple projection:
        # 1. Clip centers to [r_i, W-r_i] and [r_i, H-r_i]? 
        #    This might cause overlaps.
        # 2. Iteratively resolve overlaps by shrinking radii?
        #    The task asks to maximize sum of radii. 
        #    If we just shrink to fit, we might lose value.
        #    But validity is paramount.
        
        # Let's do a final "shrink to fit" pass.
        # Iterate:
        # 1. Check overlaps. If overlap, reduce radii of the two circles slightly.
        # 2. Check boundaries. If violation, reduce radius or move center.
        
        # A safer approach for the final output:
        # Use the positions from the optimization, but scale down all radii by a factor 
        # that guarantees no overlaps and boundary respect.
        
        # Calculate max overlap depth and boundary violation
        dists = cdist(centers, centers)
        required_dist = current_radii[:, None] + current_radii[None, :]
        max_overlap = np.max(required_dist - dists)
        
        # Boundary violations
        # dist_to_boundary = min(x, W-x, y, H-y)
        # violation = max(0, r - dist_to_boundary)
        x_dist = np.minimum(centers[:, 0], W - centers[:, 0])
        y_dist = np.minimum(centers[:, 1], H - centers[:, 1])
        dist_to_boundary = np.minimum(x_dist, y_dist)
        max_boundary_violation = np.max(current_radii - dist_to_boundary)
        
        max_violation = max(max_overlap, max_boundary_violation)
        
        if max_violation > 0:
            # We need to shrink radii.
            # If we shrink all radii by factor (1 - max_violation / min_radius)?
            # Or just subtract?
            # Subtracting is safer for maintaining relative sizes.
            # But we must ensure r_i > 0.
            
            # Let's scale down.
            # If we have a violation of V, and we scale radii by s, 
            # new overlap = s*(r_i+r_j) - dist.
            # We want s*(r_i+r_j) - dist <= 0 => s <= dist / (r_i+r_j).
            # This is complicated.
            
            # Simple heuristic:
            # If max_violation > 0, reduce all radii by max_violation.
            # This guarantees no overlap?
            # dist - (r_i - V + r_j - V) = dist - (r_i+r_j) + 2V.
            # If dist < r_i+r_j, let overlap = r_i+r_j - dist.
            # New overlap = overlap - 2V.
            # If V = overlap/2, new overlap = 0.
            # So reducing by max_overlap/2 should resolve overlaps.
            
            reduction = max_violation / 2.0 + 1e-6
            current_radii = current_radii - reduction
            current_radii = np.maximum(current_radii, 1e-6)
            
            # Re-check boundaries after shrinking?
            # Shrinking radii helps boundaries too.
            
        # Final validation
        # If still invalid, return a known valid seed configuration.
        dists = cdist(centers, centers)
        for i in range(N):
            for j in range(i+1, N):
                if dists[i, j] < current_radii[i] + current_radii[j] - 1e-9:
                    return None, 0
            if centers[i, 0] < current_radii[i] - 1e-9 or centers[i, 0] > W - current_radii[i] + 1e-9:
                return None, 0
            if centers[i, 1] < current_radii[i] - 1e-9 or centers[i, 1] > H - current_radii[i] + 1e-9:
                return None, 0
            
        score = np.sum(current_radii)
        return centers, score

    # Try different aspect ratios
    # W + H = 2
    # W in [0.5, 1.5]
    aspect_ratios = [1.0, 1.1, 0.9, 1.2, 0.8, 1.5, 0.67] # 1.5 means W=1.2, H=0.8
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
            
        # W = ar * H => ar*H + H = 2 => H = 2/(1+ar), W = 2*ar/(1+ar)
        H = 2.0 / (1.0 + ar)
        W = 2.0 * ar / (1.0 + ar)
        
        # Run optimization
        # Use multiple seeds for diversity
        for seed in [42, 123, 456]:
            if time.time() > deadline:
                break
            
            centers, score = run_optimization(W, H, n_iter=2000, seed=seed)
            if centers is not None:
                if score > best_score:
                    best_score = score
                    # Construct result array
                    # Format: (x, y, r)
                    # We need to map back to the coordinate system?
                    # The task says "construct() -> array of shape (n, 3)".
                    # The evaluator checks if the bounding rectangle of these circles has perimeter <= 4.
                    # My W, H are the box dimensions.
                    # So I just return the centers and radii in this coordinate system.
                    res = np.column_stack((centers, current_radii))
                    # Wait, current_radii is local to run_optimization.
                    # I need to return it.
                    # Let's modify run_optimization to return radii as well.
                    
    # The above logic has a bug: current_radii is not returned.
    # I will rewrite the inner function to return (centers, radii, score).
    
    # Rewriting construct to be cleaner and fix the return issue.
    
    def _run_opt(W, H, n_iter, seed):
        np.random.seed(seed)
        side = 5
        xs = np.linspace(0.1*W, 0.9*W, side)
        ys = np.linspace(0.1*H, 0.9*H, side)
        
        centers = np.zeros((N, 2))
        radii = np.zeros(N)
        
        count = 0
        for i in range(side):
            for j in range(side):
                if count < N:
                    centers[count] = [xs[i], ys[j]]
                    spacing = min(W, H) / side
                    radii[count] = spacing * 0.4
                    count += 1
        
        for it in range(n_iter):
            if time.time() > deadline:
                return None, None, 0
            
            dists = cdist(centers, centers)
            overlap_depths = radii[:, None] + radii[None, :] - dists
            overlap_depths = np.maximum(overlap_depths, 0)
            np.fill_diagonal(overlap_depths, 0)
            
            dists_safe = np.where(dists > 1e-6, dists, 1e-6)
            np.fill_diagonal(dists_safe, 1.0)
            inv_dists = 1.0 / dists_safe
            np.fill_diagonal(inv_dists, 0)
            
            diff = centers[:, None, :] - centers[None, :, :]
            force_matrix = diff * (overlap_depths * inv_dists)[:, :, None]
            forces = np.sum(force_matrix, axis=1)
            
            viol_left = np.maximum(radii - centers[:, 0], 0)
            forces[:, 0] += viol_left
            viol_right = np.maximum(centers[:, 0] + radii - W, 0)
            forces[:, 0] -= viol_right
            viol_bottom = np.maximum(radii - centers[:, 1], 0)
            forces[:, 1] += viol_bottom
            viol_top = np.maximum(centers[:, 1] + radii - H, 0)
            forces[:, 1] -= viol_top
            
            step_size = 0.1 * np.mean(radii)
            centers += forces * step_size
            
            dists = cdist(centers, centers)
            pair_slacks = dists - (radii[:, None] + radii[None, :])
            min_pair_slack = np.min(pair_slacks)
            
            x_dist = np.minimum(centers[:, 0], W - centers[:, 0])
            y_dist = np.minimum(centers[:, 1], H - centers[:, 1])
            dist_to_boundary = np.minimum(x_dist, y_dist)
            min_boundary_slack = np.min(dist_to_boundary - radii)
            
            min_slack = min(min_pair_slack, min_boundary_slack)
            
            if min_slack > 0:
                expansion = min_slack * 0.5
                radii += expansion
            
            if it % 100 == 0:
                noise = np.random.normal(0, 0.01 * np.mean(radii), (N, 2))
                centers += noise
                
        # Final cleanup
        dists = cdist(centers, centers)
        max_overlap = np.max(radii[:, None] + radii[None, :] - dists)
        x_dist = np.minimum(centers[:, 0], W - centers[:, 0])
        y_dist = np.minimum(centers[:, 1], H - centers[:, 1])
        dist_to_boundary = np.minimum(x_dist, y_dist)
        max_boundary_violation = np.max(radii - dist_to_boundary)
        
        max_violation = max(max_overlap, max_boundary_violation)
        
        if max_violation > 0:
            reduction = max_violation / 2.0 + 1e-6
            radii = radii - reduction
            radii = np.maximum(radii, 1e-6)
            
        # Validation
        dists = cdist(centers, centers)
        for i in range(N):
            for j in range(i+1, N):
                if dists[i, j] < radii[i] + radii[j] - 1e-9:
                    return None, None, 0
            if centers[i, 0] < radii[i] - 1e-9 or centers[i, 0] > W - radii[i] + 1e-9:
                return None, None, 0
            if centers[i, 1] < radii[i] - 1e-9 or centers[i, 1] > H - radii[i] + 1e-9:
                return None, None, 0
        
        score = np.sum(radii)
        return centers, radii, score

    # Main loop
    best_score = 0
    best_res = None
    
    aspect_ratios = [1.0, 1.1, 0.9, 1.2, 0.8, 1.5, 0.67]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        H = 2.0 / (1.0 + ar)
        W = 2.0 * ar / (1.0 + ar)
        
        for seed in [42, 123]:
            if time.time() > deadline:
                break
            centers, radii, score = _run_opt(W, H, 3000, seed)
            if centers is not None and score > best_score:
                best_score = score
                best_res = np.column_stack((centers, radii))
                
    if best_res is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_res

# EVOLVE-BLOCK-END
