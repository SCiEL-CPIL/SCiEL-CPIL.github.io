"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    # Safety margin: 150s limit, use 140s to be safe
    deadline = start_time + 140.0
    
    # Initial radius estimate based on area packing
    # Area of square = 1, Area of N circles = N * pi * r^2
    # r_max_theoretical = sqrt(1 / (N * pi))
    r_est = np.sqrt(1.0 / (N * np.pi))
    
    best_positions = None
    best_radius = 0.0
    best_score = -np.inf

    # Multi-start strategy
    # We will try a few different initializations:
    # 1. Jittered Grid
    # 2. Random Uniform
    # 3. Hexagonal Lattice (approx)
    
    initializers = []
    
    # 1. Jittered Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pos_grid = grid[:N].copy()
    pos_grid += np.random.uniform(-0.01, 0.01, size=pos_grid.shape)
    pos_grid = np.clip(pos_grid, -0.49, 0.49)
    initializers.append(pos_grid)
    
    # 2. Random Uniform
    pos_rand = np.random.uniform(-0.49, 0.49, size=(N, 2))
    initializers.append(pos_rand)
    
    # 3. Hexagonal Lattice (approx for square)
    # Hex lattice is denser. Let's generate a hex grid and clip/flatten.
    # Spacing s = 2*r_est / sqrt(3) * something? 
    # For hex packing, distance between centers is 2r.
    # Vertical spacing is 2r * sqrt(3)/2 = r*sqrt(3).
    # Horizontal spacing is 2r.
    # Let's just use a slightly denser grid to start.
    m_hex = int(np.ceil(np.sqrt(N * 1.2))) # slightly more points
    ticks_hex = (np.arange(m_hex) + 0.5) / m_hex - 0.5
    hex_grid = []
    for i, y in enumerate(ticks_hex):
        offset = 0.5 / m_hex if i % 2 == 0 else 0.0
        for x in ticks_hex:
            hex_grid.append((x + offset, y))
    hex_grid = np.array(hex_grid[:N])
    hex_grid += np.random.uniform(-0.005, 0.005, size=hex_grid.shape)
    hex_grid = np.clip(hex_grid, -0.49, 0.49)
    initializers.append(hex_grid)

    # Optimization parameters
    # We want to maximize r.
    # Objective: Minimize -soft_min(d_ij^2) 
    # To make it differentiable, use log-sum-exp approximation of min.
    # min(x_i) ≈ -1/beta * log(sum(exp(-beta * x_i)))
    # As beta -> inf, it approaches min.
    # We will optimize for a target radius r_target.
    # If we can place circles with radius r_target, then r_target is feasible.
    # However, direct optimization of max-min is better.
    
    # Let's define the objective function to minimize the "worst case" distance.
    # We want to maximize r = min( min_pair_dist/2, min_wall_dist )
    # Equivalent to minimizing -r.
    # Let's use a smooth approximation for the minimum.
    
    def _compute_smooth_min_radius(x_flat, beta=50.0):
        """
        Computes a smooth approximation of the maximum radius.
        x_flat: shape (2*N,)
        Returns: negative of the smooth min radius (to be minimized)
        """
        pos = x_flat.reshape(N, 2)
        
        # Wall distances
        # dist_wall_i = min(0.5 - x_i, 0.5 + x_i, 0.5 - y_i, 0.5 + y_i)
        # Smooth min of these 4 for each circle, then smooth min over all circles?
        # Actually, r is limited by the global minimum.
        # Let's compute all pairwise distances and wall distances.
        
        # 1. Pairwise distances
        # diff = pos[:, None, :] - pos[None, :, :]
        # dist_sq = sum(diff**2, axis=-1)
        # This creates a N x N matrix.
        # We only need i < j.
        # For efficiency in numpy, full matrix is fine for N=200 (40k elements).
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        
        # Get upper triangle
        iu = np.triu_indices(N, k=1)
        pair_dists_sq = dist_sq[iu]
        
        # 2. Wall distances
        # For each point, distance to 4 walls
        # x in [-0.5, 0.5], y in [-0.5, 0.5]
        # d_x_left = 0.5 + x
        # d_x_right = 0.5 - x
        # d_y_bottom = 0.5 + y
        # d_y_top = 0.5 - y
        # dist_wall_sq_i = min(d_x_left^2, d_x_right^2, d_y_bottom^2, d_y_top^2)
        # Actually, radius is limited by distance, not squared distance.
        # Let's work with squared distances for the log-sum-exp, then take sqrt at the end?
        # No, min(sqrt(A), sqrt(B)) = sqrt(min(A,B)).
        # So we can find min of squared distances, then take sqrt.
        
        # Wall dist sq for each point
        x, y = pos[:, 0], pos[:, 1]
        wall_dists_sq = np.stack([
            (0.5 + x)**2,
            (0.5 - x)**2,
            (0.5 + y)**2,
            (0.5 - y)**2
        ], axis=1)
        
        # We need the global minimum of all pair_dists_sq and wall_dists_sq (flattened)
        # Combine them
        all_dists_sq = np.concatenate([pair_dists_sq, wall_dists_sq.flatten()])
        
        # Smooth min using log-sum-exp
        # min(A) ≈ -1/beta * log(sum(exp(-beta * A)))
        # To avoid overflow, subtract max
        # Since we want min, and exp(-beta * A) is largest for smallest A.
        # Let A_min = min(all_dists_sq)
        # sum(exp(-beta * A)) = exp(-beta * A_min) * sum(exp(-beta * (A - A_min)))
        # log(sum(...)) = -beta * A_min + log(sum(exp(-beta * (A - A_min))))
        # min ≈ A_min - 1/beta * log(sum(exp(-beta * (A - A_min))))
        
        # However, calculating A_min explicitly makes it non-smooth.
        # Standard trick: use max(A) to stabilize exp.
        # But we are doing exp(-beta * A). If A is large, exp is small.
        # If A is small (close to 0), exp is large.
        # Let's just use the direct formula but be careful with scaling.
        # Since distances are bounded (max ~1.4), and beta=50, beta*A is max ~70.
        # exp(-70) is very small. exp(0) is 1.
        # So it's stable if we don't have huge A.
        
        # To be safer, we can normalize.
        # Let's try a simpler smooth min:
        # m = -1/beta * log( sum( exp(-beta * dist_sq) ) )
        
        # Potential issue: if dist_sq is 0, exp(0)=1.
        # If dist_sq is 1, exp(-50) ~ 1e-22.
        # So the sum is dominated by the smallest distances.
        
        # Calculate
        # We need to handle the case where values are very small.
        # Let's shift by the current minimum to ensure numerical stability?
        # No, the "soft min" formula without shifting is:
        # m = -1/beta * log( sum( exp(-beta * x_i) ) )
        # If x_i are large, exp(-beta*x_i) underflows to 0. This is fine, they don't contribute.
        # If x_i are small, they contribute.
        
        # However, if all x_i are large (e.g., > 10/beta), the sum might be 0 -> log(0) = -inf.
        # In our problem, min dist is usually > 0.01. beta=50. 50*0.01 = 0.5. exp(-0.5) ~ 0.6.
        # So it should be fine.
        
        # Let's add a small epsilon to avoid log(0) if everything is huge?
        # Unlikely for a packing problem.
        
        # Compute log-sum-exp
        # To prevent underflow of the sum if the min is not very small, 
        # we can factor out exp(-beta * min_approx).
        # But we don't know min_approx without a non-smooth op.
        # Let's just compute directly. N=200, pairs=~20k, walls=800. Total ~21k terms.
        
        # Optimization: 
        # The term exp(-beta * d_sq) is extremely small for d_sq > 1/beta * 10.
        # 1/50 * 10 = 0.2. d_sq = 0.2 => d ~ 0.45.
        # Most distances will be > 0.05 (d_sq = 0.0025).
        # exp(-50 * 0.0025) = exp(-0.125) ~ 0.88.
        # So many terms will contribute.
        
        # Let's compute
        vals = np.exp(-beta * all_dists_sq)
        s = np.sum(vals)
        
        if s == 0:
             # Fallback, should not happen
             return 0.0
             
        soft_min_d_sq = -1.0 / beta * np.log(s)
        
        # The actual radius is sqrt(soft_min_d_sq)
        # We want to MAXIMIZE radius.
        # So we MINIMIZE -radius.
        # Objective = -sqrt(soft_min_d_sq)
        
        # Note: soft_min_d_sq is an approximation of min_d_sq.
        # sqrt is monotonic.
        
        r_smooth = np.sqrt(soft_min_d_sq)
        
        # Add penalty if positions are outside?
        # The wall distances handle this. If x > 0.5, (0.5-x)^2 is small, 
        # but (0.5+x)^2 is large. The min will be small.
        # However, if x=0.6, (0.5-0.6)^2 = 0.01. 
        # The "distance" to wall is effectively 0.1? 
        # Wait, the distance from center to wall boundary is |0.5 - x|.
        # If x > 0.5, the circle is outside. 
        # The formula min(0.5-x, 0.5+x...) for x>0.5 gives 0.5-x which is negative.
        # Squared, it's positive.
        # But geometrically, the radius must be such that the circle is inside.
        # If the center is outside, the "maximum radius inside" is effectively 0 or negative?
        # The problem states centers must be inside.
        # We should enforce centers inside [-0.5, 0.5].
        # The optimizer might push them outside to "increase" distance to other circles?
        # No, pushing outside reduces wall distance (squared dist to wall becomes small if close to wall, 
        # but if far outside, dist is large. 
        # Example: x = 1.0. dist to right wall (0.5) is 0.5. dist to left wall (-0.5) is 1.5.
        # min is 0.5.
        # But the circle is not inside the square.
        # The score function in the task definition:
        # r = min( min_pair_dist/2, min_i min_k (1/2 - |c_ik|) )
        # If c_i > 0.5, then 1/2 - c_i is negative.
        # So r becomes negative.
        # We want to maximize r.
        # So the optimizer will naturally avoid negative r because it maximizes.
        # But our smooth function uses squared distances.
        # If x=0.6, wall_dist_sq = min((0.5-0.6)^2, (0.5+0.6)^2, ...) = 0.01.
        # r_smooth = sqrt(0.01) = 0.1.
        # This is WRONG. The actual r would be -0.1.
        # We need to penalize being outside.
        
        # Let's add a penalty for being outside the square.
        # Penalty = sum( max(0, |x| - 0.5)^2 )
        # This is smooth enough (kink at 0.5, but L-BFGS-B handles bounds).
        # We can use bounds in minimize!
        # bounds = [(-0.5, 0.5)] * (2*N)
        
        # If we use bounds, the optimizer will keep points inside.
        # Then wall distances are always positive.
        # So we just need to bound the variables.
        
        return -r_smooth

    # We will use bounds to keep points inside the square.
    bounds = [(-0.5, 0.5)] * (2 * N)

    # Run optimization for each initializer
    for idx, init_pos in enumerate(initializers):
        if time.time() > deadline - 10.0:
            break
            
        x0 = init_pos.flatten()
        
        # We can try increasing beta over time or use a fixed high beta.
        # High beta makes it closer to true min, but steeper (harder to optimize).
        # Low beta is smoother (easier to optimize) but less accurate.
        # Let's start with moderate beta and increase?
        # Or just use a fixed beta that works well.
        # Beta=50 is a good start.
        
        try:
            res = minimize(
                _compute_smooth_min_radius, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-6}
            )
            
            if res.success or res.status == 0:
                pos_res = res.x.reshape(N, 2)
                # Compute true radius
                r_true = _compute_radius(pos_res)
                
                if r_true > best_radius:
                    best_radius = r_true
                    best_positions = pos_res.copy()
                    best_score = r_true
                    
        except Exception as e:
            pass

    # If we still have time, do a local refinement on the best found positions
    # using a higher beta for better precision.
    if best_positions is not None and time.time() < deadline - 5.0:
        x0 = best_positions.flatten()
        # Use higher beta for refinement
        # We can't easily pass beta to the function without closure or args.
        # Let's define a wrapper or just accept the current precision.
        # Actually, let's just run one more minimize with the same function.
        # The previous run already converged.
        
        # A better refinement: 
        # The smooth min is an approximation. 
        # To get the exact max-min, we can do a binary search on r?
        # No, that's slow.
        
        # Let's just return the best found.
        
    if best_positions is None:
        # Fallback to initial grid if optimization failed completely
        best_positions = initializers[0]
        
    return best_positions

def _compute_radius(positions):
    """Compute the maximum possible radius for given positions."""
    if positions.shape[0] == 0:
        return 0.0
    
    x, y = positions[:, 0], positions[:, 1]
    
    # Check if any point is outside (should be clamped by bounds, but safe to check)
    # The score formula: min_i min_k (1/2 - |c_ik|)
    # If any c_ik > 0.5, this term is negative.
    # The min will be negative.
    # We want to maximize this.
    
    # Wall distances (positive if inside)
    # dist to x=0.5 is 0.5 - x
    # dist to x=-0.5 is 0.5 + x
    # dist to y=0.5 is 0.5 - y
    # dist to y=-0.5 is 0.5 + y
    
    wall_dists = np.min(np.stack([
        0.5 - x,
        0.5 + x,
        0.5 - y,
        0.5 + y
    ], axis=1), axis=1)
    
    min_wall_dist = np.min(wall_dists)
    
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dist_sq))
    
    # Radius is half the min of these
    r = min(min_wall_dist, min_pair_dist / 2.0)
    
    # If r is negative, it means a circle is outside.
    # The evaluator will likely return 0 or invalid for this, 
    # but mathematically r is negative.
    # The task says "maximize r". 
    # If we return positions that are outside, r is negative.
    # We should ensure we return valid positions (inside).
    # Since we used bounds, they should be inside.
    
    return r

# EVOLVE-BLOCK-END
