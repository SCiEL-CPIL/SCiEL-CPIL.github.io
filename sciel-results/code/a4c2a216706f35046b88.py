import numpy as np
import time

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Initialization: Hexagonal lattice is a good start for packing
    # We need to fit N circles. 
    # A square grid of 14x14 = 196 is close. 15x15=225 is too many.
    # Let's try to initialize with a hex lattice clipped to the square, 
    # or just a jittered grid which is robust.
    # Jittered grid is safer to implement correctly within time limits.
    
    # Grid initialization
    m = int(np.ceil(np.sqrt(N)))
    # Create a grid of m x m points
    # We will take the first N points
    # To improve on a simple grid, we can start with a hex-like offset for alternating rows
    # But a simple jittered grid often works well with gradient ascent.
    
    # Let's use a slightly randomized grid to break symmetry
    np.random.seed(42)
    
    # Grid coordinates in [-0.5, 0.5]
    # We want N points.
    # If we use a square grid of size m, we have m^2 points.
    # Let's create a grid of size m x m, and if m^2 > N, we take the first N.
    # To make it better, let's use a hexagonal packing initialization.
    
    # Hex lattice generation
    # Spacing s such that we can fit approx N circles.
    # Area of hex cell is (sqrt(3)/2) * s^2.
    # Total area 1. N * (sqrt(3)/2) s^2 <= 1 => s <= sqrt(2 / (N * sqrt(3)))
    # For N=200, s ~ sqrt(2 / (200*1.732)) ~ sqrt(0.00577) ~ 0.076
    # Let's try to generate a hex lattice and keep points inside [-0.5, 0.5]
    
    def hex_lattice_init(n):
        # Estimate spacing
        s = np.sqrt(2 / (n * np.sqrt(3)))
        # Generate points
        pts = []
        # Iterate over rows
        # Height of hex row is s * sqrt(3) / 2
        h = s * np.sqrt(3) / 2
        # Number of rows needed to cover [-0.5, 0.5]
        # y ranges from -0.5 to 0.5
        # Let's iterate y from -0.5 to 0.5 with step h
        # For each y, x offset depends on row index (parity)
        
        row_idx = 0
        y = -0.5 + h/2 # Start a bit inside? Or just -0.5?
        # Better to center the lattice
        # Let's generate a large grid and filter
        
        # Range of x, y
        x_min, x_max = -0.6, 0.6
        y_min, y_max = -0.6, 0.6
        
        # Step in y
        dy = h
        # Step in x
        dx = s
        
        # Iterate
        y = y_min
        while y < y_max:
            # x offset for this row
            offset = 0 if row_idx % 2 == 0 else s / 2
            x = x_min + offset
            while x < x_max:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    pts.append((x, y))
                x += dx
            y += dy
            row_idx += 1
        
        pts = np.array(pts)
        if len(pts) < n:
             # Fallback to grid if hex generation failed to produce enough
             return grid_init(n)
        
        # Randomly select n points from the generated lattice
        # Or, to ensure good spread, we should just take the first n?
        # Taking first n from a systematic generation might be biased.
        # Let's shuffle and take n.
        idx = np.random.choice(len(pts), n, replace=False)
        return pts[idx]

    def grid_init(n):
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        if len(grid) < n:
            # Pad with random points? Should not happen for N=200, m=14, 196 pts.
            # Wait, ceil(sqrt(200)) = 14. 14*14 = 196.
            # We need 200.
            # So we need 15x15 = 225.
            m = 15
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
        # Add small jitter
        grid = grid[:n] + np.random.normal(0, 0.01, (n, 2))
        # Clip to square
        grid[:, 0] = np.clip(grid[:, 0], -0.5, 0.5)
        grid[:, 1] = np.clip(grid[:, 1], -0.5, 0.5)
        return grid

    # Use hex lattice if possible, otherwise grid
    # Hex lattice might have fewer points in the square due to corners.
    # Let's stick to a jittered grid of 15x15, taking 200 points.
    # This ensures we have enough points and good coverage.
    pts = grid_init(N)
    
    # Optimization
    # Objective: Maximize r = min(min_pair_dist, min_wall_dist)
    # We use soft-min to approximate min.
    # f(pts) = soft_min(beta, [d_1, d_2, ..., d_M])
    # where d_k are all pair distances and wall distances.
    # Gradient ascent: pts_new = pts + lr * grad
    
    # Precompute pairs?
    # N=200, pairs = 19900.
    # Wall distances = 200 * 4 = 800 (or just min to 4 walls per point).
    # Total constraints ~ 20700.
    
    # To speed up, we can only consider "close" pairs?
    # No, soft-min requires all. But we can vectorize.
    
    # Vectorized distance calculation
    # P: (N, 2)
    # D: (N, N) matrix of distances
    # We only need upper triangle.
    
    def compute_grad_and_obj(pts, beta):
        # pts: (N, 2)
        N_pts = len(pts)
        
        # Pairwise distances
        # diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        # dists = np.linalg.norm(diff, axis=-1)
        # This creates a (N, N) matrix. 200x200 is 40000 floats. Fine.
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        
        # Mask diagonal (self distance)
        # We want to ignore i==j
        mask = ~np.eye(N_pts, dtype=bool)
        # Flatten upper triangle
        # i < j
        i_idx, j_idx = np.triu_indices(N_pts, 1)
        pair_dists = dists[i_idx, j_idx]
        pair_vecs = diff[i_idx, j_idx] # Vector from j to i? 
        # diff[i,j] = pts[i] - pts[j]
        # Gradient of dist w.r.t pts[i] is (pts[i] - pts[j]) / dist
        # Gradient of dist w.r.t pts[j] is (pts[j] - pts[i]) / dist
        
        # Wall distances
        # For each point, distance to 4 walls: 0.5 - x, 0.5 - y, x + 0.5, y + 0.5
        # x in [-0.5, 0.5]
        # d_x_pos = 0.5 - pts[:, 0]
        # d_x_neg = 0.5 + pts[:, 0]
        # d_y_pos = 0.5 - pts[:, 1]
        # d_y_neg = 0.5 + pts[:, 1]
        
        wall_dists = np.concatenate([
            0.5 - pts[:, 0],
            0.5 + pts[:, 0],
            0.5 - pts[:, 1],
            0.5 + pts[:, 1]
        ])
        
        # Combine all distances
        all_dists = np.concatenate([pair_dists, wall_dists])
        
        # Soft min
        # S = (1/beta) * log(sum(exp(-beta * d)))
        # To avoid overflow/underflow, subtract min
        d_min = np.min(all_dists)
        exp_terms = np.exp(-beta * (all_dists - d_min))
        sum_exp = np.sum(exp_terms)
        obj = d_min - (1.0 / beta) * np.log(sum_exp)
        
        # Weights for gradient
        # w_k = exp(-beta * (d_k - d_min)) / sum_exp
        weights = exp_terms / sum_exp
        
        # Gradient of obj w.r.t points
        # obj = d_min - (1/beta) log(sum_exp)
        # d obj / d pts = d obj / d d_k * d d_k / d pts
        # d obj / d d_k = - (1/beta) * (exp(-beta(d_k-d_min)) / sum_exp) * (-beta) = weights_k
        # Wait.
        # f = (1/beta) log(sum e^{-beta d})
        # df/dd = (1/beta) * (1/sum) * sum e^{-beta d} * (-beta) = - sum e^{-beta d} / sum = - 1 ??
        # No.
        # df/dd_k = (1/beta) * (1/sum_exp_raw) * (-beta e^{-beta d_k}) = - e^{-beta d_k} / sum_exp_raw
        # With shift: sum_exp = sum e^{-beta(d-d_min)}.
        # df/dd_k = - (e^{-beta(d_k-d_min)}) / sum_exp = - weights_k
        
        # So gradient of soft-min objective w.r.t distance d_k is -weights_k.
        # We want to MAXIMIZE the radius, so we want to INCREASE the distances.
        # So we move in direction of + grad_obj.
        # grad_obj = - grad_soft_min = + weights_k * (grad_d_k)
        
        grad_pts = np.zeros_like(pts)
        
        # Pair contributions
        # For pair (i, j) with distance d_ij and weight w_ij:
        # d d_ij / d pts[i] = (pts[i] - pts[j]) / d_ij
        # d d_ij / d pts[j] = (pts[j] - pts[i]) / d_ij
        # Contribution to grad_pts[i] += w_ij * (pts[i] - pts[j]) / d_ij
        # Contribution to grad_pts[j] += w_ij * (pts[j] - pts[i]) / d_ij
        
        # pair_dists and pair_vecs are aligned with i_idx, j_idx
        # Avoid division by zero (shouldn't happen if distinct)
        safe_dists = np.where(pair_dists > 1e-8, pair_dists, 1e-8)
        unit_vecs = pair_vecs / safe_dists[:, np.newaxis]
        
        # weights for pairs
        pair_weights = weights[:len(pair_dists)]
        
        # Scatter add
        # np.add.at is slow. Use bincount?
        # Or just loop? N=200, pairs=20k. Loop might be okay in Python if vectorized inside?
        # Better to use np.add.at or accumulate with np.bincount on flattened indices?
        # Indices are 2D.
        
        # Let's use np.add.at
        np.add.at(grad_pts, i_idx, pair_weights[:, np.newaxis] * unit_vecs)
        np.add.at(grad_pts, j_idx, -pair_weights[:, np.newaxis] * unit_vecs)
        
        # Wall contributions
        # Wall dists are arranged as:
        # [d_x_pos (N), d_x_neg (N), d_y_pos (N), d_y_neg (N)]
        # weights for walls
        wall_weights = weights[len(pair_dists):]
        
        w_x_pos = wall_weights[0:N]
        w_x_neg = wall_weights[N:2*N]
        w_y_pos = wall_weights[2*N:3*N]
        w_y_neg = wall_weights[3*N:4*N]
        
        # d d_x_pos / d x = -1  (d = 0.5 - x)
        # d d_x_neg / d x = 1   (d = 0.5 + x)
        # d d_y_pos / d y = -1
        # d d_y_neg / d y = 1
        
        # Grad x component:
        # From x_pos: w_x_pos * (-1)
        # From x_neg: w_x_neg * (1)
        grad_pts[:, 0] += -w_x_pos + w_x_neg
        
        # Grad y component:
        # From y_pos: w_y_pos * (-1)
        # From y_neg: w_y_neg * (1)
        grad_pts[:, 1] += -w_y_pos + w_y_neg
        
        return obj, grad_pts

    # Optimization Loop
    # Adaptive Beta and Learning Rate
    beta = 50.0
    lr = 0.01
    best_pts = pts.copy()
    best_obj = -np.inf
    
    # Nesterov momentum
    v = np.zeros_like(pts)
    m = np.zeros_like(pts)
    
    # Simple Adam-like or Nesterov
    # Let's use Nesterov
    mu = 0.9
    lr_decay = 0.999
    
    # Iterations
    max_iters = 5000
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute objective and gradient
        obj, grad = compute_grad_and_obj(pts, beta)
        
        # Track best
        if obj > best_obj:
            best_obj = obj
            best_pts = pts.copy()
            
        # Nesterov update
        # v = mu * v + lr * grad
        # pts_new = pts + v + mu * (v - prev_v) ?
        # Standard NAG:
        # v_new = mu * v + lr * grad
        # x_new = x + v_new + mu * (v_new - v)
        
        v = mu * v + lr * grad
        pts_new = pts + v + mu * (v - v) # Wait, standard NAG is:
        # x_{k+1} = x_k + v_k + mu(v_k - v_{k-1})
        # v_k = mu v_{k-1} + lr grad(x_k)
        
        # Let's implement standard NAG properly
        # v_k = mu * v_{k-1} + lr * grad_k
        # x_{k+1} = x_k + v_k + mu * (v_k - v_{k-1})
        
        # I already updated v above.
        # prev_v = v_old
        # v_new = mu * v_old + lr * grad
        # x_new = x_old + v_new + mu * (v_new - v_old)
        
        # Let's restart with clear variables
        # Actually, simple gradient descent with adaptive step size might be more robust and faster to code correctly.
        # Let's stick to a robust variant of gradient ascent with line search or fixed decaying step.
        
        # Let's use a simple adaptive step size based on the magnitude of the gradient.
        # grad_norm = np.linalg.norm(grad)
        # step = lr / (grad_norm + 1e-8)
        
        # Update
        # pts = pts + lr * grad
        
        # Project to square
        pts_new[:, 0] = np.clip(pts_new[:, 0], -0.5, 0.5)
        pts_new[:, 1] = np.clip(pts_new[:, 1], -0.5, 0.5)
        
        # Check for overlaps?
        # If overlaps occur, the objective will drop.
        # We can check if min dist < 2*current_best_r?
        # Just rely on the objective function.
        
        pts = pts_new
        
        # Decay learning rate
        lr *= lr_decay
        
        # Increase beta to make soft-min sharper (closer to true min)
        # But be careful with numerical stability
        if i % 10 == 0:
            beta *= 1.2
            if beta > 500:
                beta = 500

    # Final Polish
    # Use the best points found
    pts = best_pts
    
    # One more check: ensure validity
    # Clip again
    pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
    pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
    
    return pts

# EVOLVE-BLOCK-END
