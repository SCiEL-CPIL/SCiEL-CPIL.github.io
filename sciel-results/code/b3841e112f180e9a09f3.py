"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def compute_radius(centers):
        """Compute the valid radius for a given set of centers."""
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        
        # Wall distances
        min_dist_to_wall = np.min(0.5 - np.abs(centers))
        if min_dist_to_wall < 0:
            return 0.0
        
        # Pairwise distances
        # Using a loop for memory efficiency in Python if needed, but vectorized is faster for N=200
        min_pair_dist = np.inf
        for i in range(N):
            diffs = centers[i+1:] - centers[i]
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            if len(dists) > 0:
                d = np.min(dists)
                if d < min_pair_dist:
                    min_pair_dist = d
        
        if not np.isfinite(min_pair_dist):
            return 0.0
            
        r = min(min_dist_to_wall, min_pair_dist / 2.0)
        return max(0.0, r)

    def generate_hex_lattice(n_rows, n_cols):
        """Generate a hexagonal lattice inside the unit square."""
        pts = []
        if n_rows <= 0 or n_cols <= 0:
            return np.array([])
        
        # Determine spacing to fit in square
        # Width: n_cols * d <= 1.0  => d <= 1/n_cols
        # Height: (n_rows-1)*sqrt(3)/2*d + d <= 1.0 => d <= 1 / (1 + (n_rows-1)*sqrt(3)/2)
        d_max_x = 1.0 / n_cols
        d_max_h = 1.0 / (1.0 + (n_rows - 1) * math.sqrt(3) / 2.0)
        d = min(d_max_x, d_max_h)
        
        # Center the lattice
        width = (n_cols - 1) * d
        height = (n_rows - 1) * d * math.sqrt(3) / 2.0
        
        x_start = -0.5 + (1.0 - width) / 2.0
        y_start = -0.5 + (1.0 - height) / 2.0
        
        for row in range(n_rows):
            y = y_start + row * d * math.sqrt(3) / 2.0
            if row % 2 == 0:
                n_in_row = n_cols
                x_offset = 0.0
            else:
                n_in_row = n_cols - 1
                x_offset = d / 2.0
            
            for col in range(n_in_row):
                x = x_start + x_offset + col * d
                pts.append([x, y])
        
        pts = np.array(pts)
        
        # Adjust to exactly N points
        if len(pts) > N:
            pts = pts[:N]
        elif len(pts) < N:
            # Pad with random points in valid range
            while len(pts) < N:
                x = np.random.uniform(-0.5 + d/2, 0.5 - d/2)
                y = np.random.uniform(-0.5 + d/2, 0.5 - d/2)
                pts = np.vstack([pts, [[x, y]]])
        
        return pts

    def soft_min_objective_and_grad(centers, beta):
        """
        Compute the negative soft-min radius and its gradient.
        We want to maximize r = min(wall_dist, pair_dist/2).
        Let f = -r. We minimize f.
        f = -min(...) = max(-wall_dist, -pair_dist/2).
        
        Using log-sum-exp for smooth max:
        M(x) = (1/beta) * log(sum(exp(beta * x_i)))
        
        Constraints:
        1. Wall: x_w = - (0.5 - |c|) = |c| - 0.5. We want this <= 0.
           Actually, let's define the "violations" or "costs" such that minimizing them maximizes the radius.
           
        Let's define the objective as minimizing the "soft minimum radius" directly.
        r_soft = - (1/beta) log( sum( exp(-beta * d_walls) ) + sum( exp(-beta * d_pairs/2) ) )
        
        Wait, standard soft min: min(x_i) ≈ -1/beta log(sum exp(-beta x_i)).
        We want to maximize this. So we minimize -soft_min = 1/beta log(sum exp(-beta x_i)).
        
        Let x_w = 0.5 - |c| (distance to wall)
        Let x_p = ||c_i - c_j|| / 2 (half distance between pairs)
        
        Objective J(c) = 1/beta * log( sum_{walls} exp(-beta * x_w) + sum_{pairs} exp(-beta * x_p) )
        
        Gradient w.r.t c:
        dJ/dc = 1/beta * [ 1/S * sum_i (-beta * exp(-beta x_i)) * (d x_i / dc) ]
              = - [ sum_i exp(-beta x_i) * (d x_i / dc) ] / S
              where S = sum_i exp(-beta x_i)
        
        Note: x_w = 0.5 - |c_k|. dx_w/dc_k = -sign(c_k).
        x_p = ||c_i - c_j|| / 2. dx_p/dc_i = (c_i - c_j) / (2 * ||c_i - c_j||) * 1? 
        Actually d(||u||)/du = u/||u||.
        So d x_p / dc_i = (c_i - c_j) / (2 * ||c_i - c_j||)
        And d x_p / dc_j = (c_j - c_i) / (2 * ||c_i - c_j||)
        """
        c = centers
        n = len(c)
        
        # Wall terms
        # x_w = 0.5 - |c|
        abs_c = np.abs(c)
        x_w = 0.5 - abs_c
        # Avoid log(0) or exp overflow issues by clipping x_w to a small positive value if negative?
        # If x_w < 0, the point is outside. exp(-beta * x_w) becomes huge. This is good for penalty.
        
        # Pair terms
        # We need to compute all pair distances.
        # For N=200, N^2/2 = 20k pairs. This is manageable.
        
        # Initialize gradient accumulator
        grad = np.zeros_like(c)
        
        # Wall contribution to numerator and denominator
        # exp_w = exp(-beta * x_w)
        # To prevent overflow, we can subtract the max x from the exponent?
        # Actually, if x is very negative (outside), -beta*x is very positive -> overflow.
        # Let's clip x_w to a minimum value, say -0.1. Or just let it be, numpy handles inf somewhat, but grad becomes 0 if S is inf?
        # Better: use a stable log-sum-exp.
        
        # However, for gradient descent, we can just compute the weights.
        # weights_w = exp(-beta * x_w) / S
        # weights_p = exp(-beta * x_p) / S
        
        # Let's collect all x values
        x_vals = []
        # Wall values
        x_w_flat = x_w.flatten()
        x_vals.append(x_w_flat)
        
        # Pair values
        # We need indices to map gradients back.
        # Instead of storing all, let's compute S and the weighted sum of gradients in one pass?
        # Or store x_p and indices.
        
        # For N=200, storing 19900 floats is trivial.
        x_p_list = []
        idx_i_list = []
        idx_j_list = []
        
        for i in range(n):
            for j in range(i+1, n):
                diff = c[i] - c[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    dist = 1e-10
                x_p = dist / 2.0
                x_p_list.append(x_p)
                idx_i_list.append(i)
                idx_j_list.append(j)
                
        x_p_arr = np.array(x_p_list)
        x_vals.append(x_p_arr)
        
        x_all = np.concatenate(x_vals)
        
        # Stable log-sum-exp for the objective value (not strictly needed for grad, but good for monitoring)
        max_x = np.max(x_all)
        # J = 1/beta * log(sum(exp(-beta * x))) = 1/beta * ( -beta * max_x + log(sum(exp(-beta * (x - max_x)))) )
        #    = -max_x + 1/beta * log(sum(exp(-beta * (x - max_x))))
        
        # For gradient:
        # grad = - [ sum exp(-beta x_i) * dx_i/dc ] / sum exp(-beta x_i)
        # Let S = sum exp(-beta x_i)
        # weights_i = exp(-beta x_i) / S
        
        # To avoid overflow in exp, shift by max_x:
        # exp(-beta x_i) = exp(-beta max_x) * exp(-beta (x_i - max_x))
        # The factor exp(-beta max_x) cancels in the ratio.
        # So weights_i = exp(-beta (x_i - max_x)) / sum(exp(-beta (x_j - max_x)))
        
        shifted_x = x_all - max_x
        exp_shifted = np.exp(-beta * shifted_x)
        S = np.sum(exp_shifted)
        weights = exp_shifted / S
        
        # Now compute gradient
        # Wall part
        w_w = weights[:N*2] # First N*2 are walls (2 per point)
        # x_w = 0.5 - |c|. d x_w / dc = -sign(c)
        # grad_wall = - sum(weights_w * (-sign(c))) = sum(weights_w * sign(c))
        # Wait, formula was grad = - [ sum w_i dx_i/dc ] / 1 (since S is in w)
        # Let's re-verify sign.
        # J = 1/beta log S. dJ = 1/(beta S) dS.
        # dS = sum exp(-beta x_i) (-beta dx_i) = -beta sum exp(-beta x_i) dx_i
        # dJ = -1/S sum exp(-beta x_i) dx_i = - sum (exp(-beta x_i)/S) dx_i = - sum w_i dx_i
        
        # So grad_c = - sum_i w_i (dx_i/dc)
        
        # Wall: dx_w/dc = -sign(c)
        # Contribution: - w_w * (-sign(c)) = w_w * sign(c)
        
        # Reshape w_w to (N, 2)
        w_w_reshaped = w_w.reshape(N, 2)
        grad += w_w_reshaped * np.sign(c)
        
        # Pair part
        w_p = weights[N*2:]
        # dx_p/dc_i = (c_i - c_j) / (2 * dist)
        # dx_p/dc_j = (c_j - c_i) / (2 * dist)
        
        # We need to accumulate these.
        # For each pair (i, j) with weight w_p[k]:
        # grad[i] -= w_p[k] * (c[i] - c[j]) / (2 * dist)
        # grad[j] -= w_p[k] * (c[j] - c[i]) / (2 * dist)
        
        # To do this efficiently without a Python loop over 20k pairs if possible, or just use a loop since N=200 is small.
        # A loop over 19900 iterations in Python might take ~10-50ms. Acceptable for a few hundred iterations.
        
        idx_i = np.array(idx_i_list)
        idx_j = np.array(idx_j_list)
        
        # Re-compute distances and unit vectors? We have x_p_arr which is dist/2.
        # dist = 2 * x_p_arr
        # unit_vec = (c_i - c_j) / dist
        # dx_p/dc_i = unit_vec / 2
        
        # Let's use numpy advanced indexing to get c_i and c_j
        c_i = c[idx_i]
        c_j = c[idx_j]
        diffs = c_i - c_j
        dists = 2.0 * x_p_arr
        # Avoid div by zero
        dists_safe = np.where(dists < 1e-10, 1e-10, dists)
        unit_vecs = diffs / dists_safe[:, np.newaxis]
        
        # dx_p/dc_i = unit_vecs / 2
        # Contribution to grad[i] = - w_p * (unit_vecs / 2)
        # Contribution to grad[j] = - w_p * (-unit_vecs / 2) = + w_p * (unit_vecs / 2)
        
        # We need to scatter these into grad.
        # grad += scatter(idx_i, -w_p * unit_vecs / 2)
        # grad += scatter(idx_j, +w_p * unit_vecs / 2)
        
        # np.add.at is the way
        np.add.at(grad, idx_i, -0.5 * w_p[:, np.newaxis] * unit_vecs)
        np.add.at(grad, idx_j,  0.5 * w_p[:, np.newaxis] * unit_vecs)
        
        # Objective value
        obj_val = -max_x + (1.0/beta) * np.log(S)
        
        return obj_val, grad

    def optimize(centers_init, beta_schedule, iters_per_beta, lr_init, lr_decay, momentum=0.9):
        """
        Perform projected gradient descent with momentum.
        """
        centers = centers_init.copy()
        vel = np.zeros_like(centers)
        
        best_centers = centers.copy()
        best_radius = compute_radius(centers)
        
        current_lr = lr_init
        
        for beta in beta_schedule:
            for it in range(iters_per_beta):
                if time.time() > deadline - 5:
                    return best_centers
                
                obj, grad = soft_min_objective_and_grad(centers, beta)
                
                # Update velocity
                vel = momentum * vel - current_lr * grad
                centers = centers + vel
                
                # Project to square [-0.5, 0.5]
                centers = np.clip(centers, -0.5, 0.5)
                
                # Check improvement
                r = compute_radius(centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = centers.copy()
                    
                # Decay learning rate
                current_lr *= lr_decay
                
        return best_centers

    # Main logic
    # Generate multiple initializations
    inits = []
    
    # Hex lattices with different aspect ratios
    for rows in [13, 14, 15, 16]:
        cols = math.ceil(N / rows)
        pts = generate_hex_lattice(rows, cols)
        if len(pts) == N:
            # Add small jitter to break symmetry
            jitter = np.random.normal(0, 0.001, size=pts.shape)
            pts_jittered = pts + jitter
            pts_jittered = np.clip(pts_jittered, -0.5, 0.5)
            inits.append(pts_jittered)
            
    # Square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N]
    jitter = np.random.normal(0, 0.001, size=grid.shape)
    grid_jittered = grid + jitter
    grid_jittered = np.clip(grid_jittered, -0.5, 0.5)
    inits.append(grid_jittered)
    
    best_centers = None
    best_radius = 0.0
    
    # Optimization parameters
    # Start with low beta (smooth), increase to high beta (sharp)
    # beta = 10 is smooth, beta = 500 is sharp
    beta_schedule = [10, 30, 100, 300, 1000]
    iters_per_beta = 100
    lr_init = 0.005
    lr_decay = 0.99
    
    for init in inits:
        if time.time() > deadline - 10:
            break
            
        # Run optimization
        result = optimize(init, beta_schedule, iters_per_beta, lr_init, lr_decay)
        
        r = compute_radius(result)
        if r > best_radius:
            best_radius = r
            best_centers = result.copy()
            
    # Final polish: try to squeeze out more with a very high beta and low lr
    if best_centers is not None and time.time() < deadline - 5:
        # High beta polish
        beta_schedule_polish = [1000, 3000, 10000]
        iters_polish = 50
        lr_polish = 0.001
        lr_decay_polish = 0.98
        
        polished = optimize(best_centers, beta_schedule_polish, iters_polish, lr_polish, lr_decay_polish)
        r = compute_radius(polished)
        if r > best_radius:
            best_radius = r
            best_centers = polished.copy()
            
    # Fallback
    if best_centers is None:
        best_centers = inits[0]
        
    return best_centers
# EVOLVE-BLOCK-END
