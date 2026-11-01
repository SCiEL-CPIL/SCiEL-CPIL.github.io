"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    """Construct a good packing of 200 equal circles in the unit square [-1/2, 1/2]^2."""
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # 1. Initial Configuration: Hexagonal-like grid
    # A good starting point is crucial. We use a hexagonal lattice stretched to fit.
    # For 200 circles, a 14x15 grid is close (210). Let's use a hex grid with some rows.
    # Or simply a dense grid.
    # Let's try a hexagonal packing initialization.
    
    def generate_hex_init(n):
        # Approximate hexagonal packing
        # r ~ 0.0366
        # Let's just use a grid for simplicity and robustness, then let optimizer fix it.
        # Grid is easier to vectorize for LSE.
        # sqrt(200) ~ 14.1
        rows = 14
        cols = 15
        # We need 200. 14*15 = 210. Remove 10 from edges?
        # Better: Generate 210, keep 200 best? Or just use 200 in a 14x15 grid with 10 missing.
        
        # Let's use a standard hex pattern
        centers = []
        # Target area per circle ~ 1/200 = 0.005
        # Hex area per circle ~ sqrt(3)/2 * (2r)^2 = 2*sqrt(3)*r^2
        # r ~ 0.0366 -> area ~ 2*1.732*0.00134 ~ 0.00466. Fits well.
        
        # Let's just create a grid of 15x15 = 225 and take 200?
        # No, let's do a proper hex init.
        
        # Simple robust init: Grid
        m = 15
        # Create 15x15 grid
        xs = np.linspace(-0.48, 0.48, m)
        ys = np.linspace(-0.48, 0.48, m)
        xx, yy = np.meshgrid(xs, ys)
        grid = np.column_stack((xx.ravel(), yy.ravel()))
        
        # We need 200. 
        # Sort by distance from center? No, just take first 200? 
        # Better: Randomly remove 25 from the 225 grid to avoid symmetry locks?
        # Or just use 14x15 = 210 and remove 10?
        
        # Let's use 14 rows of 15, but shift odd rows.
        # Actually, for LSE optimization, a symmetric grid is a saddle point.
        # We need to break symmetry.
        
        # Strategy: Generate 200 points on a hexagonal lattice.
        # Hex lattice vectors:
        # a1 = (1, 0), a2 = (0.5, sqrt(3)/2)
        # Scale factor s.
        
        # Let's try to fit 200 circles.
        # Rows ~ 14, Cols ~ 15.
        
        centers_list = []
        # We will generate a 15x15 grid and randomly pick 200?
        # No, let's generate a hex grid of ~200 points.
        
        # Let's just use a 14x15 grid (210 points) and drop 10 random ones.
        rows = 14
        cols = 15
        dx = 0.95 / (cols - 1)
        dy = 0.95 / (rows - 1)
        
        for i in range(rows):
            for j in range(cols):
                x = -0.475 + j * dx
                y = -0.475 + i * dy
                if i % 2 == 1:
                    x += dx / 2
                # Clamp to ensure inside
                x = np.clip(x, -0.49, 0.49)
                y = np.clip(y, -0.49, 0.49)
                centers_list.append([x, y])
        
        centers = np.array(centers_list)
        # We have 210. Remove 10 random ones to get 200.
        indices = np.random.choice(len(centers), 200, replace=False)
        return centers[indices]

    centers = generate_hex_init(N)
    
    # 2. Optimization using Log-Sum-Exp (LSE) surrogate
    # Objective: Maximize min(d_ij, d_iw)
    # LSE approximation: -log(sum(exp(-d_ij/beta))) ~ min(d_ij) for large beta.
    # We want to maximize this, so we minimize -LSE = log(sum(exp(-d/beta))).
    
    def compute_clearances(c):
        """Compute all pairwise distances and wall distances."""
        n = len(c)
        # Pairwise
        dx = c[:, 0][:, None] - c[:, 0][None, :]
        dy = c[:, 1][:, None] - c[:, 1][None, :]
        dists_sq = dx**2 + dy**2
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        
        # Walls
        wall_dists = np.stack([
            0.5 - c[:, 0],
            c[:, 0] + 0.5,
            0.5 - c[:, 1],
            c[:, 1] + 0.5
        ], axis=1)
        
        return dists, wall_dists

    def lse_objective(c, beta=50.0):
        """Minimize the LSE of inverse distances (equivalent to maximizing min distance)."""
        # To avoid overflow, we use the log-sum-exp trick.
        # We want to minimize log(sum(exp(-d/beta))).
        # This is equivalent to minimizing the "soft-min" of the distances.
        
        dists, wall_dists = compute_clearances(c)
        
        # Flatten distances
        # We only care about distances < some threshold to save time?
        # For N=200, N^2 = 40000. It's fine.
        
        # Combine all distances
        # Note: exp(-d/beta) is large for small d.
        # We want to penalize small d.
        
        # d_min_soft = -beta * log( sum(exp(-d/beta)) )
        # We want to maximize d_min_soft, so minimize -d_min_soft = beta * log( sum(exp(-d/beta)) )
        
        # To be numerically stable:
        # Let m = min(d). sum(exp(-d/beta)) = exp(-m/beta) * sum(exp(-(d-m)/beta))
        # log(sum) = -m/beta + log(sum(exp(-(d-m)/beta)))
        # Objective = beta * ( -m/beta + log(sum(exp(-(d-m)/beta))) ) = -m + beta * log(sum(exp(-(d-m)/beta)))
        
        # Actually, simpler: just minimize the sum of exp(-d/beta).
        # Gradient is easy.
        
        # Let's use the sum of exp(-d/beta) as the objective.
        # This pushes all small distances up.
        
        # To save memory/time, we can compute in chunks or just use the full matrix.
        # 200x200 is small.
        
        # We exclude diagonal (inf)
        flat_dists = dists.ravel()
        flat_dists = flat_dists[flat_dists < np.inf]
        
        wall_flat = wall_dists.ravel()
        
        all_dists = np.concatenate([flat_dists, wall_flat])
        
        # Soft-min of distances
        # We want to maximize the minimum distance.
        # Let f(d) = sum(exp(-d/beta)).
        # As beta -> inf, f(d) is dominated by the smallest d.
        # Minimizing f(d) maximizes the smallest d.
        
        # Numerical stability:
        # max_val = np.max(-all_dists / beta)
        # val = np.log(np.sum(np.exp(-all_dists / beta - max_val))) + max_val
        # return val * beta  # This is the soft-min value. We want to MINIMIZE this?
        # No.
        # min(d) approx -beta * log(sum(exp(-d/beta)))
        # So we want to MAXIMIZE -beta * log(...)
        # Which is MINIMIZING beta * log(...)
        
        # Let's just return the sum of exp(-d/beta). It's monotonic with the soft-min.
        
        # To avoid underflow/overflow, shift by max.
        # But for 200 points, d is around 0.03. -d/beta ~ -0.03/50 ~ -0.0006.
        # exp(-0.0006) ~ 1.
        # Sum ~ 40000.
        # It's safe.
        
        # However, if d becomes very small (overlap), exp(-d/beta) -> 1.
        # If d becomes large, exp(-d/beta) -> 0.
        # So minimizing the sum works.
        
        # Gradient:
        # d/da sum(exp(-d(a)/beta)) = sum( -1/beta * exp(-d(a)/beta) * d/da d(a) )
        
        # We will use scipy minimize with analytic gradient.
        
        return np.sum(np.exp(-all_dists / beta))

    def lse_gradient(c, beta=50.0):
        dists, wall_dists = compute_clearances(c)
        
        flat_dists = dists.ravel()
        mask = flat_dists < np.inf
        flat_dists = flat_dists[mask]
        
        wall_flat = wall_dists.ravel()
        
        all_dists = np.concatenate([flat_dists, wall_flat])
        
        # Weights w_i = exp(-d_i / beta)
        weights = np.exp(-all_dists / beta)
        
        # Gradient of sum w_i wrt c
        # w_i depends on d_i.
        # d w_i / d c = w_i * (-1/beta) * (d d_i / d c)
        
        grad = np.zeros_like(c)
        
        # Pairwise distances
        # d_ij = sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
        # d d_ij / d x_i = (x_i - x_j) / d_ij
        # d d_ij / d x_j = (x_j - x_i) / d_ij
        
        # We need to map the flattened indices back to pairs.
        # This is tricky to do efficiently in vectorized form without loops if we want exact gradient.
        # Given N=200, a loop over pairs might be slow in Python.
        # Let's try to vectorize the gradient accumulation.
        
        # Create index arrays for i, j
        i_idx, j_idx = np.where(mask.reshape(N, N))
        
        # Get distances for these pairs
        d_pairs = dists[i_idx, j_idx]
        w_pairs = weights[:len(d_pairs)] # First part of weights are pairwise
        
        # Force on i: -1/beta * w_i * (c_i - c_j) / d_i
        # Force on j: -1/beta * w_i * (c_j - c_i) / d_i
        
        # Vectorized update
        # delta_c = (c_i - c_j)
        dc_x = c[i_idx, 0] - c[j_idx, 0]
        dc_y = c[i_idx, 1] - c[j_idx, 1]
        
        # Unit vectors
        inv_d = 1.0 / d_pairs
        ux = dc_x * inv_d
        uy = dc_y * inv_d
        
        # Contributions to gradient
        # grad[i] += -1/beta * w * u
        # grad[j] += -1/beta * w * (-u) = +1/beta * w * u
        
        coeff = -w_pairs / beta
        
        # Use np.add.at for accumulation
        np.add.at(grad, (i_idx, 0), coeff * ux)
        np.add.at(grad, (i_idx, 1), coeff * uy)
        np.add.at(grad, (j_idx, 0), -coeff * ux)
        np.add.at(grad, (j_idx, 1), -coeff * uy)
        
        # Wall distances
        # d = 0.5 - x  -> d/dx = -1
        # d = x + 0.5  -> d/dx = 1
        # d = 0.5 - y  -> d/dy = -1
        # d = y + 0.5  -> d/dy = 1
        
        w_walls = weights[len(d_pairs):]
        
        # Wall 1: 0.5 - x. grad_x = -1/beta * w * (-1) = +1/beta * w
        # Wall 2: x + 0.5. grad_x = -1/beta * w * (1) = -1/beta * w
        # Wall 3: 0.5 - y. grad_y = -1/beta * w * (-1) = +1/beta * w
        # Wall 4: y + 0.5. grad_y = -1/beta * w * (1) = -1/beta * w
        
        # wall_dists shape (N, 4)
        # w_walls is flattened (N*4)
        w_walls_reshaped = w_walls.reshape(N, 4)
        
        grad[:, 0] += (w_walls_reshaped[:, 0] / beta) - (w_walls_reshaped[:, 1] / beta)
        grad[:, 1] += (w_walls_reshaped[:, 2] / beta) - (w_walls_reshaped[:, 3] / beta)
        
        return grad

    # Run L-BFGS-B
    # Start with a reasonable beta, then anneal beta down to make it sharper.
    
    best_centers = centers.copy()
    best_r = 0.0
    
    def compute_radius(centers):
        n = len(centers)
        dx = centers[:, 0][:, None] - centers[:, 0][None, :]
        dy = centers[:, 1][:, None] - centers[:, 1][None, :]
        dists = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        wall = np.min(np.stack([
            0.5 - centers[:, 0],
            centers[:, 0] + 0.5,
            0.5 - centers[:, 1],
            centers[:, 1] + 0.5
        ]), axis=0)
        wall_r = np.min(wall)
        
        return min(wall_r, min_pair)

    current_r = compute_radius(best_centers)
    if current_r > best_r:
        best_r = current_r

    # Annealing loop
    betas = [100.0, 50.0, 20.0, 10.0, 5.0]
    
    for beta in betas:
        if time.time() > deadline - 20:
            break
            
        res = minimize(
            lse_objective, 
            best_centers, 
            method='L-BFGS-B', 
            jac=lse_gradient, 
            args=(beta,),
            options={'maxiter': 100, 'maxfun': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
        )
        
        if res.success or res.fun < lse_objective(best_centers, beta):
            # Check bounds
            if np.all(res.x >= -0.5) and np.all(res.x <= 0.5):
                r_new = compute_radius(res.x)
                if r_new > best_r:
                    best_r = r_new
                    best_centers = res.x.copy()
                else:
                    # If radius didn't improve but objective did, maybe it's a better local min?
                    # Keep the one with better radius for safety.
                    pass
            else:
                # Clip and re-evaluate
                res.x = np.clip(res.x, -0.49, 0.49)
                r_new = compute_radius(res.x)
                if r_new > best_r:
                    best_r = r_new
                    best_centers = res.x.copy()

    # 3. Active Set Refinement
    # Identify the circles that are at the minimum distance.
    # Move them slightly to increase the minimum distance.
    
    for _ in range(50):
        if time.time() > deadline - 5:
            break
            
        dists, wall_dists = compute_clearances(best_centers)
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists)
        min_wall_dist = np.min(wall_dists)
        
        current_r = min(min_pair_dist, min_wall_dist) / 2.0
        
        if current_r <= best_r:
            break
        
        best_r = current_r
        
        # Find active pairs
        # Threshold: distances within 1% of min
        thresh_pair = min_pair_dist * 1.01
        thresh_wall = min_wall_dist * 1.01
        
        active_pairs = np.argwhere(dists <= thresh_pair)
        active_walls = np.argwhere(wall_dists <= thresh_wall)
        
        # Simple local search on active points
        # We will try to push active points away from their neighbors/walls.
        
        # Create a mask of active points
        active_idx = set()
        for i, j in active_pairs:
            active_idx.add(i)
            active_idx.add(j)
        for i, k in active_walls:
            active_idx.add(i)
            
        active_idx = list(active_idx)
        if not active_idx:
            break
            
        # For each active point, compute a "push" vector
        # Sum of repulsion from active neighbors + wall repulsion
        
        improvements = 0
        for idx in active_idx:
            # Compute force
            fx, fy = 0, 0
            
            # Neighbors
            for j in range(N):
                if j == idx: continue
                dx = best_centers[idx, 0] - best_centers[j, 0]
                dy = best_centers[idx, 1] - best_centers[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                if d < min_pair_dist * 1.1:
                    # Push away
                    fx += dx / d
                    fy += dy / d
            
            # Walls
            if best_centers[idx, 0] < 0: fx += 1
            else: fx -= 1
            if best_centers[idx, 1] < 0: fy += 1
            else: fy -= 1
            
            # Normalize
            norm = np.sqrt(fx*fx + fy*fy)
            if norm > 0:
                fx /= norm
                fy /= norm
                
            # Try a small step
            step = 0.001
            new_x = best_centers[idx, 0] + fx * step
            new_y = best_centers[idx, 1] + fy * step
            
            if -0.49 <= new_x <= 0.49 and -0.49 <= new_y <= 0.49:
                # Check if it improves the minimum
                old_r = best_r
                best_centers[idx, 0] = new_x
                best_centers[idx, 1] = new_y
                
                new_r = compute_radius(best_centers)
                if new_r > old_r:
                    improvements += 1
                else:
                    best_centers[idx, 0] -= fx * step
                    best_centers[idx, 1] -= fy * step
            
        if improvements == 0:
            break

    return best_centers

# EVOLVE-BLOCK-END
