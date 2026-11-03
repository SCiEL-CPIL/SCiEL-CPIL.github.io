import numpy as np
import time
import math
from scipy.spatial.distance import cdist

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def compute_radius(centers):
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        # Wall distances
        # Distance to boundary [-0.5, 0.5]
        wx = 0.5 - np.abs(centers[:, 0])
        wy = 0.5 - np.abs(centers[:, 1])
        wall_dists = np.minimum(wx, wy)
        min_wall = np.min(wall_dists)
        
        # Pairwise distances
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix)
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def objective(x):
        """
        Minimizes a smooth approximation of -radius.
        Uses a soft-min over the critical constraints (walls and closest pairs).
        """
        centers = x.reshape(-1, 2)
        
        # 1. Wall constraints
        # d_wall_i = min(0.5 - |x_i|, 0.5 - |y_i|)
        wx = 0.5 - np.abs(centers[:, 0])
        wy = 0.5 - np.abs(centers[:, 1])
        wall_dists = np.minimum(wx, wy)
        
        # 2. Pairwise constraints
        # We need the closest pairs. Computing full cdist is O(N^2) ~ 40k ops, fast enough.
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Get the K smallest pairwise distances to define the "active set"
        # N=200, total pairs ~ 20k. 
        # Taking K=100 smallest is sufficient for the gradient signal of the minimum.
        K = 100
        # Flatten, argsort, take K
        flat_dists = dist_matrix.flatten()
        # Exclude diagonal? Diagonal is inf, so it won't be in smallest K.
        # But we need to be careful with indices if we were to use them. 
        # For the objective value, we just need the values.
        # However, for a more robust soft-min, we can use all pairs but weight by distance?
        # No, standard soft-min of the minimum is dominated by the smallest terms.
        # Let's just take the smallest K values.
        
        # Efficient way to get K smallest:
        # np.argpartition(flat_dists, K)[:K]
        # But we must ensure we don't pick diagonal (inf). Inf is largest, so safe.
        
        if len(flat_dists) > K:
            # Get indices of K smallest
            part_idx = np.argpartition(flat_dists, K)[:K]
            pair_dists = flat_dists[part_idx]
        else:
            pair_dists = flat_dists
        
        # Convert pair distances to "radius equivalent" clearances
        # Clearance c_ij = d_ij / 2
        pair_clearances = pair_dists / 2.0
        
        # Combine all clearances into one vector
        # Walls: N values
        # Pairs: K values
        all_clearances = np.concatenate([wall_dists, pair_clearances])
        
        # We want to maximize min(all_clearances).
        # Minimize -min(all_clearances).
        # Smooth approximation: -softmin(all_clearances)
        # softmin(c) = (1/beta) * log( sum( exp(-beta * c_i) ) )
        # As beta -> inf, this approaches min(c).
        
        beta = 50.0
        # Log-sum-exp trick for stability
        # s = sum(exp(-beta * c))
        # log_s = -beta * max_c + log(sum(exp(-beta * (c - max_c))))
        # softmin = (1/beta) * log_s = -max_c + (1/beta) * log(sum(exp(-beta * (c - max_c))))
        
        max_c = np.max(all_clearances)
        # If max_c is very large, exp(-beta * (c - max_c)) is exp(positive) for c < max_c?
        # Wait. c_i <= max_c. So c_i - max_c <= 0.
        # -beta * (c_i - max_c) >= 0.
        # So exp(...) >= 1. No overflow risk from large positive exponents if we shift by max.
        # Actually, if c_i is very small, -beta*(c_i - max_c) is large positive.
        # Example: c_i=0, max_c=0.1, beta=50. -50*(0-0.1) = 5. exp(5) ~ 148. Fine.
        # If c_i=0, max_c=0.001, beta=50. -50*(-0.001) = 0.05. exp(0.05) ~ 1.
        
        shifted = all_clearances - max_c
        # shifted <= 0
        exp_terms = np.exp(-beta * shifted)
        log_sum_exp = np.log(np.sum(exp_terms))
        soft_min_val = (max_c + log_sum_exp / beta)
        
        # Objective is to maximize this, so minimize negative
        return -soft_min_val

    def initial_hex_grid(n_rows, n_cols, d_scale=1.0, jitter=0.0):
        """
        Generate a hexagonal grid of points.
        """
        d = d_scale / (n_cols + 0.5)
        row_spacing = d * math.sqrt(3) / 2
        total_height = (n_rows - 1) * row_spacing + d
        if total_height > 1.0 + 1e-6:
            return None
        
        pts = []
        for row in range(n_rows):
            y = -0.5 + d / 2 + row * row_spacing
            if row % 2 == 0:
                n_in_row = n_cols
                x_start = -0.5 + d / 2
            else:
                n_in_row = n_cols - 1
                x_start = -0.5 + d
            
            for col in range(n_in_row):
                x = x_start + col * d
                pts.append([x, y])
        
        pts = np.array(pts)
        
        # Add jitter
        if jitter > 0:
            # Jitter within bounds, ensuring we don't go outside
            # Simple jitter, then clip
            pts = pts + np.random.randn(len(pts), 2) * jitter
            pts = np.clip(pts, -0.49, 0.49)
            
        if len(pts) >= N:
            return pts[:N]
        elif len(pts) < N:
            # Pad with random points if not enough? Or just return None.
            # Better to return None to avoid bad inits.
            return None
        return pts

    best_centers = None
    best_radius = 0.0

    # Generate diverse initial configurations
    inits = []
    
    # 1. Standard Hexagonal grids with varying row counts and scales
    for n_rows in range(13, 17):
        if time.time() > deadline - 60:
            break
        n_cols = math.ceil(N / n_rows)
        for d_scale in [0.95, 1.0, 1.05, 1.1]:
            for jitter in [0.0, 0.005, 0.01]:
                pts = initial_hex_grid(n_rows, n_cols, d_scale, jitter)
                if pts is not None:
                    inits.append(pts)
    
    # 2. Square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) >= N:
        inits.append(grid[:N])
        # Jittered square
        jittered_grid = grid[:N] + np.random.randn(N, 2) * 0.005
        jittered_grid = np.clip(jittered_grid, -0.49, 0.49)
        inits.append(jittered_grid)

    # 3. Random
    rand_pts = np.random.uniform(-0.49, 0.49, (N, 2))
    inits.append(rand_pts)

    # Shuffle inits to mix up quality
    np.random.shuffle(inits)

    # Run Nelder-Mead on each init
    # Nelder-Mead is gradient-free and can handle non-smooth functions better than L-BFGS-B.
    # It is slower per iteration but often finds better minima for packing problems.
    
    # To make it faster, we can limit the number of iterations and use a good initial simplex.
    # scipy.optimize.minimize with method='Nelder-Mead'
    
    for init in inits:
        if time.time() > deadline - 15:
            break
            
        x0 = init.flatten()
        
        # Nelder-Mead options
        # initial_simplex: We can let scipy generate it, or provide one.
        # Providing a simplex scaled to the problem size helps.
        # Default is usually fine but we can tune.
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 2000, 
                    'xatol': 1e-8, 
                    'fatol': 1e-10, 
                    'adaptive': True
                }
            )
        except Exception:
            continue
            
        centers = res.x.reshape(-1, 2)
        # Clip to ensure validity
        centers = np.clip(centers, -0.5, 0.5)
        
        r = compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
            
            # Optional: If we find a good one, we can try to polish it further with more iterations
            # or a different method, but NM is already doing the work.
            # We can break early if we reach a theoretical upper bound? 
            # Upper bound for 200 circles in unit square is approx 1/(2*sqrt(200)) ~ 0.0353 for square grid, 
            # but hex is denser. Best known is 0.0366.
            # If best_radius > 0.0366, we've beaten the known best.
            if best_radius > 0.036612798904:
                # We can try to squeeze more, but it's a strong signal.
                # Let's continue to see if we can improve further, but we have a good candidate.
                pass

    # Final Polish: 
    # If we have time, take the best result and run a local coordinate ascent or a very fine NM step.
    if best_centers is not None and time.time() < deadline - 5:
        # Try a few random perturbations and re-optimize? No, NM is expensive.
        # Just run NM again on the best point with tighter tolerances?
        # NM doesn't use gradients, so "tighter tolerances" just means more iterations.
        
        x0 = best_centers.flatten()
        try:
            res = minimize(
                objective, 
                x0, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 3000, 
                    'xatol': 1e-10, 
                    'fatol': 1e-12, 
                    'adaptive': True
                }
            )
            centers = res.x.reshape(-1, 2)
            centers = np.clip(centers, -0.5, 0.5)
            r = compute_radius(centers)
            if r > best_radius:
                best_radius = r
                best_centers = centers.copy()
        except Exception:
            pass

    if best_centers is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]

    return best_centers

# Import minimize at top level to avoid import overhead in function if called multiple times? 
# No, it's fine inside or outside. Let's put it outside for cleanliness.
from scipy.optimize import minimize
