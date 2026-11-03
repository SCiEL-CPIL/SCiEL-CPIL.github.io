import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def compute_radius(centers):
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix)
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def objective_and_grad(x):
        """
        Minimizes a smooth approximation of -r.
        Uses soft-min of squared clearances.
        """
        centers = x.reshape(-1, 2)
        
        # Clearances to walls
        # c_x = 0.5 - centers[:, 0], c_y = 0.5 - centers[:, 1]
        # We want to maximize min(c_x, c_y, d_ij/2)
        # Equivalently minimize -min(...)
        # Approximate min with soft-min: -log(sum(exp(-beta * val))) / beta
        # But exp can overflow. Use log-sum-exp trick.
        
        # Let's use a simpler smooth approximation:
        # f = sum( (target - val)^2 for val < target ) ? No, that's not differentiable at target.
        
        # Better: Minimize sum( softplus(target - val) )
        # softplus(z) = log(1 + exp(z))
        # If val is large (good), target - val is negative, softplus ~ 0.
        # If val is small (bad), target - val is positive, softplus ~ target - val.
        
        # However, we don't know the target 'r'. We want to maximize the minimum clearance.
        # Let's define the objective as minimizing the "worst-case" clearance.
        # A standard trick for max-min is to minimize the sum of squared "violations" relative to a moving threshold, 
        # or use a smooth approximation of the min function directly.
        
        # Let's try: Minimize sum_i ( -log(sum_j exp(beta * clearance_j)) ) / beta ?
        # No, let's use the "power sum" approximation of the minimum.
        # min(x) ≈ (sum x^p)^(-1/p) for p -> -infinity.
        # This is hard to differentiate numerically for large p.
        
        # Let's go with the gradient-based approach mentioned in inspiration 1:
        # Maximize sum of squared pairwise distances? No, that's not the objective.
        
        # Let's stick to the direct smooth approximation of the min clearance.
        # Let clearances be vector c. We want max(min(c)).
        # Approximate min(c) by: (1/beta) * log( sum( exp(-beta * c_i) ) ) is the soft MAX of -c, i.e. -softmin(c).
        # So we want to minimize: (1/beta) * log( sum( exp(-beta * c_i) ) )
        
        # Clearances:
        # 1. Walls: c_wall_i = 0.5 - |x_i|, 0.5 - |y_i|. (4 values per point)
        # 2. Pairs: c_pair_ij = |x_i - x_j| / 2.
        
        # This sum is huge (O(N^2)).
        # To keep it fast, we can restrict the "active set" or use a lower beta.
        # But for global optimization, we need the gradient to be informative.
        
        # Alternative Strategy: 
        # Use a "repulsive energy" approach.
        # Energy = sum_{i<j} (1/d_ij)^2 + sum_i (1/wall_dist_i)^2
        # Minimizing this pushes points apart.
        # This is smooth and differentiable.
        # It doesn't directly maximize the minimum distance, but it correlates strongly.
        # And it's much faster to compute gradients.
        
        eps = 1e-6
        centers = x.reshape(-1, 2)
        
        # Wall penalties
        wx = 0.5 - centers[:, 0]
        wy = 0.5 - centers[:, 1]
        # Ensure positive distance for log/sqrt if using those, but we use inverse square.
        # To avoid singularity at 0, add eps.
        wall_dist_sq = wx**2 + wy**2 
        # Actually, wall distance is min(wx, wy). 
        # Let's just penalize proximity to walls.
        # d_wall = min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y)
        # Smooth approximation: softmin of these 4 distances.
        
        # Let's try a different objective:
        # Minimize sum_{i<j} (1 / (d_ij^2 + eps)) + sum_i (1 / (w_i^2 + eps))
        # This is a standard packing heuristic.
        
        # Compute pairwise distances
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Pairwise term
        pair_terms = 1.0 / (dist_matrix**2 + eps)
        # Sum over upper triangle to avoid double counting, or just sum all and divide by 2
        pair_energy = np.sum(pair_terms) / 2.0
        
        # Wall term
        # Distance to walls
        d_left = centers[:, 0] + 0.5
        d_right = 0.5 - centers[:, 0]
        d_bottom = centers[:, 1] + 0.5
        d_top = 0.5 - centers[:, 1]
        
        # Min of these 4. Approximate with softmin or just sum of inverses?
        # If we use sum of inverses, it penalizes being close to ANY wall.
        wall_energy = np.sum(1.0 / (d_left**2 + eps) + 1.0 / (d_right**2 + eps) + 
                             1.0 / (d_bottom**2 + eps) + 1.0 / (d_top**2 + eps))
        
        total_energy = pair_energy + wall_energy
        
        # Gradient
        # d(pair_energy)/dx_i = sum_{j != i} d(1/d_ij^2)/dx_i
        # d(1/d^2)/dx_i = -2/d^4 * (x_i - x_j)
        
        # This requires O(N^2) operations for gradient too, which is fine for N=200 (40k ops).
        
        # Let's compute gradient analytically.
        grad = np.zeros_like(x)
        
        # Pairwise gradient
        # dist_matrix[i,j] is distance.
        # We need vector differences.
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        # dist_matrix is already computed.
        # factor = -2 / (d^4 + ...)? 
        # d/dx (1/d^2) = -2/d^3 * (1/d) * (x-y)/d ? No.
        # Let u = x_i - x_j. d = ||u||. f = 1/d^2 = u^-2.
        # df/du = -2 u^-3 (u/d) = -2 u / d^4.
        # So grad_i += sum_j -2 * (x_i - x_j) / d_ij^4
        
        # We have dist_matrix. 
        # denom = dist_matrix**4 + eps
        # grad_contrib = -2 * diff / denom[..., np.newaxis]
        
        # But diff is (N, N, 2).
        # We need to sum over j for each i.
        # np.sum(grad_contrib, axis=1) gives gradient for all i.
        
        # Careful with diagonal (d=0). We set dist_matrix diag to inf, so denom is inf, contrib is 0.
        # But diff diagonal is 0. 0/inf is 0. OK.
        
        # However, computing diff (N,N,2) is 200*200*2 = 80k floats. Fine.
        
        # Optimization: We already have dist_matrix.
        # Let's recompute diff to get the vector direction.
        # Or use: grad_i = -2 * sum_j (x_i - x_j) / d_ij^4
        
        # Vectorized:
        # diff is (N, N, 2)
        # denom is (N, N)
        # factor = -2.0 / (dist_matrix**4 + eps)
        # factor[:, np.newaxis] is (N, N, 1)
        # grad_pair = np.sum(diff * factor[:, :, np.newaxis], axis=1)
        
        # Wall gradient
        # d/dx (1/(x+0.5)^2) = -2/(x+0.5)^3
        # d/dx (1/(0.5-x)^2) = -2/(0.5-x)^3 * (-1) = 2/(0.5-x)^3
        
        grad_wall = np.zeros_like(centers)
        grad_wall[:, 0] += -2.0 / (d_left**3 + eps)
        grad_wall[:, 0] += 2.0 / (d_right**3 + eps)
        grad_wall[:, 1] += -2.0 / (d_bottom**3 + eps)
        grad_wall[:, 1] += 2.0 / (d_top**3 + eps)
        
        # Combine
        grad_flat = grad_pair.reshape(-1) + grad_wall.reshape(-1)
        
        return total_energy, grad_flat

    def initial_hex_grid(n_rows, n_cols, d_scale=1.0):
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
        if len(pts) >= N:
            pts = pts[:N]
        elif len(pts) < N:
            return None
        return pts

    def initial_square_grid(m):
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    best_centers = None
    best_radius = 0.0

    # Generate initial configurations
    inits = []
    
    # Hexagonal
    for n_rows in range(13, 17):
        if time.time() > deadline - 60:
            break
        n_cols = math.ceil(N / n_rows)
        for d_scale in [0.95, 1.0, 1.05]:
            pts = initial_hex_grid(n_rows, n_cols, d_scale)
            if pts is not None:
                inits.append(pts)
                
    # Square
    m = int(np.ceil(np.sqrt(N)))
    grid = initial_square_grid(m)
    inits.append(grid)
    
    # Perturbed square
    for _ in range(3):
        if time.time() > deadline - 60:
            break
        perturbed = grid + np.random.randn(N, 2) * 0.01
        perturbed = np.clip(perturbed, -0.49, 0.49)
        inits.append(perturbed)

    # Random
    for _ in range(2):
        if time.time() > deadline - 60:
            break
        rand_pts = np.random.uniform(-0.49, 0.49, (N, 2))
        inits.append(rand_pts)

    # Run L-BFGS-B on each init
    for init in inits:
        if time.time() > deadline - 20:
            break
            
        x0 = init.flatten()
        
        # L-BFGS-B with bounds
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            res = minimize(
                objective_and_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True,
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
        except Exception:
            continue
            
        if res.success or res.fun < 1e10: # Check for failure
            centers = res.x.reshape(-1, 2)
            # Ensure inside bounds (L-BFGS-B should, but clip just in case)
            centers = np.clip(centers, -0.5, 0.5)
            
            r = compute_radius(centers)
            if r > best_radius:
                best_radius = r
                best_centers = centers.copy()
                
    # If we have time, do a final polish on the best result with a different optimizer or more iterations
    if best_centers is not None and time.time() < deadline - 10:
        x0 = best_centers.flatten()
        bounds = [(-0.5, 0.5)] * (2 * N)
        try:
            # Increase iterations for polish
            res = minimize(
                objective_and_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True,
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
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
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]

    return best_centers
