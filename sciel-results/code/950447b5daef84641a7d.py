"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_score(points):
    """Calculate the score (radius) for a given set of points."""
    if len(points) != N:
        return 0.0
    # Check if points are inside the unit circle
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms >= 1.0):
        return 0.0
    
    # Calculate minimum pairwise distance
    # Using scipy.spatial.distance.pdist for efficiency
    from scipy.spatial.distance import pdist
    dists = pdist(points, 'euclidean')
    min_dist = np.min(dists)
    
    # The radius is limited by the closest pair and the distance to the boundary
    r_pair = min_dist / 2.0
    r_boundary = 1.0 - np.max(norms)
    
    return min(r_pair, r_boundary)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Generate initial points using a hexagonal lattice
    # We need to generate enough points and pick the best N
    # Estimate optimal spacing: Area of unit circle is pi. N circles of radius r.
    # r ~ sqrt(pi / (N * 2 * sqrt(3))) approx.
    # For N=300, r ~ 0.05. Spacing ~ 0.1.
    # Let's create a grid with spacing 0.05 to be safe, then select.
    
    def generate_hex_points(spacing, max_radius=1.0):
        points = []
        # Range for x and y
        x_max = max_radius
        y_max = max_radius
        # Hex grid generation
        # y step is spacing * sqrt(3)/2
        dy = spacing * np.sqrt(3.0) / 2.0
        for j in range(int(-max_radius / dy) - 1, int(max_radius / dy) + 2):
            y = j * dy
            # Offset for even/odd rows
            offset = (spacing / 2.0) if (j % 2 != 0) else 0.0
            for i in range(int(-max_radius / spacing) - 1, int(max_radius / spacing) + 2):
                x = i * spacing + offset
                if x**2 + y**2 <= max_radius**2:
                    points.append([x, y])
        return np.array(points)

    # Generate a dense grid
    spacing = 0.08
    init_points = generate_hex_points(spacing)
    
    if len(init_points) < N:
        spacing = 0.05
        init_points = generate_hex_points(spacing)
        
    # Select N points closest to the center to start with a compact configuration
    # This helps the optimizer start from a "dense" state
    norms = np.linalg.norm(init_points, axis=1)
    idx = np.argsort(norms)[:N]
    x0 = init_points[idx].copy()
    
    # Add small random jitter to break symmetry
    np.random.seed(42)
    x0 += np.random.normal(0, 0.001, x0.shape)
    
    # Ensure initial points are strictly inside
    norms = np.linalg.norm(x0, axis=1)
    x0[norms >= 1.0] *= 0.99
    
    best_score = _compute_score(x0)
    best_points = x0.copy()
    
    # 2. Optimization using L-BFGS-B
    # Objective: Minimize repulsive potential
    # E = sum_{i<j} 1/d_ij^2 + sum_i 1/(1 - ||x_i||)^2
    # This is expensive to compute for N=300 in pure Python, so we use NumPy.
    
    def objective_and_grad(flat_x):
        x = flat_x.reshape(-1, 2)
        # Calculate pairwise distances
        # dists = sqrt(sum((x_i - x_j)^2))
        # To avoid division by zero, add epsilon
        # We compute the matrix of squared distances
        x_sq = np.sum(x**2, axis=1)
        dist_sq = x_sq[:, None] + x_sq[None, :] - 2 * x @ x.T
        np.fill_diagonal(dist_sq, 1.0) # Avoid self-interaction division by zero
        
        # Clip to prevent negative due to floating point errors
        dist_sq = np.maximum(dist_sq, 1e-12)
        
        # Repulsive term: 1/d^2
        # We only sum over i < j to avoid double counting
        # mask for i < j
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        repulsive_sum = np.sum(1.0 / dist_sq[mask])
        
        # Boundary term: 1/(1 - ||x||)^2
        norms = np.sqrt(x_sq)
        # Clip norms to be slightly less than 1 to avoid division by zero or inf
        norms_clipped = np.minimum(norms, 0.999999)
        boundary_sum = np.sum(1.0 / (1.0 - norms_clipped)**2)
        
        total_energy = repulsive_sum + boundary_sum
        
        # Gradient calculation
        # d/dx_i (1/d_ij^2) = -2 * (x_i - x_j) / d_ij^4
        # This is complex to vectorize perfectly for the gradient in one go without extra memory.
        # L-BFGS-B can use numerical gradients, but that's slow.
        # Let's try to compute the gradient analytically.
        
        # Gradient of repulsive term
        # grad_i = sum_{j!=i} -2 * (x_i - x_j) / ||x_i - x_j||^4
        # dist_sq is the matrix of squared distances.
        # dist_4 = dist_sq**2
        # We need the vector (x_i - x_j).
        # diff = x[:, None, :] - x[None, :, :] # (N, N, 2)
        # This uses a lot of memory (300*300*2*8 bytes ~ 1.4 MB, acceptable).
        
        diff = x[:, None, :] - x[None, :, :] # (N, N, 2)
        # dist_sq is (N, N)
        dist_4 = dist_sq**2
        # We need to sum over j for each i
        # term_ij = -2 * diff_ij / dist_4_ij
        # Set diagonal to 0
        np.fill_diagonal(dist_4, 1.0)
        # Avoid division by zero on diagonal in diff/dist_4
        # We can just ignore diagonal in the sum
        inv_dist_4 = 1.0 / dist_4
        np.fill_diagonal(inv_dist_4, 0.0)
        
        # grad_repulsive_i = sum_j (-2 * diff_ij * inv_dist_4_ij)
        # diff is (N, N, 2), inv_dist_4 is (N, N)
        # We want to sum over axis 1 (j)
        # Multiply diff by inv_dist_4[..., None]
        weighted_diff = diff * inv_dist_4[..., None]
        grad_repulsive = -2.0 * np.sum(weighted_diff, axis=1)
        
        # Gradient of boundary term
        # f_i = (1 - ||x_i||)^-2
        # df_i/dx_i = -2 * (1 - ||x_i||)^-3 * (-1) * (x_i / ||x_i||)
        #            = 2 * x_i / (||x_i|| * (1 - ||x_i||)^3)
        # Let r_i = ||x_i||
        # term_i = 2 * x_i / (r_i * (1 - r_i)^3)
        
        # Handle r_i = 0
        r_safe = np.where(norms > 1e-8, norms, 1.0)
        boundary_factor = 2.0 / (r_safe * (1.0 - r_safe)**3)
        # Set factor to 0 where norms were small to avoid instability?
        # Actually if r is small, (1-r)^3 is ~1, r is small, factor is large.
        # But x_i is small. So x_i / r_i is bounded.
        # Let's just use the formula.
        grad_boundary = x * boundary_factor[:, None]
        
        total_grad = grad_repulsive + grad_boundary
        
        return total_energy, total_grad.flatten()

    # Run optimization
    # Bounds: -1 to 1 for all coordinates
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    try:
        result = minimize(
            objective_and_grad,
            x0.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8, 'maxfun': 5000}
        )
        if result.success or result.fun < objective_and_grad(x0.flatten())[0]:
            new_points = result.x.reshape(-1, 2)
            # Check validity
            new_score = _compute_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
    except Exception as e:
        pass # Keep best found so far

    # If time remains, try a second run with different initialization or refinement
    if time.time() < deadline - 20:
        # Try random initialization
        x0_rand = np.random.uniform(-0.8, 0.8, (N, 2))
        # Project inside circle
        norms = np.linalg.norm(x0_rand, axis=1)
        mask = norms > 0.9
        x0_rand[mask] *= 0.9 / norms[mask]
        
        try:
            result2 = minimize(
                objective_and_grad,
                x0_rand.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-7, 'gtol': 1e-7}
            )
            if result2.success:
                new_points2 = result2.x.reshape(-1, 2)
                new_score2 = _compute_score(new_points2)
                if new_score2 > best_score:
                    best_score = new_score2
                    best_points = new_points2
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
