"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # --- Stage 1: Hexagonal Lattice Initialization ---
    # Generate a hexagonal lattice of points.
    # We generate more points than needed and then select the best N.
    # This ensures a dense, valid starting configuration close to optimal bulk packing.
    
    # Estimate spacing for hexagonal packing
    # Area per point ~ 1/sqrt(N) * some factor
    # For N=300, r~0.053, spacing d = 2r ~ 0.106
    # Let's generate a grid with spacing slightly larger than 2*0.053 to start valid
    d_init = 0.105 
    
    # Hexagonal lattice vectors
    a = d_init
    b = d_init * np.sqrt(3) / 2
    
    # Generate points in a bounding box
    max_radius = 1.0
    # Number of lattice points needed
    # x range: -max_radius to max_radius
    # y range: -max_radius to max_radius
    
    points_list = []
    
    # Iterate over rows
    # Row spacing is b
    num_rows = int(np.ceil(max_radius / b)) + 2
    for i in range(-num_rows, num_rows + 1):
        y = i * b
        # Offset every other row
        offset = 0 if i % 2 == 0 else a / 2
        
        # x range for this row
        # We need points within the unit circle, so x^2 + y^2 <= 1
        # x_max = sqrt(1 - y^2)
        if abs(y) > 1.0:
            continue
        x_max = np.sqrt(1.0 - y**2)
        
        num_cols = int(np.ceil(x_max / a)) + 1
        for j in range(-num_cols, num_cols + 1):
            x = j * a + offset
            if x**2 + y**2 <= 1.0:
                points_list.append([x, y])
                
    points = np.array(points_list)
    
    if len(points) == 0:
        # Fallback if lattice generation failed (unlikely)
        points = np.zeros((N, 2))
    elif len(points) < N:
        # If we didn't generate enough, add random points or increase density
        # For N=300 and d=0.105, we should have plenty.
        # Let's just pad with random points if necessary
        missing = N - len(points)
        rng = np.random.default_rng(42)
        for _ in range(missing):
            r = np.sqrt(rng.random())
            theta = rng.random() * 2 * np.pi
            points = np.vstack([points, [r * np.cos(theta), r * np.sin(theta)]])
    else:
        # We have more points than N.
        # Strategy: Select the N points that maximize the minimum distance.
        # A greedy approach: Sort by distance to center? No.
        # Better: Keep the points that form the densest valid subset.
        # Since the lattice is regular, the "best" N points are roughly the ones
        # closest to the center, or we can just take the first N.
        # However, taking arbitrary N might leave "holes" at the boundary.
        # A better heuristic for lattice: Take the N points with the smallest
        # "hexagonal distance" from origin, or simply the first N generated
        # in a spiral order.
        
        # Let's sort by radius from center to ensure we fill the disk evenly
        norms = np.linalg.norm(points, axis=1)
        sorted_indices = np.argsort(norms)
        points = points[sorted_indices[:N]]
        
    # Ensure exactly N points
    if len(points) > N:
        points = points[:N]
    elif len(points) < N:
        # This shouldn't happen with the padding logic above, but just in case
        points = np.vstack([points, np.zeros((N - len(points), 2))])

    # --- Stage 2: Smooth Energy Minimization (Soft-Min Maximization) ---
    # We want to maximize r = min( min_pair_dist/2, min_boundary_dist )
    # Let f(x) = min( min_pair_dist/2, min_boundary_dist )
    # We approximate the min with a soft-min (LogSumExp) to get a smooth landscape.
    # Minimize E(x) = -log( sum( exp(-alpha * d_i) ) ) / alpha
    # Where d_i are the "clearances".
    # Clearances:
    # 1. Pairwise: (dist_ij - 2r_target) ? No, we want to maximize the min dist.
    # Let's define the objective to maximize a smooth approximation of the minimum clearance.
    # Let C_pair_ij = dist_ij / 2
    # Let C_bound_i = 1 - |p_i|
    # We want to maximize min(C).
    # Soft-Min: S(C) = -1/alpha * log( sum( exp(-alpha * C_k) ) )
    # We want to maximize S(C). Equivalent to minimizing -S(C).
    
    alpha = 50.0  # Sharpness of the min approximation. Higher = closer to true min.
    # Start with lower alpha for global movement, increase for local refinement.
    
    def compute_energy(pts, alpha):
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        
        # Get upper triangle indices
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_dists = dist[i_idx, j_idx]
        
        # Clearances for pairs: dist/2
        # We want to penalize small distances.
        # C_pair = pair_dists / 2
        
        # Boundary clearances
        norms = np.sqrt(np.sum(pts**2, axis=1))
        C_bound = 1.0 - norms
        
        # Combine all clearances into one array
        # C = [C_pair, C_bound]
        C = np.concatenate([pair_dists / 2.0, C_bound])
        
        # Soft-Min
        # To avoid overflow/underflow, subtract max
        C_max = np.max(C)
        # If C is very negative, exp(-alpha*C) can overflow.
        # But C should be positive for valid packings.
        # If C is small, -alpha*C is large positive.
        # Let's use the stable logsumexp formula:
        # logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
        # Here x = -alpha * C
        x = -alpha * C
        x_max = np.max(x)
        log_sum_exp = x_max + np.log(np.sum(np.exp(x - x_max)))
        
        # Soft-Min value: -1/alpha * log_sum_exp
        # We want to MAXIMIZE this, so we minimize its negative.
        energy = -log_sum_exp / alpha
        
        # Compute gradient for optimization?
        # Manual gradient is complex. Let's use a numerical or simple gradient-free
        # approach if SciPy minimize is too slow or unstable with the soft-min.
        # However, SciPy L-BFGS-B is fast. Let's provide the gradient analytically.
        
        # dE/dC_k = (1/alpha) * exp(-alpha*C_k) / sum(exp(-alpha*C_j))
        # Let P_k = exp(-alpha*C_k) / sum(exp(-alpha*C_j))  (Probability)
        # dE/dC_k = P_k
        
        # Chain rule:
        # dC_pair_ij / d p_i = -0.5 * (p_i - p_j) / dist_ij
        # dC_bound_i / d p_i = -p_i / |p_i|
        
        # grad_E = sum_k (dE/dC_k * dC_k/dp)
        
        # Vectorized gradient calculation
        # P_pair for each pair
        # P_bound for each point
        
        # This requires careful indexing.
        # Given the time constraint and N=300, a numerical gradient might be too slow
        # for L-BFGS-B (which needs many evals).
        # Let's stick to a simpler optimization: Coordinate Descent or Nelder-Mead?
        # No, L-BFGS-B is best if we have gradient.
        
        # Let's implement the analytic gradient.
        
        # Probabilities
        exp_x = np.exp(x - x_max)
        sum_exp = np.sum(exp_x)
        P = exp_x / sum_exp
        
        # Split P into pair and bound parts
        num_pairs = len(i_idx)
        P_pair = P[:num_pairs]
        P_bound = P[num_pairs:]
        
        # Gradient accumulation
        grad = np.zeros_like(pts)
        
        # Pair contributions
        # dC_pair / dp_i = 0.5 * (p_j - p_i) / dist_ij  (Note: C = dist/2, so dC/dp = 0.5 * dir)
        # Wait, C = dist/2. d(dist)/dp_i = (p_i - p_j)/dist.
        # So dC/dp_i = 0.5 * (p_i - p_j)/dist.
        # dE/dp_i = sum_j P_ij * 0.5 * (p_i - p_j)/dist_ij
        
        # We need to accumulate for each i.
        # For each pair (i, j), P_ij contributes to grad[i] and grad[j].
        # grad[i] += P_ij * 0.5 * (p_i - p_j) / dist_ij
        # grad[j] += P_ij * 0.5 * (p_j - p_i) / dist_ij
        
        # Vectorized update is tricky with scatter.
        # For N=300, loop over pairs might be okay if we do it in Python? No, 45k pairs * iterations is too slow.
        # Let's use np.add.at
        
        # diff_vec = p_i - p_j
        diff_vec = pts[i_idx] - pts[j_idx]
        dists = pair_dists
        
        # Unit vectors
        # Handle zero distance
        mask = dists > 1e-9
        unit_vecs = np.zeros_like(diff_vec)
        unit_vecs[mask] = diff_vec[mask] / dists[mask][:, np.newaxis]
        
        # Coefficients: 0.5 * P_pair
        coeffs = 0.5 * P_pair
        
        # grad_i += coeffs * unit_vecs
        # grad_j -= coeffs * unit_vecs
        
        np.add.at(grad, i_idx, coeffs[:, np.newaxis] * unit_vecs)
        np.add.at(grad, j_idx, -coeffs[:, np.newaxis] * unit_vecs)
        
        # Boundary contributions
        # C_bound_i = 1 - |p_i|
        # dC_bound_i / dp_i = -p_i / |p_i|
        # dE/dp_i += P_bound_i * (-p_i / |p_i|)
        
        norms_safe = np.where(norms > 1e-9, norms, 1.0)
        unit_norms = pts / norms_safe[:, np.newaxis]
        
        grad += -P_bound[:, np.newaxis] * unit_norms
        
        return energy, grad

    # Optimization Loop
    # We will use L-BFGS-B from scipy.optimize
    from scipy.optimize import minimize
    
    # Initial alpha
    current_alpha = 20.0
    
    # We will run a few phases:
    # Phase 1: Low alpha (smooth), global movement
    # Phase 2: High alpha (sharp), local refinement
    
    # Define objective function for minimize
    def obj_func(pts_flat):
        pts = pts_flat.reshape(N, 2)
        # Project to disk to ensure validity during optimization steps
        norms = np.linalg.norm(pts, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            pts[mask] *= (1.0 / norms[mask])[:, np.newaxis]
        
        e, _ = compute_energy(pts, current_alpha)
        return e
    
    def grad_func(pts_flat):
        pts = pts_flat.reshape(N, 2)
        # Project to disk
        norms = np.linalg.norm(pts, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            pts[mask] *= (1.0 / norms[mask])[:, np.newaxis]
        
        _, g = compute_energy(pts, current_alpha)
        return g.flatten()

    # Start optimization
    # Use L-BFGS-B
    # maxiter depends on time.
    # Let's estimate iterations.
    
    # Phase 1
    current_alpha = 10.0
    res1 = minimize(
        obj_func, 
        points.flatten(), 
        method='L-BFGS-B', 
        jac=grad_func, 
        options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
    )
    
    if time.time() > deadline - 30:
        points = res1.x.reshape(N, 2)
    else:
        points = res1.x.reshape(N, 2)
        
        # Phase 2: Increase alpha for sharper minimum
        current_alpha = 50.0
        res2 = minimize(
            obj_func, 
            points.flatten(), 
            method='L-BFGS-B', 
            jac=grad_func, 
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if time.time() > deadline - 10:
            points = res2.x.reshape(N, 2)
        else:
            points = res2.x.reshape(N, 2)
            
            # Phase 3: Very high alpha for final polish
            current_alpha = 200.0
            res3 = minimize(
                obj_func, 
                points.flatten(), 
                method='L-BFGS-B', 
                jac=grad_func, 
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            points = res3.x.reshape(N, 2)

    # Final projection to ensure strict validity
    norms = np.linalg.norm(points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        points[mask] *= (1.0 / norms[mask])[:, np.newaxis]
        
    # Check for coincident points or invalid values
    # If any points are too close, we might need to perturb them, 
    # but the soft-min should have prevented this.
    
    return points

# EVOLVE-BLOCK-END
