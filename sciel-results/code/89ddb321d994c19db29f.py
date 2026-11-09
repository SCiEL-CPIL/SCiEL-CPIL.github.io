"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Returns array of shape (N, 2) with centers of N equal circles inside unit circle.
    Uses a multi-stage approach:
    1. Initialize with a good hexagonal lattice packing
    2. Refine using gradient-based optimization with soft-min objective
    3. Final polish with tighter tolerances
    """
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    # Step 1: Generate initial hexagonal lattice packing
    def hex_lattice_init(n_points):
        """Generate n_points centers from a hexagonal lattice clipped to unit disk."""
        # Try different spacings to find one that gives at least n_points
        best_points = None
        for spacing_factor in np.linspace(0.8, 1.5, 30):
            spacing = 2.0 / (np.sqrt(n_points) * spacing_factor)
            k = int(1.0 / spacing) + 3
            rows = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    r = np.hypot(x, y)
                    if r <= 1.0 - spacing / 2.0:
                        rows.append((x, y))
            points = np.array(rows)
            if len(points) >= n_points:
                # Sort by distance from center, take closest n_points
                dists = np.hypot(points[:, 0], points[:, 1])
                idx = np.argsort(dists, kind='stable')[:n_points]
                if best_points is None or len(points) > len(best_points):
                    best_points = points[idx]
        
        if best_points is not None:
            return best_points
        # Fallback: random points
        return np.random.uniform(-0.8, 0.8, size=(n_points, 2))
    
    points = hex_lattice_init(N)
    
    # Step 2: Gradient-based optimization with soft-min
    def soft_min(values, beta):
        """Compute soft minimum using log-sum-exp trick."""
        m = np.min(values)
        return m + np.log(np.sum(np.exp(-beta * (values - m)))) / beta
    
    def soft_min_grad(values, beta):
        """Compute gradient of soft minimum w.r.t. values."""
        m = np.min(values)
        exp_vals = np.exp(-beta * (values - m))
        weights = exp_vals / np.sum(exp_vals)
        return -beta * weights
    
    def compute_score_and_grad(points, beta):
        """
        Compute soft-min objective and its gradient w.r.t. point positions.
        Objective: maximize min(min_{i<j} |c_i - c_j|/2, min_i (1 - |c_i|))
        We minimize the negative of this.
        """
        n = len(points)
        
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Half-distances between pairs (i<j)
        i_idx, j_idx = np.triu_indices(n, 1)
        pair_dists = dists[i_idx, j_idx] / 2.0
        
        # Boundary distances
        radii = np.hypot(points[:, 0], points[:, 1])
        boundary_dists = 1.0 - radii
        
        # Combine all constraints
        all_dists = np.concatenate([pair_dists, boundary_dists])
        
        # Soft min
        m = np.min(all_dists)
        exp_vals = np.exp(-beta * (all_dists - m))
        weights = exp_vals / np.sum(exp_vals)
        
        soft_min_val = m + np.log(np.sum(exp_vals)) / beta
        
        # Gradient w.r.t. points
        grad = np.zeros_like(points)
        
        # Pairwise distance gradients
        pair_weights = weights[:len(pair_dists)]
        for k in range(len(pair_dists)):
            i, j = i_idx[k], j_idx[k]
            w = pair_weights[k]
            diff_ij = points[i] - points[j]
            dist_ij = np.sqrt(np.sum(diff_ij ** 2))
            if dist_ij > 1e-10:
                # d/dp_i (|p_i - p_j|/2) = (p_i - p_j) / (2 * |p_i - p_j|)
                # But we have -soft_min, so gradient of -soft_min w.r.t. p_i
                # = -sum_k w_k * d(d_k)/dp_i
                # For pair (i,j): d(pair_dist)/dp_i = (p_i - p_j) / (2 * |p_i - p_j|) * 2 / 2 = (p_i-p_j)/|p_i-p_j| / 2 * 2
                # Actually pair_dist = |p_i - p_j| / 2
                # d(pair_dist)/dp_i = (p_i - p_j) / (2 * |p_i - p_j|)
                # So d(-soft_min)/dp_i = -w * (p_i - p_j) / (2 * |p_i - p_j|)
                # Wait, let me reconsider. We want to maximize soft_min, so we minimize -soft_min.
                # grad_{p_i}(-soft_min) = -sum_k w_k * grad_{p_i}(d_k)
                # For pair (i,j): grad_{p_i}(|p_i-p_j|/2) = (p_i - p_j) / (2 * |p_i - p_j|)
                # So grad_{p_i}(-soft_min) includes -w * (p_i - p_j) / (2 * |p_i - p_j|)
                # To maximize, we move in direction -grad = +w * (p_i - p_j) / (2 * |p_i - p_j|)
                # But wait, that would move p_i away from p_j, which is wrong for maximizing min distance.
                # Let me re-derive.
                # soft_min = m + (1/beta) * log(sum exp(-beta * d_k))
                # d(soft_min)/dp_i = sum_k w_k * d(-d_k)/dp_i = -sum_k w_k * d(d_k)/dp_i
                # where w_k = exp(-beta*(d_k-m)) / sum_j exp(-beta*(d_j-m))
                # For pair (i,j): d_k = |p_i - p_j| / 2
                # d(d_k)/dp_i = (p_i - p_j) / (2 * |p_i - p_j|)
                # So d(soft_min)/dp_i = -w * (p_i - p_j) / (2 * |p_i - p_j|)
                # To maximize soft_min, we move in direction d(soft_min)/dp_i
                # = -w * (p_i - p_j) / (2 * |p_i - p_j|)
                # This moves p_i towards p_j? No: -(p_i - p_j) = p_j - p_i, so it moves p_i towards p_j.
                # That's wrong! We want to maximize the minimum distance, so we should move apart.
                
                # I think I have a sign error. Let me reconsider.
                # We want to maximize min(d_1, d_2, ..., d_m)
                # soft_min is a smooth approximation that is <= true min
                # d(soft_min)/dp = -sum_k w_k * d(d_k)/dp
                # For the pair constraint d_k = |p_i - p_j|/2, increasing this means moving apart
                # d(d_k)/dp_i = (p_i - p_j)/(2*|p_i-p_j|)
                # d(soft_min)/dp_i = -w_k * (p_i-p_j)/(2*|p_i-p_j|)
                # This is negative in the direction away from p_j, meaning soft_min decreases as we move apart?
                # No wait: if p_i moves away from p_j, |p_i-p_j| increases, d_k increases.
                # d(d_k)/dp_i dotted with (p_i-p_j)/|p_i-p_j| = 1/2 > 0
                # So d(soft_min)/dp_i dotted with (p_i-p_j)/|p_i-p_j| = -w_k * 1/2 < 0
                # This means soft_min decreases as p_i moves away from p_j, which is wrong!
                
                # I must have the soft_min formula wrong. Let me use a different formulation.
                # Actually, the standard soft-min is: softmin(x) = -1/beta * log(sum exp(-beta * x_i))
                # This satisfies: softmin(x) <= min(x), and softmin(x) -> min(x) as beta -> inf
                # d(softmin)/dx_i = -exp(-beta*x_i) / sum_j exp(-beta*x_j) = -w_i
                # So d(softmin)/dp = -sum_k w_k * d(d_k)/dp
                # For pair (i,j): d(d_k)/dp_i = (p_i-p_j)/(2*|p_i-p_j|)
                # d(softmin)/dp_i = -w_k * (p_i-p_j)/(2*|p_i-p_j|)
                # Moving in direction -d(softmin)/dp_i = w_k * (p_i-p_j)/(2*|p_i-p_j|) increases softmin
                # This moves p_i away from p_j, which is correct!
                
                # So to maximize soft_min, we update: p_i += lr * w_k * (p_i-p_j) / (2*|p_i-p_j|)
                # In code, we compute grad = -d(softmin)/dp, and do p += lr * grad
                # grad_{p_i} = w_k * (p_i-p_j) / (2*|p_i-p_j|) for each pair involving i
                # Plus boundary terms
                
                grad[i] += w * diff_ij / (2.0 * dist_ij)
                grad[j] -= w * diff_ij / (2.0 * dist_ij)
        
        # Boundary distance gradients
        boundary_weights = weights[len(pair_dists):]
        for i in range(n):
            w = boundary_weights[i]
            r = radii[i]
            if r > 1e-10:
                # d(1 - |p_i|)/dp_i = -p_i / |p_i|
                # d(softmin)/dp_i = -w * (-p_i/|p_i|) = w * p_i/|p_i|
                # To maximize, move in direction w * p_i/|p_i|, i.e., towards center
                # grad (for ascent) = w * p_i / |p_i|
                grad[i] += w * points[i] / r
        
        return soft_min_val, grad
    
    # Optimization loop
    best_points = points.copy()
    best_score = -np.inf
    
    # Compute initial score
    initial_beta = 50.0
    soft_min_val, _ = compute_score_and_grad(points, initial_beta)
    best_score = soft_min_val
    
    # Adaptive learning rate
    lr = 0.01
    beta = initial_beta
    
    for iteration in range(2000):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            break
        
        # Increase beta over time for tighter approximation
        beta = min(initial_beta * (1.0 + iteration * 0.001), 200.0)
        
        # Compute score and gradient
        soft_min_val, grad = compute_score_and_grad(points, beta)
        
        # Track best
        if soft_min_val > best_score:
            best_score = soft_min_val
            best_points = points.copy()
        
        # Update points (gradient ascent)
        points = points + lr * grad
        
        # Project back into unit circle (soft projection)
        radii = np.hypot(points[:, 0], points[:, 1])
        mask = radii > 0.99
        if np.any(mask):
            points[mask] *= 0.99 / radii[mask]
        
        # Adaptive learning rate
        if iteration % 100 == 0 and iteration > 0:
            lr *= 0.95
        
        # Check for convergence
        if iteration % 50 == 0:
            # Compute true min for monitoring
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            i_idx, j_idx = np.triu_indices(N, 1)
            pair_dists = dists[i_idx, j_idx] / 2.0
            radii = np.hypot(points[:, 0], points[:, 1])
            boundary_dists = 1.0 - radii
            true_min = np.min(np.concatenate([pair_dists, boundary_dists]))
            if true_min > best_score:
                best_score = true_min
                best_points = points.copy()
    
    # Step 3: Final polish with L-BFGS from scipy
    elapsed = time.time() - start_time
    if elapsed < deadline - 10.0:
        try:
            from scipy.optimize import minimize
            
            def objective_and_grad(x, beta=100.0):
                pts = x.reshape(N, 2)
                # Compute soft min
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff ** 2, axis=2))
                np.fill_diagonal(dists, np.inf)
                i_idx, j_idx = np.triu_indices(N, 1)
                pair_dists = dists[i_idx, j_idx] / 2.0
                radii = np.hypot(pts[:, 0], pts[:, 1])
                boundary_dists = 1.0 - radii
                all_dists = np.concatenate([pair_dists, boundary_dists])
                
                m = np.min(all_dists)
                exp_vals = np.exp(-beta * (all_dists - m))
                weights = exp_vals / np.sum(exp_vals)
                soft_min_val = m + np.log(np.sum(exp_vals)) / beta
                
                # Gradient
                grad_x = np.zeros_like(x)
                pair_weights = weights[:len(pair_dists)]
                for k in range(len(pair_dists)):
                    i, j = i_idx[k], j_idx[k]
                    w = pair_weights[k]
                    diff_ij = pts[i] - pts[j]
                    dist_ij = np.sqrt(np.sum(diff_ij ** 2))
                    if dist_ij > 1e-10:
                        grad_x[2*i:2*i+2] += w * diff_ij / (2.0 * dist_ij)
                        grad_x[2*j:2*j+2] -= w * diff_ij / (2.0 * dist_ij)
                
                boundary_weights = weights[len(pair_dists):]
                for i in range(N):
                    w = boundary_weights[i]
                    r = radii[i]
                    if r > 1e-10:
                        grad_x[2*i:2*i+2] += w * pts[i] / r
                
                # Return negative for minimization
                return -soft_min_val, -grad_x
            
            x0 = best_points.flatten()
            result = minimize(objective_and_grad, x0, method='L-BFGS-B', 
                              jac=True, options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result.success or result.fun < -best_score:
                polished = result.x.reshape(N, 2)
                # Verify validity
                diff = polished[:, np.newaxis, :] - polished[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff ** 2, axis=2))
                np.fill_diagonal(dists, np.inf)
                i_idx, j_idx = np.triu_indices(N, 1)
                pair_dists = dists[i_idx, j_idx] / 2.0
                radii = np.hypot(polished[:, 0], polished[:, 1])
                boundary_dists = 1.0 - radii
                true_min = np.min(np.concatenate([pair_dists, boundary_dists]))
                
                if true_min > best_score and np.all(radii <= 1.0) and np.all(np.isfinite(polished)):
                    best_points = polished
                    best_score = true_min
        except Exception:
            pass
    
    # Final validation
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    i_idx, j_idx = np.triu_indices(N, 1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    radii = np.hypot(best_points[:, 0], best_points[:, 1])
    boundary_dists = 1.0 - radii
    final_min = np.min(np.concatenate([pair_dists, boundary_dists]))
    
    # Ensure all points are inside and valid
    if np.any(radii > 1.0) or np.any(~np.isfinite(best_points)):
        # Fallback to hex lattice
        best_points = hex_lattice_init(N)
    
    return best_points
# EVOLVE-BLOCK-END
