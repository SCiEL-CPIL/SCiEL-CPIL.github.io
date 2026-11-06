"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Find 50 circle centers in [-0.5, 0.5]^2 that maximize the minimum radius.
    
    Strategy: Smooth differentiable objective with soft-min penalty,
    initialized from jittered hexagonal grids, refined via L-BFGS-B.
    """
    deadline = time.time() + 140  # 10s safety margin
    
    def compute_radius(c):
        """Compute the actual radius for a set of centers."""
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists)
        min_boundary = 0.5 - np.max(np.abs(c), axis=1).max()
        return min(min_pairwise / 2.0, min_boundary)
    
    def soft_min_objective(c_flat, sharpness=50.0):
        """
        Smooth differentiable objective that penalizes:
        1. Pairwise distances below a target (soft min of pairwise distances)
        2. Boundary violations
        
        Returns negative of a smooth approximation to the radius.
        """
        c = c_flat.reshape(N, 2)
        
        # Boundary penalty: penalize any coordinate exceeding 0.5
        abs_c = np.abs(c)
        boundary_violation = np.maximum(0.0, abs_c - 0.5)
        boundary_penalty = np.sum(boundary_violation**2)
        
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq + 1e-300)
        np.fill_diagonal(dists, np.inf)
        
        # Soft minimum of pairwise distances using log-sum-exp trick
        # We want to penalize when min distance is small
        # Use: -softmin = -(-1/sharpness * log(sum(exp(-sharpness * d))))
        # This is smooth and differentiable
        # To avoid overflow, subtract max first
        d_min_val = np.min(dists)
        # Soft min approximation: for large sharpness, approaches true min
        # Use a simpler approach: penalize pairs that are too close
        target_r = 0.07  # approximate target radius
        # Penalty for pairs closer than 2*target_r
        close_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = dists[i, j]
                if d < 2.0 * target_r:
                    close_penalty += (2.0 * target_r - d)**2
        
        # Boundary penalty weighted
        total_penalty = close_penalty + 10.0 * boundary_penalty
        
        # Also add a term that encourages larger minimum distance
        # Use soft min of all pairwise distances and boundary clearances
        boundary_clearances = 0.5 - abs_c
        min_boundary_clearance = np.min(boundary_clearances)
        
        # Combined: we want to maximize min(min_pairwise/2, min_boundary)
        # Smooth approximation: use soft min
        # Let's use a different approach: minimize the sum of squared violations
        # where violation = max(0, 2*r - d_ij) for some target r
        
        # Simpler: just return negative of a smooth proxy
        # The soft min of pairwise distances / 2, combined with boundary
        # Use: -0.5 * softmin(dists) + boundary_penalty
        # Soft min via log-sum-exp with negative sign
        # softmin(d) = -1/k * log(sum(exp(-k*d)))
        k = sharpness
        # For numerical stability
        neg_k_dists = -k * dists
        max_neg_k_dists = np.max(neg_k_dists)
        log_sum_exp = max_neg_k_dists + np.log(np.sum(np.exp(neg_k_dists - max_neg_k_dists)))
        softmin_dists = -log_sum_exp / k
        
        # Soft min of boundary clearances
        neg_k_boundary = -k * boundary_clearances
        max_neg_k_boundary = np.max(neg_k_boundary)
        log_sum_exp_boundary = max_neg_k_boundary + np.log(np.sum(np.exp(neg_k_boundary - max_neg_k_boundary)))
        softmin_boundary = -log_sum_exp_boundary / k
        
        # The radius is min(softmin_dists/2, softmin_boundary)
        # Approximate the min with another soft min
        # softmin(a, b) = -1/k * log(exp(-k*a) + exp(-k*b))
        a = softmin_dists / 2.0
        b = softmin_boundary
        neg_k_ab = -k * np.array([a, b])
        max_neg_k_ab = np.max(neg_k_ab)
        log_sum_exp_ab = max_neg_k_ab + np.log(np.sum(np.exp(neg_k_ab - max_neg_k_ab)))
        softmin_radius = -log_sum_exp_ab / k
        
        return -softmin_radius + 0.1 * boundary_penalty
    
    def generate_hex_initial(seed=42, jitter_scale=0.01):
        """Generate a hexagonal grid initial configuration."""
        np.random.seed(seed)
        
        # Hexagonal packing: rows offset by half spacing
        # For 50 points in unit square
        # Try different aspect ratios
        best_c = None
        best_r = -np.inf
        
        for n_rows in range(5, 11):
            n_cols = int(np.ceil(N / n_rows))
            total = n_rows * n_cols
            if total < N:
                continue
            
            spacing_x = 1.0 / (n_cols + 0.5)
            spacing_y = 1.0 / (n_rows + 0.5)
            
            # Hexagonal offset
            hex_offset = spacing_x * 0.5
            
            points = []
            for row in range(n_rows):
                for col in range(n_cols):
                    x = -0.5 + spacing_x * (col + 0.75) + (row % 2) * hex_offset * 0.5
                    y = -0.5 + spacing_y * (row + 0.75)
                    points.append([x, y])
            
            if len(points) < N:
                continue
            
            c = np.array(points[:N])
            
            # Add jitter
            c += np.random.normal(0, jitter_scale, size=c.shape)
            c = np.clip(c, -0.49, 0.49)
            
            r = compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
        
        if best_c is None:
            # Fallback to grid
            m = int(np.ceil(np.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            best_c = grid[:N].copy()
            best_c += np.random.normal(0, 0.01, size=best_c.shape)
            best_c = np.clip(best_c, -0.49, 0.49)
        
        return best_c
    
    # Start with best hex initial
    best_centers = generate_hex_initial(seed=42, jitter_scale=0.005)
    best_val = compute_radius(best_centers)
    
    # Multiple restarts with different seeds and jitter levels
    seeds = [42, 123, 456, 789, 1024, 2048, 3141, 5678, 9101, 12345]
    jitter_scales = [0.005, 0.01, 0.015, 0.02, 0.003, 0.008, 0.012, 0.018]
    
    n_attempts = 0
    for seed in seeds:
        for js in jitter_scales[:3]:
            if time.time() > deadline - 15:
                break
            
            c_init = generate_hex_initial(seed=seed, jitter_scale=js)
            
            try:
                result = minimize(
                    soft_min_objective,
                    c_init.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'maxfun': 50000, 'gtol': 1e-5, 'ftol': 1e-8},
                    bounds=[(-0.5, 0.5)] * (2 * N)
                )
                
                c_curr = result.x.reshape(N, 2)
                c_curr = np.clip(c_curr, -0.5 + 1e-8, 0.5 - 1e-8)
                
                # Verify feasibility
                if np.any(np.abs(c_curr) > 0.5):
                    continue
                
                r_curr = compute_radius(c_curr)
                if r_curr > best_val:
                    best_val = r_curr
                    best_centers = c_curr.copy()
                
                n_attempts += 1
                
            except Exception:
                pass
        
        if time.time() > deadline - 15:
            break
    
    # Additional refinement: local search on the best configuration
    if time.time() < deadline - 10:
        c = best_centers.copy()
        
        # Gradient-based local refinement with smooth objective
        for refinement_pass in range(3):
            if time.time() > deadline - 5:
                break
            
            try:
                result = minimize(
                    soft_min_objective,
                    c.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'maxfun': 100000, 'gtol': 1e-6, 'ftol': 1e-9},
                    bounds=[(-0.5, 0.5)] * (2 * N)
                )
                
                c_curr = result.x.reshape(N, 2)
                c_curr = np.clip(c_curr, -0.5 + 1e-8, 0.5 - 1e-8)
                
                r_curr = compute_radius(c_curr)
                if r_curr > best_val:
                    best_val = r_curr
                    best_centers = c_curr.copy()
                    c = best_centers.copy()
                else:
                    break
                    
            except Exception:
                break
        
        # Pairwise local search: push closest pairs apart
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = min_idx[0], min_idx[1]
            if i >= j:
                i, j = j, i
            
            d = c[j] - c[i]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                continue
            
            direction = d / dist
            step = 0.0005
            new_i = c[i] - direction * step
            new_j = c[j] + direction * step
            
            if np.max(np.abs(new_i)) <= 0.5 and np.max(np.abs(new_j)) <= 0.5:
                old_r = compute_radius(c)
                c[i] = new_i
                c[j] = new_j
                new_r = compute_radius(c)
                if new_r > old_r:
                    if new_r > best_val:
                        best_val = new_r
                        best_centers = c.copy()
                else:
                    c[i] = c[i] + direction * step
                    c[j] = c[j] - direction * step
    
    return best_centers
# EVOLVE-BLOCK-END
