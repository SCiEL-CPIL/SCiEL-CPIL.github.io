"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # Initial positions: simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    x0 = grid[:N].flatten()
    
    def compute_radius(x):
        """Compute the maximum radius for given positions."""
        positions = x.reshape(-1, 3)
        # Minimum distance to boundary
        boundary_dists = 0.5 - np.abs(positions)
        min_boundary = np.min(boundary_dists)
        
        # Minimum pairwise distance / 2
        min_pair = np.min(pdist(positions)) / 2.0
        
        return min(min_boundary, min_pair)
    
    def soft_min(values, beta=50.0):
        """Smooth approximation of min using log-sum-exp."""
        # Use -logsumexp(-values) / beta for soft min
        # Clip to avoid overflow
        vals = np.clip(values, -1e10, 1e10)
        return -np.logaddexp.reduce(-vals / beta) * beta
    
    def negative_soft_radius(x, beta=50.0):
        """Smooth objective: negative of soft-min radius."""
        positions = x.reshape(-1, 3)
        # Boundary distances
        boundary_dists = 0.5 - np.abs(positions)
        # Pairwise distances / 2
        pair_dists = pdist(positions) / 2.0
        
        # Combine all constraints into a single soft-min
        all_dists = np.concatenate([boundary_dists, pair_dists])
        return -soft_min(all_dists, beta)
    
    def negative_soft_radius_grad(x, beta=50.0):
        """Gradient of the smooth objective."""
        positions = x.reshape(-1, 3)
        boundary_dists = 0.5 - np.abs(positions)
        pair_dists = pdist(positions) / 2.0
        all_dists = np.concatenate([boundary_dists, pair_dists])
        
        # Soft min gradient: softmax of -values
        neg_vals = -all_dists / beta
        max_neg = np.max(neg_vals)
        exp_vals = np.exp(neg_vals - max_neg)
        weights = exp_vals / np.sum(exp_vals)
        
        # Gradient w.r.t. each distance
        grad_dists = -weights  # d(-soft_min)/d(dist_i) = -w_i
        
        # Build gradient w.r.t. x
        grad = np.zeros_like(x)
        
        # Boundary part
        # d(boundary_dist_i)/dx_j for each coordinate
        # boundary_dist_i = 0.5 - |x_i|
        # d/dx_i = -sign(x_i)
        sign_x = np.sign(positions)
        grad_boundary = -sign_x * grad_dists[:N*3].reshape(-1, 3)
        grad[:N*3] += grad_boundary.flatten()
        
        # Pairwise part
        # For each pair (i,j), dist_ij = |x_i - x_j|/2
        # d(dist_ij)/dx_i = (x_i - x_j) / (2 * |x_i - x_j|)
        # d(dist_ij)/dx_j = -(x_i - x_j) / (2 * |x_i - x_j|)
        idx_i, idx_j = np.triu_indices(N, 1)
        diff = positions[idx_i] - positions[idx_j]
        dists = np.linalg.norm(diff, axis=1)
        dists = np.maximum(dists, 1e-12)
        
        # Unit vectors
        unit_vecs = diff / dists[:, None]
        
        # Gradient contribution from each pair
        # weight for pair (i,j) is grad_dists[N*3 + pair_idx]
        pair_weights = grad_dists[N*3:]
        
        for k in range(len(idx_i)):
            i, j = idx_i[k], idx_j[k]
            w = pair_weights[k]
            grad[3*i:3*i+3] += w * unit_vecs[k] / 2.0
            grad[3*j:3*j+3] -= w * unit_vecs[k] / 2.0
        
        return grad
    
    # Initial radius
    r0 = compute_radius(x0)
    best_x = x0.copy()
    best_r = r0
    
    # Bounds: positions must be in [-0.5, 0.5]^3
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Try multiple random restarts with local optimization
    rng = np.random.RandomState(42)
    num_restarts = 15
    
    for restart in range(num_restarts):
        if time.time() > deadline - 40.0:
            break
        
        # For first restart, use the grid; for others, perturb
        if restart == 0:
            x_current = x0.copy()
        else:
            x_current = x0 + rng.randn(N * 3) * 0.08
            # Clip to bounds
            x_current = np.clip(x_current, -0.49, 0.49)
        
        # Local optimization using L-BFGS-B with smooth objective
        try:
            result = minimize(negative_soft_radius, x_current, method='L-BFGS-B',
                             bounds=bounds,
                             jac=lambda x: negative_soft_radius_grad(x, beta=50.0),
                             options={'maxiter': 300, 'gtol': 1e-7, 'ftol': 1e-9})
            
            if result.success or (-result.fun) > best_r:
                r_new = compute_radius(result.x)
                if r_new > best_r:
                    best_r = r_new
                    best_x = result.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish pass with tighter beta and more iterations
    if time.time() < deadline - 15.0:
        try:
            # Use tighter beta for more accurate min approximation
            result = minimize(negative_soft_radius, best_x, method='L-BFGS-B',
                             bounds=bounds,
                             jac=lambda x: negative_soft_radius_grad(x, beta=100.0),
                             options={'maxiter': 500, 'gtol': 1e-9, 'ftol': 1e-11})
            
            r_new = compute_radius(result.x)
            if r_new > best_r:
                best_r = r_new
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Additional polish with COBYLA for final refinement
    if time.time() < deadline - 5.0:
        try:
            def negative_radius_exact(x):
                return -compute_radius(x)
            
            result = minimize(negative_radius_exact, best_x, method='COBYLA',
                             options={'maxiter': 1000, 'rhobeg': 1e-7, 'catol': 1e-10})
            
            r_new = compute_radius(result.x)
            if r_new > best_r:
                best_r = r_new
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Ensure final result is within bounds
    best_x = np.clip(best_x, -0.5, 0.5)
    
    return best_x.reshape(-1, 3)
# EVOLVE-BLOCK-END
