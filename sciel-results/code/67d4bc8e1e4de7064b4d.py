"""
Packing 200 equal circles in a unit square using smooth gradient-based optimization.
Uses Log-Sum-Exp approximation of the minimum distance to create a differentiable objective.
Optimized with L-BFGS-B from SciPy.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 200

def _compute_radius(pts):
    """Compute the exact maximum common radius for a configuration."""
    if pts is None or pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    # Check bounds
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        # Clip for evaluation, but technically invalid if outside. 
        # The problem states: "A centre on or outside the square ... make the candidate invalid."
        # So we must ensure pts are strictly inside or on boundary.
        # However, the evaluator calculates r = min(...). If pts are outside, wall dist is negative?
        # "min_i min_k (1/2 - |c_ik|)" -> if c_ik > 0.5, this is negative.
        # So we just calculate the value.
        pass
        
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    # Square is [-0.5, 0.5] x [-0.5, 0.5]
    # Distance to nearest wall for point (x,y) is min(0.5-x, x+0.5, 0.5-y, y+0.5)
    # Which is 0.5 - max(|x|, |y|)
    wall_dists = 0.5 - np.max(np.abs(pts), axis=1)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _soft_min_beta(beta, pts):
    """
    Calculate the smooth approximation of the minimum distance and its gradient.
    Objective to minimize: -soft_min(distances)
    soft_min(d) = -1/beta * log(sum(exp(-beta * d)))
    """
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12) # epsilon for stability
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    wall_dists = 0.5 - np.max(np.abs(pts), axis=1)
    
    # Combine all distances into a single vector
    # We need to be careful with memory. N=200, pairs ~ 20k.
    # dists is 200x200. We can flatten the upper triangle.
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # To avoid overflow/underflow in exp, shift by min
    min_d = np.min(all_dists)
    exp_terms = np.exp(-beta * (all_dists - min_d))
    sum_exp = np.sum(exp_terms)
    
    # soft_min = -1/beta * log(sum_exp) + min_d
    soft_min_val = -1.0 / beta * np.log(sum_exp) + min_d
    
    # Gradient of soft_min w.r.t. pts
    # d(soft_min)/d(pts) = sum_k ( -beta * exp(-beta * d_k) / sum_exp * (-d d_k / d pts) )
    #                     = sum_k ( beta * w_k * d d_k / d pts )
    # where w_k = exp(-beta * d_k) / sum_exp
    
    weights = beta * exp_terms / sum_exp
    
    # Gradient of pairwise distances
    # d_{ij} = ||p_i - p_j||
    # d d_{ij} / d p_i = (p_i - p_j) / d_{ij}
    # d d_{ij} / d p_j = (p_j - p_i) / d_{ij}
    
    grad = np.zeros_like(pts)
    
    # Process pairs
    # weights[:N_pairs] correspond to pair_dists
    pair_weights = weights[:len(pair_dists)]
    
    # Vectorized gradient accumulation for pairs
    # diff is pts[:, None, :] - pts[None, :, :]
    # We need to sum over j for each i
    # grad_i = sum_{j != i} w_{ij} * (p_i - p_j) / d_{ij}
    
    # We can use the full matrix for easier vectorization, then take upper triangle logic?
    # Or just loop over the non-zero weights? N=200, 20k pairs. Loop is slow in Python.
    # Let's use matrix operations.
    
    # Create a matrix of weights W where W[i,j] = w_{ij} if i<j else 0? 
    # Actually, the weight for distance d_ij is the same for the gradient contribution to i and j (with opposite sign).
    # Let's build a weight matrix W (N x N) initialized to 0.
    W = np.zeros((N, N))
    W[i_idx, j_idx] = pair_weights
    W[j_idx, i_idx] = pair_weights # Symmetric
    
    # Normalize distances for gradient direction
    # dists has inf on diagonal. Replace with 1 to avoid nan, but W is 0 there so it won't matter.
    safe_dists = np.where(dists > 0, dists, 1.0)
    
    # grad_i = sum_j W[i,j] * (p_i - p_j) / d_ij
    # Let's compute the vector sum
    # (p_i - p_j) is diff[i,j]
    # We want sum_j W[i,j] * diff[i,j] / safe_dists[i,j]
    
    # term = W * (diff / safe_dists[:, :, None])
    # This creates a (N, N, 2) array.
    # grad = sum(term, axis=1)
    
    # To save memory, we can do:
    # grad = np.sum(W[:, :, None] * (diff / safe_dists[:, :, None]), axis=1)
    
    # Let's try this vectorized approach.
    # diff is (N, N, 2)
    # safe_dists is (N, N)
    # W is (N, N)
    
    # Optimization: Only compute for non-zero W? 
    # With N=200, the full matrix is 40k elements. 40k * 2 floats = 80k floats. Very small.
    
    grad_pairs = np.sum(W[:, :, np.newaxis] * (diff / safe_dists[:, :, np.newaxis]), axis=1)
    
    # Gradient of wall distances
    # w_i = 0.5 - max(|x_i|, |y_i|)
    # d w_i / d x_i = -sign(x_i) if |x_i| > |y_i| else 0
    # d w_i / d y_i = -sign(y_i) if |y_i| >= |x_i| else 0
    
    wall_weights = weights[len(pair_dists):] # Weights for wall distances
    
    abs_x = np.abs(pts[:, 0])
    abs_y = np.abs(pts[:, 1])
    
    # Determine which wall is closest (max absolute coord)
    # If |x| > |y|, closest wall is vertical. Gradient w.r.t x is -sign(x), w.r.t y is 0.
    # If |y| >= |x|, closest wall is horizontal. Gradient w.r.t y is -sign(y), w.r.t x is 0.
    
    grad_walls = np.zeros_like(pts)
    
    mask_x = abs_x > abs_y
    mask_y = ~mask_x
    
    # For x-dominated points
    grad_walls[mask_x, 0] = -wall_weights[mask_x] * np.sign(pts[mask_x, 0])
    # For y-dominated points
    grad_walls[mask_y, 1] = -wall_weights[mask_y] * np.sign(pts[mask_y, 1])
    
    # Total gradient of soft_min
    grad_soft_min = grad_pairs + grad_walls
    
    # We want to MINIMIZE -soft_min
    # So objective = -soft_min_val
    # Gradient of objective = -grad_soft_min
    
    return -soft_min_val, -grad_soft_min

def _initialize_hexagonal(n):
    """Generate a hexagonal lattice initialization."""
    points = []
    # Hex lattice parameters
    # Spacing s
    s = 1.0
    dx = s
    dy = s * np.sqrt(3) / 2
    
    rows = int(np.ceil(np.sqrt(n / 0.866))) + 2
    for row in range(rows):
        for col in range(rows + 2):
            x = col * dx + (0.5 * dx if row % 2 else 0.0)
            y = row * dy
            points.append([x, y])
            if len(points) >= n:
                break
        if len(points) >= n:
            break
            
    pts = np.array(points[:n])
    # Center and scale
    pts -= pts.mean(axis=0)
    max_extent = np.max(np.abs(pts))
    if max_extent > 0:
        pts *= 0.45 / max_extent
        
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initialize
    pts = _initialize_hexagonal(N)
    
    # We will run L-BFGS-B with increasing beta (sharpness)
    # Start with lower beta to untangle, then increase.
    betas = [10.0, 30.0, 100.0, 300.0]
    
    best_pts = pts.copy()
    best_radius = _compute_radius(pts)
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        def objective(x):
            val, _ = _soft_min_beta(beta, x.reshape(N, 2))
            return val
            
        def gradient(x):
            _, grad = _soft_min_beta(beta, x.reshape(N, 2))
            return grad.ravel()
            
        try:
            res = minimize(
                objective, 
                pts.ravel(), 
                jac=gradient, 
                method='L-BFGS-B', 
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < objective(pts.ravel()):
                new_pts = res.x.reshape(N, 2)
                # Clip to valid range
                new_pts = np.clip(new_pts, -0.5, 0.5)
                
                current_radius = _compute_radius(new_pts)
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_pts = new_pts
                pts = new_pts
        except Exception as e:
            pass # Keep previous pts if optimization fails

    # Final polish with very high beta if time permits
    if time.time() < deadline - 10.0:
        beta_final = 1000.0
        def objective_final(x):
            val, _ = _soft_min_beta(beta_final, x.reshape(N, 2))
            return val
        def gradient_final(x):
            _, grad = _soft_min_beta(beta_final, x.reshape(N, 2))
            return grad.ravel()
            
        try:
            res = minimize(
                objective_final, 
                best_pts.ravel(), 
                jac=gradient_final, 
                method='L-BFGS-B', 
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            new_pts = res.x.reshape(N, 2)
            new_pts = np.clip(new_pts, -0.5, 0.5)
            current_radius = _compute_radius(new_pts)
            if current_radius > best_radius:
                best_radius = current_radius
                best_pts = new_pts
        except Exception:
            pass

    return best_pts

# Test/Run
if __name__ == "__main__":
    pts = construct()
    r = _compute_radius(pts)
    print(f"Radius: {r}")
