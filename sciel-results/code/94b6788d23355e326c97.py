# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    time_limit = 600.0  # Seconds
    safety_margin = 5.0
    
    best_points = None
    best_score = -1.0

    def compute_score(pts):
        """Compute the raw score R = (d_min / d_max)^2"""
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def smooth_objective_and_grad(x, beta):
        """
        Computes the negative of the smooth ratio objective and its gradient.
        x is the flattened array of points (n * d).
        beta is the sharpness parameter.
        """
        pts = x.reshape(n, d)
        
        # Compute pairwise differences
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        
        # We need sums over all i != j
        # To avoid division by zero in gradients, add a small epsilon to dists where it is 0
        dists_safe = np.where(dists > 1e-8, dists, 1.0)
        
        # Calculate softmin and softmax sums
        # exp(-beta * dists) for min
        # exp(beta * dists) for max
        # Clip exponents to avoid overflow
        exp_min_args = -beta * dists_safe
        exp_max_args = beta * dists_safe
        
        # Mask out diagonal (dists=0) from sums
        exp_min_vals = np.where(mask, np.exp(np.clip(exp_min_args, -700, 700)), 0.0)
        exp_max_vals = np.where(mask, np.exp(np.clip(exp_max_args, -700, 700)), 0.0)
        
        S_min = np.sum(exp_min_vals)
        S_max = np.sum(exp_max_vals)
        
        if S_min == 0 or S_max == 0:
            return 0.0, np.zeros_like(x)
        
        m = -1.0 / beta * np.log(S_min)
        M = 1.0 / beta * np.log(S_max)
        
        # Objective to maximize: Phi = m / M
        # We want to minimize -Phi
        if M == 0:
            return 0.0, np.zeros_like(x)
            
        Phi = m / M
        obj = -Phi
        
        # Gradient calculation
        # grad_{p_i} Phi = (2 / M^2) * sum_{j != i} ( M * w_min_ij - m * w_max_ij ) * (p_i - p_j) / D_ij
        # w_min_ij = exp_min_vals[i,j] / S_min
        # w_max_ij = exp_max_vals[i,j] / S_max
        
        w_min = exp_min_vals / S_min
        w_max = exp_max_vals / S_max
        
        # Weights for the gradient term
        # term_ij = M * w_min_ij - m * w_max_ij
        weights = M * w_min - m * w_max
        
        # Gradient for each point i
        grad_pts = np.zeros_like(pts)
        
        # For each i, sum over j
        # (p_i - p_j) * weights[i,j] / dists[i,j]
        # Note: weights[i,i] is 0, so we can sum over all j
        # dists_safe is used for division
        
        # Vectorized computation:
        # diff[i,j] = p_i - p_j
        # We want sum_j weights[i,j] * diff[i,j] / dists_safe[i,j]
        
        # Create a matrix of unit vectors scaled by 1/d
        # unit_vecs[i,j] = diff[i,j] / dists_safe[i,j]
        # Handle dists=0 carefully (should be 0 due to mask in weights)
        
        inv_dists = np.where(dists_safe > 1e-8, 1.0 / dists_safe, 0.0)
        unit_vecs = diff * inv_dists[:, :, np.newaxis]
        
        # grad_pts[i] = (2 / M^2) * sum_j weights[i,j] * unit_vecs[i,j]
        # Sum over axis 1
        sum_grad = np.sum(weights[:, :, np.newaxis] * unit_vecs, axis=1)
        
        grad_pts = (2.0 / (M * M)) * sum_grad
        
        # Flatten gradient
        grad = grad_pts.flatten()
        
        return obj, grad

    def run_optimization(init_points, beta_schedule):
        """
        Runs the optimization starting from init_points with the given beta schedule.
        Returns the best points and score found.
        """
        current_points = init_points.copy()
        x = current_points.flatten()
        
        for beta in beta_schedule:
            if time.time() - start_time > time_limit - safety_margin:
                break
                
            # Use L-BFGS-B
            try:
                res = minimize(
                    smooth_objective_and_grad,
                    x,
                    args=(beta,),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'gtol': 1e-8, 'maxfun': 1000}
                )
                if res.success or res.fun < 0: # res.fun is -Phi
                    x = res.x
                    current_points = x.reshape(n, d)
            except Exception:
                pass
            
        # Final score calculation with exact min/max
        score = compute_score(current_points)
        return current_points, score

    # Define beta schedule
    # Start with low beta (smooth) and go to high beta (sharp)
    betas = [10, 50, 100, 200, 500, 1000, 2000]
    
    # Define initial configurations
    inits = []
    
    # 1. Circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Random
    np.random.seed(42)
    inits.append(np.random.randn(n, d))
    
    # 3. Grid-like
    x_grid = np.linspace(-1, 1, 4)
    y_grid = np.linspace(-1, 1, 4)
    grid_pts = np.array([(x, y) for x in x_grid for y in y_grid])
    inits.append(grid_pts)
    
    # 4. Another random
    np.random.seed(123)
    inits.append(np.random.randn(n, d) * 2)
    
    # 5. Square
    angles_sq = np.linspace(0, 2 * np.pi, n, endpoint=False)
    inits.append(np.column_stack([np.sign(np.cos(angles_sq)), np.sign(np.sin(angles_sq))] * 1.0)) # Just corners? No, this is sparse.
    # Let's use a perturbed circle
    inits.append(np.column_stack([np.cos(angles) + np.random.randn(n)*0.1, np.sin(angles) + np.random.randn(n)*0.1]))

    for init in inits:
        if time.time() - start_time > time_limit - safety_margin:
            break
            
        pts, score = run_optimization(init, betas)
        
        if score > best_score:
            best_score = score
            best_points = pts

    # If no optimization was run or failed, fallback to best init
    if best_points is None:
        best_points = inits[0]
        best_score = compute_score(best_points)

    # Final polish: Try a few steps of exact gradient ascent on the raw ratio
    # This helps if the smooth approximation converged to a local min that is slightly better with exact moves
    if time.time() - start_time < time_limit - safety_margin and best_points is not None:
        current_pts = best_points.copy()
        current_score = best_score
        lr = 0.01
        for step in range(100):
            if time.time() - start_time > time_limit - safety_margin:
                break
            
            dists = pdist(current_pts)
            d_min = np.min(dists)
            d_max = np.max(dists)
            
            if d_max == 0:
                break
                
            # Identify pairs
            # For simplicity, we just use the smooth gradient with a very high beta
            # which approximates the exact gradient
            obj, grad = smooth_objective_and_grad(current_pts.flatten(), 10000)
            
            # Gradient ascent on Phi (minimize -Phi)
            new_pts = current_pts - lr * grad.reshape(n, d)
            
            new_score = compute_score(new_pts)
            
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_pts
            else:
                lr *= 0.5
                if lr < 1e-6:
                    break

    return best_points

# EVOLVE-BLOCK-END
