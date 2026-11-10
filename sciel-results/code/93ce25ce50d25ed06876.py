# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the points.

    """
    n = 16
    d = 2
    T = 0.1  # Temperature for soft min/max

    def objective_and_grad(x_flat):
        """
        Objective: minimize -log(soft_min) + log(soft_max)
        This is equivalent to maximizing log(soft_min / soft_max)
        """
        x = x_flat.reshape(n, d)
        
        # Compute pairwise distances and their gradients
        # d_{ij} = ||x_i - x_j||
        # We need gradients of d_{ij} w.r.t x_k
        # Instead of storing all gradients, we can compute the weighted sum directly.
        
        # Compute all pairwise distances
        dists = pdist(x)
        
        # Handle degenerate case where all points are the same
        if np.allclose(dists, 0):
            return 0.0, np.zeros_like(x_flat)
        
        # Soft min and max
        # m = -T * log(sum(exp(-d/T)))
        # M = T * log(sum(exp(d/T)))
        
        # To avoid overflow/underflow in exp, we can use log-sum-exp tricks, 
        # but for T=0.1 and reasonable distances, direct exp might be okay if we normalize.
        # However, let's stick to the formula and be careful.
        
        # Weights for soft min
        # w_min_ij = exp(-d_ij / T) / sum(exp(-d_ij / T))
        # Weights for soft max
        # w_max_ij = exp(d_ij / T) / sum(exp(d_ij / T))
        
        # Calculate exponents
        # To prevent overflow, subtract max from d for max calc, add min to d for min calc?
        # Actually, let's just compute the sums.
        
        # For stability, let's use the fact that we only need the gradient direction.
        # But we need the value too.
        
        # Let's compute the terms
        exp_min_terms = np.exp(-dists / T)
        exp_max_terms = np.exp(dists / T)
        
        sum_min = np.sum(exp_min_terms)
        sum_max = np.sum(exp_max_terms)
        
        if sum_min == 0 or sum_max == 0:
            return 0.0, np.zeros_like(x_flat)
            
        w_min = exp_min_terms / sum_min
        w_max = exp_max_terms / sum_max
        
        m = -T * np.log(sum_min)
        M = T * np.log(sum_max)
        
        if m <= 0 or M <= 0:
            return 0.0, np.zeros_like(x_flat)
            
        # Objective value: - (log(m) - log(M)) = log(M) - log(m)
        val = np.log(M) - np.log(m)
        
        # Gradient calculation
        # grad_x (log(m) - log(M)) = (1/m) * grad_x(m) - (1/M) * grad_x(M)
        # grad_x(m) = sum_{ij} w_min_ij * grad_x(d_ij)
        # grad_x(M) = sum_{ij} w_max_ij * grad_x(d_ij)
        
        # grad_x(d_ij) is non-zero only for x_i and x_j
        # grad_x_i(d_ij) = (x_i - x_j) / d_ij
        # grad_x_j(d_ij) = (x_j - x_i) / d_ij
        
        grad = np.zeros_like(x_flat)
        
        # We need to iterate over all pairs (i, j)
        # pdist returns distances in condensed form: (0,1), (0,2), ..., (0,n-1), (1,2), ...
        # We can generate the indices or use a loop.
        
        # Efficient way: 
        # For each pair (i, j), we add to grad[i] and grad[j].
        
        # Let's get the indices of the condensed distances
        # i < j
        # Number of distances is n*(n-1)/2
        
        # We can use np.triu_indices
        i_idx, j_idx = np.triu_indices(n, k=1)
        
        # d_ij for each pair
        # dists is already in this order
        
        # Vectorized gradient accumulation is tricky because we need to scatter.
        # Let's loop over the pairs. 120 iterations is very fast.
        
        for k in range(len(dists)):
            i = i_idx[k]
            j = j_idx[k]
            d_ij = dists[k]
            
            if d_ij < 1e-10:
                continue
                
            # Unit vector from j to i
            # diff = x_i - x_j
            diff = x[i] - x[j]
            # grad_d_ij_wrt_x_i = diff / d_ij
            # grad_d_ij_wrt_x_j = -diff / d_ij
            
            # Coefficient for this pair in the total gradient
            # coeff_min = w_min[k] / m
            # coeff_max = w_max[k] / M
            # Total coeff for grad_x(d_ij) in the objective gradient:
            # We are minimizing val = log(M) - log(m)
            # grad_val = (1/M) * grad_M - (1/m) * grad_m
            # grad_m = sum w_min * grad_d
            # grad_M = sum w_max * grad_d
            # So grad_val = sum ( (w_max/M) - (w_min/m) ) * grad_d
            
            coeff = (w_max[k] / M) - (w_min[k] / m)
            
            # grad_d_ij w.r.t x_i is diff/d_ij
            # grad_d_ij w.r.t x_j is -diff/d_ij
            
            grad_vec = diff / d_ij
            
            # Add to grad[i] and grad[j]
            # Note: x_flat is flattened x. x[i] corresponds to indices 2*i and 2*i+1
            idx_i = slice(2*i, 2*i+2)
            idx_j = slice(2*j, 2*j+2)
            
            grad[idx_i] += coeff * grad_vec
            grad[idx_j] -= coeff * grad_vec
            
        return val, grad

    best_points = None
    best_score = -np.inf

    # Multi-start
    # 1. Regular polygon
    # 2. Random
    # 3. Grid perturbed
    
    initializations = []
    
    # Regular polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    initializations.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Random points
    rng = np.random.RandomState(42)
    for _ in range(5):
        x0 = rng.randn(n, d)
        # Normalize to unit scale roughly
        dists = pdist(x0)
        if np.max(dists) > 0:
            x0 = x0 / np.max(dists)
        initializations.append(x0)
    
    # Perturbed regular polygon
    for _ in range(5):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radii = 1.0 + rng.uniform(-0.2, 0.2, n)
        x0 = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initializations.append(x0)
        
    # Grid-like
    for _ in range(5):
        # 4x4 grid
        xs = np.linspace(-1, 1, 4)
        ys = np.linspace(-1, 1, 4)
        xx, yy = np.meshgrid(xs, ys)
        x0 = np.column_stack([xx.flatten(), yy.flatten()])
        x0 = x0 + rng.randn(n, d) * 0.1
        initializations.append(x0)

    for x0 in initializations:
        try:
            res = minimize(
                objective_and_grad,
                x0.flatten(),
                jac=True, # Returns (val, grad)
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            points = res.x.reshape(n, d)
            
            # Evaluate actual score
            dists = pdist(points)
            if len(dists) > 0 and np.max(dists) > 0:
                m = np.min(dists)
                M = np.max(dists)
                score = (m / M) ** 2
            else:
                score = 0
                
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
        except Exception:
            continue

    if best_points is None:
        # Fallback
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points


# EVOLVE-BLOCK-END
