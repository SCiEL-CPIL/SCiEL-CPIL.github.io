import numpy as np
import time
from scipy.optimize import minimize

N = 300

def compute_score(points):
    """Compute the radius for a given set of points."""
    # Distance between all pairs
    # This is O(N^2) which is 300^2 = 90000, very fast.
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Mask diagonal
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    # Distance to boundary
    dist_to_origin = np.linalg.norm(points, axis=1)
    min_boundary_dist = np.min(1.0 - dist_to_origin)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)

def get_hex_grid(N):
    """Generate a hexagonal grid and select N points closest to the center."""
    # Estimate spacing to fit N points in unit circle
    # Area of hex cell ~ (sqrt(3)/2) * s^2
    # Area of circle = pi
    # N * (sqrt(3)/2) * s^2 ~ pi
    # s ~ sqrt(2*pi / (N*sqrt(3)))
    s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    
    # Generate grid
    k = int(2.0 / s) + 5
    rows = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) < 1.0:
                rows.append((x, y))
    
    points = np.array(rows)
    if len(points) < N:
        # If not enough, shrink spacing
        while len(points) < N:
            s *= 0.95
            k = int(2.0 / s) + 5
            rows = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * s
                    y = j * s * np.sqrt(3.0) / 2.0
                    if np.hypot(x, y) < 1.0:
                        rows.append((x, y))
            points = np.array(rows)
            
    # Select N closest to center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    return points[idx]

def soft_min_objective(x, lambda_val):
    """
    Minimize this objective to maximize the minimum distance.
    Uses log-sum-exp to approximate the min.
    """
    points = x.reshape(-1, 2)
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    # Boundary distances
    dist_to_origin = np.linalg.norm(points, axis=1)
    boundary_dist = 1.0 - dist_to_origin
    
    # We want to maximize min( dist/2, boundary_dist )
    # So we minimize - min(...)
    # Approximate min(a, b) with -1/lambda * log(exp(-lambda a) + exp(-lambda b))
    
    # To avoid overflow, we can subtract the max (which is negative here, so min of negs is max of exps? No.)
    # Let's just compute the terms.
    # terms = exp(-lambda * dist/2) for pairs, exp(-lambda * boundary_dist) for boundaries
    # Sum them, log, divide by -lambda.
    
    # Numerical stability:
    # The values are distances. Max dist ~ 2. Min dist ~ 0.1.
    # lambda * dist ~ 50 * 0.1 = 5. exp(-5) is small.
    # If dist is very small, exp(-lambda*dist) is close to 1.
    
    pair_terms = np.exp(-lambda_val * dist / 2.0)
    boundary_terms = np.exp(-lambda_val * boundary_dist)
    
    # Sum all terms
    # dist matrix is N x N, boundary is N.
    # We only need the sum of all terms.
    total_sum = np.sum(pair_terms) + np.sum(boundary_terms)
    
    # The objective is -1/lambda * log(total_sum)
    # We minimize this.
    return -1.0 / lambda_val * np.log(total_sum)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Initial guess
    x0 = get_hex_grid(N).flatten()
    current_best = x0.copy()
    best_score = compute_score(current_best.reshape(-1, 2))
    
    # 2. Optimization loop with increasing lambda
    # Start with a moderate lambda to find the basin, then sharpen.
    lambdas = [20, 50, 100, 200, 400, 800]
    
    for lam in lambdas:
        if time.time() > deadline:
            break
            
        # Define objective and gradient
        # Gradient of soft min is complex, but L-BFGS-B can approximate it.
        # For better performance, we could provide gradient, but finite diff is okay for 600 dims if steps are few.
        # However, finite diff on 600 dims is 600 evals per step.
        # Let's try providing the gradient analytically if possible, or just use L-BFGS-B with finite diff.
        
        # Actually, L-BFGS-B with finite differences is very slow for N=300.
        # Let's try to compute the gradient analytically.
        
        def objective_and_grad(x):
            points = x.reshape(-1, 2)
            
            # Pairwise
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            
            # Boundary
            dist_to_origin = np.linalg.norm(points, axis=1)
            boundary_dist = 1.0 - dist_to_origin
            
            # Weights for soft min
            # f(x) = -1/lam * log( sum exp(-lam * d_k) )
            # grad f = -1/lam * ( sum exp(-lam d_k) * (-lam * grad d_k) ) / sum exp(-lam d_k)
            #        = sum ( exp(-lam d_k) / sum_exp ) * grad d_k
            # Let w_k = exp(-lam d_k) / sum_exp
            
            pair_terms = np.exp(-lam * dist / 2.0)
            boundary_terms = np.exp(-lam * boundary_dist)
            
            total_sum = np.sum(pair_terms) + np.sum(boundary_terms)
            
            # Weights
            pair_weights = pair_terms / total_sum
            boundary_weights = boundary_terms / total_sum
            
            # Objective value
            obj = -1.0 / lam * np.log(total_sum)
            
            # Gradient
            grad = np.zeros_like(x)
            
            # Contribution from pairs
            # d(dist_ij) / d(x_i) = (x_i - x_j) / dist_ij
            # d(dist_ij) / d(x_j) = (x_j - x_i) / dist_ij
            # We need to sum over all j != i
            # grad_i = sum_{j != i} w_ij * (x_i - x_j) / dist_ij
            
            # To do this efficiently, we can use matrix operations or loops.
            # N=300, N^2=90k. A loop in Python is too slow.
            # Let's use numpy.
            
            # dist is NxN. pair_weights is NxN.
            # We want for each i: sum_j w_ij * (x_i - x_j) / dist_ij
            # Note that w_ij is symmetric.
            
            # Let's compute the vector sum for each i.
            # term_ij = w_ij / dist_ij * (x_i - x_j)
            # This is element-wise division, which is problematic where dist is inf.
            # We masked diagonal to inf, so w_ii is 0 (exp(-inf)=0).
            # But division by inf is 0. So it's safe.
            
            # However, doing this in a vectorized way for all i and j is tricky because of the (x_i - x_j) term.
            
            # Alternative: 
            # grad_i = sum_j (w_ij / dist_ij) * x_i - sum_j (w_ij / dist_ij) * x_j
            # Let M_ij = w_ij / dist_ij
            # grad_i = (sum_j M_ij) * x_i - sum_j (M_ij * x_j)
            
            M = pair_weights / dist
            # Handle inf in dist -> M is 0.
            M[np.isinf(dist)] = 0.0
            
            # Sum of M along axis 1 (for each i, sum over j)
            row_sums = np.sum(M, axis=1)
            
            # Sum of M_ij * x_j along axis 0 (for each i, sum over j of M_ij * x_j)
            # M is NxN, x is N x 2.
            # result[i] = sum_j M[i,j] * x[j]
            weighted_sum = M @ points
            
            grad_pairs = row_sums[:, np.newaxis] * points - weighted_sum
            
            # Contribution from boundary
            # d(boundary_dist_i) / d(x_i) = - x_i / |x_i|
            # grad_i += boundary_weight_i * (-x_i / |x_i|)
            
            # dist_to_origin is N.
            # Avoid division by zero
            safe_dist = np.where(dist_to_origin > 1e-12, dist_to_origin, 1e-12)
            grad_boundary = -boundary_weights[:, np.newaxis] * points / safe_dist[:, np.newaxis]
            
            grad += grad_pairs + grad_boundary
            
            # Flatten grad
            return obj, grad.flatten()

        # Run minimize
        try:
            res = minimize(
                objective_and_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True, # We return (f, grad)
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < best_score: # Note: res.fun is the objective, not the score.
                # We need to compute the actual score
                new_points = res.x.reshape(-1, 2)
                new_score = compute_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    current_best = res.x.copy()
                    x0 = res.x.copy() # Use this as start for next lambda
        except Exception as e:
            pass # Keep current best

    # Final check on time
    if time.time() < deadline - 5:
        # Maybe do one more polish with high lambda?
        pass

    return current_best.reshape(-1, 2)

# EVOLVE-BLOCK-END
