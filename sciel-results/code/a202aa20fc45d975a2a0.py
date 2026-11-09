import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 300

# EVOLVE-BLOCK-START

def _compute_score(points):
    """
    Computes the score (common radius r) for a given set of points.
    r = min( min_{i<j} ||c_i - c_j|| / 2, min_i (1 - ||c_i||) )
    """
    if points is None or len(points) != N:
        return 0.0
    
    # Check for non-finite values
    if not np.all(np.isfinite(points)):
        return 0.0

    # Boundary constraint: points must be inside unit circle
    norms = np.linalg.norm(points, axis=1)
    boundary_r = 1.0 - np.max(norms)
    
    if boundary_r < 0:
        return 0.0

    # Pairwise distances
    d = pdist(points)
    if len(d) == 0:
        pair_r = float('inf')
    else:
        pair_r = np.min(d) / 2.0

    r = min(pair_r, boundary_r)
    return max(0.0, r)

def _softmin_obj_and_grad(x, beta):
    """
    Computes the objective function and gradient for the soft-minimum of distances.
    We want to maximize the minimum distance, so we minimize the negative soft-minimum.
    
    Constraints:
    1. Pairwise distances d_ij >= 2r (implicitly, we maximize the min distance)
    2. Boundary distances 1 - ||x_i|| >= r
    
    We define a set of 'distances' D which includes:
    - All pairwise distances d_ij
    - All boundary distances b_i = 1 - ||x_i||
    
    The score is min(D) / 2? No, the score is min( min_{i<j} d_ij / 2, min_i (1 - ||x_i||) ).
    Let's define the "radius constraints" directly:
    R_ij = d_ij / 2
    R_i = 1 - ||x_i||
    
    We want to maximize min(R_ij, R_i).
    
    Soft-minimum of R:
    S(R) = -1/beta * log(sum(exp(-beta * R)))
    
    We minimize -S(R).
    """
    points = x.reshape(-1, 2)
    n = len(points)
    
    # 1. Boundary distances
    norms = np.linalg.norm(points, axis=1)
    # To avoid log(0) or issues, we can clip norms slightly or handle gradients carefully.
    # R_i = 1 - norms
    # If norm > 1, R_i is negative. This is fine for the exponential, it will dominate.
    
    # 2. Pairwise distances
    # d_ij = ||x_i - x_j||
    # R_ij = d_ij / 2
    
    # We need to compute sum(exp(-beta * R)) for all R.
    # There are N boundary terms and N*(N-1)/2 pairwise terms.
    # Total ~ 45,000 terms for N=300.
    
    # Efficient computation:
    # Boundary part
    # exp(-beta * (1 - norms))
    
    # Pairwise part
    # diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    # dists = np.linalg.norm(diff, axis=2)
    # R_pair = dists / 2.0
    # We only need upper triangle.
    
    # Let's compute the log-sum-exp for numerical stability.
    # log(sum(exp(-beta * R))) = -beta * max(R) + log(sum(exp(-beta * (R - max(R)))))
    # Then S(R) = -1/beta * log_sum_exp = max(R) - 1/beta * log(sum(exp(-beta * (R - max(R)))))
    
    # Wait, maximizing min(R) is equivalent to minimizing -min(R).
    # As beta -> inf, -1/beta * log(sum(exp(-beta R))) -> min(R).
    # So we minimize f(x) = 1/beta * log(sum(exp(-beta R))).
    
    # Let's collect all R values.
    # R_boundary = 1 - norms
    # R_pair = d_ij / 2
    
    # For gradient, we need:
    # grad f = -1/sum(exp) * sum ( -beta * exp(-beta R_k) * grad R_k )
    #         = 1/sum(exp) * sum ( beta * exp(-beta R_k) * grad R_k )
    # Let w_k = exp(-beta R_k) / sum(exp(-beta R_j))
    # grad f = beta * sum( w_k * grad R_k )
    
    # This is a weighted average of the gradients of the individual radius constraints,
    # weighted by how "tight" they are (smaller R -> larger weight).
    
    # 1. Compute R values
    R_boundary = 1.0 - norms
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    R_pair = dists / 2.0
    
    # Mask for upper triangle (i < j) to avoid double counting and self-distance (R=0)
    # Self distance is 0, R=0. exp(0)=1. This would be the largest term if all R>0.
    # But we want min R. If self R=0 is included, min is 0.
    # We MUST exclude i=j.
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # Flatten the valid R_pair
    R_pair_valid = R_pair[mask]
    R_boundary_valid = R_boundary
    
    # Combine all R
    # Total terms: N + N*(N-1)/2
    # R_all = concat(R_boundary, R_pair_valid)
    
    # For numerical stability, find max R
    R_max = np.max(np.concatenate([R_boundary, R_pair_valid]))
    
    # Compute exp(-beta * (R - R_max))
    exp_boundary = np.exp(-beta * (R_boundary - R_max))
    exp_pair = np.exp(-beta * (R_pair_valid - R_max))
    
    sum_exp = np.sum(exp_boundary) + np.sum(exp_pair)
    
    # Objective: f = 1/beta * log(sum_exp) + R_max
    # Wait, S(R) = -1/beta * log(sum(exp(-beta R)))
    # We want to maximize S(R), so minimize -S(R).
    # -S(R) = 1/beta * log(sum(exp(-beta R)))
    # Using stability:
    # log(sum(exp(-beta R))) = -beta * R_max + log(sum(exp(-beta (R - R_max))))
    # -S(R) = 1/beta * (-beta * R_max + log(sum_exp_stable))
    #        = -R_max + 1/beta * log(sum_exp_stable)
    
    obj_val = -R_max + (1.0 / beta) * np.log(sum_exp)
    
    # Gradient
    # grad f = beta * sum( w_k * grad R_k )
    # where w_k = exp(-beta (R_k - R_max)) / sum_exp
    
    w_boundary = exp_boundary / sum_exp
    w_pair = exp_pair / sum_exp
    
    # grad R_boundary[i] = - (x_i / ||x_i||)
    # grad R_pair[i,j] = (x_i - x_j) / (2 * ||x_i - x_j||) for term i,j
    # Note: R_pair is symmetric. The term (i,j) contributes to grad x_i and grad x_j.
    # R_ij = 0.5 * ||x_i - x_j||
    # grad_xi R_ij = 0.5 * (x_i - x_j) / ||x_i - x_j||
    # grad_xj R_ij = 0.5 * (x_j - x_i) / ||x_i - x_j||
    
    # We need to accumulate gradients.
    grad_x = np.zeros_like(points)
    
    # Boundary contribution
    # grad_x[i] += w_boundary[i] * (-x_i / ||x_i||)
    norms_safe = np.where(norms < 1e-8, 1e-8, norms)
    grad_boundary = -w_boundary[:, np.newaxis] * (points / norms_safe[:, np.newaxis])
    grad_x += grad_boundary
    
    # Pairwise contribution
    # We have weights w_pair corresponding to the flattened upper triangle.
    # We need to map these back to (i, j) indices.
    # This is expensive if done naively.
    
    # Alternative: Compute gradient of the pairwise part directly using matrix operations?
    # Let G be the gradient matrix.
    # G[i,j] = w_pair[i,j] * (x_i - x_j) / (2 * d_ij)
    # grad_x[i] = sum_j G[i,j] - sum_j G[j,i] ?
    # Actually, for each pair (i,j), the contribution to x_i is +w * (x_i - x_j)/(2 d)
    # and to x_j is -w * (x_i - x_j)/(2 d) = w * (x_j - x_i)/(2 d).
    
    # Let's construct the full weight matrix W_pair (N x N) from the flattened upper triangle.
    W_pair_full = np.zeros((n, n))
    W_pair_full[mask] = w_pair
    
    # Compute diff / dists_safe
    dists_safe = np.where(dists < 1e-8, 1e-8, dists)
    # Unit vector from j to i: (x_i - x_j) / d_ij
    # Term for R_ij is 0.5 * d_ij. Grad is 0.5 * (x_i - x_j) / d_ij.
    
    # Gradient contribution from pair (i,j) to x_i:
    # w_pair[i,j] * 0.5 * (x_i - x_j) / d_ij
    # Note: W_pair_full[i,j] is the weight for the term R_ij.
    # The term R_ij appears in the sum.
    # grad_x[i] = sum_j W_pair_full[i,j] * 0.5 * (x_i - x_j) / d_ij
    #             - sum_j W_pair_full[j,i] * 0.5 * (x_j - x_i) / d_ij  (from terms where i is the second index)
    # Wait, the weight W_pair_full[i,j] corresponds to the constraint R_ij.
    # The constraint R_ij depends on x_i and x_j.
    # d R_ij / d x_i = 0.5 * (x_i - x_j) / d_ij
    # d R_ij / d x_j = 0.5 * (x_j - x_i) / d_ij
    
    # So, grad_x[i] = sum_j [ W_pair_full[i,j] * d R_ij / d x_i ]
    #                = sum_j [ W_pair_full[i,j] * 0.5 * (x_i - x_j) / d_ij ]
    
    # We can compute this efficiently.
    # Let A = W_pair_full * 0.5 / dists_safe
    # A is N x N. A[i,j] = w_ij * 0.5 / d_ij
    # grad_x[i] = sum_j A[i,j] * (x_i - x_j)
    #            = sum_j A[i,j] * x_i - sum_j A[i,j] * x_j
    #            = x_i * sum_j A[i,j] - sum_j A[i,j] * x_j
    
    A = W_pair_full * 0.5 / dists_safe
    
    # Row sums of A
    A_row_sums = np.sum(A, axis=1)
    
    # Matrix-vector product A @ points
    A_x = A @ points
    
    grad_pair = A_row_sums[:, np.newaxis] * points - A_x
    grad_x += grad_pair
    
    total_grad = (beta * grad_x).flatten()
    
    return obj_val, total_grad

def _initial_hex_lattice(r_est):
    """
    Generate a hexagonal lattice of points inside the unit disk,
    scaled such that the minimum distance is approximately 2 * r_est.
    """
    spacing = 2.0 * r_est
    if spacing <= 0:
        return None
        
    k = int(1.0 / spacing) + 5
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0:
                pts.append((x, y))
    
    if len(pts) < N:
        return None
    
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    return pts[idx]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (10s buffer)
    
    # Initial guess for radius
    # Best known is ~0.05315. Area bound is 1/sqrt(300) ~ 0.0577.
    # Let's start with a slightly smaller radius to ensure feasibility, then optimize.
    r_start = 0.04
    
    init_pts = _initial_hex_lattice(r_start)
    if init_pts is None:
        # Fallback: random points in a smaller disk
        init_pts = np.random.rand(N, 2) * 0.8 - 0.4
        init_pts = np.clip(init_pts, -0.9, 0.9)
        
    best_points = init_pts.copy()
    best_score = _compute_score(best_points)
    
    # Soft-minimum optimization
    # Start with a low beta and increase it
    betas = [20, 50, 100, 200, 400, 800]
    
    x_current = init_pts.flatten().copy()
    
    for beta in betas:
        if time.time() > deadline - 10.0:
            break
            
        # Define objective
        def obj_grad(x):
            e, g = _softmin_obj_and_grad(x, beta)
            return e, g
            
        try:
            # Use L-BFGS-B
            # maxiter needs to be high enough to converge for high beta
            max_iter = 100 if beta < 100 else 50
            result = minimize(obj_grad, x_current, method='L-BFGS-B', jac=True, 
                              options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
            
            x_current = result.x
            opt_points = x_current.reshape(-1, 2)
            opt_score = _compute_score(opt_points)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
                
        except Exception as e:
            # If optimization fails, keep previous best
            pass
            
    return best_points

# EVOLVE-BLOCK-END
