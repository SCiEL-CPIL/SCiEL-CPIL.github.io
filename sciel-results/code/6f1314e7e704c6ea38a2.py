"""
Strategy: Deterministic Gradient-Based Optimization using Log-Sum-Exp Soft-Min.

The current program uses Simulated Annealing (SA), which is a stochastic local search.
This program replaces that with a deterministic, differentiable approach:
1. Approximate the "min" operation in the score function using a Log-Sum-Exp (soft-min)
   transformation. This creates a smooth, differentiable objective function.
2. Use L-BFGS-B (a quasi-Newton method) to maximize this smooth objective.
3. Perform multiple restarts from different high-quality initializations (hexagonal lattice
   variants) to explore different basins of attraction.

Why this is different:
- SA relies on random moves and temperature schedules. It can get stuck in local minima
  and requires many evaluations to converge.
- Gradient-based methods use the shape of the landscape to take large, directed steps.
  For smooth approximations of packing problems, these often converge to higher quality
  solutions faster and more reliably than random perturbations.
"""

import numpy as np
import time
import scipy.optimize as opt

N = 300
TIME_LIMIT = 115.0  # Safety margin for 150s kill switch

def _hex_init(rotation=0.0, jitter=0.0):
    """
    Generate a hexagonal lattice initialization.
    Returns an array of shape (N, 2).
    """
    # Estimate spacing. Best r ~ 0.053. Spacing ~ 0.106.
    # We want to fit 300 points. Area of unit circle ~ 3.14.
    # Area per point ~ 3.14 / 300 ~ 0.0105.
    # Hexagon area for spacing d is ~ d^2 * sqrt(3)/2.
    # d^2 ~ 0.0105 * 2 / 1.732 ~ 0.012. d ~ 0.11.
    # Let's use a slightly tighter spacing to ensure we have enough points,
    # then select the best N.
    spacing = 0.105
    
    # Generate points in a grid large enough to cover the unit circle
    k = int(1.1 / spacing) + 2
    points = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            points.append([x, y])
    
    points = np.array(points)
    
    # Rotate points
    if rotation != 0.0:
        c, s = np.cos(rotation), np.sin(rotation)
        R = np.array([[c, -s], [s, c]])
        points = points @ R.T
        
    # Add jitter if requested
    if jitter > 0.0:
        points += np.random.normal(0, jitter, points.shape)
        
    # Filter points strictly inside unit circle (with small margin)
    radii = np.linalg.norm(points, axis=1)
    mask = radii < 0.99
    points = points[mask]
    radii = radii[mask]
    
    # Sort by radius (closest to center first) and take N
    idx = np.argsort(radii)
    points = points[idx]
    
    if len(points) < N:
        # Fallback: if we don't have enough points (shouldn't happen), 
        # pad with random points
        theta = np.random.uniform(0, 2 * np.pi, N - len(points))
        rho = np.sqrt(np.random.uniform(0, 0.5, N - len(points)))
        pad = np.column_stack((rho * np.cos(theta), rho * np.sin(theta)))
        points = np.vstack([points, pad])
    else:
        points = points[:N]
        
    return points

def _objective_and_grad(flat_x, beta=50.0):
    """
    Objective: Minimize negative soft-min of distances.
    Soft-min of a set of values v is approximated by -log(sum(exp(-beta * v))) / beta.
    We want to maximize min(distances), so we minimize -soft_min(distances).
    
    Distances include:
    1. Pairwise distances between circles / 2 (radius constraint)
    2. Distance to boundary (1 - |c_i|)
    
    Actually, the score is min( min_{i<j} d_ij/2, min_i (1-|c_i|) ).
    Let R_pair = min_{i<j} d_ij/2.
    Let R_bound = min_i (1-|c_i|).
    Score = min(R_pair, R_bound).
    
    We can treat all "constraints" as a single set of values to maximize the minimum of.
    Values V:
    - For each pair (i,j): d_ij / 2
    - For each point i: 1 - |c_i|
    
    We want to maximize min(V).
    Equivalent to minimizing -min(V).
    Using soft-min: f(x) = - (1/beta) * log( sum( exp(-beta * v_k) ) )
    
    Gradient:
    d f / d x = - (1/beta) * d/dx [ log( sum( exp(-beta * v_k) ) ) ]
              = - (1/beta) * [ 1 / sum(exp(-beta v_k)) * sum( exp(-beta v_k) * d(-beta v_k)/dx ) ]
              = - (1/beta) * [ 1 / S * sum( exp(-beta v_k) * (-beta) * dv_k/dx ) ]
              = sum( (exp(-beta v_k) / S) * dv_k/dx )
    
    Where S = sum(exp(-beta v_k)).
    The term w_k = exp(-beta v_k) / S acts as a weight, emphasizing the smallest v_k.
    
    dv_k/dx:
    - For pair (i,j) value v_k = 0.5 * ||c_i - c_j||:
      dv_k/dx_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
      dv_k/dx_j = 0.5 * (c_j - c_i) / ||c_i - c_j||
    - For boundary i value v_k = 1 - ||c_i||:
      dv_k/dx_i = - c_i / ||c_i||
    """
    points = flat_x.reshape(N, 2)
    
    # Calculate pairwise distances
    # To save memory/time, we can compute gradients in a loop or vectorized chunks,
    # but for N=300, full matrix is 90k pairs. 
    # exp(-beta * v) might underflow if beta*v is large.
    # We should normalize v by subtracting the max? No, soft-min is sensitive to absolute values.
    # However, if v is small (bad packing), exp(-beta*v) is large.
    # If v is large (good packing), exp(-beta*v) is small.
    # We only care about the smallest v's.
    
    # Let's compute all v_k and their gradients contributions.
    # For efficiency, we'll iterate over pairs? No, vectorize.
    
    # Pairwise
    # diffs shape (N, N, 2)
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(dists, np.inf)
    
    # Values for pairs: dists / 2
    v_pairs = dists / 2.0
    
    # Boundary
    radii = np.linalg.norm(points, axis=1)
    v_bounds = 1.0 - radii
    
    # Combine all values into a single list conceptually
    # But for gradient, we need to accumulate.
    
    # To avoid overflow in exp(-beta * v), we can use the log-sum-exp trick or just rely on 
    # the fact that we are minimizing. 
    # Actually, if v is close to 0, exp(-beta*v) is ~1. If v is 0.05, exp(-50*0.05)=exp(-2.5)~0.08.
    # The weights will be dominated by the smallest distances.
    
    # We compute weights w_k = exp(-beta * v_k)
    # To prevent underflow for large v_k (which contribute 0 anyway), we can clip v_k?
    # Or just let them underflow to 0.
    
    # However, calculating exp for 45k pairs + 300 bounds is fast.
    
    # We need to be careful with the sum S.
    # Let's compute log_weights = -beta * v
    # Then log_S = logsumexp(log_weights)
    # w_k = exp(log_weights_k - log_S)
    
    log_w_pairs = -beta * v_pairs
    log_w_bounds = -beta * v_bounds
    
    # Mask diagonal for pairs
    log_w_pairs_flat = log_w_pairs.flatten()
    # Set diagonal to -inf so they don't contribute
    # The diagonal in the flattened (N*N) array is at indices 0, N+1, 2N+2...
    # Easier: create a mask
    mask_diag = np.eye(N, dtype=bool)
    log_w_pairs_valid = np.where(mask_diag, -np.inf, log_w_pairs).flatten()
    
    all_log_w = np.concatenate([log_w_pairs_valid, log_w_bounds])
    
    # Compute log_S
    # scipy.special.logsumexp is available
    from scipy.special import logsumexp
    log_S = logsumexp(all_log_w)
    
    # Weights
    w_pairs = np.exp(log_w_pairs - log_S)
    w_pairs[mask_diag] = 0.0
    w_bounds = np.exp(log_w_bounds - log_S)
    
    # Objective value: -soft_min = -( -1/beta * log_S ) = log_S / beta
    # Wait. 
    # soft_min(v) = -1/beta * log(sum(exp(-beta v))) = -1/beta * log_S
    # We want to minimize -soft_min = 1/beta * log_S.
    obj = log_S / beta
    
    # Gradient
    # grad = sum( w_k * dv_k/dx )
    
    grad = np.zeros_like(points)
    
    # Pair contributions
    # dv_k/dx_i = 0.5 * (c_i - c_j) / dist_ij
    # dv_k/dx_j = 0.5 * (c_j - c_i) / dist_ij
    #
    # grad_i = sum_j w_ij * 0.5 * (c_i - c_j) / dist_ij
    #
    # We can vectorize this.
    # w_pairs shape (N, N)
    # dists shape (N, N)
    # 
    # factor_ij = w_ij * 0.5 / dist_ij
    # We need to handle dist=0 (diagonal) -> factor=0.
    
    with np.errstate(divide='ignore', invalid='ignore'):
        factor_pairs = np.where(dists > 1e-8, w_pairs * 0.5 / dists, 0.0)
        
    # grad_i = sum_j factor_ij * (c_i - c_j)
    # = sum_j factor_ij * c_i - sum_j factor_ij * c_j
    # = c_i * sum_j factor_ij - sum_j (factor_ij * c_j)
    
    sum_factor = np.sum(factor_pairs, axis=1) # Shape (N,)
    
    # sum_j (factor_ij * c_j)
    # factor_pairs shape (N, N), points shape (N, 2)
    # Result shape (N, 2)
    # We can use matrix multiplication: factor_pairs @ points
    # But factor_pairs is (N, N). points is (N, 2).
    # (factor_pairs @ points) gives for each i: sum_j factor_ij * c_j. Correct.
    
    sum_factor_c = factor_pairs @ points
    
    grad_pairs = points * sum_factor[:, np.newaxis] - sum_factor_c
    
    # Boundary contributions
    # dv_k/dx_i = - c_i / ||c_i||
    # grad_i += w_bounds_i * (- c_i / ||c_i||)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        grad_bound_factors = np.where(radii > 1e-8, w_bounds / radii, 0.0)
        
    grad_bounds = -points * grad_bound_factors[:, np.newaxis]
    
    grad = grad_pairs + grad_bounds
    
    # Flatten gradient
    grad_flat = grad.flatten()
    
    return obj, grad_flat

def construct():
    start_time = time.time()
    best_points = None
    best_score = -np.inf
    
    # Define a few different initializations to restart from
    # 1. Standard Hex
    # 2. Rotated Hex
    # 3. Jittered Hex
    inits = [
        _hex_init(rotation=0.0, jitter=0.0),
        _hex_init(rotation=np.pi/6, jitter=0.0),
        _hex_init(rotation=0.0, jitter=0.01),
        _hex_init(rotation=np.pi/3, jitter=0.01),
    ]
    
    # Time per restart
    # We have ~115s.
    # L-BFGS-B is fast. Let's try to run as many as possible or set a time limit per run.
    # Let's say 20s per run. That allows ~5 runs.
    time_per_run = 20.0
    
    for i, init_points in enumerate(inits):
        if time.time() - start_time > TIME_LIMIT - 5.0:
            break
            
        # Flatten for optimizer
        x0 = init_points.flatten()
        
        # Run L-BFGS-B
        # We maximize the soft-min, so we minimize the negative soft-min.
        # Our objective returns (obj, grad) where obj is the value to minimize (positive log_S/beta).
        # Wait, higher soft-min is better. 
        # soft_min = -1/beta log_S.
        # Maximize soft_min -> Minimize -soft_min -> Minimize 1/beta log_S.
        # So yes, we minimize obj = log_S / beta.
        
        # Increase beta over time? 
        # Start with lower beta (smoother) and increase to sharper (closer to true min).
        # Or just use a fixed high beta? 
        # High beta makes the function very peaked, which can be hard for L-BFGS-B.
        # Low beta is smoother but less accurate.
        # Let's try a moderate beta, e.g., 50.0.
        
        beta = 50.0
        
        # Wrap objective to capture beta
        def obj_func(x):
            return _objective_and_grad(x, beta)[0]
            
        def grad_func(x):
            return _objective_and_grad(x, beta)[1]
        
        # Actually, calling _objective_and_grad twice is inefficient.
        # Let's create a closure that caches or just accept the cost.
        # For N=300, the cost is dominated by the matrix ops. 
        # L-BFGS-B calls func and grad separately. 
        # We can optimize by combining, but scipy requires separate or a tuple return.
        # scipy.optimize.minimize with jac=True expects func to return (f, g) if using 
        # a specific wrapper? No, standard is f and jac(x).
        # We can just use the same function. The overhead of Python function call is small 
        # compared to NumPy ops.
        
        def f_and_g(x):
            return _objective_and_grad(x, beta)
        
        # Use L-BFGS-B
        # maxiter: 1000 should be enough for convergence or limit.
        # ftol/gtol: standard.
        
        try:
            res = opt.minimize(
                f_and_g,
                x0,
                method='L-BFGS-B',
                jac=True,  # Use the gradient returned by the function
                options={
                    'maxiter': 200, # Limit iterations to ensure time bound
                    'ftol': 1e-8,
                    'gtol': 1e-5,
                    'maxls': 20
                }
            )
        except Exception as e:
            # If optimization fails, skip
            continue
            
        x_opt = res.x.reshape(N, 2)
        
        # Calculate true score (hard min)
        # This is the actual metric
        diffs = x_opt[:, np.newaxis, :] - x_opt[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        radii = np.linalg.norm(x_opt, axis=1)
        min_bound_dist = np.min(1.0 - radii)
        
        score = min(min_pair_dist / 2.0, min_bound_dist)
        
        if score > best_score:
            best_score = score
            best_points = x_opt.copy()
            
        # Check time
        if time.time() - start_time > TIME_LIMIT - 10.0:
            break
            
    # Final validity check
    if best_points is None:
        # Fallback
        best_points = inits[0]
        
    radii = np.linalg.norm(best_points, axis=1)
    valid = np.isfinite(best_points).all(axis=1) & (radii < 1.0)
    if not np.all(valid):
        # If invalid, return a fallback random valid configuration
        theta = np.random.uniform(0, 2 * np.pi, N)
        rho = np.sqrt(np.random.uniform(0, 0.9, N))
        best_points = np.column_stack((rho * np.cos(theta), rho * np.sin(theta)))
        
    return best_points

# Helper for testing
if __name__ == "__main__":
    pts = construct()
    # Calculate score
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    radii = np.linalg.norm(pts, axis=1)
    min_bound_dist = np.min(1.0 - radii)
    score = min(min_pair_dist / 2.0, min_bound_dist)
    print(f"Score: {score}")
