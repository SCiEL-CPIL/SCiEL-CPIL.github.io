# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time


def _compute_ratio(points: np.ndarray) -> float:
    """Compute R = (d_min / d_max)^2 for the given points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2


def _objective_and_grad(x: np.ndarray, n: int, beta: float) -> tuple:
    """
    Objective: minimize -log(sum(exp(-beta * d_i))) which approximates -log(d_min)
    plus a term for d_max to keep points bounded.
    Actually, let's use a different formulation:
    We want to maximize d_min / d_max.
    Equivalent to minimizing d_max / d_min, or maximizing log(d_min) - log(d_max).
    
    A smooth surrogate for log(d_min) is: -1/beta * log(sum(exp(-beta * d_i)))
    A smooth surrogate for log(d_max) is: 1/beta * log(sum(exp(beta * d_i)))
    
    So we maximize: -1/beta * log(sum(exp(-beta * d_i))) - 1/beta * log(sum(exp(beta * d_i)))
    Or minimize the negative of that.
    
    Let's use: minimize 1/beta * log(sum(exp(beta * d_i))) + 1/beta * log(sum(exp(-beta * d_i)))
    This is the sum of the smooth max and smooth min (negated).
    Wait, log-sum-exp(beta * d) approximates log(exp(beta * d_max)) = beta * d_max.
    So 1/beta * log(sum(exp(beta * d))) approximates d_max.
    log-sum-exp(-beta * d) approximates log(exp(-beta * d_min)) = -beta * d_min.
    So 1/beta * log(sum(exp(-beta * d))) approximates -d_min.
    
    So minimizing (1/beta * LSE(beta*d) + 1/beta * LSE(-beta*d)) approximates minimizing (d_max - d_min).
    But we want to maximize d_min / d_max.
    
    Let's try: minimize log(d_max) - log(d_min).
    Surrogate: (1/beta) * log(sum(exp(beta * d))) - (-1/beta * log(sum(exp(-beta * d))))
    = (1/beta) * log(sum(exp(beta * d))) + (1/beta) * log(sum(exp(-beta * d)))
    This is the same as above, which minimizes d_max - d_min.
    
    Let's try a different approach. Minimize f(x) = log(d_max) - log(d_min) is non-smooth.
    
    Alternative: Minimize the ratio d_max / d_min directly? Also non-smooth.
    
    Let's use the soft-min and soft-max approach to maximize d_min - epsilon * d_max?
    No, the ratio is key.
    
    Let's stick with minimizing: (1/beta) * log(sum(exp(beta * d))) - (1/beta) * log(sum(exp(-beta * d)))
    This approximates d_max - d_min. Minimizing this keeps the spread small, but doesn't explicitly maximize the ratio.
    However, if we scale the points, the ratio is invariant. So we can fix d_max = 1 and maximize d_min.
    
    Let's fix the scale by normalizing the points such that d_max = 1.
    Then we just need to maximize d_min.
    Surrogate for d_min: (1/beta) * log(sum(exp(-beta * d))) is an approximation of -d_min?
    No, LSE(-beta * d) ≈ -beta * d_min + log(1 + ...).
    So (1/beta) * LSE(-beta * d) ≈ -d_min.
    Minimizing this is maximizing d_min.
    
    So the strategy:
    1. Start with some points.
    2. Normalize so d_max = 1.
    3. Minimize (1/beta) * log(sum(exp(-beta * d))) to maximize d_min.
    4. Repeat with higher beta for better accuracy.
    
    But normalization breaks the gradient. We need to incorporate normalization into the optimization.
    
    Better: Minimize f(x) = log(d_max) - log(d_min) using smooth surrogates.
    f_smooth(x) = (1/beta) * log(sum(exp(beta * d))) + (1/beta) * log(sum(exp(-beta * d)))
    This approximates d_max - d_min.
    
    Actually, let's look at the observed change in p43. It used:
    _solve_with_lower_bound: minimize d_max subject to d_min >= lb.
    This is a feasibility problem.
    
    Let's try a different structural approach:
    Use a multi-start optimization with a smooth objective that approximates the ratio.
    
    Objective: minimize g(x) = log(d_max) - log(d_min)
    Smooth approximation:
    log(d_max) ≈ (1/beta) * log(sum(exp(beta * d_i)))
    log(d_min) ≈ -(1/beta) * log(sum(exp(-beta * d_i)))
    So g_smooth = (1/beta) * log(sum(exp(beta * d))) + (1/beta) * log(sum(exp(-beta * d)))
    
    Let's compute the gradient of this.
    Let S_plus = sum(exp(beta * d_i)), S_minus = sum(exp(-beta * d_i))
    f = (1/beta) * (log(S_plus) + log(S_minus))
    df/dx_k = (1/beta) * ( (1/S_plus) * sum_i exp(beta * d_i) * beta * (d_i/||d_i||) * (dx_k/d_i) 
                          + (1/S_minus) * sum_i exp(-beta * d_i) * (-beta) * (d_i/||d_i||) * (dx_k/d_i) )
    
    This is getting complex. Let's use a simpler objective: minimize d_max - d_min, which is scale-invariant? No, it's not.
    
    Let's try: minimize (d_max / d_min).
    
    Actually, the most robust approach for this type of problem is often:
    1. Use a set of initial configurations (e.g., grid, circle, random).
    2. For each, run L-BFGS-B to minimize a smooth surrogate of log(d_max) - log(d_min).
    3. Keep the best result.
    
    Let's implement the smooth surrogate:
    f(x) = (1/beta) * log(sum(exp(beta * d))) + (1/beta) * log(sum(exp(-beta * d)))
    
    Gradient:
    Let v_i = exp(beta * d_i), w_i = exp(-beta * d_i)
    S_v = sum(v_i), S_w = sum(w_i)
    f = (1/beta) * (log(S_v) + log(S_w))
    
    df/dx_k = (1/beta) * [ (1/S_v) * sum_i v_i * beta * (d_i / ||d_i||) * (dx_k / d_i) 
                          + (1/S_w) * sum_i w_i * (-beta) * (d_i / ||d_i||) * (dx_k / d_i) ]
    
    Note: dx_k / d_i = (x_k - x_i) / d_i for the k-th coordinate of point k?
    Wait, x is a flattened array of shape (n*2,).
    d_i = ||x_{2i} - x_{2j}|| for pair (i,j).
    
    This is computationally expensive but doable.
    
    Let's simplify by using a fixed beta and multiple restarts.
    """
    n = x.shape[0] // 2
    pts = x.reshape(n, 2)
    d = pdist(pts)
    
    # Soft max and min
    # To avoid overflow, use the log-sum-exp trick
    beta_d = beta * d
    max_beta_d = np.max(beta_d)
    S_plus = np.exp(max_beta_d) * np.sum(np.exp(beta_d - max_beta_d))
    log_S_plus = max_beta_d + np.log(np.sum(np.exp(beta_d - max_beta_d)))
    
    min_beta_d = np.min(-beta_d)  # which is -max(beta_d)
    # For S_minus = sum(exp(-beta * d))
    # -beta * d has max at min d
    max_neg_beta_d = np.max(-beta_d)
    S_minus = np.exp(max_neg_beta_d) * np.sum(np.exp(-beta_d - max_neg_beta_d))
    log_S_minus = max_neg_beta_d + np.log(np.sum(np.exp(-beta_d - max_neg_beta_d)))
    
    f_val = (1.0 / beta) * (log_S_plus + log_S_minus)
    
    # Gradient
    # df/dx_k = (1/beta) * [ (1/S_v) * sum_i v_i * beta * grad_d_i[k] 
    #                       + (1/S_w) * sum_i w_i * (-beta) * grad_d_i[k] ]
    # where v_i = exp(beta * d_i), w_i = exp(-beta * d_i)
    # grad_d_i[k] is the gradient of d_i with respect to x_k
    
    # This is O(n^2 * 2) which is fine for n=16.
    
    # Compute weights
    v_weights = np.exp(beta_d - log_S_plus)  # v_i / S_v
    w_weights = np.exp(-beta_d - log_S_minus)  # w_i / S_w
    
    # v_weights and w_weights are arrays of shape (120,)
    
    # Now compute gradient
    grad = np.zeros_like(x)
    
    # Map pair indices to point indices
    # pdist returns distances in order: (0,1), (0,2), ..., (0,15), (1,2), ...
    # We need to map each of the 120 distances to the two points involved.
    
    # This is a bit tricky to do efficiently without loops.
    # For n=16, 120 pairs is small, so a loop is acceptable.
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            di = d[idx]
            if di < 1e-10:
                idx += 1
                continue
            
            # Gradient of d_i w.r.t. x_i and x_j
            # d_i = ||x_i - x_j||
            # grad_x_i = (x_i - x_j) / d_i
            # grad_x_j = (x_j - x_i) / d_i
            
            xi = pts[i]
            xj = pts[j]
            diff = xi - xj
            grad_di_i = diff / di
            grad_di_j = -diff / di
            
            # Contribution from S_plus term: (1/beta) * (1/S_v) * v_i * beta * grad_di
            # = (v_i / S_v) * grad_di
            # Contribution from S_minus term: (1/beta) * (1/S_w) * w_i * (-beta) * grad_di
            # = -(w_i / S_w) * grad_di
            
            coeff = v_weights[idx] - w_weights[idx]
            
            grad[2*i:2*i+2] += coeff * grad_di_i
            grad[2*j:2*j+2] += coeff * grad_di_j
            
            idx += 1
    
    return f_val, grad


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the points.

    """
    n = 16
    d = 2
    
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit, use 580 for safety
    
    best_score = 0.0
    best_points = None
    
    # Multiple initializations
    initializations = []
    
    # 1. Regular polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    initializations.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Grid
    grid_x = np.linspace(-1, 1, 4)
    grid_y = np.linspace(-1, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    initializations.append(np.column_stack([gx.flatten(), gy.flatten()]))
    
    # 3. Random
    np.random.seed(42)
    for _ in range(5):
        initializations.append(np.random.randn(n, d) * 0.5)
    
    # 4. Two concentric circles
    angles1 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angles2 = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
    circle1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
    circle2 = np.column_stack([1.5 * np.cos(angles2), 1.5 * np.sin(angles2)])
    initializations.append(np.vstack([circle1, circle2]))
    
    # 5. Hexagonal packing approximation
    # Place points in a hexagonal pattern
    hex_points = []
    # Center
    hex_points.append([0, 0])
    # First ring (6 points)
    for i in range(6):
        angle = i * np.pi / 3
        hex_points.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
    # Second ring (12 points, but we only need 9 more)
    for i in range(12):
        angle = i * np.pi / 6
        r = 1.0 if i % 2 == 0 else np.sqrt(3)
        hex_points.append([r * np.cos(angle), r * np.sin(angle)])
    hex_points = np.array(hex_points[:n])
    initializations.append(hex_points)
    
    # Optimization parameters
    betas = [10.0, 50.0, 100.0, 200.0]
    
    for init in initializations:
        if time.time() > deadline - 60:
            break
            
        x0 = init.flatten()
        
        for beta in betas:
            if time.time() > deadline - 30:
                break
            
            try:
                result = minimize(
                    _objective_and_grad,
                    x0,
                    args=(n, beta),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < 1e10:
                    x_opt = result.x
                    pts_opt = x_opt.reshape(n, 2)
                    score = _compute_ratio(pts_opt)
                    
                    if score > best_score:
                        best_score = score
                        best_points = pts_opt.copy()
                        
                    x0 = x_opt  # Use the result as the next initialization
                    
            except Exception:
                pass
    
    if best_points is None:
        # Fallback
        best_points = initializations[0]
    
    return best_points


# EVOLVE-BLOCK-END
