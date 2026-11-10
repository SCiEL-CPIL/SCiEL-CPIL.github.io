# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize

def _score(points):
    """Compute R = (d_min / d_max)^2."""
    n = points.shape[0]
    if n < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max < 1e-15:
        return 0.0
    return (d_min / d_max) ** 2

def _soft_objective(x, alpha, lambda_rep):
    """
    Smooth surrogate objective.
    We want to maximize the minimum distance and minimize the maximum distance.
    
    Term 1: Soft minimum of distances. 
    As alpha increases, this approaches min(d_ij). We want to maximize this, so we minimize -soft_min.
    However, -log(sum(exp(-alpha*d))) is not quite right for maximizing min.
    
    Let's use:
    f(x) = - (1/alpha) * log( sum_{i<j} exp(-alpha * d_ij) )  -- This approximates -min(d_ij) as alpha->inf.
    Wait, min(d) is approximated by (1/alpha) log(sum exp(-alpha d)).
    So we want to maximize this. Thus we minimize: - (1/alpha) log(sum exp(-alpha d)).
    
    Term 2: We want to minimize the spread, i.e., max distance.
    A common smooth approximation for max is log-sum-exp: (1/beta) log(sum exp(beta d)).
    So we minimize: (1/beta) log(sum exp(beta d)).
    
    However, log-sum-exp is numerically unstable for large beta or d.
    Alternative: Just minimize the sum of squared distances? No, that minimizes the whole cluster size, which helps d_max but might collapse d_min to 0.
    
    Actually, the problem is scale-invariant. We can fix the scale.
    Let's fix the sum of squared distances to a constant? Or fix d_max?
    
    Better approach:
    Objective = - soft_min(d) + lambda * soft_max(d)
    
    Let's use:
    1. soft_min(d) = (1/alpha) * log( sum exp(-alpha * d) )
    2. soft_max(d) = (1/beta) * log( sum exp(beta * d) )
    
    We want to maximize soft_min and minimize soft_max.
    So minimize: -soft_min(d) + soft_max(d)
    
    But soft_max is sensitive to the largest distance.
    Also, we need to handle the fact that the configuration can be scaled.
    If we scale points by k, d_ij -> k * d_ij.
    soft_min -> k * soft_min
    soft_max -> k * soft_max
    Objective -> k * (-soft_min + soft_max).
    If (-soft_min + soft_max) is negative, we want k -> inf.
    If it is positive, we want k -> 0.
    This means the objective is unbounded unless the optimal configuration has soft_min = soft_max, which implies all distances are equal (impossible for 16 points).
    
    So we must fix the scale.
    Let's normalize the points such that the sum of squared distances is 1, or the max norm is 1.
    But normalization inside the objective makes it non-smooth.
    
    Alternative: Add a penalty for large scale.
    Or, simply minimize: (soft_max - soft_min) / soft_max ? No, that's the original ratio.
    
    Let's try:
    Minimize: (1/alpha) log(sum exp(-alpha d)) is NOT correct.
    min(d) approx = (1/alpha) log(sum exp(-alpha d)) is WRONG.
    min(d) = - (1/alpha) log(sum exp(-alpha d)) is WRONG.
    
    Recall: min(a,b) = - (1/alpha) log( exp(-alpha a) + exp(-alpha b) ).
    So soft_min = - (1/alpha) log( sum exp(-alpha d_ij) ).
    
    max(a,b) = (1/alpha) log( exp(alpha a) + exp(alpha b) ).
    So soft_max = (1/alpha) log( sum exp(alpha d_ij) ).
    
    We want to maximize soft_min and minimize soft_max.
    So we minimize: -soft_min + soft_max
    = (1/alpha) log( sum exp(-alpha d_ij) ) + (1/alpha) log( sum exp(alpha d_ij) )
    
    Let's check scale invariance.
    If d -> k d:
    Term 1: (1/alpha) log( sum exp(-alpha k d) )
    Term 2: (1/alpha) log( sum exp(alpha k d) )
    
    For large alpha, Term 1 ~ min(-k d) = -k min(d).
    Term 2 ~ max(k d) = k max(d).
    Objective ~ k (max(d) - min(d)).
    This is still scale dependent. If max(d) > min(d), which is always true, we want k -> 0 to minimize the objective.
    This will collapse all points to the origin.
    
    We need to prevent collapse.
    We can add a penalty term that penalizes small distances (to keep d_min > 0) or penalize large distances (to keep d_max < inf).
    But we want to maximize the RATIO.
    
    Let's fix the scale by constraining the sum of squared distances to be 1.
    Constraint: sum_{i<j} d_ij^2 = 1.
    This is a non-convex constraint, but we can use Lagrange multipliers or just add a penalty.
    
    Objective:
    f(x) = -soft_min(d) + soft_max(d) + mu * (sum d_ij^2 - 1)^2
    
    With large mu, this enforces the scale.
    With large alpha, soft_min and soft_max approximate min and max.
    
    Let's implement this.
    """
    pts = x.reshape(16, 2)
    dists = spd.pdist(pts)
    
    # Numerical stability for log-sum-exp
    # Term 1: -soft_min = (1/alpha) log(sum exp(-alpha d))
    # Note: min(d) = - (1/alpha) log(sum exp(-alpha d))
    # So -soft_min = (1/alpha) log(sum exp(-alpha d))
    
    # We want to maximize min(d), so we minimize -min(d).
    # -min(d) approx = (1/alpha) log(sum exp(-alpha d))
    
    # Term 2: soft_max = (1/alpha) log(sum exp(alpha d))
    # We want to minimize max(d), so we minimize soft_max.
    
    # To avoid overflow in exp(alpha d), we can shift by max(d).
    d_max_val = np.max(dists)
    d_min_val = np.min(dists)
    
    # If points are identical, dists are 0.
    if d_max_val < 1e-12:
        return 1e6
    
    # Calculate Term 1: (1/alpha) log(sum exp(-alpha d))
    # exp(-alpha d) is stable for d >= 0.
    # But if d is large, exp(-alpha d) -> 0.
    # log(sum exp(-alpha d)) ~ -alpha * min(d).
    # So Term 1 ~ -min(d).
    # We want to MINIMIZE Term 1? No.
    # We want to MAXIMIZE min(d).
    # So we want to MINIMIZE -min(d).
    # -min(d) is approximated by (1/alpha) log(sum exp(-alpha d)).
    # Let's call this obj_min_part.
    
    # Calculate Term 2: (1/alpha) log(sum exp(alpha d))
    # To be stable, subtract max(d).
    # sum exp(alpha d) = exp(alpha d_max) * sum exp(alpha (d - d_max))
    # log(sum exp(alpha d)) = alpha d_max + log(sum exp(alpha (d - d_max)))
    # Term 2 = d_max + (1/alpha) log(sum exp(alpha (d - d_max)))
    # As alpha -> inf, Term 2 -> d_max.
    
    # We want to MINIMIZE Term 2.
    
    # Scale Penalty
    sum_d_sq = np.sum(dists ** 2)
    scale_penalty = 100.0 * (sum_d_sq - 1.0) ** 2
    
    # Total Objective
    # We want to minimize: obj_min_part + obj_max_part + scale_penalty
    # obj_min_part approximates -min(d). Minimizing it maximizes min(d).
    # obj_max_part approximates max(d). Minimizing it minimizes max(d).
    
    try:
        # Term 1
        exp_term1 = np.exp(-alpha * dists)
        sum_exp1 = np.sum(exp_term1)
        if sum_exp1 < 1e-300:
            obj_min_part = 1e6
        else:
            obj_min_part = (1.0 / alpha) * np.log(sum_exp1)
            
        # Term 2
        shifted_dists = dists - d_max_val
        exp_term2 = np.exp(alpha * shifted_dists)
        sum_exp2 = np.sum(exp_term2)
        if sum_exp2 < 1e-300:
            obj_max_part = 1e6
        else:
            obj_max_part = d_max_val + (1.0 / alpha) * np.log(sum_exp2)
            
    except Exception:
        return 1e6
        
    return obj_min_part + obj_max_part + scale_penalty

def _generate_hexagonal_16():
    """
    Generate 16 points from a hexagonal lattice, centered and normalized.
    """
    points = []
    # Hex lattice basis
    s = 1.0
    # Generate a sufficient grid
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * s + 0.5 * j * s
            y = j * s * np.sqrt(3) / 2
            points.append([x, y])
    
    points = np.array(points)
    # Center
    points -= points.mean(axis=0)
    
    # Sort by distance to origin
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    
    # Take 16 closest
    pts = points[idx[:16]]
    
    # Normalize sum of squared distances to 1
    d_sq = np.sum(spd.pdist(pts) ** 2)
    if d_sq > 1e-10:
        pts = pts / np.sqrt(d_sq)
        
    return pts

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    n = 16
    d = 2
    deadline = time.time() + 590
    
    best_points = None
    best_score = -1.0
    
    rng = np.random.default_rng(42)
    
    # Parameters for the soft objective
    alpha = 20.0  # Sharpness of soft min/max
    # We don't need lambda_rep if we use the scale penalty
    
    initializers = [
        _generate_hexagonal_16
    ]
    
    # Add random initializers for robustness
    for _ in range(3):
        pts = rng.normal(size=(16, 2))
        pts -= pts.mean(axis=0)
        d_sq = np.sum(spd.pdist(pts) ** 2)
        if d_sq > 1e-10:
            pts = pts / np.sqrt(d_sq)
        initializers.append(lambda p=pts: p)
        
    num_rounds = 8
    
    for r in range(num_rounds):
        if time.time() > deadline - 20:
            break
            
        init_func = initializers[r % len(initializers)]
        init_pts = init_func()
        
        # Flatten
        x0 = init_pts.flatten()
        
        # Define objective for L-BFGS-B
        # We use a slightly smaller alpha for optimization to keep it smooth,
        # then evaluate with the true score.
        opt_alpha = 10.0
        
        def objective(x):
            return _soft_objective(x, opt_alpha, 0.0)
            
        # Run L-BFGS-B
        # Bounds: not strictly necessary if we normalize, but helps stability.
        # The points can be anywhere, but the scale is fixed by the penalty.
        # We can leave bounds as None.
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.x is not None:
                opt_pts = res.x.reshape(16, 2)
                # Normalize again to be safe
                d_sq = np.sum(spd.pdist(opt_pts) ** 2)
                if d_sq > 1e-10:
                    opt_pts = opt_pts / np.sqrt(d_sq)
                    
                cur_score = _score(opt_pts)
                
                if cur_score > best_score:
                    best_score = cur_score
                    best_points = opt_pts.copy()
                    
        except Exception:
            continue
            
    if best_points is None:
        return _generate_hexagonal_16()
        
    return best_points

# EVOLVE-BLOCK-END
