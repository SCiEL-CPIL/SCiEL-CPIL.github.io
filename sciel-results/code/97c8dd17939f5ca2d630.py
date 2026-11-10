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
    timeout = 600.0  # Safety margin from 630s limit
    
    best_score = 0.0
    best_points = None

    # Helper to compute score
    def compute_score(pts):
        if pts.shape != (16, 2):
            return 0.0
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    # Objective: Maximize soft-min of distances
    # S(x) = -1/k * log(sum(exp(-k * d_ij)))
    # This is a smooth approximation of min(d_ij)
    k_soft = 10.0
    
    def soft_min_obj(x):
        pts = x.reshape(-1, 2)
        dists = pdist(pts)
        # Avoid underflow/overflow issues in exp
        # If dists are very large, exp(-k*d) is 0. Sum is 0. log(0) is -inf.
        # We want to maximize S, so minimize -S.
        # -S = 1/k * log(sum(exp(-k*d)))
        # If all d are large, sum ~ 0, log ~ -inf, -S ~ -inf. We want to minimize -S, so large d is good?
        # Wait. 
        # If d -> inf, exp(-kd) -> 0. Sum -> 0. log(0) -> -inf.
        # S = -1/k * (-inf) = +inf.
        # We want to MAXIMIZE S. So we MINIMIZE -S.
        # -S = 1/k * log(sum(exp(-k*d))).
        # If d is large, -S is very negative.
        # If d is small, exp(-kd) is large, sum is large, log is large, -S is positive.
        # So minimizing -S encourages large d. Correct.
        
        # Numerical stability:
        # Use log-sum-exp trick
        # log(sum(exp(a_i))) = max(a) + log(sum(exp(a_i - max(a))))
        # Here a_i = -k * d_ij
        a = -k_soft * dists
        a_max = np.max(a)
        if a_max < -700: # exp(a) underflows to 0
            return 0.0 # effectively -S is very negative (good), but we return 0 for gradient? 
            # Actually if a_max is very negative, it means d is very large. 
            # The objective value should be very negative (good for minimization).
            # But we need a finite value.
            # Let's just return a large negative number? No, L-BFGS-B expects finite.
            # If d is large, the "cost" is low.
            # Let's return a value proportional to a_max.
            return a_max / k_soft
            
        log_sum_exp = a_max + np.log(np.sum(np.exp(a - a_max)))
        return log_sum_exp / k_soft

    def soft_min_grad(x):
        pts = x.reshape(-1, 2)
        # Compute distances and vectors
        # We need gradient of -S = 1/k * log(sum(exp(-k*d)))
        # d(-S)/dx_i = 1/k * 1/sum(exp(-k*d)) * sum_j d/dx_i (exp(-k*d_ij))
        # = 1/k * 1/Z * sum_j exp(-k*d_ij) * (-k) * (x_i - x_j)/d_ij
        # = -1/Z * sum_j exp(-k*d_ij) * (x_i - x_j)/d_ij
        # where Z = sum_{j!=i} exp(-k*d_ij) ? No, Z is global sum over all pairs.
        # Wait, the derivative of log(sum_{all pairs} exp(...)) w.r.t x_i only involves terms with i.
        # Let Z = sum_{a<b} exp(-k d_ab).
        # d/dx_i (log Z) = (1/Z) * sum_{j != i} d/dx_i (exp(-k d_ij))
        # = (1/Z) * sum_{j != i} exp(-k d_ij) * (-k) * (x_i - x_j)/d_ij
        # So grad(-S) = (1/k) * (1/Z) * sum_{j != i} exp(-k d_ij) * (-k) * (x_i - x_j)/d_ij
        # = -(1/Z) * sum_{j != i} exp(-k d_ij) * (x_i - x_j)/d_ij
        
        dists = pdist(pts)
        a = -k_soft * dists
        a_max = np.max(a)
        if a_max < -700:
            # Gradients are effectively 0 if distances are huge
            return np.zeros_like(x)
        
        # Compute weights w_ij = exp(-k d_ij) / Z
        # Z = sum exp(a)
        # To avoid underflow in weights, we can compute relative weights.
        # w_ij = exp(a_ij - a_max) / sum(exp(a - a_max))
        exp_a = np.exp(a - a_max)
        Z_norm = np.sum(exp_a)
        
        # Map pdist indices to (i, j)
        # pdist index k corresponds to pair (i, j) with i < j
        # i = int((2*n - 3 + sqrt(9 - 8*(2*n - 2 - 2*k))) / 2) ? 
        # Easier: loop or use helper
        # Since n=16, pairs=120. Loop is fine.
        grad = np.zeros_like(x)
        
        # Precompute inverse distances and vectors for all pairs
        # We need to sum contributions for each i
        # For each pair (i, j) with weight w_k:
        # grad_i += -w_k * (x_i - x_j) / d_k
        # grad_j += -w_k * (x_j - x_i) / d_k = +w_k * (x_i - x_j) / d_k
        
        # Let's iterate over pdist indices
        for k in range(len(dists)):
            d_k = dists[k]
            if d_k < 1e-9:
                continue
            w_k = exp_a[k] / Z_norm
            
            # Find i, j for index k
            # Formula for i-th row in condensed vector:
            # k = i*n - i*(i+1)/2 + (j - i - 1)
            # Inverse:
            # i = floor((2*n - 3 + sqrt((2*n-3)^2 - 8*(2*n - 2 - 2*k))) / 2)
            # j = k - i*n + i*(i+1)/2 + i + 1
            i = int((2*n - 3 + np.sqrt((2*n - 3)**2 - 8 * (2*n - 2 - 2*k))) / 2)
            j = k - i * n + i * (i + 1) // 2 + i + 1
            
            vec = pts[i] - pts[j]
            term = (w_k / d_k) * vec
            
            grad[2*i:2*i+2] -= term
            grad[2*j:2*j+2] += term
            
        return grad

    # Initialization strategies
    initializations = []
    
    # 1. Random points
    np.random.seed(42)
    for _ in range(10):
        initializations.append(np.random.randn(n, 2))
        
    # 2. Regular polygon (Circle)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    initializations.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 3. Hexagonal lattice subset
    points_hex = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i + 0.5 * j
            y = j * np.sqrt(3) / 2
            points_hex.append([x, y])
    points_hex = np.array(points_hex)
    dists_center = np.linalg.norm(points_hex, axis=1)
    idx = np.argsort(dists_center)[:n]
    initializations.append(points_hex[idx])
    
    # 4. Grid
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    initializations.append(grid_pts)

    # 5. Perturbed circle
    for _ in range(5):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        radii = 1 + 0.1 * np.random.randn(n)
        initializations.append(np.column_stack([np.cos(angles)*radii, np.sin(angles)*radii]))

    candidates = []

    for init in initializations:
        if time.time() - start_time > timeout - 20:
            break
            
        # Normalize initial points to avoid scale issues
        init_norm = init - np.mean(init, axis=0)
        d_max_init = np.max(pdist(init_norm))
        if d_max_init > 0:
            init_norm = init_norm / d_max_init
        else:
            continue
            
        x0 = init_norm.flatten()
        
        # Optimize with L-BFGS-B
        try:
            res = minimize(soft_min_obj, x0, method='L-BFGS-B', jac=soft_min_grad, 
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
        except Exception:
            continue
            
        opt_points = res.x.reshape(-1, 2)
        
        # Normalize to d_max = 1
        opt_points = opt_points - np.mean(opt_points, axis=0)
        d_max_opt = np.max(pdist(opt_points))
        if d_max_opt > 0:
            opt_points = opt_points / d_max_opt
            
        score = compute_score(opt_points)
        candidates.append((score, opt_points))
        
        # If we have time, we can do a local polish with Nelder-Mead on the score itself
        # But Nelder-Mead on 32D is expensive. Let's only do it for the top candidates later.

    # Sort candidates by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 5 candidates and polish them with Nelder-Mead on the actual score (or soft min)
    # Using actual score (d_min/d_max)^2 is non-smooth. 
    # Let's use a very sharp soft-min for polishing to stay smooth.
    k_polish = 50.0
    
    def polish_obj(x):
        pts = x.reshape(-1, 2)
        # Normalize internally to make it scale invariant? 
        # No, Nelder-Mead will change scale. 
        # We should normalize the points before computing the objective?
        # If we normalize inside the objective, the gradient (if we had one) would be wrong, 
        # but Nelder-Mead doesn't use gradients.
        # However, if we normalize inside, the function is flat along the scale direction.
        # This might confuse Nelder-Mead.
        # Better: Just optimize the shape, and normalize at the end.
        # But if the optimizer expands the points, d_min increases, but d_max increases too.
        # The ratio is invariant.
        # So we can just maximize d_min? No, if we maximize d_min, we can expand indefinitely.
        # We need to constrain scale.
        # Let's constrain the sum of squared distances to be 1? Or just normalize the input x in the objective?
        
        # Let's try: Normalize pts to have unit max distance inside the objective.
        pts_norm = pts - np.mean(pts, axis=0)
        d_max = np.max(pdist(pts_norm))
        if d_max < 1e-9:
            return 0.0
        pts_norm = pts_norm / d_max
        dists = pdist(pts_norm)
        a = -k_polish * dists
        a_max = np.max(a)
        if a_max < -700:
            return 0.0
        log_sum_exp = a_max + np.log(np.sum(np.exp(a - a_max)))
        return log_sum_exp / k_polish

    # Polish top 5
    for score, pts in candidates[:5]:
        if time.time() - start_time > timeout - 5:
            break
            
        x0 = pts.flatten()
        try:
            res = minimize(polish_obj, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            polished_points = res.x.reshape(-1, 2)
            
            # Normalize and score
            polished_points = polished_points - np.mean(polished_points, axis=0)
            d_max_pol = np.max(pdist(polished_points))
            if d_max_pol > 0:
                polished_points = polished_points / d_max_pol
                
            pol_score = compute_score(polished_points)
            
            if pol_score > best_score:
                best_score = pol_score
                best_points = polished_points.copy()
        except Exception:
            pass

    # If no points were found or all failed, return the best from L-BFGS-B
    if best_points is None:
        if candidates:
            best_score, best_points = candidates[0]
        else:
            np.random.seed(42)
            best_points = np.random.randn(n, 2)
            best_points = best_points - np.mean(best_points, axis=0)
            d_max = np.max(pdist(best_points))
            if d_max > 0:
                best_points = best_points / d_max

    return best_points
