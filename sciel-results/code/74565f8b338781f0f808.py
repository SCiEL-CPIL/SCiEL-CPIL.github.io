import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0
    
    n = 14
    d = 3
    
    # Pre-compute indices for pairwise distances
    # We will compute distances manually for gradient computation efficiency
    # or use pdist and map back, but manual is faster for gradients.
    
    def get_pairwise_indices(n):
        idx = []
        for i in range(n):
            for j in range(i+1, n):
                idx.append((i, j))
        return np.array(idx)
    
    pairs = get_pairwise_indices(n)
    i_idx, j_idx = pairs[:, 0], pairs[:, 1]
    
    def compute_distances(points):
        # points: (n, 3)
        # returns: (91,) array of squared distances? No, Euclidean distances.
        # diff = points[i_idx] - points[j_idx]
        # dists = np.linalg.norm(diff, axis=1)
        # To be efficient:
        diff = points[i_idx] - points[j_idx]
        sq_dists = np.sum(diff**2, axis=1)
        dists = np.sqrt(sq_dists)
        return dists, diff, sq_dists

    def objective_and_grad(points, beta):
        """
        Minimizes -log(soft_min^2 / soft_max^2)
        Which is -2*log(soft_min) + 2*log(soft_max)
        """
        dists, diff, sq_dists = compute_distances(points)
        
        # Avoid log(0) or inf
        # If points coincide, dist is 0.
        # We add a small epsilon to distances for numerical stability in soft ops
        # But for the final metric, we use exact.
        
        # Softmin: m = -1/beta * log(sum(exp(-beta * dists)))
        # Log Softmin: log(m) = -log(sum(exp(-beta * dists))) - log(1/beta) ... 
        # Actually, let's just maximize m^2 / M^2.
        # log(m^2/M^2) = 2 log(m) - 2 log(M)
        
        # m_beta = -1/beta * log(sum(exp(-beta * d)))
        # M_beta =  1/beta * log(sum(exp( beta * d)))
        
        # Let's work with log-sum-exp for stability.
        # LSE(x) = max(x) + log(sum(exp(x - max(x))))
        
        # For softmin: x = -beta * d
        # LSE_neg = max(-beta*d) + log(sum(exp(-beta*d - max(-beta*d))))
        # softmin_val = -1/beta * LSE_neg
        
        # For softmax: x = beta * d
        # LSE_pos = max(beta*d) + log(sum(exp(beta*d - max(beta*d))))
        # softmax_val = 1/beta * LSE_pos
        
        neg_bd = -beta * dists
        pos_bd = beta * dists
        
        max_neg = np.max(neg_bd)
        max_pos = np.max(pos_bd)
        
        # Stable log-sum-exp
        lse_neg = max_neg + np.log(np.sum(np.exp(neg_bd - max_neg)))
        lse_pos = max_pos + np.log(np.sum(np.exp(pos_bd - max_pos)))
        
        log_softmin = -lse_neg / beta # log(m) = log(-1/beta * exp(lse_neg)) = -log(beta) + lse_neg ... wait.
        # m = -1/beta * exp(lse_neg). 
        # log(m) = -log(beta) + lse_neg.
        # Wait, exp(-beta d) sums to S. m = -1/beta log S.
        # This is NOT exp(lse).
        # Let's stick to the definition:
        # m = -1/beta * log( sum exp(-beta d) )
        # log(m) = log( -1/beta ) + log( log( sum exp(-beta d) ) ) ? No.
        
        # Let's just compute m and M directly with clipping.
        # For large beta, exp(-beta d) underflows for large d.
        # We can shift.
        
        # m = -1/beta * log( sum exp(-beta d) )
        # Let D_max = max(d).
        # sum exp(-beta d) = exp(-beta D_max) * sum exp(-beta (d - D_max))
        # log(sum ...) = -beta D_max + log( sum exp(-beta (d - D_max)) )
        # m = -1/beta * ( -beta D_max + log(...) ) = D_max - 1/beta * log( sum exp(-beta (d - D_max)) )
        
        # M = 1/beta * log( sum exp(beta d) )
        # Let D_min = min(d).
        # sum exp(beta d) = exp(beta D_min) * sum exp(beta (d - D_min))
        # log(sum ...) = beta D_min + log( sum exp(beta (d - D_min)) )
        # M = D_min + 1/beta * log( sum exp(beta (d - D_min)) )
        
        # This is stable.
        
        d_max = np.max(dists)
        d_min = np.min(dists)
        
        # If all points are same, d_max=0, d_min=0.
        if d_max < 1e-10:
            return 1e10, np.zeros_like(points)
            
        # Softmin
        # exp(-beta * (d - d_max))
        exp_neg = np.exp(-beta * (dists - d_max))
        log_sum_neg = np.log(np.sum(exp_neg))
        soft_min = d_max - (1.0 / beta) * log_sum_neg
        
        # Softmax
        exp_pos = np.exp(beta * (dists - d_min))
        log_sum_pos = np.log(np.sum(exp_pos))
        soft_max = d_min + (1.0 / beta) * log_sum_pos
        
        # Objective: Minimize -2 * log(soft_min) + 2 * log(soft_max)
        # Gradient wrt points
        # d(soft_min)/d(points) ?
        # soft_min = d_max - 1/beta * log(sum exp(-beta (d_k - d_max)))
        # Let S_neg = sum exp(-beta (d_k - d_max))
        # d(soft_min)/d(x_i) = -1/beta * (1/S_neg) * d(S_neg)/d(x_i)
        # d(S_neg)/d(x_i) = sum_k exp(-beta (d_k - d_max)) * d(-beta (d_k - d_max))/d(x_i)
        # Note: d_max depends on x_i if x_i is involved in the max distance.
        # This makes the gradient of the "shifted" term complicated if we treat d_max as variable.
        # However, the unshifted form:
        # m = -1/beta log(sum exp(-beta d_k))
        # dm/dx_i = -1/beta * (1/S) * sum_k exp(-beta d_k) * (-beta) * dd_k/dx_i
        # dm/dx_i = (1/S) * sum_k exp(-beta d_k) * dd_k/dx_i
        # Let w_k = exp(-beta d_k) / S. Sum w_k = 1.
        # dm/dx_i = sum_k w_k * dd_k/dx_i
        
        # Similarly for M:
        # M = 1/beta log(sum exp(beta d_k))
        # dM/dx_i = 1/beta * (1/T) * sum_k exp(beta d_k) * beta * dd_k/dx_i
        # dM/dx_i = (1/T) * sum_k exp(beta d_k) * dd_k/dx_i
        # Let v_k = exp(beta d_k) / T. Sum v_k = 1.
        # dM/dx_i = sum_k v_k * dd_k/dx_i
        
        # Objective J = -2 log(m) + 2 log(M)
        # dJ/dx_i = -2/m * dm/dx_i + 2/M * dM/dx_i
        # dJ/dx_i = -2/m * sum_k w_k * dd_k/dx_i + 2/M * sum_k v_k * dd_k/dx_i
        
        # dd_k/dx_i:
        # d_k = ||x_i - x_j||
        # dd_k/dx_i = (x_i - x_j) / d_k
        # dd_k/dx_j = (x_j - x_i) / d_k = -dd_k/dx_i
        
        # We need to accumulate gradients for all points.
        # grad_points = np.zeros_like(points)
        
        # Calculate weights
        # Using stable exps
        # w_k proportional to exp(-beta d_k)
        # v_k proportional to exp(beta d_k)
        
        # To avoid overflow/underflow in weights, we can reuse the shifted exps?
        # w_k = exp(-beta d_k) / sum(exp(-beta d_k))
        # This is just the softmax of -beta d_k.
        # v_k = softmax(beta d_k).
        
        # We already have exp_neg and exp_pos shifted.
        # sum_exp_neg = sum(exp_neg)
        # w_k = exp(-beta d_k) / sum_exp_neg
        # Note: exp(-beta d_k) = exp(-beta d_max) * exp_neg_k
        # So w_k = exp_neg_k / sum_exp_neg
        
        sum_exp_neg = np.sum(exp_neg)
        w = exp_neg / sum_exp_neg
        
        sum_exp_pos = np.sum(exp_pos)
        v = exp_pos / sum_exp_pos
        
        # Gradient of m: sum_k w_k * (x_i - x_j)/d_k for pair k involving i
        # Gradient of M: sum_k v_k * (x_i - x_j)/d_k
        
        # We can compute the vector sum for each point.
        # For each pair k (i, j):
        # term_k = w_k / d_k * (x_i - x_j)  for dm
        # term_k = v_k / d_k * (x_i - x_j)  for dM
        
        # grad_m = zeros
        # for each k:
        #   grad_m[i_idx[k]] += w[k] / d_k * diff[k]
        #   grad_m[j_idx[k]] -= w[k] / d_k * diff[k]
        
        # Same for grad_M with v.
        
        # J_grad = -2/m * grad_m + 2/M * grad_M
        
        if soft_min < 1e-10 or soft_max < 1e-10:
             # Degenerate case, push apart
             return 1e10, np.random.randn(*points.shape)
             
        coeff_m = -2.0 / soft_min
        coeff_M = 2.0 / soft_max
        
        # Compute weighted differences
        # diff is (91, 3)
        # d_k is (91,)
        # w is (91,)
        
        # Handle d_k = 0
        safe_d = np.where(dists < 1e-10, 1e-10, dists)
        
        w_over_d = w / safe_d
        v_over_d = v / safe_d
        
        # Accumulate gradients
        grad_m = np.zeros_like(points)
        grad_M = np.zeros_like(points)
        
        # np.add.at is slow but robust. 
        # Since n=14, 91 pairs, we can just loop or use scatter.
        # Loop is fine for 91 iterations.
        
        for k in range(len(pairs)):
            i = i_idx[k]
            j = j_idx[k]
            d_k = safe_d[k]
            diff_k = diff[k]
            
            wm = w[k] / d_k
            vm = v[k] / d_k
            
            grad_m[i] += wm * diff_k
            grad_m[j] -= wm * diff_k
            
            grad_M[i] += vm * diff_k
            grad_M[j] -= vm * diff_k
            
        grad_J = coeff_m * grad_m + coeff_M * grad_M
        
        val = -2.0 * np.log(soft_min) + 2.0 * np.log(soft_max)
        
        return val, grad_J

    def true_metric(points):
        dists, _, _ = compute_distances(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    best_score = 0.0
    best_points = None
    
    # Initializations
    inits = []
    
    # 1. Icosahedron + 2 Poles
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico_verts /= np.linalg.norm(ico_verts[0]) # Normalize
    poles = np.array([[0, 0, 1], [0, 0, -1]])
    init_ico = np.vstack([ico_verts, poles])
    inits.append(init_ico)
    
    # 2. Random Sphere
    np.random.seed(42)
    for _ in range(5):
        rand_pts = np.random.randn(n, d)
        rand_pts /= np.linalg.norm(rand_pts, axis=1, keepdims=True)
        inits.append(rand_pts)
        
    # 3. Random Gaussian
    for _ in range(5):
        rand_pts = np.random.randn(n, d)
        inits.append(rand_pts)

    # Optimization Loop
    # We will use a sequence of beta values (annealing)
    # Start with low beta (smooth), increase to high beta (sharp)
    betas = [10, 50, 100, 500, 1000, 2000]
    
    current_points = inits[0].copy()
    
    for beta in betas:
        for init in inits:
            if time.time() > deadline - 60.0:
                break
                
            try:
                res = minimize(
                    objective_and_grad, 
                    init.copy(), 
                    args=(beta,),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.success:
                    score = true_metric(res.x)
                    if score > best_score:
                        best_score = score
                        best_points = res.x.copy()
                        
                # Also check the initial point itself
                score_init = true_metric(init)
                if score_init > best_score:
                    best_score = score_init
                    best_points = init.copy()
                    
            except Exception as e:
                pass
        
        # If we found a good point, use it as the base for the next beta
        # Actually, just restart from best_points for the next beta?
        # Or restart from all inits?
        # Restarting from all inits with higher beta is safer to escape local minima of the smooth approx.
        # But using the best found so far is also good.
        # Let's add best_points to inits for the next round if it's not already there.
        if best_points is not None:
             # Simple check: if best_points is not in inits (by norm)
             # Just append it
             inits.append(best_points.copy())
             
    # Final polish with Nelder-Mead on the true metric?
    # The true metric is non-smooth. Nelder-Mead can handle it.
    if best_points is not None and time.time() < deadline - 30.0:
        try:
            # Maximize true_metric, so minimize -true_metric
            def neg_true_metric(points):
                return -true_metric(points)
            
            res_nm = minimize(
                neg_true_metric, 
                best_points, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-12}
            )
            
            if res_nm.success:
                score_nm = true_metric(res_nm.x)
                if score_nm > best_score:
                    best_score = score_nm
                    best_points = res_nm.x.copy()
        except Exception:
            pass

    if best_points is None:
        return inits[0]
        
    return best_points

# EVOLVE-BLOCK-END
