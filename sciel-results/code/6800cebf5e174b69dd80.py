import numpy as np
import time
from scipy.optimize import minimize

N = 300

def _hex_lattice_init(n):
    # Generate hex lattice points
    # Spacing s such that we get at least n points in unit circle
    # Approx area per point s^2 * sqrt(3)/2. Total area pi.
    # n * s^2 * sqrt(3)/2 <= pi -> s <= sqrt(2*pi / (n*sqrt(3)))
    s = np.sqrt(2 * np.pi / (n * np.sqrt(3))) * 0.95 # 5% margin
    
    # Generate grid
    k = int(1.0 / s) + 2
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0:
                points.append((x, y, r))
                
    points = np.array(points)
    # Sort by distance to center
    points = points[points[:, 2].argsort()]
    
    # If we have more than N, take the first N
    if len(points) >= n:
        return points[:n, :2]
    else:
        # Fallback: random or scaled
        # This shouldn't happen with 0.95 factor, but just in case
        # Scale down to fit
        r_max = np.max(points[:, 2])
        scale = 1.0 / r_max
        return points[:, :2] * scale

def construct():
    start_time = time.time()
    time_limit = 140.0 # Safety margin
    
    x0 = _hex_lattice_init(N)
    
    # Precompute indices for pairs
    idx_i, idx_j = np.triu_indices(N, 1)
    
    def objective_and_grad(x):
        # x is shape (N, 2)
        # Calculate distances
        # d_ij
        diff = x[idx_i] - x[idx_j]
        d_ij = np.linalg.norm(diff, axis=1)
        
        # Avoid division by zero
        d_ij_safe = np.where(d_ij < 1e-12, 1e-12, d_ij)
        
        # Boundary distances
        r_i = np.linalg.norm(x, axis=1)
        d_bound = 1.0 - r_i
        
        # Combine all constraints y
        # y_k = d_ij / 2
        # y_m = d_bound
        y_pairs = d_ij / 2.0
        y_bound = d_bound
        
        # Soft min
        beta = 100.0 # Adaptive or fixed? Fixed 100 is usually good for this scale
        
        # Shift for numerical stability
        all_y = np.concatenate([y_pairs, y_bound])
        y_min = np.min(all_y)
        
        # Weights
        exp_terms = np.exp(-beta * (all_y - y_min))
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp
        
        # Objective: -softmin = -(-1/beta * log(sum_exp)) = 1/beta * log(sum_exp)
        # Wait, softmin(y) = -1/beta * log(sum(exp(-beta*y)))
        # We want to maximize softmin, so minimize -softmin.
        # -softmin = 1/beta * log(sum(exp(-beta*y)))
        # Note: The shift y_min cancels out in log(sum(exp(-beta*(y-y_min))))?
        # sum(exp(-beta*(y-y_min))) = exp(beta*y_min) * sum(exp(-beta*y))
        # log(...) = beta*y_min + log(sum(exp(-beta*y)))
        # So -softmin = 1/beta * (beta*y_min + log(sum(exp(-beta*y)))) = y_min + 1/beta * log(sum(exp(-beta*y)))
        # The term 1/beta * log(sum(exp(-beta*y))) is the "log-sum-exp" correction.
        # For large beta, it approaches 0.
        # So minimizing -softmin is roughly minimizing y_min.
        
        obj_val = y_min + (1.0 / beta) * np.log(sum_exp)
        
        # Gradient
        # grad_obj = grad(-softmin) = -grad(softmin)
        # grad(softmin) = sum w_k * grad(y_k)
        # So grad_obj = - sum w_k * grad(y_k)
        
        grad = np.zeros_like(x)
        
        # Gradient of y_pairs
        # w_pairs = weights[:len(y_pairs)]
        w_pairs = weights[:len(y_pairs)]
        
        # grad_y_pairs_i = (c_i - c_j) / (2 * d_ij)
        # grad_y_pairs_j = (c_j - c_i) / (2 * d_ij)
        
        # Vectorized accumulation
        # diff is (M, 2)
        # w_pairs is (M,)
        # term = w_pairs[:, np.newaxis] * diff / (2 * d_ij_safe[:, np.newaxis])
        terms = w_pairs[:, np.newaxis] * diff / (2.0 * d_ij_safe[:, np.newaxis])
        
        # Accumulate to i
        np.add.at(grad, idx_i, -terms) # Because we are minimizing -softmin, and grad(softmin) has +terms for i?
        # Let's check signs.
        # y = d/2. grad_c_i (d/2) = (c_i - c_j) / (2d).
        # softmin grad = sum w * grad(y).
        # So grad(softmin)_i gets contribution w * (c_i - c_j)/(2d).
        # We minimize -softmin. So grad(-softmin)_i = - w * (c_i - c_j)/(2d).
        # So we subtract the terms.
        
        # Accumulate to j
        # grad_c_j (d/2) = (c_j - c_i) / (2d) = - (c_i - c_j)/(2d) = -term
        # grad(softmin)_j = sum w * (-term)
        # grad(-softmin)_j = - sum w * (-term) = sum w * term
        # So we add the terms to j.
        
        np.add.at(grad, idx_j, terms)
        
        # Gradient of y_bound
        # w_bound = weights[len(y_pairs):]
        w_bound = weights[len(y_pairs):]
        
        # y = 1 - |c|. grad_c (1 - |c|) = - c / |c|
        # r_i_safe = np.where(r_i < 1e-12, 1e-12, r_i)
        r_i_safe = np.where(r_i < 1e-12, 1e-12, r_i)
        grad_bound = -x / r_i_safe[:, np.newaxis]
        
        # grad(softmin) += w_bound * grad_bound
        # grad(-softmin) -= w_bound * grad_bound
        grad -= (w_bound[:, np.newaxis] * grad_bound)
        
        return obj_val, grad

    # Optimization
    # Use L-BFGS-B
    # We don't have strict bounds, but the objective penalizes going out.
    # However, to be safe, we can provide bounds (-1.1, 1.1) or similar.
    # Actually, if a point goes far out, d_bound becomes very negative, exp(-beta * y) becomes huge,
    # so the objective becomes very large (bad for minimization).
    # So it should stay in.
    
    result = None
    best_x = x0
    best_score = -np.inf
    
    # Run optimization
    try:
        # maxiter limit to prevent hanging
        # 200 iterations usually enough for 600 vars with L-BFGS-B
        res = minimize(
            objective_and_grad, 
            x0.flatten(), 
            method='L-BFGS-B', 
            jac=True, 
            options={'maxiter': 100, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < objective_and_grad(x0.flatten())[0]:
            best_x = res.x.reshape(N, 2)
            
    except Exception as e:
        print(f"Optimization failed: {e}")
        
    # Calculate true score for best_x
    def calc_score(x):
        diff = x[idx_i] - x[idx_j]
        d_ij = np.linalg.norm(diff, axis=1)
        min_d_ij = np.min(d_ij) / 2.0
        
        r_i = np.linalg.norm(x, axis=1)
        min_d_bound = np.min(1.0 - r_i)
        
        return np.min([min_d_ij, min_d_bound])
    
    current_score = calc_score(best_x)
    if current_score > best_score:
        best_score = current_score
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed < time_limit - 10:
        # Try a second pass with higher beta or from the previous result
        # Or just refine with more iterations?
        # Let's try increasing beta and running again from best_x
        # Modify global beta? No, closure.
        # We can just run minimize again with a different function.
        
        beta2 = 200.0
        def objective_and_grad2(x):
             # Same logic, but with beta2
             # ... (copy paste logic with beta2)
             # To save code, I'll just reuse the structure but this is inefficient.
             # Better: just run one good pass.
             pass
             
        # Actually, the first pass with beta=100 should be decent.
        # If time permits, we can do a "polish" pass with a very high beta (e.g., 1000)
        # which approximates the hard min closely.
        
        def objective_and_grad_polish(x):
            diff = x[idx_i] - x[idx_j]
            d_ij = np.linalg.norm(diff, axis=1)
            d_ij_safe = np.where(d_ij < 1e-12, 1e-12, d_ij)
            r_i = np.linalg.norm(x, axis=1)
            d_bound = 1.0 - r_i
            y_pairs = d_ij / 2.0
            y_bound = d_bound
            all_y = np.concatenate([y_pairs, y_bound])
            y_min = np.min(all_y)
            beta = 1000.0
            exp_terms = np.exp(-beta * (all_y - y_min))
            sum_exp = np.sum(exp_terms)
            weights = exp_terms / sum_exp
            obj_val = y_min + (1.0 / beta) * np.log(sum_exp)
            grad = np.zeros_like(x)
            w_pairs = weights[:len(y_pairs)]
            terms = w_pairs[:, np.newaxis] * diff / (2.0 * d_ij_safe[:, np.newaxis])
            np.add.at(grad, idx_i, -terms)
            np.add.at(grad, idx_j, terms)
            w_bound = weights[len(y_pairs):]
            r_i_safe = np.where(r_i < 1e-12, 1e-12, r_i)
            grad_bound = -x / r_i_safe[:, np.newaxis]
            grad -= (w_bound[:, np.newaxis] * grad_bound)
            return obj_val, grad

        try:
            res2 = minimize(
                objective_and_grad_polish, 
                best_x.flatten(), 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 50, 'gtol': 1e-10}
            )
            if res2.success or res2.fun < objective_and_grad_polish(best_x.flatten())[0]:
                candidate_x = res2.x.reshape(N, 2)
                score2 = calc_score(candidate_x)
                if score2 > best_score:
                    best_x = candidate_x
                    best_score = score2
        except Exception:
            pass

    return best_x
