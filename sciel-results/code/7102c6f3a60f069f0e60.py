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
    M = len(idx_i)
    
    def calc_score(x):
        diff = x[idx_i] - x[idx_j]
        d_ij = np.linalg.norm(diff, axis=1)
        min_d_ij = np.min(d_ij) / 2.0
        
        r_i = np.linalg.norm(x, axis=1)
        min_d_bound = np.min(1.0 - r_i)
        
        return np.min([min_d_ij, min_d_bound])
    
    def make_objective(beta):
        def objective_and_grad(x_flat):
            x = x_flat.reshape(N, 2)
            # Calculate distances
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
            all_y = np.concatenate([y_pairs, y_bound])
            
            # Soft min
            # softmin(y) = -1/beta * log(sum(exp(-beta*y)))
            # We want to maximize softmin, so minimize -softmin.
            # -softmin = 1/beta * log(sum(exp(-beta*y)))
            
            # Shift for numerical stability
            y_min = np.min(all_y)
            exp_terms = np.exp(-beta * (all_y - y_min))
            sum_exp = np.sum(exp_terms)
            weights = exp_terms / sum_exp
            
            # Objective value
            obj_val = y_min + (1.0 / beta) * np.log(sum_exp)
            
            # Gradient
            # grad(-softmin) = - sum w_k * grad(y_k)
            
            grad = np.zeros_like(x)
            
            # Gradient of y_pairs
            # y = d/2. grad_c_i (d/2) = (c_i - c_j) / (2 * d)
            # grad_c_j (d/2) = (c_j - c_i) / (2 * d) = - grad_c_i
            w_pairs = weights[:M]
            
            # term_i = w * (c_i - c_j) / (2d)
            # term_j = w * (c_j - c_i) / (2d)
            
            # We need to accumulate -w * grad(y)
            # So for i: -w * (c_i - c_j) / (2d)
            # For j: -w * (c_j - c_i) / (2d) = w * (c_i - c_j) / (2d)
            
            terms = w_pairs[:, np.newaxis] * diff / (2.0 * d_ij_safe[:, np.newaxis])
            
            # Accumulate to i: -terms
            np.add.at(grad, idx_i, -terms)
            # Accumulate to j: +terms
            np.add.at(grad, idx_j, terms)
            
            # Gradient of y_bound
            # y = 1 - |c|. grad_c (1 - |c|) = - c / |c|
            w_bound = weights[M:]
            
            r_i_safe = np.where(r_i < 1e-12, 1e-12, r_i)
            grad_bound = -x / r_i_safe[:, np.newaxis]
            
            # grad(-softmin) -= w_bound * grad_bound
            # So we add w_bound * grad_bound? No.
            # grad(-softmin) = - sum w * grad(y)
            # Term for bound: - w_bound * (-c/|c|) = w_bound * c/|c|
            # Wait.
            # y_bound = 1 - r_i
            # grad_c(y_bound) = - c_i / r_i
            # grad(-softmin)_bound_part = - w_bound * grad_c(y_bound) = - w_bound * (-c_i/r_i) = w_bound * c_i/r_i
            # So we ADD w_bound * c_i/r_i to grad.
            
            # My previous code: grad -= (w_bound[:, np.newaxis] * grad_bound)
            # grad_bound = -x/r_i
            # So grad -= w * (-x/r_i) = grad += w * x/r_i.
            # This matches.
            
            grad += (w_bound[:, np.newaxis] * (x / r_i_safe[:, np.newaxis]))
            
            return obj_val, grad.flatten()
        return objective_and_grad

    best_x = x0
    best_score = calc_score(best_x)
    
    # Stage 1: Coarse
    if time.time() - start_time < time_limit - 20:
        obj1 = make_objective(50.0)
        try:
            res1 = minimize(
                obj1, 
                best_x.flatten(), 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 100, 'gtol': 1e-6, 'ftol': 1e-6}
            )
            if res1.fun < obj1(best_x.flatten())[0]:
                candidate = res1.x.reshape(N, 2)
                s = calc_score(candidate)
                if s > best_score:
                    best_x = candidate
                    best_score = s
        except Exception:
            pass

    # Stage 2: Fine
    if time.time() - start_time < time_limit - 10:
        obj2 = make_objective(100.0)
        try:
            res2 = minimize(
                obj2, 
                best_x.flatten(), 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-8}
            )
            if res2.fun < obj2(best_x.flatten())[0]:
                candidate = res2.x.reshape(N, 2)
                s = calc_score(candidate)
                if s > best_score:
                    best_x = candidate
                    best_score = s
        except Exception:
            pass

    # Stage 3: Polish
    if time.time() - start_time < time_limit - 2:
        obj3 = make_objective(500.0)
        try:
            res3 = minimize(
                obj3, 
                best_x.flatten(), 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 50, 'gtol': 1e-10, 'ftol': 1e-10}
            )
            if res3.fun < obj3(best_x.flatten())[0]:
                candidate = res3.x.reshape(N, 2)
                s = calc_score(candidate)
                if s > best_score:
                    best_x = candidate
                    best_score = s
        except Exception:
            pass

    return best_x
