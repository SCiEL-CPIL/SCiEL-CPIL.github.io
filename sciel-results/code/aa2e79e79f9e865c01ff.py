import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = 110.0  # Keep a 10s safety margin

    # Initial grid: 5x5x4
    # x, y: 5 points, z: 4 points
    # x, y in [-0.4, -0.2, 0, 0.2, 0.4]
    # z in [-0.375, -0.125, 0.125, 0.375]
    x_vals = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    y_vals = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    z_vals = np.array([-0.375, -0.125, 0.125, 0.375])
    
    points = np.array([(x, y, z) for z in z_vals for y in y_vals for x in x_vals])
    # points shape (100, 3)
    
    def objective(params):
        c = params[:-1].reshape(-1, 3)
        r = params[-1]
        
        # Calculate pairwise distances
        # Use broadcasting
        # diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        # dists = np.linalg.norm(diff, axis=2)
        # This is O(N^2) memory, 100x100 is fine.
        
        # To save memory/time, we can compute upper triangle
        i, j = np.triu_indices(N, k=1)
        diff = c[i] - c[j]
        dists = np.linalg.norm(diff, axis=1)
        
        # Pair terms: log(1 + exp(-lambda * (dist - 2r)))
        # To avoid overflow in exp, use log1p(exp(x)) where x is negative
        # x = -lambda * (dists - 2r)
        # If x > 0, log1p(exp(x)) is large.
        # If x < 0, log1p(exp(x)) is small.
        
        lam = 50.0
        x_pairs = -lam * (dists - 2.0 * r)
        # Clip x_pairs to avoid overflow in exp if x is very large positive
        # But we want to minimize, so large values are bad.
        # log1p(exp(x)) for x > 0 is approx x.
        # We can just use np.log1p(np.exp(np.clip(x_pairs, -500, 500)))
        
        pair_terms = np.log1p(np.exp(np.clip(x_pairs, -500, 500)))
        
        # Boundary terms
        # min_k (0.5 - |c_ik| - r)
        # We want to penalize if |c_ik| > 0.5 - r
        # Term: log(1 + exp(-lambda * (0.5 - |c_ik| - r)))
        
        abs_c = np.abs(c)
        # For each point and each dimension
        # dist_to_boundary = 0.5 - abs_c
        # margin = dist_to_boundary - r
        margins = 0.5 - abs_c - r
        
        x_bound = -lam * margins
        bound_terms = np.log1p(np.exp(np.clip(x_bound, -500, 500)))
        
        sum_pairs = np.sum(pair_terms)
        sum_bound = np.sum(bound_terms)
        
        # Objective: Minimize sum of penalties - alpha * r
        # We want to maximize r, so subtract alpha * r
        alpha = 1.0
        return sum_pairs + sum_bound - alpha * r

    def gradient(params):
        # Numerical gradient or analytic?
        # Analytic is better but complex.
        # Let's use finite differences for now, or let L-BFGS-B estimate it.
        # L-BFGS-B can use finite differences if jac is not provided.
        # With 301 variables, finite differences might be slow (301 evaluations per step).
        # Let's provide an analytic gradient.
        
        c = params[:-1].reshape(-1, 3)
        r = params[-1]
        
        i, j = np.triu_indices(N, k=1)
        diff = c[i] - c[j]
        dists = np.linalg.norm(diff, axis=1)
        
        lam = 50.0
        x_pairs = -lam * (dists - 2.0 * r)
        
        # sigmoid(x) = exp(x) / (1 + exp(x))
        # d/dx log(1 + exp(x)) = sigmoid(x)
        # x = -lam * (d - 2r)
        # d/d(d) x = -lam * (1/d) * diff
        # d/d(r) x = -lam * (-2) = 2lam
        
        sig_pairs = 1.0 / (1.0 + np.exp(-np.clip(x_pairs, -500, 500)))
        
        # Gradient w.r.t c
        # d_obj/d_c_i = sum_{j != i} d_obj/d_dist_ij * d_dist_ij/d_c_i
        # d_obj/d_dist_ij = sigmoid(x) * d_x/d_dist = sigmoid(x) * (-lam)
        # d_dist_ij/d_c_i = diff / dist
        
        grad_c = np.zeros_like(c)
        # We need to sum contributions for each i
        # For each pair (i, j), diff = c_i - c_j
        # d/d_c_i = -lam * sig * diff / dist
        # d/d_c_j = -lam * sig * (-diff) / dist = lam * sig * diff / dist
        
        coeff = -lam * sig_pairs / dists
        # coeff shape (N_pairs,)
        # diff shape (N_pairs, 3)
        # contrib_i = coeff * diff
        # contrib_j = -coeff * diff
        
        np.add.at(grad_c, i, coeff[:, np.newaxis] * diff)
        np.add.at(grad_c, j, -coeff[:, np.newaxis] * diff)
        
        # Gradient w.r.t r
        # d_obj/d_r = sum_pairs d/d_r + sum_bound d/d_r - alpha
        # d/d_r log(1+exp(x)) = sigmoid(x) * d_x/d_r
        # x_pairs = -lam * (d - 2r) -> d_x/d_r = 2lam
        # x_bound = -lam * (0.5 - |c| - r) -> d_x/d_r = lam
        
        grad_r = np.sum(sig_pairs * 2.0 * lam)
        
        # Boundary gradient
        abs_c = np.abs(c)
        margins = 0.5 - abs_c - r
        x_bound = -lam * margins
        sig_bound = 1.0 / (1.0 + np.exp(-np.clip(x_bound, -500, 500)))
        
        # d_obj/d_c_k = sum_{i} sigmoid(x_ik) * d_x_ik/d_c_k
        # x_ik = -lam * (0.5 - |c_ik| - r)
        # d_x_ik/d_c_k = -lam * d(0.5 - |c_ik|)/d_c_k = -lam * (-sign(c_ik)) = lam * sign(c_ik)
        
        sign_c = np.sign(c)
        grad_c_bound = sig_bound * lam * sign_c
        grad_c += np.sum(grad_c_bound, axis=1) # Wait, sig_bound is (N, 3)
        # grad_c_bound is (N, 3). Sum over points? No, each point i has 3 terms.
        # grad_c[i] += sig_bound[i] * lam * sign_c[i]
        # This is element-wise addition.
        
        # Correction:
        # grad_c[i] accumulates from pairs and boundary.
        # Boundary part for point i:
        # term_k = sig_bound[i,k] * lam * sign_c[i,k]
        # grad_c[i] += term (vector of 3)
        
        # My previous line `grad_c += np.sum(grad_c_bound, axis=1)` is wrong.
        # grad_c_bound is already (N, 3).
        grad_c += sig_bound * lam * sign_c
        
        grad_r += np.sum(sig_bound * lam)
        
        grad_r -= alpha
        
        # Combine gradients
        grad = np.concatenate([grad_c.flatten(), [grad_r]])
        return grad

    # Initial parameters
    r0 = 0.1
    x0 = np.concatenate([points.flatten(), [r0]])
    
    bounds = [(-0.5, 0.5)] * (N * 3) + [(0.0, 0.5)]
    
    best_r = r0
    best_c = points.copy()
    
    # Run optimization
    try:
        res = minimize(objective, x0, method='L-BFGS-B', jac=gradient, bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-5, 'disp': False})
        
        if res.success or res.fun < objective(x0):
            c_opt = res.x[:-1].reshape(-1, 3)
            r_opt = res.x[-1]
            
            # Verify validity
            if np.all(np.isfinite(c_opt)) and np.all(np.isfinite(r_opt)):
                # Check constraints strictly to be safe
                i, j = np.triu_indices(N, k=1)
                dists = np.linalg.norm(c_opt[i] - c_opt[j], axis=1)
                min_dist = np.min(dists)
                min_bound = np.min(0.5 - np.abs(c_opt))
                
                # The radius is limited by min(min_dist/2, min_bound)
                actual_r = min(min_dist / 2.0, min_bound)
                
                if actual_r > best_r:
                    best_r = actual_r
                    best_c = c_opt
    except Exception as e:
        pass

    # If we have time, try a few random restarts
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        for _ in range(5):
            if time.time() - start_time > deadline - 10:
                break
            # Random perturbation
            np.random.seed(int(time.time() * 1000) % 10000)
            c_init = points + np.random.normal(0, 0.05, points.shape)
            # Clip to bounds
            c_init = np.clip(c_init, -0.45, 0.45)
            
            x0 = np.concatenate([c_init.flatten(), [best_r]])
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', jac=gradient, bounds=bounds, 
                               options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-5, 'disp': False})
                
                if res.success or res.fun < objective(x0):
                    c_opt = res.x[:-1].reshape(-1, 3)
                    r_opt = res.x[-1]
                    
                    if np.all(np.isfinite(c_opt)) and np.all(np.isfinite(r_opt)):
                        i, j = np.triu_indices(N, k=1)
                        dists = np.linalg.norm(c_opt[i] - c_opt[j], axis=1)
                        min_dist = np.min(dists)
                        min_bound = np.min(0.5 - np.abs(c_opt))
                        actual_r = min(min_dist / 2.0, min_bound)
                        
                        if actual_r > best_r:
                            best_r = actual_r
                            best_c = c_opt
            except Exception:
                pass

    return best_c

if __name__ == "__main__":
    construct()
