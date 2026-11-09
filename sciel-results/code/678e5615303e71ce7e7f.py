import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 130
DEADLINE = 110.0  # Seconds

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _covering_radius(points, test_points):
    """Compute the covering radius in radians."""
    # Use KD-Tree for efficient nearest neighbor search
    tree = cKDTree(points)
    dists, _ = tree.query(test_points)
    return np.max(dists)

def construct():
    start_time = time.time()
    
    # Initial points: Fibonacci spiral
    points = _fibonacci_sphere(N)
    
    # Test points for the surrogate objective
    # Use a dense set of points on the sphere to approximate the max distance
    n_test = 5000
    test_points = _fibonacci_sphere(n_test)
    
    # Precompute test point norms (should be 1)
    # test_points is already normalized
    
    def objective_and_gradient(v):
        # v is shape (3*N,)
        p = v.reshape(N, 3)
        
        # Normalize points to unit sphere
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        p = p / norms
        
        # Find nearest test point for each config point? No, for each test point, find nearest config point.
        # We need min_i ||y_j - p_i||^2 for each j.
        # Efficient way: 
        # dists_sq[j, i] = ||y_j - p_i||^2 = ||y_j||^2 + ||p_i||^2 - 2 y_j . p_i = 1 + 1 - 2 y_j . p_i = 2 - 2 y_j . p_i
        # So we just need the max dot product y_j . p_i.
        
        # dot products shape (n_test, N)
        dots = test_points @ p.T
        max_dots = np.max(dots, axis=1)
        min_dists_sq = 2.0 - 2.0 * max_dots
        
        # Soft-max surrogate: log(sum(exp(min_dists_sq / T)))
        # To avoid overflow, subtract max
        m = np.max(min_dists_sq)
        if m > 500: # Overflow check, though dists are small
             exp_vals = np.exp((min_dists_sq - m) / T)
        else:
             exp_vals = np.exp(min_dists_sq / T)
        
        sum_exp = np.sum(exp_vals)
        obj_val = m + T * np.log(sum_exp)
        
        # Gradient calculation
        # w_j = exp(min_dists_sq_j / T) / sum_exp
        w = exp_vals / sum_exp
        
        # For each config point p_i, we need to know which test points j are assigned to it.
        # i_star(j) = argmax_i dots[j, i]
        i_star = np.argmax(dots, axis=1)
        
        grad_p = np.zeros_like(p)
        # Accumulate gradients
        # grad_p[i] = sum_{j: i_star(j)==i} w_j * (-2 * (y_j - p_i))
        # Note: The derivative of min_dist_sq w.r.t p_i is -2(y_j - p_i)
        
        # Efficient accumulation using np.add.at
        # w is (n_test,), i_star is (n_test,)
        # We need to sum w_j * (y_j - p_i) for each i.
        
        # term_j = w_j * y_j  (shape n_test, 3)
        # term_p_i = w_j * p_i (shape n_test, 3)
        
        # Sum w_j * y_j for each i
        sum_w_y = np.zeros((N, 3))
        np.add.at(sum_w_y, i_star, w[:, np.newaxis] * test_points)
        
        # Sum w_j for each i
        sum_w = np.zeros(N)
        np.add.at(sum_w, i_star, w)
        
        # grad_p[i] = -2 * (sum_w_y[i] - p_i * sum_w[i])
        grad_p = -2.0 * (sum_w_y - p * sum_w[:, np.newaxis])
        
        # Project gradient onto tangent space
        # g_proj = g - (g . p) p
        dot_g_p = np.sum(grad_p * p, axis=1, keepdims=True)
        grad_p = grad_p - dot_g_p * p
        
        # Reshape gradient to (3*N,)
        grad_v = grad_p.flatten()
        
        return obj_val, grad_v

    # Annealing schedule
    # Start with large T, reduce to small T
    temperatures = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    
    v_current = points.flatten()
    
    for T in temperatures:
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Wrap T in a closure for the optimizer
        def obj_grad(v, T=T):
            return objective_and_gradient(v)
            
        # L-BFGS-B
        res = minimize(
            obj_grad, 
            v_current, 
            method='L-BFGS-B',
            jac=True, # We return (f, g)
            options={'maxiter': 50, 'maxls': 10, 'ftol': 1e-8, 'gtol': 1e-5}
        )
        
        if res.success or res.status == 0:
            v_current = res.x
        else:
            # If it fails, keep previous best
            pass
            
        # Check time
        if time.time() - start_time > DEADLINE - 5:
            break

    # Final result
    p_final = v_current.reshape(N, 3)
    norms = np.linalg.norm(p_final, axis=1, keepdims=True)
    p_final = p_final / norms
    
    # Optional: Final polish with very small T or direct local search
    # Given time constraints, the annealed result should be good.
    # Let's do one more quick optimization with T=0.001 if time permits
    if time.time() - start_time < DEADLINE - 2:
         def obj_grad_final(v):
             return objective_and_gradient(v)
         # Monkey patch T? No, objective_and_gradient uses T from outer scope.
         # We need to pass T. Let's refactor objective to take T.
         
         # Re-implement with T passed explicitly for the final step
         def obj_grad_t(v, T):
             p = v.reshape(N, 3)
             norms = np.linalg.norm(p, axis=1, keepdims=True)
             norms[norms == 0] = 1.0
             p = p / norms
             
             dots = test_points @ p.T
             max_dots = np.max(dots, axis=1)
             min_dists_sq = 2.0 - 2.0 * max_dots
             
             m = np.max(min_dists_sq)
             exp_vals = np.exp((min_dists_sq - m) / T)
             sum_exp = np.sum(exp_vals)
             obj_val = m + T * np.log(sum_exp)
             
             w = exp_vals / sum_exp
             i_star = np.argmax(dots, axis=1)
             
             sum_w_y = np.zeros((N, 3))
             np.add.at(sum_w_y, i_star, w[:, np.newaxis] * test_points)
             
             sum_w = np.zeros(N)
             np.add.at(sum_w, i_star, w)
             
             grad_p = -2.0 * (sum_w_y - p * sum_w[:, np.newaxis])
             
             dot_g_p = np.sum(grad_p * p, axis=1, keepdims=True)
             grad_p = grad_p - dot_g_p * p
             
             return obj_val, grad_p.flatten()

         res_final = minimize(
             lambda v: obj_grad_t(v, 0.001), 
             v_current, 
             method='L-BFGS-B',
             jac=True,
             options={'maxiter': 30, 'ftol': 1e-10, 'gtol': 1e-6}
         )
         if res_final.success:
             v_current = res_final.x
             p_final = v_current.reshape(N, 3)
             norms = np.linalg.norm(p_final, axis=1, keepdims=True)
             p_final = p_final / norms

    # Ensure valid output
    if not np.all(np.isfinite(p_final)):
        return _fibonacci_sphere(N)
        
    return p_final
