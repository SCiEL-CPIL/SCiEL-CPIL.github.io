import numpy as np
import scipy.integrate
import scipy.optimize
import time

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit

    # Define a grid of cosmological parameters for validation
    # w0 in [-1.2, 0.5], w1 in [-1.5, 1.5]
    # Increase resolution to better capture worst-case error
    w0_vals = np.linspace(-1.2, 0.5, 25)
    w1_vals = np.linspace(-1.5, 1.5, 25)
    
    # Define a grid of redshifts for evaluation
    # The error should be minimized across a range of z
    # Include lower z to capture behavior near 0, and higher z for tail
    z_vals = np.concatenate([
        np.linspace(0.01, 0.1, 10),
        np.linspace(0.2, 1.0, 20),
        np.linspace(1.5, 3.0, 20),
        np.linspace(3.5, 6.0, 10)
    ])
    
    # Precompute exact integrals for all combinations
    # I_exact = int_0^1 du / E(z*u)
    # E(t) = (1+t)^(3(1+w0)) * exp(3*w1*(1 - 1/(1+t)))
    # 1/E(t) = exp(-log_E)
    
    # Use high-order Gauss-Legendre for fast, accurate reference integrals
    # 128 points is typically sufficient for smooth functions like this to ~1e-12 accuracy
    gl_ref_nodes, gl_ref_weights = np.polynomial.legendre.leggauss(128)
    # Map from [-1, 1] to [0, 1]
    u_ref = 0.5 + 0.5 * gl_ref_nodes
    w_ref = 0.5 * gl_ref_weights
    
    exact_ints = []
    model_params = []
    
    # Vectorize the exact integral computation
    # We can process batches or just loop if N is small enough, but 25*25*60 = 37500 models.
    # Vectorizing over the grid dimensions is better.
    
    # Reshape for broadcasting: w0 (25,1,1), w1 (1,25,1), z (1,1,60)
    w0_grid = w0_vals[:, None, None]
    w1_grid = w1_vals[None, :, None]
    z_grid = z_vals[None, None, :]
    
    # u_ref shape (128,). We need to broadcast to (25, 25, 60, 128)
    # This might be memory heavy: 25*25*60*128 * 8 bytes = 37500 * 128 * 8 ~ 38 MB. Acceptable.
    
    t_grid = z_grid * u_ref[None, None, None, :] # Shape (25, 25, 60, 128)
    
    # Calculate log_E
    # log_E = 3*(1+w0)*log1p(t) + 3*w1*(1 - 1/(1+t))
    log_E_grid = 3.0 * (1.0 + w0_grid) * np.log1p(t_grid) + 3.0 * w1_grid * (1.0 - 1.0 / (1.0 + t_grid))
    
    # Calculate 1/E(t)
    f_grid = np.exp(-log_E_grid) # Shape (25, 25, 60, 128)
    
    # Perform quadrature sum over the last axis
    # Weights w_ref shape (128,)
    exact_ints_grid = np.sum(f_grid * w_ref[None, None, None, :], axis=3) # Shape (25, 25, 60)
    
    # Flatten for optimization objective
    exact_ints = exact_ints_grid.flatten()
    
    # Extract w0, w1, z arrays for vectorized objective
    # We need to match the flattened order.
    # numpy flatten is C-order: last index changes fastest.
    # So we need to repeat w0 for each w1,z, etc.
    # Or just keep them as grids and reshape in objective?
    # It's faster to pre-compute the broadcasted arrays for the objective function inputs.
    
    # Let's keep the grid structure for the objective to avoid flattening/repeating large arrays if possible,
    # or just flatten them once.
    w0_flat = np.repeat(np.tile(w0_vals, len(w1_vals)), len(z_vals)) # 25*25*60
    w1_flat = np.tile(np.repeat(w1_vals, len(z_vals)), len(w0_vals))
    z_flat = np.tile(z_vals, len(w0_vals)*len(w1_vals))
    
    # Verify lengths
    assert len(w0_flat) == len(exact_ints)
    
    # Parameterization
    # 16 parameters: 8 for nodes (theta), 8 for weights (alpha)
    # u_i = sigmoid(theta_i)
    # w_i = softmax(alpha_i)
    
    def params_to_nodes_weights(params):
        theta = params[:8]
        alpha = params[8:16]
        
        # Nodes
        u = 1.0 / (1.0 + np.exp(-theta))
        
        # Weights
        # Shift alpha for numerical stability
        alpha_max = np.max(alpha)
        exp_alpha = np.exp(alpha - alpha_max)
        w = exp_alpha / np.sum(exp_alpha)
        
        return u, w

    def objective(params):
        u, w = params_to_nodes_weights(params)
        
        # Calculate quadrature sum for all models
        # Create arrays of shape (n_models, 8)
        # n_models is 37500. 37500 * 8 = 300,000 floats. Very fast.
        t = z_flat[:, None] * u[None, :]
        
        # Calculate E(t)
        log_E = 3.0 * (1.0 + w0_flat)[:, None] * np.log1p(t) + 3.0 * w1_flat[:, None] * (1.0 - 1.0 / (1.0 + t))
        
        f_vals = np.exp(-log_E) # 1/E(t)
        
        # Quadrature sum: sum_i w_i f(t_i)
        quad_sums = np.sum(f_vals * w[None, :], axis=1)
        
        # Relative error
        rel_errors = np.abs(quad_sums - exact_ints) / exact_ints
        
        # Return max error
        return np.max(rel_errors)

    def objective_smooth(params):
        # Use a soft-max or similar to make it differentiable for BFGS/L-BFGS-B
        u, w = params_to_nodes_weights(params)
        t = z_flat[:, None] * u[None, :]
        log_E = 3.0 * (1.0 + w0_flat)[:, None] * np.log1p(t) + 3.0 * w1_flat[:, None] * (1.0 - 1.0 / (1.0 + t))
        f_vals = np.exp(-log_E)
        quad_sums = np.sum(f_vals * w[None, :], axis=1)
        rel_errors = np.abs(quad_sums - exact_ints) / exact_ints
        
        # Soft max with temperature
        # Use a lower temperature for sharper max approximation during refinement
        temp = 0.005
        return np.log(np.sum(np.exp(rel_errors / temp))) * temp

    # Initial guess: Gauss-Legendre nodes and weights on [0,1]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    u0 = 0.5 + 0.5 * gl_nodes
    w0_init = 0.5 * gl_weights
    
    # Inverse transform for parameters
    u0_safe = np.clip(u0, 1e-9, 1-1e-9)
    theta0 = np.log(u0_safe / (1 - u0_safe))
    
    w0_init_safe = np.clip(w0_init, 1e-9, 1.0)
    alpha0 = np.log(w0_init_safe)
    
    p0 = np.concatenate([theta0, alpha0])
    
    best_params = p0
    best_score = objective(p0)
    
    # Strategy:
    # 1. Try L-BFGS-B on the smooth objective to get close to a local minimum
    # 2. Refine with Powell on the raw max objective
    
    try:
        # Step 1: L-BFGS-B on smooth objective
        # We have plenty of time now, so we can be more aggressive with iterations
        res1 = scipy.optimize.minimize(
            objective_smooth, 
            p0, 
            method='L-BFGS-B', 
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        if time.time() - start_time < time_limit - 10:
            if objective(res1.x) < best_score:
                best_params = res1.x
                best_score = objective(res1.x)
                
        # Step 2: Powell on raw objective starting from best so far
        # Powell is derivative-free and good for max functions
        res2 = scipy.optimize.minimize(
            objective, 
            best_params, 
            method='Powell', 
            options={'maxiter': 500, 'xtol': 1e-12, 'ftol': 1e-14}
        )
        
        if time.time() - start_time < time_limit - 5:
            if res2.fun < best_score:
                best_params = res2.x
                best_score = res2.fun
                
    except Exception:
        pass

    # Check time
    if time.time() - start_time > time_limit - 2:
        u_final, w_final = params_to_nodes_weights(best_params)
        sort_idx = np.argsort(u_final)
        u_sorted = u_final[sort_idx]
        w_sorted = w_final[sort_idx]
        w_norm = w_sorted / np.sum(w_sorted)
        return {"nodes": u_sorted, "weights": w_norm}

    u_final, w_final = params_to_nodes_weights(best_params)
    
    # Ensure strict increasing order of nodes
    sort_idx = np.argsort(u_final)
    u_sorted = u_final[sort_idx]
    w_sorted = w_final[sort_idx]
    
    # Normalize weights to sum to 1 exactly
    w_norm = w_sorted / np.sum(w_sorted)
    
    return {"nodes": u_sorted, "weights": w_norm}
