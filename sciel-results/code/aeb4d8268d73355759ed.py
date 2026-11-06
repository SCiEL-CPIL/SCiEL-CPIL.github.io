import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

def construct():
    start_time = time.time()
    
    # 1. Define the CPL model E(z)
    def E_of_z(z, w0, wa, Om=0.3):
        # Om is fixed to 0.3 for simplicity, or we can vary it. 
        # The task says "spatially flat", so Ok=0. Ode = 1-Om.
        # The formula for E(z) in CPL:
        # integral term = (w0+wa)*ln(1+z) - wa*z/(1+z)
        # E(z)^2 = Om*(1+z)^3 + (1-Om)*exp(-3 * integral_term)
        
        z = np.asarray(z)
        if z.ndim == 0:
            z = np.array([z])
            
        # Handle z=0 case carefully for log
        log_term = np.zeros_like(z)
        mask = z > 0
        z_masked = z[mask]
        log_term[mask] = (w0 + wa) * np.log1p(z_masked) - wa * z_masked / (1 + z_masked)
        
        # For z=0, integral is 0, so exp(0)=1.
        # The term is (1-Om) * exp(-3 * 0) = 1-Om.
        
        exp_term = np.exp(-3 * log_term)
        
        E2 = Om * (1 + z)**3 + (1 - Om) * exp_term
        return np.sqrt(E2)

    def exact_dL(z, w0, wa, Om=0.3):
        if z <= 0:
            return 0.0
        # Integrate 1/E(z') from 0 to z
        # Use quad for accuracy
        integral, _ = quad(lambda zp: 1.0 / E_of_z(zp, w0, wa, Om), 0, z, limit=100)
        return (1 + z) * integral

    # 2. Define Design Set
    # w0 in [-1.2, -0.8], wa in [-0.5, 0.5]
    w0_vals = np.linspace(-1.2, -0.8, 5)
    wa_vals = np.linspace(-0.5, 0.5, 5)
    z_vals = np.logspace(-1, 0, 20) # 0.1 to 1.0
    
    design_models = []
    design_zs = []
    design_dLs = []
    
    for w0 in w0_vals:
        for wa in wa_vals:
            for z in z_vals:
                design_models.append((w0, wa))
                design_zs.append(z)
                design_dLs.append(exact_dL(z, w0, wa))
                
    design_models = np.array(design_models)
    design_zs = np.array(design_zs)
    design_dLs = np.array(design_dLs)

    # 3. Define Quadrature Rule Evaluation
    def evaluate_rule(z_nodes, weights, z_target, w0, wa):
        # z_nodes: shape (8,)
        # weights: shape (8,)
        # z_target: scalar or array
        # Formula: z*(1+z) * sum_i w_i / E(a_i) where a_i = 1/(1+z*z_i)
        
        z_target = np.asarray(z_target)
        if z_target.ndim == 0:
            z_target = np.array([z_target])
            
        # a_i = 1 / (1 + z_target * z_nodes)
        # z_target shape (N, 1), z_nodes shape (1, 8)
        z_expanded = z_target[:, np.newaxis]
        nodes_expanded = z_nodes[np.newaxis, :]
        
        a = 1.0 / (1.0 + z_expanded * nodes_expanded)
        
        # Calculate E(a) for all a
        # a is shape (N, 8)
        # w0, wa are scalars
        Om = 0.3
        # Reuse E_of_z logic but vectorized for a
        # Note: E_of_z expects z. Here 'a' is the argument to E.
        # E(a) = H(a)/H0.
        # The formula for E(z) was derived for z. Here the variable is 'a'.
        # So we just plug 'a' into the E(z) formula.
        
        a_flat = a.flatten()
        log_term = np.zeros_like(a_flat)
        mask = a_flat > 0
        # a is always > 0 since z, nodes > 0
        # If z=0, a=1.
        log_term[mask] = (w0 + wa) * np.log1p(a_flat[mask]) - wa * a_flat[mask] / (1 + a_flat[mask])
        exp_term = np.exp(-3 * log_term)
        E2 = Om * (1 + a_flat)**3 + (1 - Om) * exp_term
        E_vals = np.sqrt(E2).reshape(a.shape)
        
        # Sum w_i / E(a_i)
        # weights shape (8,) -> (1, 8)
        w_expanded = weights[np.newaxis, :]
        inv_E = 1.0 / E_vals
        sum_term = np.sum(w_expanded * inv_E, axis=1)
        
        # Result: z*(1+z) * sum_term
        result = z_target * (1 + z_target) * sum_term
        return result

    # 4. Objective Function for Optimization
    # Variables: u (logit nodes), y (log weights)
    def objective(params):
        u = params[:8]
        y = params[8:]
        
        # Nodes: sigmoid(u)
        z_nodes = 1.0 / (1.0 + np.exp(-u))
        # Ensure strictly increasing? Sigmoid is monotonic.
        # But if u is not sorted, nodes might not be sorted.
        # The formula doesn't require sorted nodes, but the task says "Nodes must be strictly increasing".
        # We must enforce this.
        # We can sort z_nodes and corresponding weights.
        # But sorting changes the mapping.
        # Better to constrain u to be increasing?
        # Or just sort at the end and check if it improved?
        # For now, let's assume the optimizer finds a set. 
        # If we sort, we must sort weights too.
        
        # Weights: softmax(y)
        y_max = np.max(y)
        w_unnorm = np.exp(y - y_max)
        weights = w_unnorm / np.sum(w_unnorm)
        
        # Evaluate on design set
        # Vectorize over design points
        # z_nodes shape (8,), weights (8,)
        # design_zs shape (N,), design_models shape (N, 2)
        
        N = len(design_zs)
        z_target = design_zs
        w0_arr = design_models[:, 0]
        wa_arr = design_models[:, 1]
        
        # This loop is slow in Python. Need to vectorize or use a fast approximation.
        # Given N=100, 8 nodes, it's 800 operations per model.
        # We can just loop over models, it's fast enough.
        
        max_rel_err = 0.0
        for i in range(N):
            z_t = z_target[i]
            w0 = w0_arr[i]
            wa = wa_arr[i]
            
            # Compute quadrature
            a = 1.0 / (1.0 + z_t * z_nodes)
            # Compute E(a)
            log_term = (w0 + wa) * np.log1p(a) - wa * a / (1 + a)
            exp_term = np.exp(-3 * log_term)
            E2 = 0.3 * (1 + a)**3 + 0.7 * exp_term
            E_vals = np.sqrt(E2)
            
            sum_term = np.sum(weights / E_vals)
            dL_quad = z_t * (1 + z_t) * sum_term
            
            dL_exact = design_dLs[i]
            
            if dL_exact > 0:
                rel_err = abs(dL_quad - dL_exact) / dL_exact
                if rel_err > max_rel_err:
                    max_rel_err = rel_err
            
        return max_rel_err

    # 5. Optimization
    # Initial guess
    # Nodes: Gauss-Legendre on [0,1]
    x_gl, _ = np.polynomial.legendre.leggauss(8)
    z_init = 0.5 * (x_gl + 1)
    # Convert to logit u
    # u = log(z / (1-z))
    # Avoid 0 or 1
    z_init = np.clip(z_init, 1e-6, 1-1e-6)
    u_init = np.log(z_init / (1 - z_init))
    
    # Weights: uniform
    y_init = np.zeros(8)
    
    params_init = np.concatenate([u_init, y_init])
    
    best_params = params_init
    best_err = objective(params_init)
    
    # Use Nelder-Mead or L-BFGS-B
    # L-BFGS-B is good for smooth problems.
    # Objective is max of absolute errors, so it's non-smooth.
    # Nelder-Mead is more robust to non-smoothness.
    
    try:
        res = minimize(objective, params_init, method='Nelder-Mead', 
                       options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 500, 'adaptive': True})
        if res.fun < best_err:
            best_err = res.fun
            best_params = res.x
    except Exception as e:
        print(f"Optimization failed: {e}")
        
    # Extract best nodes and weights
    u_best = best_params[:8]
    y_best = best_params[8:]
    
    z_nodes = 1.0 / (1.0 + np.exp(-u_best))
    y_max = np.max(y_best)
    w_unnorm = np.exp(y_best - y_max)
    weights = w_unnorm / np.sum(w_unnorm)
    
    # Enforce strictly increasing nodes
    # If nodes are not increasing, sort them and weights
    sort_idx = np.argsort(z_nodes)
    z_nodes = z_nodes[sort_idx]
    weights = weights[sort_idx]
    
    # Ensure nodes are strictly in (0,1)
    z_nodes = np.clip(z_nodes, 1e-10, 1 - 1e-10)
    
    # Normalize weights to sum to 1 exactly (floating point)
    weights = weights / np.sum(weights)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 85:
        # Return a safe fallback if optimization took too long?
        # The current best is likely valid.
        pass

    return {"nodes": z_nodes, "weights": weights}
