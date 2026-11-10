import numpy as np
import scipy.integrate
import scipy.optimize
import time

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit

    # Define a grid of cosmological parameters for validation
    w0_vals = np.linspace(-1.2, 0.5, 12)
    w1_vals = np.linspace(-1.5, 1.5, 12)
    
    # Define a grid of redshifts for evaluation
    z_vals = np.concatenate([
        np.linspace(0.02, 0.15, 8),
        np.linspace(0.2, 1.0, 10),
        np.linspace(1.5, 3.0, 10),
        np.linspace(3.5, 6.0, 5)
    ])
    
    # Precompute exact integrals
    exact_ints = []
    model_params = []
    
    for w0 in w0_vals:
        for w1 in w1_vals:
            for z in z_vals:
                def integrand(u, w0=w0, w1=w1, z=z):
                    t = z * u
                    log_E = 3.0 * (1.0 + w0) * np.log1p(t) + 3.0 * w1 * (1.0 - 1.0 / (1.0 + t))
                    return np.exp(-log_E)
                
                try:
                    val, _ = scipy.integrate.quad(integrand, 0, 1, limit=100, epsabs=1e-12, epsrel=1e-12)
                    exact_ints.append(val)
                    model_params.append((w0, w1, z))
                except Exception:
                    us = np.linspace(0, 1, 10001)
                    ts = z * us
                    log_E = 3.0 * (1.0 + w0) * np.log1p(ts) + 3.0 * w1 * (1.0 - 1.0 / (1.0 + ts))
                    vals = np.exp(-log_E)
                    val = np.trapz(vals, us)
                    exact_ints.append(val)
                    model_params.append((w0, w1, z))

    exact_ints = np.array(exact_ints)
    n_models = len(exact_ints)
    
    w0_arr = np.array([p[0] for p in model_params])
    w1_arr = np.array([p[1] for p in model_params])
    z_arr = np.array([p[2] for p in model_params])
    
    def compute_integrand_matrix(nodes):
        """Compute f(t_i) for all models and nodes. Shape: (n_models, 8)"""
        t = z_arr[:, None] * nodes[None, :]
        log_E = 3.0 * (1.0 + w0_arr)[:, None] * np.log1p(t) + 3.0 * w1_arr[:, None] * (1.0 - 1.0 / (1.0 + t))
        f_vals = np.exp(-log_E)
        return f_vals
    
    def solve_weights_linear(nodes):
        """
        Given fixed nodes, solve for optimal weights via linear programming.
        Minimize epsilon subject to:
          -epsilon <= sum_j w_j f_ij - exact_i <= epsilon for all i
          sum_j w_j = 1
          w_j >= 0
        """
        f_mat = compute_integrand_matrix(nodes)  # (n_models, 8)
        n_nodes = 8
        
        # Variables: w_0, ..., w_7, epsilon
        # Minimize: epsilon
        c = np.zeros(n_nodes + 1)
        c[n_nodes] = 1.0
        
        # Inequality constraints
        A_ub = np.zeros((2 * n_models, n_nodes + 1))
        b_ub = np.zeros(2 * n_models)
        
        for i in range(n_models):
            # Row 2i: sum_j f_ij * w_j - epsilon <= exact_i
            A_ub[2*i, :n_nodes] = f_mat[i, :]
            A_ub[2*i, n_nodes] = -1.0
            b_ub[2*i] = exact_ints[i]
            
            # Row 2i+1: -sum_j f_ij * w_j - epsilon <= -exact_i
            A_ub[2*i+1, :n_nodes] = -f_mat[i, :]
            A_ub[2*i+1, n_nodes] = -1.0
            b_ub[2*i+1] = -exact_ints[i]
        
        # Equality constraint: sum_j w_j = 1
        A_eq = np.zeros((1, n_nodes + 1))
        A_eq[0, :n_nodes] = 1.0
        b_eq = np.array([1.0])
        
        # Bounds: w_j >= 0, epsilon >= 0
        bounds = [(0, None)] * n_nodes + [(0, None)]
        
        try:
            result = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                                            bounds=bounds, method='highs')
            if result.success:
                weights = result.x[:n_nodes]
                weights = np.maximum(weights, 0)
                weights = weights / np.sum(weights)
                return weights, result.x[n_nodes]
            else:
                _, gl_w = np.polynomial.legendre.leggauss(8)
                gl_w = 0.5 * gl_w
                return gl_w, None
        except Exception:
            _, gl_w = np.polynomial.legendre.leggauss(8)
            gl_w = 0.5 * gl_w
            return gl_w, None

    def evaluate_nodes(nodes):
        """Evaluate the worst-case relative error for given nodes (with optimal weights)."""
        nodes = np.sort(nodes)
        weights, eps = solve_weights_linear(nodes)
        if weights is None:
            return np.inf
        
        f_mat = compute_integrand_matrix(nodes)
        quad_sums = f_mat @ weights
        rel_errors = np.abs(quad_sums - exact_ints) / exact_ints
        return np.max(rel_errors)
    
    # Parameterization for joint optimization of nodes and weights
    # Nodes: u_i = sigmoid(theta_i)
    # Weights: w_i = softmax(alpha_i)
    
    def params_to_nodes_weights(params):
        theta = params[:8]
        alpha = params[8:16]
        
        u = 1.0 / (1.0 + np.exp(-theta))
        
        alpha_max = np.max(alpha)
        exp_alpha = np.exp(alpha - alpha_max)
        w = exp_alpha / np.sum(exp_alpha)
        
        return u, w

    def objective(params):
        u, w = params_to_nodes_weights(params)
        # Ensure nodes are sorted for consistent evaluation
        sort_idx = np.argsort(u)
        u_sorted = u[sort_idx]
        w_sorted = w[sort_idx]
        
        f_mat = compute_integrand_matrix(u_sorted)
        quad_sums = f_mat @ w_sorted
        rel_errors = np.abs(quad_sums - exact_ints) / exact_ints
        return np.max(rel_errors)

    def objective_smooth(params, temp=0.01):
        u, w = params_to_nodes_weights(params)
        sort_idx = np.argsort(u)
        u_sorted = u[sort_idx]
        w_sorted = w[sort_idx]
        
        f_mat = compute_integrand_matrix(u_sorted)
        quad_sums = f_mat @ w_sorted
        rel_errors = np.abs(quad_sums - exact_ints) / exact_ints
        
        # Soft max approximation
        scaled = rel_errors / temp
        max_scaled = np.max(scaled)
        return temp * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))

    # Initial guess: Gauss-Legendre nodes and weights on [0,1]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    u0 = 0.5 + 0.5 * gl_nodes
    w0_init = 0.5 * gl_weights
    
    u0_safe = np.clip(u0, 1e-9, 1-1e-9)
    theta0 = np.log(u0_safe / (1 - u0_safe))
    
    w0_init_safe = np.clip(w0_init, 1e-9, 1.0)
    alpha0 = np.log(w0_init_safe)
    
    p0 = np.concatenate([theta0, alpha0])
    
    best_params = p0
    best_score = objective(p0)
    
    # Strategy:
    # 1. L-BFGS-B on smooth objective
    # 2. Powell on raw objective
    
    try:
        # Step 1: L-BFGS-B on smooth objective
        res1 = scipy.optimize.minimize(
            lambda p: objective_smooth(p, temp=0.01), 
            p0, 
            method='L-BFGS-B', 
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if time.time() - start_time < time_limit - 10:
            if objective(res1.x) < best_score:
                best_params = res1.x
                best_score = objective(res1.x)
                
        # Step 2: Powell on raw objective
        res2 = scipy.optimize.minimize(
            objective, 
            best_params, 
            method='Powell', 
            options={'maxiter': 200, 'xtol': 1e-9, 'ftol': 1e-12}
        )
        
        if time.time() - start_time < time_limit - 5:
            if res2.fun < best_score:
                best_params = res2.x
                best_score = res2.fun
                
    except Exception:
        pass

    # Extract final nodes and weights from best parameters
    u_final, w_final = params_to_nodes_weights(best_params)
    
    # Ensure strict increasing order of nodes
    sort_idx = np.argsort(u_final)
    u_sorted = u_final[sort_idx]
    w_sorted = w_final[sort_idx]
    
    # Now, for the best nodes, solve for the exact optimal weights via LP
    # This ensures the weights are truly optimal for these nodes
    weights_optimal, _ = solve_weights_linear(u_sorted)
    
    if weights_optimal is not None:
        final_weights = weights_optimal
    else:
        final_weights = w_sorted / np.sum(w_sorted)
    
    # Ensure nodes are strictly increasing in [0,1]
    final_nodes = np.sort(u_sorted)
    for i in range(1, 8):
        if final_nodes[i] <= final_nodes[i-1]:
            final_nodes[i] = final_nodes[i-1] + 1e-10
    final_nodes = np.clip(final_nodes, 1e-10, 1-1e-10)
    
    # Ensure weights are positive and sum to 1
    final_weights = np.maximum(final_weights, 1e-15)
    final_weights = final_weights / np.sum(final_weights)
    
    return {"nodes": final_nodes, "weights": final_weights}
