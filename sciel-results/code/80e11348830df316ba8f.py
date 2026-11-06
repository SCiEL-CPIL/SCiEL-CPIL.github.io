"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as so
import scipy.integrate as si

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # We want to find 8 nodes u_i and 8 weights w_i
    # such that sum(w_i / E(a_i)) approximates int_0^1 dt / E(a(t))
    # for a_i = 1/(1+z*u_i)
    
    # Strategy: Moment Matching against a set of "basis" integrands.
    # We select a set of (w0, wa, z) combinations that represent the extremes of the parameter space.
    # For each combination, we compute the exact integral I_k.
    # We want to choose nodes/weights such that the quadrature Q_k matches I_k as closely as possible.
    # Instead of minimizing max error, we minimize the sum of squared relative errors against these basis integrals.
    # This is a least-squares problem, which is often easier to solve robustly than a min-max problem.
    
    # Basis set: 
    # w0 in [-1.5, -1.0, -0.5, 0.0]
    # wa in [-1.0, 0.0, 1.0]
    # z in [0.1, 0.5, 1.0, 2.0, 5.0]
    # This gives 4*3*5 = 60 basis cases.
    
    w0_vals = [-1.5, -1.0, -0.5, 0.0]
    wa_vals = [-1.0, 0.0, 1.0]
    z_vals = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    basis_cases = []
    for w0 in w0_vals:
        for wa in wa_vals:
            for z in z_vals:
                basis_cases.append((z, w0, wa))
    
    def _compute_exact(z, w0, wa):
        Omega_m = 0.3
        Omega_L = 0.7
        
        def integrand(t):
            a = 1.0 / (1.0 + z * t)
            if a <= 0:
                return 0.0
            log_term = 3.0 * (1.0 + w0 + wa) * np.log(a) - 3.0 * wa * (a - 1.0)
            if log_term < -700:
                term_L = 0.0
            else:
                term_L = Omega_L * np.exp(log_term)
            term_m = Omega_m * (a ** -3)
            E2 = term_m + term_L
            if E2 <= 0:
                return 0.0
            E = np.sqrt(E2)
            return 1.0 / E

        res, err = si.quad(integrand, 0.0, 1.0, limit=100, epsabs=1e-10, epsrel=1e-10)
        return res

    # Precompute exact values for basis cases
    exact_vals = []
    for z, w0, wa in basis_cases:
        if time.time() > deadline:
            break
        try:
            val = _compute_exact(z, w0, wa)
            exact_vals.append(val)
        except Exception:
            exact_vals.append(1.0) # Fallback
    
    if not exact_vals:
        # Fallback to Gauss-Legendre
        nodes, weights = np.polynomial.legendre.leggauss(8)
        nodes = (nodes + 1.0) / 2.0
        weights = weights / 2.0
        return {"nodes": nodes, "weights": weights}

    def _quadrature_vec(nodes, weights, z, w0, wa):
        Omega_m = 0.3
        Omega_L = 0.7
        total = 0.0
        for u, w in zip(nodes, weights):
            a = 1.0 / (1.0 + z * u)
            if a <= 0:
                continue
            log_term = 3.0 * (1.0 + w0 + wa) * np.log(a) - 3.0 * wa * (a - 1.0)
            if log_term < -700:
                term_L = 0.0
            else:
                term_L = Omega_L * np.exp(log_term)
            term_m = Omega_m * (a ** -3)
            E2 = term_m + term_L
            if E2 <= 0:
                continue
            E = np.sqrt(E2)
            total += w / E
        return total

    def _objective_ls(params):
        """
        Objective: Sum of squared relative errors against basis cases.
        params: [u1..u8, w1..w8]
        """
        nodes = params[:8]
        weights = params[8:]
        
        # Check constraints
        if np.any(np.diff(nodes) <= 0):
            return 1e10
        if np.any(weights <= 0):
            return 1e10
        if abs(np.sum(weights) - 1.0) > 1e-6:
            return 1e10
            
        err_sum = 0.0
        for i, (z, w0, wa) in enumerate(basis_cases):
            if time.time() > deadline:
                return 1e10
            exact_val = exact_vals[i]
            if exact_val == 0:
                continue
            quad_val = _quadrature_vec(nodes, weights, z, w0, wa)
            rel_err = (quad_val - exact_val) / exact_val
            err_sum += rel_err * rel_err
            
        return err_sum

    # Initial guess: Gauss-Legendre
    nodes_init, weights_init = np.polynomial.legendre.leggauss(8)
    nodes_init = (nodes_init + 1.0) / 2.0
    weights_init = weights_init / 2.0
    params_init = np.concatenate([nodes_init, weights_init])
    
    # Define constraints for least_squares
    # We need to ensure nodes are in [0,1], increasing, and weights positive summing to 1.
    # Least squares doesn't support bounds directly in the same way, but we can use 'trf' with bounds.
    
    # Bounds:
    # Nodes: 1e-6 to 1-1e-6
    # Weights: 1e-6 to 1
    lower_bounds = np.concatenate([np.full(8, 1e-6), np.full(8, 1e-6)])
    upper_bounds = np.concatenate([np.full(8, 1.0 - 1e-6), np.full(8, 1.0)])
    
    # To enforce sum(weights)=1 and ordering, we can use a penalty in the objective or transform parameters.
    # Transforming parameters is more robust.
    # Let nodes be cumulative sums of positive deltas?
    # u_1 = d_1
    # u_2 = u_1 + d_2
    # ...
    # u_8 = u_7 + d_8
    # Sum u_i < 1.
    # This is complex. Let's stick to bounds and penalty for ordering/sum.
    
    def _residuals(params):
        nodes = params[:8]
        weights = params[8:]
        
        # Penalty for ordering
        order_pen = 0.0
        for i in range(7):
            if nodes[i+1] <= nodes[i]:
                order_pen += 1.0
        
        # Penalty for sum weights
        sum_pen = abs(np.sum(weights) - 1.0)
        
        # Penalty for negative weights
        neg_pen = 0.0
        for w in weights:
            if w <= 0:
                neg_pen += 1.0
        
        # Compute relative errors
        rel_errs = []
        for i, (z, w0, wa) in enumerate(basis_cases):
            exact_val = exact_vals[i]
            if exact_val == 0:
                continue
            quad_val = _quadrature_vec(nodes, weights, z, w0, wa)
            rel_err = (quad_val - exact_val) / exact_val
            rel_errs.append(rel_err)
            
        # Combine penalties and errors
        # Scale penalties to be significant
        final_res = np.concatenate([np.array(rel_errs), [order_pen * 10.0, sum_pen * 10.0, neg_pen * 10.0]])
        return final_res

    best_params = params_init
    best_cost = _objective_ls(params_init)
    
    # Run least_squares
    try:
        res = so.least_squares(_residuals, params_init, method='trf', 
                               bounds=(lower_bounds, upper_bounds),
                               max_nfev=2000, xtol=1e-12, ftol=1e-12)
        if res.cost < best_cost:
            best_params = res.x
            best_cost = res.cost
    except Exception:
        pass
    
    # Refine with a second pass if time permits
    if time.time() < deadline - 5.0:
        try:
            res2 = so.least_squares(_residuals, best_params, method='trf', 
                                    bounds=(lower_bounds, upper_bounds),
                                    max_nfev=1000, xtol=1e-14, ftol=1e-14)
            if res2.cost < best_cost:
                best_params = res2.x
                best_cost = res2.cost
        except Exception:
            pass

    final_nodes = best_params[:8]
    final_weights = best_params[8:]
    
    # Final cleanup: enforce strict increase and positive weights
    # Sort nodes if necessary (though they should be sorted)
    # Ensure they are strictly increasing
    for i in range(1, 8):
        if final_nodes[i] <= final_nodes[i-1]:
            final_nodes[i] = final_nodes[i-1] + 1e-12
    # Clip to [0,1]
    final_nodes = np.clip(final_nodes, 1e-12, 1.0 - 1e-12)
    
    # Ensure weights are positive and sum to 1
    final_weights = np.clip(final_weights, 1e-12, None)
    final_weights = final_weights / np.sum(final_weights)
    
    return {"nodes": final_nodes, "weights": final_weights}

# EVOLVE-BLOCK-END
