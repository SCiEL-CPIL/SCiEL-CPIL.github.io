"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as so

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for safety

    Omega_m = 0.3
    Omega_L = 0.7

    # Define a set of representative CPL cosmologies to optimize against.
    # We assume the validation set is within a reasonable physical range.
    # w0 in [-2, 1], wa in [-2, 2] covers phantom, cosmological constant, and quintessence.
    # We use a grid of parameters to define the "worst case" in the search space.
    w0_vals = np.linspace(-1.5, 0.5, 5)
    wa_vals = np.linspace(-1.0, 1.0, 5)
    params_grid = [(w0, wa) for w0 in w0_vals for wa in wa_vals]
    
    # We optimize for z=1.0 primarily because the integrand shape 
    # f(t) = 1/E(a(t)) is scaled by z. 
    # The relative error of the quadrature integral I(z) = ∫_0^1 (1+z t) f_z(t) dt 
    # is dominated by the approximation of f_z(t).
    # For small z, f_z(t) ≈ f_0(t). For large z, it varies.
    # However, the task says "same nodes and weights... work across all models".
    # It does not say "same nodes for all z". Wait, "The same nodes and weights must work across all models."
    # Does "models" include z? "disjoint public validation cosmologies".
    # Usually, a quadrature rule for ∫_0^1 g(t) dt is fixed.
    # Here g(t) depends on z.
    # If we fix nodes t_i, we are approximating ∫ h_z(t) dt.
    # We should optimize for a range of z as well.
    
    z_vals = [0.5, 1.0, 2.0, 5.0]
    
    # Total set of test cases (z, w0, wa)
    test_cases = []
    for z in z_vals:
        for w0, wa in params_grid:
            test_cases.append((z, w0, wa))
            
    # To keep evaluation fast, we might subsample or use a representative subset for the DE objective,
    # but refine with the full set or a dense set.
    # Let's use a dense set for the final check, but a smaller set for the DE objective to fit in time.
    # 5 * 5 * 4 = 100 cases. 
    # Evaluating 100 cases * 8 nodes is very fast (vectorizable).
    
    def _compute_exact(z, w0, wa):
        """Compute exact integral using high-precision adaptive quadrature."""
        def integrand(t):
            a = 1.0 / (1.0 + z * t)
            # E^2 = Omega_m / a^3 + Omega_L * (a/(1-z))^(3(1+w0)) * (1 - wa*log(a/(1-z)))^3
            # Using the form: E^2 = Omega_m (1+zt)^3 + Omega_L (1+zt)^(3(1+w0)) (1 - wa log(1+zt))^3
            # This form is valid for CPL where a_ref = 1.
            term_m = Omega_m * (1.0 + z * t) ** 3
            log_term = np.log(1.0 + z * t)
            term_L_base = (1.0 + z * t) ** (3.0 * (1.0 + w0))
            term_L_corr = (1.0 - wa * log_term) ** 3
            term_L = Omega_L * term_L_base * term_L_corr
            E2 = term_m + term_L
            return 1.0 / np.sqrt(E2)
        
        # Use quad with high precision
        try:
            from scipy.integrate import quad
            res, _ = quad(integrand, 0.0, 1.0, limit=100, epsabs=1e-12, epsrel=1e-12)
            return res
        except Exception:
            return 1.0

    # Precompute exact values for the test cases
    # This takes some time, so we do it once.
    exact_vals = np.zeros(len(test_cases))
    for i, (z, w0, wa) in enumerate(test_cases):
        if time.time() > deadline - 10:
            break
        exact_vals[i] = _compute_exact(z, w0, wa)
        
    # If we ran out of time, fallback
    if np.any(exact_vals == 0):
         # Fill zeros with a rough estimate or abort? 
         # If we can't compute exacts, we can't optimize well.
         # But quad is usually fast.
         pass

    # Vectorized Quadrature Evaluation
    # nodes: (8,)
    # weights: (8,)
    # z: scalar, w0: scalar, wa: scalar
    def _quad_eval(nodes, weights, z, w0, wa):
        t = nodes
        a_term = 1.0 + z * t
        term_m = Omega_m * a_term ** 3
        log_term = np.log(a_term)
        term_L_base = a_term ** (3.0 * (1.0 + w0))
        term_L_corr = (1.0 - wa * log_term) ** 3
        term_L = Omega_L * term_L_base * term_L_corr
        E2 = term_m + term_L
        E = np.sqrt(E2)
        integrand_vals = 1.0 / E
        return np.dot(weights, integrand_vals)

    def _get_rel_errors(nodes, weights):
        """Compute relative errors for all test cases."""
        errs = np.zeros(len(test_cases))
        for i, (z, w0, wa) in enumerate(test_cases):
            exact = exact_vals[i]
            if exact == 0:
                continue
            quad_val = _quad_eval(nodes, weights, z, w0, wa)
            errs[i] = np.abs((quad_val - exact) / exact)
        return errs

    # Parameterization for Differential Evolution
    # We need 8 nodes in (0,1) strictly increasing and 8 weights > 0 summing to 1.
    # DE works on a flat vector. 
    # Let's use a simple parameterization:
    # 8 nodes: raw values in (0,1). We will sort them? No, DE needs to respect the order or we sort in the objective.
    # If we sort, the landscape is non-smooth at permutations.
    # Better: Parameterize gaps.
    # x_0 in (0, 0.1), x_i in (0, 1) for gaps?
    # Simpler: Use 8 variables for nodes, constrain them to be sorted in the objective function by sorting.
    # This effectively searches over the set of all 8-node rules, regardless of input order.
    # Weights: 8 variables, normalize in objective.
    
    def _param_to_rule(params):
        # params: 16 values
        # First 8: node positions (unsorted)
        # Last 8: weight magnitudes
        raw_nodes = params[:8]
        raw_weights = params[8:]
        
        # Map raw nodes to (0,1) using sigmoid to stay in bounds? 
        # DE bounds can be set directly. Let's use bounds (1e-6, 1-1e-6) for nodes.
        # But we must ensure strict increase.
        nodes = np.sort(raw_nodes)
        # Enforce strict increase by adding tiny epsilon if duplicates
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 1e-9
        
        # Weights: make positive and normalize
        weights = np.abs(raw_weights) + 1e-12
        weights = weights / np.sum(weights)
        
        return nodes, weights

    def _objective(params):
        nodes, weights = _param_to_rule(params)
        errs = _get_rel_errors(nodes, weights)
        return np.max(errs)

    # Bounds for DE
    # Nodes: (1e-5, 1 - 1e-5)
    # Weights: (1e-5, 10) (will be normalized)
    bounds = [(1e-5, 1 - 1e-5)] * 8 + [(1e-5, 10.0)] * 8

    # Initial population: include Gauss-Legendre and Chebyshev nodes
    # DE accepts `init` as a string or array. We use 'latinhypercube' or random.
    # To be safe, we run DE with a reasonable number of iterations.
    
    try:
        # DE is stochastic. We want to maximize the chance of finding a good solution.
        # popsize=15, maxiter=200.
        # 15 * 16 * 200 = 48000 evaluations.
        # Each eval: 100 test cases * 8 nodes. Vectorized numpy.
        # 100 * 8 = 800 ops per case? No, _get_rel_errors loops over 100 cases.
        # Inside _get_rel_errors, _quad_eval is vectorized over 8 nodes.
        # So 100 * (vector op on 8). This is very fast.
        # 48000 * 100 * 8 * (few flops) ~ 10^7 flops. < 0.1s.
        # So we can afford more iterations.
        
        res = so.differential_evolution(
            _objective, 
            bounds, 
            maxiter=300, 
            popsize=20, 
            tol=1e-8, 
            seed=42,
            polish=True, # Uses L-BFGS-B at the end
            init='sobol' # Better coverage than random
        )
        
        best_params = res.x
        best_cost = res.fun
        
    except Exception as e:
        # Fallback to Gauss-Legendre if DE fails
        best_params = np.zeros(16)
        nodes_gl, weights_gl = np.polynomial.legendre.leggauss(8)
        nodes_gl = (nodes_gl + 1.0) / 2.0
        weights_gl = weights_gl / 2.0
        best_params[:8] = nodes_gl
        best_params[8:] = weights_gl
        best_cost = _objective(best_params)

    # Refine with Nelder-Mead on the Minimax objective
    # DE polish might be enough, but NM can squeeze out more.
    if time.time() < deadline - 15:
        try:
            res_nm = so.minimize(
                _objective, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res_nm.fun < best_cost:
                best_params = res_nm.x
                best_cost = res_nm.fun
        except Exception:
            pass

    # Final Refinement: Local search on nodes and weights separately?
    # Or just extract the result.
    
    final_nodes, final_weights = _param_to_rule(best_params)
    
    # Ensure strict constraints
    final_nodes = np.sort(final_nodes)
    for i in range(1, 8):
        if final_nodes[i] <= final_nodes[i-1]:
            final_nodes[i] = final_nodes[i-1] + 1e-12
    final_nodes = np.clip(final_nodes, 1e-12, 1.0 - 1e-12)
    
    final_weights = np.clip(final_weights, 1e-12, None)
    final_weights = final_weights / np.sum(final_weights)
    
    return {"nodes": final_nodes, "weights": final_weights}

# EVOLVE-BLOCK-END
