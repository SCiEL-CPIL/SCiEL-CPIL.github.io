"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad


def _E_of_a(a, w0, wa):
    """E(a) = H(a)/H0 for flat CPL model. a = 1/(1+z)."""
    # H(z)^2/H0^2 = (1+z)^3 + Omega_m * (1+z)^3 - Omega_m - Omega_de * (1+z)^{3(1+w0+wa*z)}
    # In terms of a: E(a)^2 = a^{-3} + Omega_m * a^{-3} - Omega_m - Omega_de * a^{-3(1+w0+wa*(1/a - 1))}
    # z = 1/a - 1
    # E(a)^2 = a^{-3} + Omega_m * a^{-3} - Omega_m - Omega_de * a^{-3(1+w0+wa*(1/a-1))}
    # Actually: E(a)^2 = a^{-3} + Omega_m * a^{-3} - Omega_m - Omega_de * a^{-3(1+w0+wa*z)}
    # Wait, let me be more careful.
    # E(z)^2 = (1+z)^3 + Omega_m*(1+z)^3 - Omega_m - Omega_de*(1+z)^{3(1+w0+wa*z)}
    # Hmm, that's not right either. Let me recall the CPL parametrization.
    # E(z)^2 = (1+z)^3 + Omega_m*(1+z)^3 - Omega_m - Omega_de*(1+z)^{3(1+w0+wa*z)}
    # Actually the standard form is:
    # E(z)^2 = (1+z)^3 + Omega_m*(1+z)^3 - Omega_m - Omega_de*(1+z)^{3(1+w0+wa*z)}
    # No wait. For flat universe: Omega_m + Omega_de = 1
    # E(z)^2 = Omega_m*(1+z)^3 + Omega_de*(1+z)^{3(1+w0+wa*z)}
    # In terms of a = 1/(1+z):
    # E(a)^2 = Omega_m*a^{-3} + Omega_de*a^{-3(1+w0+wa*(1/a-1))}
    z = 1.0 / a - 1.0
    # Avoid log issues for very small a
    exponent = 3.0 * (1.0 + w0 + wa * z)
    # E(a)^2 = Omega_m * a^{-3} + Omega_de * a^{-exponent}
    # But a^{-exponent} = exp(-exponent * log(a))
    log_a = np.log(a)
    E2 = Omega_m * np.exp(-3.0 * log_a) + Omega_de * np.exp(-exponent * log_a)
    return np.sqrt(E2)


def _integrand(z, w0, wa):
    """Integrand for dL/(c/H0) = (1+z) * integral_0^z dz'/E(z')
    Actually dL/(c/H0) = (1+z) * integral_0^z dz'/E(z')
    Let u = z, then a = 1/(1+u), da = -du/(1+u)^2
    The quadrature is: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    where a_i = 1/(1+z*nodes[i])
    So the integral being approximated is: integral_0^1 du / E(1/(1+z*u)) * ... 
    Wait, let me re-read the task.
    
    "dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i), where a_i=1/(1+z*nodes[i])"
    
    So the integral is: (1+z) * integral_0^1 du / E(1/(1+z*u))
    And we approximate integral_0^1 du / E(1/(1+z*u)) ≈ z * sum_i w_i / E(1/(1+z*x_i))
    Wait that doesn't dimensionally work. Let me re-read.
    
    Actually: dL/(c/H0) = (1+z) * integral_0^z dz'/E(z')
    Change variable: let u = z'/z, so z' = z*u, dz' = z*du
    dL/(c/H0) = (1+z) * z * integral_0^1 du / E(z*u)
    And a = 1/(1+z'), so a(z*u) = 1/(1+z*u)
    E(a) = E(1/(1+z*u))
    
    So the integral to approximate is: integral_0^1 du / E(1/(1+z*u))
    Using quadrature: sum_i w_i / E(1/(1+z*x_i)) where x_i are nodes in [0,1]
    
    The task says: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    So yes, the quadrature approximates integral_0^1 du / E(1/(1+z*u))
    """
    a = 1.0 / (1.0 + z)
    return 1.0 / _E_of_a(a, w0, wa)


def _true_distance(z, w0, wa):
    """Compute true dL/(c/H0) using adaptive quadrature."""
    if z < 1e-15:
        return 0.0
    # dL/(c/H0) = (1+z) * integral_0^z dz'/E(z')
    def integrand(zp):
        a = 1.0 / (1.0 + zp)
        return 1.0 / _E_of_a(a, w0, wa)
    result, _ = quad(integrand, 0, z, limit=200, epsabs=1e-12, epsrel=1e-12)
    return (1.0 + z) * result


def _quadrature_distance(z, nodes, weights, w0, wa):
    """Compute dL/(c/H0) using the quadrature rule."""
    if z < 1e-15:
        return 0.0
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_of_a(a_vals, w0, wa)
    return (1.0 + z) * z * np.sum(weights / E_vals)


def _relative_error(nodes, weights, w0, wa, z_test):
    """Compute worst relative error over test z values."""
    max_err = 0.0
    for z in z_test:
        true_val = _true_distance(z, w0, wa)
        quad_val = _quadrature_distance(z, nodes, weights, w0, wa)
        if abs(true_val) > 1e-15:
            err = abs(quad_val - true_val) / abs(true_val)
            max_err = max(max_err, err)
    return max_err


def _objective(params, w0, wa, z_test):
    """Objective: worst relative error. params = [x1..x8, w1..w8] with constraints handled."""
    x = params[:8]
    w = params[8:]
    # Enforce ordering: x_i strictly increasing in [0,1]
    # Use soft penalty for violations
    penalty = 0.0
    for i in range(7):
        if x[i+1] <= x[i]:
            penalty += 100.0 * (x[i] - x[i+1])
    # Enforce x in [0,1]
    penalty += 100.0 * max(0, -x[0]) + 100.0 * max(0, x[-1] - 1.0)
    # Enforce positive weights
    for wi in w:
        if wi <= 0:
            penalty += 100.0 * (-wi)
    # Enforce sum of weights = 1
    penalty += 100.0 * (np.sum(w) - 1.0)**2
    
    if penalty > 1.0:
        return penalty
    
    return _relative_error(x, w, w0, wa, z_test) + penalty


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Test cosmologies (w0, wa) pairs - typical CPL parameters
    test_cosmos = [
        (0.0, 0.0),    # Lambda CDM
        (-0.5, 0.0),   # w=-0.5
        (-1.0, 0.5),   # CPL with wa=0.5
        (-1.0, -0.5),  # CPL with wa=-0.5
        (-0.9, 0.3),   # Another CPL
        (-1.2, 0.2),   # Phantom-ish
    ]
    
    # Test z values for error evaluation
    z_test = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0])
    
    # Start with a reasonable initial guess: Gauss-Legendre-like nodes
    from numpy.polynomial.legendre import leggauss
    gl_nodes, gl_weights = leggauss(8)
    # Map from [-1,1] to [0,1]
    x0 = 0.5 * (gl_nodes + 1.0)
    w0 = 0.5 * gl_weights
    
    best_params = np.concatenate([x0, w0])
    best_err = _relative_error(x0, w0, 0.0, 0.0, z_test)
    
    # Multi-start optimization over different node families
    node_families = []
    
    # Family 1: Gauss-Legendre mapped
    node_families.append((x0.copy(), w0.copy()))
    
    # Family 2: Beta-distributed nodes
    for alpha, beta in [(0.5, 0.5), (1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (0.5, 2.0), (2.0, 0.5)]:
        # Generate 8 nodes from Beta distribution, sorted
        rng = np.random.default_rng(42 + int(alpha*10 + beta))
        nodes_beta = np.sort(rng.beta(alpha, beta, 8))
        # Ensure strictly in (0,1)
        nodes_beta = np.clip(nodes_beta, 1e-6, 1-1e-6)
        # Equal weights as initial guess
        weights_beta = np.full(8, 0.125)
        node_families.append((nodes_beta.copy(), weights_beta.copy()))
    
    # Family 3: Power-law spaced
    for p in [0.5, 1.0, 2.0, 3.0]:
        t = np.linspace(0, 1, 9)
        nodes_power = t**p
        nodes_power = np.clip(nodes_power, 1e-6, 1-1e-6)
        weights_power = np.full(8, 0.125)
        node_families.append((nodes_power.copy(), weights_power.copy()))
    
    # Family 4: Log-spaced
    nodes_log = np.logspace(-2, 0, 8)
    nodes_log = np.clip(nodes_log, 1e-6, 1-1e-6)
    weights_log = np.full(8, 0.125)
    node_families.append((nodes_log.copy(), weights_log.copy()))
    
    # Family 5: Chebyshev nodes mapped to [0,1]
    k = np.arange(8)
    nodes_cheb = 0.5 * (1 - np.cos(np.pi * (k + 0.5) / 8))
    weights_cheb = np.full(8, 0.125)
    node_families.append((nodes_cheb.copy(), weights_cheb.copy()))
    
    # For each family, optimize weights given nodes using least squares
    # This is the key structural change: fix nodes, solve for optimal weights
    
    for fam_idx, (nodes_fixed, weights_init) in enumerate(node_families):
        if time.time() > deadline - 10:
            break
        
        # For a fixed set of nodes, find optimal weights that minimize worst-case error
        # Use a grid of cosmologies and z values
        # The quadrature value is linear in weights, so we can use a different approach
        
        # For each (w0, wa, z) combination, the quadrature error is:
        # |sum_i w_i * f_i - I| where f_i = 1/E(a_i) and I = integral_0^1 du/E(1+z*u)
        # We want to minimize max over all test cases of |sum_i w_i * f_i - I| / |I|
        
        # This is a linear programming problem in w (with sum(w)=1, w>0)
        # Let's set it up: minimize t subject to
        #   sum_i w_i * f_i - I >= -t * |I|  for all test cases
        #   sum_i w_i * f_i - I <= t * |I|   for all test cases
        #   sum_i w_i = 1
        #   w_i >= 0
        
        # Collect all test cases
        all_cases = []
        for w0_c, wa_c in test_cosmos:
            for z in z_test:
                a_vals = 1.0 / (1.0 + z * nodes_fixed)
                f_vals = 1.0 / _E_of_a(a_vals, w0_c, wa_c)
                I = _true_distance(z, w0_c, wa_c) / ((1.0 + z) * z) if z > 1e-15 else 0.0
                if I > 1e-15:
                    all_cases.append((f_vals, I))
        
        if len(all_cases) == 0:
            continue
        
        # Set up linear program using scipy.optimize.linprog
        # Variables: w_0, ..., w_7, t
        # Minimize: t
        # Constraints:
        #   For each case k: sum_i w_i * f_{k,i} - I_k >= -t * |I_k|
        #   => sum_i w_i * f_{k,i} + t * |I_k| >= I_k
        #   => -sum_i w_i * f_{k,i} - t * |I_k| <= -I_k
        #   And: sum_i w_i * f_{k,i} - I_k <= t * |I_k|
        #   => sum_i w_i * f_{k,i} - t * |I_k| <= I_k
        
        n_cases = len(all_cases)
        # A_ub matrix: shape (2*n_cases, 9)
        # b_ub vector: shape (2*n_cases,)
        A_ub = np.zeros((2 * n_cases, 9))
        b_ub = np.zeros(2 * n_cases)
        
        for k, (f_vals, I_k) in enumerate(all_cases):
            abs_I = abs(I_k)
            # Constraint 1: -sum_i w_i*f_i - t*|I| <= -I
            A_ub[2*k, :8] = -f_vals
            A_ub[2*k, 8] = -abs_I
            b_ub[2*k] = -I_k
            # Constraint 2: sum_i w_i*f_i - t*|I| <= I
            A_ub[2*k+1, :8] = f_vals
            A_ub[2*k+1, 8] = -abs_I
            b_ub[2*k+1] = I_k
        
        # Equality constraint: sum w_i = 1
        A_eq = np.zeros((1, 9))
        A_eq[0, :8] = 1.0
        b_eq = np.array([1.0])
        
        # Bounds: w_i >= 0, t >= 0
        bounds = [(0, None)] * 8 + [(0, None)]
        
        c = np.zeros(9)
        c[8] = 1.0  # minimize t
        
        try:
            result = minimize(
                lambda x: x[8],  # objective is just t
                np.concatenate([weights_init, [0.1]]),
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'eq', 'fun': lambda x: np.sum(x[:8]) - 1.0},
                    {'type': 'ineq', 'fun': lambda x: x[8]},  # t >= 0
                ] + [
                    {'type': 'ineq', 'fun': lambda x, k=k: -np.dot(x[:8], all_cases[k][0]) - x[8]*abs(all_cases[k][1]) + all_cases[k][1]} for k in range(n_cases)
                ] + [
                    {'type': 'ineq', 'fun': lambda x, k=k: np.dot(x[:8], all_cases[k][0]) - x[8]*abs(all_cases[k][1]) - all_cases[k][1]} for k in range(n_cases)
                ],
                options={'maxiter': 50, 'ftol': 1e-8}
            )
            
            if result.success:
                w_opt = result.x[:8]
                # Ensure positivity and normalization
                w_opt = np.maximum(w_opt, 1e-12)
                w_opt = w_opt / np.sum(w_opt)
                
                # Evaluate error
                err = _relative_error(nodes_fixed, w_opt, 0.0, 0.0, z_test)
                # Evaluate over all test cosmologies
                total_err = 0.0
                for w0_c, wa_c in test_cosmos:
                    total_err = max(total_err, _relative_error(nodes_fixed, w_opt, w0_c, wa_c, z_test))
                
                if total_err < best_err:
                    best_err = total_err
                    best_params = np.concatenate([nodes_fixed, w_opt])
                    
        except Exception:
            pass
    
    # Also try direct optimization of both nodes and weights for the best family found
    if time.time() < deadline - 15:
        x_best = best_params[:8]
        w_best = best_params[8:]
        
        def full_objective(params):
            x = params[:8]
            w = params[8:]
            # Soft constraints
            penalty = 0.0
            for i in range(7):
                if x[i+1] <= x[i]:
                    penalty += 1000.0 * (x[i] - x[i+1])
            penalty += 1000.0 * max(0, -x[0]) + 1000.0 * max(0, x[-1] - 1.0)
            for wi in w:
                if wi <= 0:
                    penalty += 1000.0 * (-wi)
            penalty += 1000.0 * (np.sum(w) - 1.0)**2
            
            if penalty > 10.0:
                return penalty
            
            # Compute worst error over all test cosmologies
            max_err = 0.0
            for w0_c, wa_c in test_cosmos:
                max_err = max(max_err, _relative_error(x, w, w0_c, wa_c, z_test))
            return max_err + penalty
        
        # Try a few optimization methods
        for method in ['Nelder-Mead', 'Powell']:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(
                    full_objective,
                    best_params,
                    method=method,
                    options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if res.fun < best_err:
                    # Validate
                    x_new = res.x[:8]
                    w_new = res.x[8:]
                    # Enforce constraints
                    w_new = np.maximum(w_new, 1e-12)
                    w_new = w_new / np.sum(w_new)
                    # Sort nodes
                    order = np.argsort(x_new)
                    x_new = x_new[order]
                    w_new = w_new[order]
                    # Ensure strictly increasing
                    for i in range(7):
                        if x_new[i+1] <= x_new[i]:
                            x_new[i+1] = x_new[i] + 1e-6
                    x_new = np.clip(x_new, 1e-6, 1-1e-6)
                    
                    # Re-evaluate
                    total_err = 0.0
                    for w0_c, wa_c in test_cosmos:
                        total_err = max(total_err, _relative_error(x_new, w_new, w0_c, wa_c, z_test))
                    
                    if total_err < best_err:
                        best_err = total_err
                        best_params = np.concatenate([x_new, w_new])
            except Exception:
                pass
    
    # Final extraction
    nodes = best_params[:8]
    weights = best_params[8:]
    
    # Ensure validity
    nodes = np.clip(nodes, 1e-6, 1 - 1e-6)
    # Ensure strictly increasing
    for i in range(7):
        if nodes[i+1] <= nodes[i]:
            nodes[i+1] = nodes[i] + 1e-6
    nodes = np.clip(nodes, 1e-6, 1 - 1e-6)
    
    weights = np.maximum(weights, 1e-12)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
