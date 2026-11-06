"""Search-based approach: generate diverse candidates, evaluate against reference, keep best."""
import numpy as np
import time
import scipy.integrate as si
import scipy.optimize as so

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill

    Omega_m = 0.3
    Omega_L = 0.7

    # Reference integration using high-order Gauss-Legendre (64 points is very accurate)
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(64)
    gl_nodes = (gl_nodes + 1.0) / 2.0
    gl_weights = gl_weights / 2.0

    def _reference_integral(z, w0, wa):
        """Compute integral of 1/E(a) da from a=1 to a=1/(1+z) using 64-point GL."""
        a_vals = 1.0 / (1.0 + z * gl_nodes)
        a_vals = np.clip(a_vals, 1e-15, 1.0)
        log_term = 3.0 * (1.0 + w0 + wa) * np.log(a_vals) - 3.0 * wa * (a_vals - 1.0)
        log_term = np.clip(log_term, -700, 700)
        term_L = Omega_L * np.exp(log_term)
        term_m = Omega_m * (a_vals ** -3)
        E2 = np.maximum(term_m + term_L, 1e-300)
        E = np.sqrt(E2)
        inv_E = 1.0 / E
        return np.dot(gl_weights, inv_E)

    # Expanded and denser test cases to cover likely validation space
    # z: log spaced 0.1 to 10
    # w0: -2.0 to 1.0
    # wa: -2.0 to 2.0
    z_vals = np.logspace(-1, np.log10(10.0), 12)
    w0_vals = np.linspace(-2.0, 1.0, 7)
    wa_vals = np.linspace(-2.0, 2.0, 7)

    test_cases = []
    for z in z_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                test_cases.append((z, w0, wa))
    
    # Convert to numpy arrays for vectorization
    z_arr = np.array([tc[0] for tc in test_cases])
    w0_arr = np.array([tc[1] for tc in test_cases])
    wa_arr = np.array([tc[2] for tc in test_cases])
    n_cases = len(test_cases)

    # Precompute reference values
    ref_vals = np.zeros(n_cases)
    for i in range(n_cases):
        if time.time() > deadline:
            ref_vals[i:] = 1.0
            break
        try:
            ref_vals[i] = _reference_integral(z_arr[i], w0_arr[i], wa_arr[i])
        except Exception:
            ref_vals[i] = 1.0

    def _quadrature_eval_vec(nodes, weights, z_arr, w0_arr, wa_arr):
        """Vectorized evaluation of the 8-point quadrature for all test cases."""
        # z_arr shape (N,), nodes shape (8,)
        # We need to compute for each case i: sum_j weights[j] * f(a_ij)
        # a_ij = 1 / (1 + z_i * nodes_j)
        
        # Broadcasting: z_arr[:, None] * nodes[None, :] -> (N, 8)
        z_nodes = z_arr[:, None] * nodes[None, :]
        a_vals = 1.0 / (1.0 + z_nodes)
        a_vals = np.clip(a_vals, 1e-12, 1.0)
        
        # log_term = 3(1+w0+wa)log(a) - 3wa(a-1)
        # w0_arr, wa_arr shape (N,) -> (N, 1)
        w0_col = w0_arr[:, None]
        wa_col = wa_arr[:, None]
        
        log_term = 3.0 * (1.0 + w0_col + wa_col) * np.log(a_vals) - 3.0 * wa_col * (a_vals - 1.0)
        log_term = np.clip(log_term, -700, 700)
        term_L = Omega_L * np.exp(log_term)
        term_m = Omega_m * (a_vals ** -3)
        E2 = np.maximum(term_m + term_L, 1e-300)
        E = np.sqrt(E2)
        inv_E = 1.0 / E
        
        # Dot product with weights for each row
        # weights shape (8,) -> (1, 8)
        qvals = np.sum(inv_E * weights[None, :], axis=1)
        return qvals

    def _worst_rel_error(nodes, weights):
        """Compute worst-case relative error in percent."""
        try:
            qvals = _quadrature_eval_vec(nodes, weights, z_arr, w0_arr, wa_arr)
            # Avoid division by zero or negative ref
            valid = ref_vals > 1e-10
            if not np.any(valid):
                return 1000.0
            rel_errs = np.abs(qvals[valid] - ref_vals[valid]) / ref_vals[valid] * 100.0
            return np.max(rel_errs)
        except Exception:
            return 1000.0

    def _smoothed_worst_rel_error(nodes, weights, beta=10.0):
        """Compute smoothed worst-case error (Log-Sum-Exp) for gradient-based optimization."""
        try:
            qvals = _quadrature_eval_vec(nodes, weights, z_arr, w0_arr, wa_arr)
            valid = ref_vals > 1e-10
            if not np.any(valid):
                return 1000.0
            rel_errs = np.abs(qvals[valid] - ref_vals[valid]) / ref_vals[valid] * 100.0
            # Log-sum-exp trick for stability
            max_err = np.max(rel_errs)
            if max_err == 0:
                return 0.0
            # Scale by beta to control sharpness
            # sum(exp(beta * err)) -> log(sum(exp(beta * err))) / beta
            # We want to approximate max, so large beta.
            # To avoid overflow, subtract max: log(sum(exp(beta*(err-max)))) + beta*max
            scaled = beta * (rel_errs - max_err)
            # Clip to avoid underflow/overflow in exp
            scaled = np.clip(scaled, -50, 50)
            log_sum_exp = np.log(np.sum(np.exp(scaled)))
            return (log_sum_exp + beta * max_err) / beta
        except Exception:
            return 1000.0

    # Generate diverse candidate rules
    candidates = []

    # 1. Standard Gauss-Legendre on [0,1]
    nodes, weights = np.polynomial.legendre.leggauss(8)
    nodes = (nodes + 1.0) / 2.0
    weights = weights / 2.0
    candidates.append((nodes, weights, "gauss_legendre"))

    # 2. Gauss-Jacobi with various parameters
    from scipy.special import roots_jacobi
    for alpha, beta in [(0, 0), (1, 0), (0, 1), (1, 1), (0.5, 0.5), (2, 1), (1, 2), (0.5, 2), (2, 0.5), (3, 0), (0, 3)]:
        try:
            nodes_j, weights_j = roots_jacobi(8, alpha, beta)
            nodes_j = (nodes_j + 1.0) / 2.0
            weights_j = weights_j / 2.0
            if np.all(np.diff(nodes_j) > 0) and np.all(weights_j > 0):
                candidates.append((nodes_j, weights_j, f"jacobi_{alpha}_{beta}"))
        except Exception:
            pass

    # 3. Beta-weighted Gauss-Legendre
    for beta in [0.5, 1.0, 2.0, 0.3, 0.7, 1.5, 3.0, 0.2, 5.0]:
        try:
            nodes_g, weights_g = np.polynomial.legendre.leggauss(8)
            nodes_g = (nodes_g + 1.0) / 2.0
            nodes_w = nodes_g ** beta
            weights_w = weights_g * beta * (nodes_g ** (beta - 1))
            weights_w = np.abs(weights_w)
            weights_w = weights_w / np.sum(weights_w)
            if np.all(np.diff(nodes_w) > 0):
                candidates.append((nodes_w, weights_w, f"beta_{beta}"))
        except Exception:
            pass

    # 4. Random candidates
    rng = np.random.RandomState(42)
    for seed in range(100):
        rng_local = np.random.RandomState(seed)
        gaps = rng_local.exponential(1.0, size=9)
        gaps = gaps / gaps.sum()
        nodes_r = np.cumsum(gaps)
        nodes_r = np.clip(nodes_r, 1e-6, 1.0 - 1e-6)
        for i in range(1, 8):
            if nodes_r[i] <= nodes_r[i-1]:
                nodes_r[i] = nodes_r[i-1] + 1e-6
        weights_r = np.ones(8)
        weights_r[0] = (nodes_r[1] - nodes_r[0]) / 2
        weights_r[-1] = (nodes_r[-1] - nodes_r[-2]) / 2
        for i in range(1, 7):
            weights_r[i] = (nodes_r[i+1] - nodes_r[i-1]) / 2
        weights_r = np.maximum(weights_r, 1e-12)
        weights_r = weights_r / np.sum(weights_r)
        candidates.append((nodes_r, weights_r, f"random_{seed}"))

    # Evaluate all candidates
    best_worst_err = float('inf')
    best_nodes = candidates[0][0]
    best_weights = candidates[0][1]
    best_name = candidates[0][2]

    for nodes_c, weights_c, name in candidates:
        if time.time() > deadline - 20.0:
            break
        try:
            worst_err = _worst_rel_error(nodes_c, weights_c)
            if worst_err < best_worst_err:
                best_worst_err = worst_err
                best_nodes = nodes_c.copy()
                best_weights = weights_c.copy()
                best_name = name
        except Exception:
            pass

    # Refine the best candidate using optimization
    def _param_to_nodes_weights(params):
        """Parameterize nodes and weights to ensure constraints."""
        # params: [x1, y1..y7, logw1..logw8]
        x1 = params[0]
        y = params[1:8]
        logw = params[8:16]

        x = np.zeros(8)
        x[0] = x1
        for i in range(1, 8):
            x[i] = x[i-1] + np.exp(y[i-1])

        nodes = 1.0 / (1.0 + np.exp(-x))
        nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)

        w = np.exp(logw)
        w = w / np.sum(w)
        return nodes, w

    def _nodes_weights_to_param(nodes, weights):
        """Convert nodes/weights to parameter space."""
        x = np.log(nodes / (1.0 - nodes))
        params = np.zeros(16)
        params[0] = x[0]
        for i in range(1, 8):
            gap = x[i] - x[i-1]
            if gap <= 1e-12:
                gap = 1e-12
            params[1+i-1] = np.log(gap)
        params[8:16] = np.log(np.maximum(weights, 1e-12))
        return params

    # Optimization Phase 1: Powell on Max Error (Robust, derivative-free)
    if time.time() < deadline - 10.0:
        start_params = _nodes_weights_to_param(best_nodes, best_weights)

        def _objective_powell(params):
            nodes, weights = _param_to_nodes_weights(params)
            return _worst_rel_error(nodes, weights)

        try:
            res = so.minimize(_objective_powell, start_params, method='Powell',
                              options={'maxiter': 500, 'ftol': 1e-12, 'xtol': 1e-12})
            if res.fun < best_worst_err:
                best_nodes, best_weights = _param_to_nodes_weights(res.x)
                best_worst_err = res.fun
        except Exception:
            pass

    # Optimization Phase 2: L-BFGS-B on Smoothed Error (Faster local convergence)
    if time.time() < deadline - 2.0:
        start_params = _nodes_weights_to_param(best_nodes, best_weights)

        def _objective_lbfgs(params):
            nodes, weights = _param_to_nodes_weights(params)
            return _smoothed_worst_rel_error(nodes, weights, beta=20.0)

        try:
            res = so.minimize(_objective_lbfgs, start_params, method='L-BFGS-B',
                              options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-8})
            # Check actual max error, not smoothed
            cand_nodes, cand_weights = _param_to_nodes_weights(res.x)
            actual_err = _worst_rel_error(cand_nodes, cand_weights)
            if actual_err < best_worst_err:
                best_nodes, best_weights = cand_nodes, cand_weights
                best_worst_err = actual_err
        except Exception:
            pass

    # Final cleanup: ensure nodes strictly increasing, weights positive and sum to 1
    best_nodes = np.clip(best_nodes, 1e-12, 1.0 - 1e-12)
    for i in range(1, 8):
        if best_nodes[i] <= best_nodes[i-1]:
            best_nodes[i] = best_nodes[i-1] + 1e-12
    best_weights = np.clip(best_weights, 1e-12, None)
    best_weights = best_weights / np.sum(best_weights)

    return {"nodes": best_nodes, "weights": best_weights}
