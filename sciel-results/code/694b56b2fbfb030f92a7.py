"""Chebyshev spectral collocation quadrature design for luminosity distance."""
import numpy as np
import time
import scipy.optimize as so

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    Omega_m = 0.3
    Omega_L = 0.7

    # High-accuracy reference using 128-point Gauss-Legendre
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(128)
    gl_nodes = (gl_nodes + 1.0) / 2.0
    gl_weights = gl_weights / 2.0

    # Diverse test set
    z_vals = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
    w0_vals = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
    wa_vals = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])

    test_cases = []
    for z in z_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                test_cases.append((z, w0, wa))

    n_tests = len(test_cases)

    # Precompute reference values
    ref_vals = np.zeros(n_tests)
    for i, (z, w0, wa) in enumerate(test_cases):
        if time.time() > deadline - 60.0:
            break
        a_vals = 1.0 / (1.0 + z * gl_nodes)
        a_vals = np.clip(a_vals, 1e-15, 1.0)
        log_term = 3.0 * (1.0 + w0 + wa) * np.log(a_vals) - 3.0 * wa * (a_vals - 1.0)
        log_term = np.clip(log_term, -700, 700)
        term_L = Omega_L * np.exp(log_term)
        term_m = Omega_m * (a_vals ** -3)
        E2 = np.maximum(term_m + term_L, 1e-300)
        E = np.sqrt(E2)
        inv_E = 1.0 / E
        ref_vals[i] = np.dot(gl_weights, inv_E)

    # Precompute test case arrays
    z_arr = np.array([tc[0] for tc in test_cases])
    w0_arr = np.array([tc[1] for tc in test_cases])
    wa_arr = np.array([tc[2] for tc in test_cases])

    def _compute_inv_E_matrix(nodes):
        """Compute 1/E(a) matrix for all test cases at given nodes."""
        a_vals = 1.0 / (1.0 + z_arr[:, None] * nodes[None, :])
        a_vals = np.clip(a_vals, 1e-12, 1.0)
        log_terms = 3.0 * (1.0 + w0_arr[:, None] + wa_arr[:, None]) * np.log(a_vals) \
                   - 3.0 * wa_arr[:, None] * (a_vals - 1.0)
        log_terms = np.clip(log_terms, -700, 700)
        term_L = Omega_L * np.exp(log_terms)
        term_m = Omega_m * (a_vals ** -3)
        E2 = np.maximum(term_m + term_L, 1e-300)
        E = np.sqrt(E2)
        return 1.0 / E

    def _worst_rel_error(nodes, weights):
        """Worst-case relative error in percent."""
        inv_E = _compute_inv_E_matrix(nodes)
        qvals = inv_E @ weights
        rel_errs = np.abs(qvals - ref_vals) / ref_vals * 100.0
        return np.max(rel_errs)

    # Start with Gauss-Legendre as initial guess
    gl8_nodes, gl8_weights = np.polynomial.legendre.leggauss(8)
    gl8_nodes = (gl8_nodes + 1.0) / 2.0
    gl8_weights = gl8_weights / 2.0

    best_nodes = gl8_nodes.copy()
    best_weights = gl8_weights.copy()
    best_err = _worst_rel_error(best_nodes, best_weights)

    # === CHEBYSHEV SPECTRAL COLLOCATION APPROACH ===
    # Key insight: The integrand 1/E(a) for CPL dark energy models has a smooth
    # structure that can be well-approximated by Chebyshev polynomials. We use
    # Chebyshev nodes (which minimize the Lebesgue constant for interpolation)
    # and solve a linear system to find weights that exactly integrate the
    # first N Chebyshev polynomials. This gives near-minimax properties.
    
    def _chebyshev_nodes(n):
        """Chebyshev-Gauss-Lobatto nodes on [0,1] (increasing order)."""
        k = np.arange(n)
        # Standard Chebyshev nodes on [-1,1]: cos(pi*k/(n-1)) for k=0..n-1
        # Map to [0,1]: (1 + cos(pi*k/(n-1)))/2, then reverse for increasing
        nodes = 0.5 * (1.0 + np.cos(np.pi * k / (n - 1)))
        return nodes[::-1]  # Reverse for increasing order

    def _chebyshev_weights(n):
        """Chebyshev quadrature weights on [0,1] for exact integration of Cheb polys."""
        # For Chebyshev-Gauss-Lobatto, the weights for integrating f(x) dx
        # where x is in [-1,1] are: w_k = (2/(n-1)) * c_k, with c_0=c_{n-1}=0.5
        # After mapping to [0,1], multiply by 1/2
        k = np.arange(n)
        w = np.ones(n) * 2.0 / (n - 1)
        w[0] *= 0.5
        w[-1] *= 0.5
        w *= 0.5  # Mapping from [-1,1] to [0,1]
        return w

    def _build_chebyshev_matrix(nodes, n_terms):
        """Build Chebyshev polynomial values at given nodes.
        T_k(x) = cos(k * arccos(x)) where x is mapped to [-1,1]."""
        # Map nodes from [0,1] to [-1,1]
        x = 2.0 * nodes - 1.0
        x = np.clip(x, -1.0, 1.0)
        theta = np.arccos(x)
        
        # Build matrix of T_k(x) for k = 0, ..., n_terms-1
        k_vals = np.arange(n_terms)
        # T_k(x) = cos(k*theta)
        matrix = np.cos(np.outer(k_vals, theta))
        return matrix

    def _find_chebyshev_weights(nodes, n_terms=8):
        """Find weights that exactly integrate the first n_terms Chebyshev polynomials.
        
        We want: sum_i w_i * T_k(x_i) = integral_0^1 T_k(x) dx for k=0,...,n_terms-1
        
        The integral of T_k(x) over [0,1] can be computed analytically.
        """
        # Map to [-1,1]: integral_0^1 T_k(x) dx = 0.5 * integral_{-1}^1 T_k(u) du
        # where u = 2x-1
        # integral_{-1}^1 T_k(u) du = 0 if k is odd, = 2/(1-k^2) if k is even and k>0, = 2 if k=0
        
        rhs = np.zeros(n_terms)
        for k in range(n_terms):
            if k == 0:
                rhs[k] = 1.0  # integral_0^1 1 dx = 1
            elif k % 2 == 1:
                rhs[k] = 0.0
            else:
                # integral_0^1 T_k(x) dx = 0.5 * integral_{-1}^1 T_k(u) du
                # integral_{-1}^1 T_k(u) du = 2/(1-k^2) for even k > 0
                rhs[k] = 1.0 / (1.0 - k * k)
        
        # Build Chebyshev matrix
        cheb_mat = _build_chebyshev_matrix(nodes, n_terms)
        
        # Solve: cheb_mat @ w = rhs
        try:
            w, _, _, _ = np.linalg.lstsq(cheb_mat, rhs, rcond=None)
            return w
        except:
            return None

    # Try Chebyshev nodes with various numbers of terms
    for n_terms in range(5, 9):
        if time.time() > deadline - 30.0:
            break
        try:
            nodes_cheb = _chebyshev_nodes(8)
            nodes_cheb = np.clip(nodes_cheb, 1e-6, 1.0 - 1e-6)
            
            # Ensure strictly increasing
            for i in range(1, 8):
                if nodes_cheb[i] <= nodes_cheb[i-1]:
                    nodes_cheb[i] = nodes_cheb[i-1] + 1e-6
            nodes_cheb = np.clip(nodes_cheb, 1e-6, 1.0 - 1e-6)
            
            w_cheb = _find_chebyshev_weights(nodes_cheb, n_terms)
            
            if w_cheb is not None and np.all(w_cheb > 0):
                w_cheb = w_cheb / np.sum(w_cheb)
                err = _worst_rel_error(nodes_cheb, w_cheb)
                if err < best_err:
                    best_err = err
                    best_nodes = nodes_cheb.copy()
                    best_weights = w_cheb.copy()
        except Exception:
            pass

    # Also try: optimize nodes using Chebyshev-based objective
    # Use the fact that Chebyshev nodes minimize the Lebesgue constant,
    # so perturb them slightly and re-optimize weights
    
    def _param_to_nodes_weights(params):
        """Convert parameter vector to nodes and weights."""
        log_gaps = params[:7]
        log_first = params[7]
        log_weights = params[8:16]
        
        gap_sizes = np.exp(log_gaps)
        x = np.zeros(8)
        x[0] = log_first
        for i in range(7):
            x[i+1] = x[i] + gap_sizes[i]
        
        nodes = 1.0 / (1.0 + np.exp(-x))
        nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 1e-10
        
        w = np.exp(log_weights)
        w = w / np.sum(w)
        
        return nodes, w

    def _soft_max_rel_error(nodes, weights, beta=50.0):
        """Smooth approximation of worst-case relative error."""
        inv_E = _compute_inv_E_matrix(nodes)
        qvals = inv_E @ weights
        rel_errs = np.abs(qvals - ref_vals) / ref_vals * 100.0
        max_err = np.max(rel_errs)
        if max_err < 1e-15:
            return 0.0
        scaled = beta * (rel_errs - max_err)
        log_sum_exp = max_err + (1.0 / beta) * np.log(np.sum(np.exp(scaled)))
        return log_sum_exp

    # Phase 2: Local refinement from Chebyshev solution
    if time.time() < deadline - 15.0:
        try:
            # Convert best to parameter space
            x_best = np.log(best_nodes / (1.0 - best_nodes))
            params_init = np.zeros(16)
            params_init[7] = x_best[0]
            for i in range(7):
                gap = x_best[i+1] - x_best[i]
                if gap < 1e-12:
                    gap = 1e-12
                params_init[i] = np.log(gap)
            params_init[8:16] = np.log(np.maximum(best_weights, 1e-12))
            
            bounds = [(-4.0, 4.0)] * 7 + [(-10.0, 10.0)] + [(-10.0, 10.0)] * 8
            
            def _objective_smooth(params):
                nodes, weights = _param_to_nodes_weights(params)
                return _soft_max_rel_error(nodes, weights, beta=50.0)
            
            res = so.minimize(
                _objective_smooth,
                params_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-11},
            )
            
            if res.fun < best_err * 1.01:
                nodes_smooth, weights_smooth = _param_to_nodes_weights(res.x)
                err_smooth = _worst_rel_error(nodes_smooth, weights_smooth)
                if err_smooth < best_err:
                    best_err = err_smooth
                    best_nodes = nodes_smooth.copy()
                    best_weights = weights_smooth.copy()
        except Exception:
            pass

    # Phase 3: Final weight optimization
    if time.time() < deadline - 2.0:
        try:
            inv_E = _compute_inv_E_matrix(best_nodes)
            w_final, _, _, _ = np.linalg.lstsq(inv_E, ref_vals, rcond=None)
            if np.all(w_final > 0):
                w_final = w_final / np.sum(w_final)
                err_final = _worst_rel_error(best_nodes, w_final)
                if err_final < best_err:
                    best_err = err_final
                    best_weights = w_final.copy()
        except Exception:
            pass

    # Final cleanup
    best_nodes = np.clip(best_nodes, 1e-10, 1.0 - 1e-10)
    for i in range(1, 8):
        if best_nodes[i] <= best_nodes[i-1]:
            best_nodes[i] = best_nodes[i-1] + 1e-10
    best_weights = np.clip(best_weights, 1e-12, None)
    best_weights = best_weights / np.sum(best_weights)

    return {"nodes": best_nodes, "weights": best_weights}
