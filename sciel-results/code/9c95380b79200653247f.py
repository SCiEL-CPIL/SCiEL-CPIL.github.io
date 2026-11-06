"""Improved approach: Multi-start Nelder-Mead optimization on a representative test set."""
import numpy as np
import time
import scipy.optimize as so

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin

    Omega_m = 0.3
    Omega_L = 0.7

    # Reference integration using high-order Gauss-Legendre (64 points)
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(64)
    gl_nodes = (gl_nodes + 1.0) / 2.0
    gl_weights = gl_weights / 2.0

    # Define a representative set of test cases for optimization.
    # We want to cover the likely validation space: z in [0.1, 10], w0 in [-2, 1], wa in [-2, 2].
    # We use a smaller grid here to speed up the optimization objective.
    z_vals = np.logspace(-1, np.log10(10.0), 8)
    w0_vals = np.linspace(-2.0, 1.0, 5)
    wa_vals = np.linspace(-2.0, 2.0, 5)

    opt_test_cases = []
    for z in z_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                opt_test_cases.append((z, w0, wa))

    z_opt = np.array([tc[0] for tc in opt_test_cases])
    w0_opt = np.array([tc[1] for tc in opt_test_cases])
    wa_opt = np.array([tc[2] for tc in opt_test_cases])

    # Precompute reference values for the optimization set
    a_vals_ref = 1.0 / (1.0 + z_opt[:, None] * gl_nodes[None, :])
    a_vals_ref = np.clip(a_vals_ref, 1e-15, 1.0)
    log_term_ref = 3.0 * (1.0 + w0_opt[:, None] + wa_opt[:, None]) * np.log(a_vals_ref) - 3.0 * wa_opt[:, None] * (a_vals_ref - 1.0)
    log_term_ref = np.clip(log_term_ref, -700, 700)
    term_L_ref = Omega_L * np.exp(log_term_ref)
    term_m_ref = Omega_m * (a_vals_ref ** -3)
    E2_ref = np.maximum(term_m_ref + term_L_ref, 1e-300)
    E_ref = np.sqrt(E2_ref)
    inv_E_ref = 1.0 / E_ref
    ref_vals_opt = np.dot(inv_E_ref, gl_weights)

    def _eval_worst_err(nodes, weights):
        """Vectorized worst relative error on the optimization set."""
        a_vals = 1.0 / (1.0 + z_opt[:, None] * nodes[None, :])
        a_vals = np.clip(a_vals, 1e-12, 1.0)
        log_term = 3.0 * (1.0 + w0_opt[:, None] + wa_opt[:, None]) * np.log(a_vals) - 3.0 * wa_opt[:, None] * (a_vals - 1.0)
        log_term = np.clip(log_term, -700, 700)
        term_L = Omega_L * np.exp(log_term)
        term_m = Omega_m * (a_vals ** -3)
        E2 = np.maximum(term_m + term_L, 1e-300)
        E = np.sqrt(E2)
        inv_E = 1.0 / E
        q_vals = np.dot(inv_E, weights)
        rel_errs = np.abs(q_vals - ref_vals_opt) / ref_vals_opt * 100.0
        return np.max(rel_errs)

    # Generate initial candidates
    candidates = []

    # 1. Standard Gauss-Legendre
    nodes, weights = np.polynomial.legendre.leggauss(8)
    nodes = (nodes + 1.0) / 2.0
    weights = weights / 2.0
    candidates.append((nodes, weights))

    # 2. Gauss-Jacobi variants
    from scipy.special import roots_jacobi
    for alpha, beta in [(0, 0), (1, 0), (0, 1), (1, 1), (0.5, 0.5), (2, 1), (1, 2), (0.5, 2), (2, 0.5), (3, 0), (0, 3)]:
        try:
            nodes_j, weights_j = roots_jacobi(8, alpha, beta)
            nodes_j = (nodes_j + 1.0) / 2.0
            weights_j = weights_j / 2.0
            if np.all(np.diff(nodes_j) > 0) and np.all(weights_j > 0):
                candidates.append((nodes_j, weights_j))
        except Exception:
            pass

    # 3. Clenshaw-Curtis
    N = 8
    theta = np.pi * np.arange(N) / (N - 1)
    nodes_cc = 0.5 * (1.0 + np.cos(theta))
    nodes_cc = nodes_cc[::-1]
    weights_cc = np.zeros(N)
    weights_cc[0] = 1.0 / (N - 1) ** 2
    weights_cc[-1] = 1.0 / (N - 1) ** 2
    for j in range(1, N - 1):
        s = 0.0
        for k in range(2, N - 1, 2):
            s += 2.0 * np.cos(k * theta[j]) / (k ** 2 - 1)
        weights_cc[j] = s / (N - 1) ** 2
    weights_cc = weights_cc[::-1]
    weights_cc = np.abs(weights_cc)
    weights_cc = weights_cc / np.sum(weights_cc)
    candidates.append((nodes_cc, weights_cc))

    # 4. Random smooth nodes
    rng = np.random.RandomState(42)
    for seed in range(10):
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
        candidates.append((nodes_r, weights_r))

    # Parameterization: 
    # nodes: 8 values in (0,1), strictly increasing.
    # We use logit transform for nodes to handle bounds and ordering?
    # Or just use the raw values and rely on Nelder-Mead's ability to handle bounds if we penalize?
    # Better: Transform parameters to ensure validity.
    # p[0:8] -> logit(nodes), but they must be increasing.
    # Let p[0] = logit(n0)
    # p[i] = log(n_i - n_{i-1}) for i=1..7
    # p[8:16] -> log(weights)
    
    def _params_to_nodes_weights(params):
        x0 = params[0]
        gaps = np.exp(params[1:8])
        x = np.zeros(8)
        x[0] = x0
        for i in range(1, 8):
            x[i] = x[i-1] + gaps[i-1]
        # Map x to (0,1) via sigmoid? No, x is in logit space?
        # If x is logit, nodes = 1/(1+exp(-x)).
        # But we need them to be in [0,1].
        # Let's use a simpler mapping:
        # nodes = 1 / (1 + exp(-x))
        # This ensures (0,1).
        nodes = 1.0 / (1.0 + np.exp(-x))
        # Clip to avoid 0 or 1
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        
        w = np.exp(params[8:16])
        w = w / np.sum(w)
        return nodes, w

    def _nodes_weights_to_params(nodes, weights):
        params = np.zeros(16)
        # Inverse sigmoid
        x = np.log(nodes / (1.0 - nodes))
        params[0] = x[0]
        for i in range(1, 8):
            gap = x[i] - x[i-1]
            if gap <= 1e-12: gap = 1e-12
            params[i] = np.log(gap)
        params[8:16] = np.log(np.maximum(weights, 1e-12))
        return params

    def _objective(params):
        nodes, weights = _params_to_nodes_weights(params)
        return _eval_worst_err(nodes, weights)

    best_nodes = candidates[0][0].copy()
    best_weights = candidates[0][1].copy()
    best_err = _eval_worst_err(best_nodes, best_weights)

    # Multi-start optimization
    for nodes_c, weights_c in candidates:
        if time.time() > deadline:
            break
        
        try:
            params_c = _nodes_weights_to_params(nodes_c, weights_c)
            # Use Nelder-Mead for non-smooth max error
            res = so.minimize(_objective, params_c, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.fun < best_err:
                best_nodes, best_weights = _params_to_nodes_weights(res.x)
                best_err = res.fun
        except Exception:
            pass

    # Final cleanup
    best_nodes = np.clip(best_nodes, 1e-12, 1.0 - 1e-12)
    for i in range(1, 8):
        if best_nodes[i] <= best_nodes[i-1]:
            best_nodes[i] = best_nodes[i-1] + 1e-12
    best_weights = np.clip(best_weights, 1e-12, None)
    best_weights = best_weights / np.sum(best_weights)

    return {"nodes": best_nodes, "weights": best_weights}
