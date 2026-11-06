import numpy as np
import time
import scipy.optimize as opt
from scipy.integrate import quad

def _E_cpl(z, w0, w1, Om):
    one_plus_z = 1.0 + z
    exp_out = 3.0 * (1.0 + w0) / 2.0
    exp_in = 3.0 * (1.0 - w0 - w1)
    term_m = Om * one_plus_z ** exp_in
    term_de = 1.0 - Om
    return (one_plus_z ** exp_out) * np.sqrt(term_m + term_de)

def _exact_integral(z, w0, w1, Om):
    f = lambda zp: 1.0 / _E_cpl(zp, w0, w1, Om)
    result, _ = quad(f, 0, z, limit=100)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    models = [
        (-1.0, 0.0, 0.3),
        (-1.2, 0.0, 0.3),
        (-0.8, 0.0, 0.3),
        (-1.0, -0.5, 0.3),
        (-1.0, 0.5, 0.3),
        (-0.8, -1.0, 0.3),
        (-1.2, 1.0, 0.3),
        (-1.0, 0.0, 0.2),
        (-1.0, 0.0, 0.35),
        (-0.9, 0.5, 0.25),
    ]
    z_test = np.array([0.5, 1.0, 2.0, 5.0])

    # Precompute exact integrals
    exact_vals = np.zeros((len(models), len(z_test)))
    for i, (w0, w1, Om) in enumerate(models):
        for j, z in enumerate(z_test):
            exact_vals[i, j] = _exact_integral(z, w0, w1, Om)

    # Precompute z * nodes for each model and z
    # nodes shape: (8,), z_test shape: (4,)
    # We'll vectorize over models and z_test

    def _compute_quad_errors(params):
        """
        params: 16-dim vector. First 8 are logit-space node params, last 8 are log-weight params.
        Returns: array of relative errors, shape (n_models, n_z)
        """
        node_params = params[:8]
        weight_params = params[8:]

        # Transform to nodes in (0,1) via sigmoid, then sort
        nodes_raw = 1.0 / (1.0 + np.exp(-node_params))
        # Ensure strict increasing: add small linear trend to guarantee ordering
        # Actually, sigmoid of sorted inputs gives sorted outputs if inputs are sorted.
        # We'll sort the params to ensure nodes are sorted.
        sort_idx = np.argsort(node_params)
        nodes_raw = nodes_raw[sort_idx]
        # Clamp to avoid exact 0 or 1
        nodes_raw = np.clip(nodes_raw, 1e-8, 1.0 - 1e-8)

        # Transform to weights via softmax
        weights_raw = np.exp(weight_params - np.max(weight_params))
        weights_raw = weights_raw / np.sum(weights_raw)
        # Reorder weights to match sorted nodes
        weights_raw = weights_raw[sort_idx]

        # Compute quadrature values for all models and z
        # nodes_raw: (8,), z_test: (4,)
        # For each z in z_test, z_vals = z * nodes_raw
        # E_vals for each model
        errors = np.zeros((len(models), len(z_test)))
        for i, (w0, w1, Om) in enumerate(models):
            for j, z in enumerate(z_test):
                z_vals = z * nodes_raw
                E_vals = _E_cpl(z_vals, w0, w1, Om)
                quad_val = z * np.dot(weights_raw, 1.0 / E_vals)
                exact_val = exact_vals[i, j]
                errors[i, j] = abs(quad_val - exact_val) / abs(exact_val)
        return errors

    def _smooth_max_error(params):
        """
        Smooth approximation of max error using log-sum-exp.
        """
        errors = _compute_quad_errors(params)
        # Use log-sum-exp for smooth max
        flat_errors = errors.ravel()
        # Scale to avoid overflow
        scale = 100.0  # temperature for soft-max
        max_e = np.max(flat_errors)
        if max_e < 1e-15:
            return 0.0
        # log-sum-exp: max_e + log(sum(exp((e - max_e)/scale)))
        log_sum = max_e / scale + np.log(np.sum(np.exp((flat_errors - max_e) / scale)))
        return scale * log_sum

    # Initialize from Gauss-Legendre
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes_gl = (x_gl + 1.0) / 2.0
    weights_gl = w_gl / 2.0
    weights_gl = weights_gl / np.sum(weights_gl)

    # Inverse transform: nodes -> logit space
    node_params_init = np.log(nodes_gl / (1.0 - nodes_gl))
    weight_params_init = np.log(weights_gl) - np.log(np.sum(weights_gl))

    best_params = np.concatenate([node_params_init, weight_params_init])
    best_err = _smooth_max_error(best_params)

    # Optimize using L-BFGS-B
    try:
        res = opt.minimize(_smooth_max_error, best_params, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
        if res.fun < best_err:
            best_params = res.x
            best_err = res.fun
    except:
        pass

    # Second pass with Nelder-Mead for fine-tuning
    if time.time() < deadline - 10:
        try:
            res2 = opt.minimize(_smooth_max_error, best_params, method='Nelder-Mead',
                                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12})
            if res2.fun < best_err:
                best_params = res2.x
                best_err = res2.fun
        except:
            pass

    # Extract final nodes and weights
    node_params = best_params[:8]
    weight_params = best_params[8:]

    sort_idx = np.argsort(node_params)
    nodes_raw = 1.0 / (1.0 + np.exp(-node_params))
    nodes_raw = nodes_raw[sort_idx]
    nodes_raw = np.clip(nodes_raw, 1e-10, 1.0 - 1e-10)

    weights_raw = np.exp(weight_params - np.max(weight_params))
    weights_raw = weights_raw / np.sum(weights_raw)
    weights_raw = weights_raw[sort_idx]

    # Ensure strictly increasing
    for i in range(1, 8):
        if nodes_raw[i] <= nodes_raw[i-1]:
            nodes_raw[i] = nodes_raw[i-1] + 1e-10

    # Final normalization
    weights_raw = weights_raw / np.sum(weights_raw)

    return {"nodes": nodes_raw, "weights": weights_raw}
