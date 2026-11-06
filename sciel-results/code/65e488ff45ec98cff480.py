"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

def _E_sq(a, Om, w0, wa):
    """Compute E^2(a) for flat CPL dark energy model."""
    w = w0 + wa * (1.0 - a)
    return Om * a**(-3) + (1.0 - Om) * a**(-3.0 * (1.0 + w))

def _E(a, Om, w0, wa):
    return np.sqrt(_E_sq(a, Om, w0, wa))

def _exact_integral(z, Om, w0, wa):
    """Compute the exact integral using adaptive quadrature (high accuracy)."""
    def integrand(x):
        a = 1.0 / (1.0 + z * x)
        return 1.0 / _E(a, Om, w0, wa)
    val, err = quad(integrand, 0, 1, limit=200, epsabs=1e-12, epsrel=1e-12)
    return (1.0 + z) * z * val

def _quadrature_value(nodes, weights, z, Om, w0, wa):
    """Compute quadrature approximation."""
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E(a_vals, Om, w0, wa)
    return (1.0 + z) * z * np.sum(weights / E_vals)

def _nodes_from_params(params):
    """Convert 7 params to 8 strictly increasing nodes in (0,1) via cumulative exp."""
    raw = np.exp(params)
    nodes = np.concatenate([[0.0], np.cumsum(raw)])
    nodes = nodes / nodes[-1]
    nodes = np.clip(nodes, 1e-8, 1.0 - 1e-8)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-8
    return np.clip(nodes, 1e-8, 1.0 - 1e-8)

def _weights_from_params(params):
    """Convert 8 params to positive weights summing to 1 via softmax."""
    w = np.exp(params)
    return w / w.sum()

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Diverse cosmological models for training
    cosmo_params = []
    for Om in [0.25, 0.28, 0.30, 0.32, 0.35]:
        for w0 in [-1.5, -1.2, -1.0, -0.8, -0.5]:
            for wa in [-0.5, -0.3, 0.0, 0.3, 0.5, 1.0]:
                cosmo_params.append((Om, w0, wa))

    z_list = [0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    # Create a diverse test set
    test_cases = []
    rng = np.random.RandomState(42)
    n_cosmo = len(cosmo_params)
    n_z = len(z_list)
    max_cases = 500
    indices = rng.choice(n_cosmo * n_z, size=min(max_cases, n_cosmo * n_z), replace=False)
    for idx in indices:
        ci = idx % n_cosmo
        zi = idx // n_cosmo
        test_cases.append((z_list[zi], cosmo_params[ci]))

    # Pre-compute exact values
    exact_vals = []
    for z, (Om, w0, wa) in test_cases:
        exact_vals.append(_exact_integral(z, Om, w0, wa))

    # Gauss-Legendre starting point
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    gl_nodes_mapped = (gl_nodes + 1.0) / 2.0
    gl_weights_mapped = gl_weights / 2.0

    # Inverse transform to get initial params
    def _params_from_nodes_weights(nodes, weights):
        # For nodes: use log of differences
        diffs = np.diff(np.concatenate([[0.0], nodes]))
        node_params = np.log(np.maximum(diffs, 1e-10))
        # For weights: use log
        weight_params = np.log(np.maximum(weights, 1e-10))
        return np.concatenate([node_params, weight_params])

    initial_params = _params_from_nodes_weights(gl_nodes_mapped, gl_weights_mapped)

    def _objective(params):
        if time.time() > deadline - 5.0:
            return 1e10
        nodes = _nodes_from_params(params[:7])
        weights = _weights_from_params(params[7:15])

        worst_err = 0.0
        for j, (z, (Om, w0, wa)) in enumerate(test_cases):
            exact = exact_vals[j]
            if exact < 1e-10:
                continue
            quad_val = _quadrature_value(nodes, weights, z, Om, w0, wa)
            rel_err = abs(quad_val - exact) / exact
            if rel_err > worst_err:
                worst_err = rel_err
        return worst_err

    best_obj = _objective(initial_params)
    best_result = initial_params.copy()

    # Primary optimization: L-BFGS-B
    try:
        result = minimize(
            _objective,
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_result = result.x.copy()
    except Exception:
        pass

    # Try a second optimization from a different starting point
    if time.time() < deadline - 15.0:
        rng_init2 = np.random.RandomState(1)
        initial_params2 = np.concatenate([
            rng_init2.normal(0, 1.0, 7),
            np.log(gl_weights_mapped) + rng_init2.normal(0, 0.5, 8)
        ])

        try:
            result2 = minimize(
                _objective,
                initial_params2,
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'maxfev': 200,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )

            if result2.fun < best_obj:
                best_obj = result2.fun
                best_result = result2.x.copy()
        except Exception:
            pass

    # Extract final nodes and weights
    if best_result is not None:
        nodes = _nodes_from_params(best_result[:7])
        weights = _weights_from_params(best_result[7:15])
    else:
        # Fallback to Gauss-Legendre
        nodes = gl_nodes_mapped.copy()
        weights = gl_weights_mapped.copy()

    # Final validation and cleanup
    nodes = np.sort(nodes)
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-6

    weights = np.maximum(weights, 1e-10)
    weights = weights / weights.sum()

    if not np.all(np.diff(nodes) > 0) or not np.all(weights > 0) or abs(weights.sum() - 1.0) > 1e-12:
        return {"nodes": (np.arange(8) + 0.5) / 8.0, "weights": np.full(8, 1.0 / 8.0)}

    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
