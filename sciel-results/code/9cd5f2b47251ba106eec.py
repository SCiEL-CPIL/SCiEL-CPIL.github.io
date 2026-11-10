"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import roots_legendre
from scipy.optimize import minimize, differential_evolution


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    # Gauss-Legendre 8-point nodes and weights on [-1, 1]
    t_gl = np.array([
        -0.9602898564975363, -0.7966664774136267,
        -0.5255324099163290, -0.1834346425004484,
         0.1834346425004484,  0.5255324099163290,
         0.7966664774136267,  0.9602898564975363
    ])
    w_gl = np.array([
        0.1012285362903763, 0.2223810344533745,
        0.3137066458778873, 0.3626837833783620,
        0.3626837833783620, 0.3137066458778873,
        0.2223810344533745, 0.1012285362903763
    ])

    # Map to [0, 1]
    best_nodes = 0.5 * t_gl + 0.5
    best_weights = 0.5 * w_gl
    best_weights = best_weights / np.sum(best_weights)

    # Chebyshev-Gauss nodes as alternative starting point
    # Chebyshev nodes on [-1,1]: cos((2k+1)*pi/(2n)), k=0..n-1
    k = np.arange(8)
    t_cheb = np.cos((2 * k + 1) * np.pi / 16.0)
    t_cheb = -t_cheb  # reverse to be increasing
    w_cheb = np.full(8, np.pi / 8.0)
    # Adjust weights for Chebyshev-Gauss quadrature
    # For Chebyshev-Gauss: w_k = pi/n for all k
    nodes_cheb = 0.5 * t_cheb + 0.5
    weights_cheb = 0.5 * w_cheb
    weights_cheb = weights_cheb / np.sum(weights_cheb)

    # Best solution tracking
    best_nodes = nodes_cheb.copy()
    best_weights = weights_cheb.copy()
    best_obj = np.inf

    # CPL cosmology models: (w0, wa) pairs covering typical range
    cpl_models = [
        (0, 0),       # Lambda CDM (w0=0 is not standard, use -1,0)
        (-1, 0),      # Cosmological constant
        (-1, 1),      # Quintessence-like
        (-0.5, 0),    # Mild dark energy
        (-1, -1),     # Phantom-ish
        (0.5, 0),     # Matter-dominated-ish
        (-1, 0.5),    # Evolving DE
        (-0.8, 0.2),  # Intermediate
        (-1, -0.5),   # Slightly phantom
        (-0.7, -0.3), # Mild evolution
        (-1, 1.5),    # Strong evolution
        (0, 1),       # Matter + evolving
        (-1.2, 0.3),  # Phantom with evolution
        (-0.9, -0.4), # Near -1 with negative evolution
        (-1, 2),      # Very strong evolution
        (-1, -2),     # Strong phantom evolution
        (-0.3, 0.5),  # Mild DE with evolution
        (-1.5, 0),    # Strong phantom
    ]

    # Redshifts to test at - include a wide range
    z_values = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0])

    def _E(a, w0, wa):
        """H(a)/H0 for CPL model. a = 1/(1+z)"""
        Om_m = 0.3
        Om_La = 1.0 - Om_m
        exponent_de = 3.0 * (1.0 + w0 + wa * (1.0 - a))
        term_m = Om_m * a**(-3.0)
        term_de = Om_La * a**(exponent_de)
        E_sq = term_m + term_de
        return np.sqrt(np.maximum(E_sq, 1e-30))

    def _true_integral(w0, wa, z):
        """Compute true integral using dense quadrature"""
        u = np.linspace(0, 1, 20001)
        a = 1.0 / (1.0 + z * u)
        E_vals = _E(a, w0, wa)
        integrand = 1.0 / E_vals
        true_int = np.trapz(integrand, u)
        return (1.0 + z) * z * true_int

    def _quadrature_value(nodes, weights, w0, wa, z):
        """Compute quadrature approximation"""
        a = 1.0 / (1.0 + z * nodes)
        E_vals = _E(a, w0, wa)
        integral_approx = np.sum(weights / E_vals)
        return (1.0 + z) * z * integral_approx

    def _worst_case_error(nodes, weights):
        """Compute worst-case relative error across all models and redshifts"""
        max_rel_err = 0.0
        for w0, wa in cpl_models:
            for z in z_values:
                true_val = _true_integral(w0, wa, z)
                quad_val = _quadrature_value(nodes, weights, w0, wa, z)
                if true_val > 1e-10:
                    rel_err = abs(quad_val - true_val) / true_val
                    if rel_err > max_rel_err:
                        max_rel_err = rel_err
        return max_rel_err

    # Evaluate initial solutions
    gl_err = _worst_case_error(0.5 * t_gl + 0.5, 0.5 * w_gl / np.sum(0.5 * w_gl))
    cheb_err = _worst_case_error(nodes_cheb, weights_cheb)
    
    if gl_err < cheb_err:
        best_nodes = 0.5 * t_gl + 0.5
        best_weights = 0.5 * w_gl / np.sum(0.5 * w_gl)
        best_obj = gl_err
    else:
        best_obj = cheb_err

    # Encoding/decoding for optimization
    def _encode(nodes, weights):
        """Encode nodes and weights into parameter vector using log-spacings and log-weights"""
        spacings = np.diff(np.concatenate([[0], nodes, [1]]))
        # Ensure positive
        spacings = np.maximum(spacings, 1e-15)
        log_spacings = np.log(spacings)
        log_weights = np.log(np.maximum(weights, 1e-15))
        return np.concatenate([log_spacings, log_weights])

    def _decode(x):
        """Decode parameter vector into nodes and weights"""
        log_spacings = x[:9]
        spacings = np.exp(log_spacings)
        total = np.sum(spacings)
        nodes = np.cumsum(spacings) / total
        # Remove the last point (should be 1)
        nodes = nodes[:-1]

        log_weights = x[9:17]
        w_max = np.max(log_weights)
        exp_w = np.exp(log_weights - w_max)
        weights = exp_w / np.sum(exp_w)

        return nodes, weights

    def _objective(x):
        """Objective: minimize worst-case relative error"""
        nodes, weights = _decode(x)

        # Check constraints
        if not np.all(nodes > 1e-15) or not np.all(nodes < 1 - 1e-15):
            return 1e10
        if not np.all(np.diff(nodes) > 1e-15):
            return 1e10
        if not np.all(weights > 0):
            return 1e10
        if abs(np.sum(weights) - 1.0) > 1e-10:
            return 1e10

        return _worst_case_error(nodes, weights)

    # Try optimization from multiple starting points
    starting_points = [
        _encode(best_nodes, best_weights),
        _encode(0.5 * t_gl + 0.5, 0.5 * w_gl / np.sum(0.5 * w_gl)),
    ]

    # Add some perturbed starting points
    rng = np.random.default_rng(42)
    for _ in range(3):
        x_base = starting_points[0].copy()
        x_perturbed = x_base + rng.normal(0, 0.1, size=x_base.shape)
        starting_points.append(x_perturbed)

    # Run L-BFGS-B from each starting point
    for x0 in starting_points:
        if time.time() - start_time > deadline - 20:
            break
        try:
            res = minimize(_objective, x0, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_obj:
                best_obj = res.fun
                best_nodes, best_weights = _decode(res.x)
        except Exception:
            pass

    # Try differential evolution if time permits
    if time.time() - start_time < deadline - 30:
        try:
            bounds = [(-10, 10)] * 9 + [(-10, 10)] * 8
            de_res = differential_evolution(_objective, bounds, maxiter=50,
                                            popsize=15, tol=1e-10, seed=42,
                                            polish=True)
            if de_res.fun < best_obj:
                best_obj = de_res.fun
                best_nodes, best_weights = _decode(de_res.x)
        except Exception:
            pass

    # Final refinement with Nelder-Mead
    if time.time() - start_time < deadline - 10:
        try:
            x_current = _encode(best_nodes, best_weights)
            nm_res = minimize(_objective, x_current, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12})
            if nm_res.fun < best_obj:
                best_obj = nm_res.fun
                best_nodes, best_weights = _decode(nm_res.x)
        except Exception:
            pass

    # Ensure constraints are met
    best_nodes = np.sort(best_nodes)
    best_weights = best_weights / np.sum(best_weights)

    # Final validation
    if not np.all(np.diff(best_nodes) > 0) or not np.all(best_weights > 0) or \
       abs(np.sum(best_weights) - 1.0) > 1e-12:
        # Fallback to Gauss-Legendre
        best_nodes = 0.5 * t_gl + 0.5
        best_weights = 0.5 * w_gl / np.sum(0.5 * w_gl)

    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
