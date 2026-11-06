"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import quad
from scipy.optimize import differential_evolution

def construct():
    """
    Discover an eight-node positive quadrature rule for dimensionless luminosity distance
    in spatially flat, time-varying dark-energy models.
    
    Strategy: Use differential evolution (global optimizer) over bounded (nodes, weights)
    space with a large diverse CPL model set to minimize worst-case relative error.
    """
    start_time = time.time()
    deadline = start_time + 85.0

    def E_cpl(z, w0, wa, Omega_m=0.3):
        Omega_DE = 1.0 - Omega_m
        exp_arg = 3.0 * ((w0 + wa) * z - wa * np.log1p(z))
        exp_arg = np.clip(exp_arg, -700, 700)
        return np.sqrt(Omega_m * (1.0 + z)**3 + Omega_DE * np.exp(exp_arg))

    # Large diverse set of CPL models
    test_models = [
        (-1.0, 0.0), (-1.0, 0.3), (-1.0, 0.6), (-1.0, -0.3),
        (-0.5, 0.0), (-0.5, 0.4), (-0.5, -0.2),
        (-1.5, 0.0), (-1.5, 0.3), (-1.5, -0.3),
        (-0.8, 0.3), (-0.8, -0.3),
        (-1.2, 0.4), (-1.2, -0.2),
        (-0.3, 0.2), (-0.3, -0.1),
        (-2.0, 0.2), (-2.0, -0.1),
        (0.0, 0.0), (0.0, 0.3),
        (-1.8, 0.2), (-1.8, -0.2),
        (-0.7, 0.5), (-0.7, -0.4),
        (-1.3, 0.3), (-1.3, -0.3),
        (-0.2, 0.3), (-0.2, -0.2),
        (-1.6, 0.1), (-1.6, -0.1),
        (-1.0, 0.8), (-1.0, -0.5),
    ]

    # Test redshifts
    z_values = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])

    # Precompute true integrals
    n_models = len(test_models)
    n_z = len(z_values)
    true_vals = np.zeros((n_models, n_z))
    for i, (w0, wa) in enumerate(test_models):
        for j, z in enumerate(z_values):
            def integrand(zp):
                return 1.0 / E_cpl(zp, w0, wa)
            result, _ = quad(integrand, 0, z, limit=200)
            true_vals[i, j] = (1.0 + z) * result

    def objective_bounded(x):
        """x = [n1,...,n8, w1,...,w8] with 0<n_i<1, w_i>0, sum(w)=1.
        We optimize over nodes (8) and log-weights (8), then normalize."""
        if time.time() > deadline:
            return 1e10
        nodes = x[:8]
        log_w = x[8:16]
        # Ensure strictly increasing
        for k in range(1, 8):
            if nodes[k] <= nodes[k-1]:
                return 1e10
        if nodes[0] <= 0 or nodes[7] >= 1:
            return 1e10
        w = np.exp(log_w)
        w = w / w.sum()

        # Vectorized evaluation
        max_err = 0.0
        for i in range(n_models):
            for j in range(n_z):
                z = z_values[j]
                z_nodes = z * nodes
                E_vals = E_cpl(z_nodes, test_models[i][0], test_models[i][1])
                approx = (1.0 + z) * z * np.dot(w, 1.0 / E_vals)
                tv = true_vals[i, j]
                if tv > 1e-10:
                    err = abs(approx - tv) / tv
                    if err > max_err:
                        max_err = err
        return max_err

    # Bounds for differential evolution
    # Nodes: strictly increasing in (0,1). Use small margins.
    node_bounds = [(1e-6, 0.125), (0.125, 0.25), (0.25, 0.375), (0.375, 0.5),
                   (0.5, 0.625), (0.625, 0.75), (0.75, 0.875), (0.875, 1.0 - 1e-6)]
    # Log-weights: allow wide range
    w_bounds = [(-10, 10)] * 8
    bounds = node_bounds + w_bounds

    # Initial population: Gauss-Legendre based
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    gl_nodes = (gl_nodes + 1.0) / 2.0
    gl_weights = gl_weights / 2.0
    init_x = np.concatenate([gl_nodes, np.log(gl_weights)])

    # Run differential evolution
    np.random.seed(42)
    best_x = init_x.copy()
    best_obj = objective_bounded(init_x)

    try:
        result = differential_evolution(
            objective_bounded,
            bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-12,
            mutation=(0.3, 1.0),
            recombination=0.8,
            polish=False,
            init='sobol',
            x0=init_x,
            updating='immediate',
            workers=1,
            callback=lambda xk, fk: None,
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_x = result.x
    except Exception:
        pass

    # Polish with a few Nelder-Mead-like local refinements using scipy
    from scipy.optimize import minimize
    if time.time() < deadline - 10.0:
        for attempt in range(3):
            if time.time() > deadline - 5.0:
                break
            try:
                # Local refinement: minimize over the last 8 (log-weights) with nodes fixed
                # Then over nodes with weights fixed
                # Simple: just do a bounded minimize on the full vector
                res = minimize(
                    objective_bounded, best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_x = res.x
            except Exception:
                break

    nodes = best_x[:8]
    log_w = best_x[8:16]
    weights = np.exp(log_w)
    weights = weights / weights.sum()

    # Ensure validity
    if not np.all(np.diff(nodes) > 0) or nodes[0] <= 0 or nodes[7] >= 1:
        nodes = (np.arange(8) + 0.5) / 8.0
        weights = np.full(8, 1.0 / 8.0)
    if not np.all(weights > 0):
        weights = np.full(8, 1.0 / 8.0)

    weights = weights / weights.sum()

    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
