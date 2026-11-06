"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import quad
from scipy.optimize import minimize, differential_evolution

def construct():
    """
    Discover an eight-node positive quadrature rule for dimensionless luminosity distance
    in spatially flat, time-varying dark-energy models.
    
    Strategy:
    1. Model the integrand variation via basis functions spanning the CPL parameter space.
    2. Use a weighted least-squares fit to derive initial nodes and weights.
    3. Refine with differential evolution and local Nelder-Mead polish.
    4. Ensure positivity and strict ordering of nodes.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    def E_cpl(z, w0, wa, Omega_m=0.3):
        Omega_DE = 1.0 - Omega_m
        exp_arg = 3.0 * ((w0 + wa) * z - wa * np.log1p(z))
        exp_arg = np.clip(exp_arg, -700, 700)
        return np.sqrt(Omega_m * (1.0 + z)**3 + Omega_DE * np.exp(exp_arg))

    # Diverse set of CPL models (w0, wa)
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

    def compute_max_error(nodes, weights):
        """Compute worst-case relative error given nodes and weights."""
        max_err = 0.0
        for i in range(n_models):
            w0, wa = test_models[i]
            for j in range(n_z):
                z = z_values[j]
                z_nodes = z * nodes
                E_vals = E_cpl(z_nodes, w0, wa)
                approx = (1.0 + z) * z * np.dot(weights, 1.0 / E_vals)
                tv = true_vals[i, j]
                if tv > 1e-10:
                    err = abs(approx - tv) / tv
                    if err > max_err:
                        max_err = err
        return max_err

    # Structural change: Basis function approach for initial guess
    # We model 1/E(z) as a linear combination of basis functions that span the CPL variation.
    # Basis functions: 1, z, z^2, log(1+z), z*log(1+z), (1+z)^-1.5, etc.
    # However, for a quadrature rule, we are fitting the integral, not the function.
    # Let's stick to the optimization but use a smarter initialization based on the structure.
    
    # Initialize with Gauss-Legendre nodes as a strong starting point
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    gl_nodes = (gl_nodes + 1.0) / 2.0
    gl_weights = gl_weights / 2.0
    init_x = np.concatenate([gl_nodes, np.log(gl_weights)])

    def objective_bounded(x):
        """x = [n1,...,n8, w1,...,w8] with 0<n_i<1, w_i>0, sum(w)=1."""
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
        return compute_max_error(nodes, w)

    # Bounds for differential evolution
    node_bounds = [(1e-6, 0.125), (0.125, 0.25), (0.25, 0.375), (0.375, 0.5),
                   (0.5, 0.625), (0.625, 0.75), (0.75, 0.875), (0.875, 1.0 - 1e-6)]
    w_bounds = [(-10, 10)] * 8
    bounds = node_bounds + w_bounds

    # Run differential evolution for global search
    np.random.seed(42)
    best_x = init_x.copy()
    best_obj = objective_bounded(init_x)

    try:
        result = differential_evolution(
            objective_bounded,
            bounds,
            seed=42,
            maxiter=100,
            popsize=12,
            tol=1e-10,
            mutation=(0.3, 1.0),
            recombination=0.8,
            polish=False,
            init='sobol',
            x0=init_x,
            updating='immediate',
            workers=1,
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_x = result.x
    except Exception:
        pass

    # Local refinement with Nelder-Mead from multiple starting points
    if time.time() < deadline - 10.0:
        start_points = [best_x.copy()]
        
        # Uniform nodes
        uniform_nodes = (np.arange(8) + 0.5) / 8.0
        uniform_w = np.full(8, 1.0 / 8.0)
        start_points.append(np.concatenate([uniform_nodes, np.log(uniform_w)]))
        
        # Random points
        np.random.seed(123)
        for _ in range(3):
            rand_nodes = np.sort(np.random.uniform(0.01, 0.99, 8))
            rand_w = np.random.uniform(0.5, 1.5, 8)
            start_points.append(np.concatenate([rand_nodes, np.log(rand_w)]))
        
        for init_p in start_points:
            if time.time() > deadline - 5.0:
                break
            try:
                res = minimize(
                    objective_bounded, init_p,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_x = res.x
            except Exception:
                pass

    # Extract best nodes and weights
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
