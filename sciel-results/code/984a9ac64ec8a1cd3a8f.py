"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import quad
from scipy.optimize import minimize

def construct():
    """
    Discover an eight-node positive quadrature rule for dimensionless luminosity distance
    in spatially flat, time-varying dark-energy models.
    
    Strategy: Multi-start local optimization (Nelder-Mead/Powell) from diverse initial guesses
    (Gauss-Legendre, Uniform, etc.) to minimize worst-case relative error.
    """
    start_time = time.time()
    deadline = start_time + 85.0

    def E_cpl(z, w0, wa, Omega_m=0.3):
        Omega_DE = 1.0 - Omega_m
        exp_arg = 3.0 * ((w0 + wa) * z - wa * np.log1p(z))
        exp_arg = np.clip(exp_arg, -700, 700)
        return np.sqrt(Omega_m * (1.0 + z)**3 + Omega_DE * np.exp(exp_arg))

    # Large diverse set of CPL models covering extreme w0, wa
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
        # Add more extremes
        (-2.5, 0.5), (-2.5, -0.5),
        (0.5, 0.0), (0.5, -0.5),
    ]

    # Test redshifts: include high z where error might be larger
    z_values = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])

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
        """x = [n1,...,n8, log_w1,...,log_w8]"""
        if time.time() > deadline:
            return 1e10
        nodes = x[:8]
        log_w = x[8:16]
        
        # Check validity
        if nodes[0] <= 0 or nodes[7] >= 1:
            return 1e10
        if not np.all(np.diff(nodes) > 1e-12):
            return 1e10
            
        w = np.exp(log_w)
        w_sum = np.sum(w)
        if w_sum <= 0:
            return 1e10
        w = w / w_sum

        # Vectorized evaluation
        max_err = 0.0
        for i in range(n_models):
            w0, wa = test_models[i]
            for j in range(n_z):
                z = z_values[j]
                z_nodes = z * nodes
                E_vals = E_cpl(z_nodes, w0, wa)
                approx = (1.0 + z) * z * np.dot(w, 1.0 / E_vals)
                tv = true_vals[i, j]
                if tv > 1e-10:
                    err = abs(approx - tv) / tv
                    if err > max_err:
                        max_err = err
        return max_err

    # Generate diverse initial guesses
    inits = []
    
    # 1. Gauss-Legendre
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    gl_nodes = (gl_nodes + 1.0) / 2.0
    gl_weights = gl_weights / 2.0
    inits.append(np.concatenate([gl_nodes, np.log(gl_weights)]))
    
    # 2. Uniform
    uni_nodes = (np.arange(8) + 0.5) / 8.0
    uni_weights = np.full(8, 1.0/8.0)
    inits.append(np.concatenate([uni_nodes, np.log(uni_weights)]))
    
    # 3. Clumped to left (low z weight)
    left_nodes = np.linspace(0.01, 0.5, 8)
    left_weights = np.full(8, 1.0/8.0)
    inits.append(np.concatenate([left_nodes, np.log(left_weights)]))
    
    # 4. Clumped to right (high z weight)
    right_nodes = np.linspace(0.5, 0.99, 8)
    right_weights = np.full(8, 1.0/8.0)
    inits.append(np.concatenate([right_nodes, np.log(right_weights)]))
    
    # 5. Random perturbations of GL
    np.random.seed(42)
    for _ in range(5):
        pert_nodes = gl_nodes + np.random.uniform(-0.02, 0.02, 8)
        pert_nodes = np.sort(pert_nodes)
        pert_nodes = np.clip(pert_nodes, 1e-4, 1-1e-4)
        # Ensure strictly increasing
        for k in range(1, 8):
            if pert_nodes[k] <= pert_nodes[k-1]:
                pert_nodes[k] = pert_nodes[k-1] + 1e-4
        pert_weights = gl_weights * np.exp(np.random.uniform(-0.5, 0.5, 8))
        inits.append(np.concatenate([pert_nodes, np.log(pert_weights)]))

    best_x = None
    best_obj = float('inf')

    # Run local optimization from each start
    for init_x in inits:
        if time.time() > deadline - 5.0:
            break
        
        try:
            # Use Nelder-Mead for robustness in small dimensions
            res = minimize(
                objective_bounded, init_x,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
        except Exception:
            pass

    # If no valid solution found, fallback to GL
    if best_x is None:
        best_x = inits[0]

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
