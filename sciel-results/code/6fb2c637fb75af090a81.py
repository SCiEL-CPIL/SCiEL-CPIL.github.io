"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad


def construct():
    """
    Construct an 8-node positive quadrature rule for the luminosity distance integral.
    
    Uses a basis-function projection approach:
    1. Pre-compute exact integrals for a diverse set of cosmologies and redshifts
    2. Build a vectorized objective that avoids repeated quad calls
    3. Optimize nodes and weights to minimize worst-case relative error
    4. Use multiple optimization strategies and restarts
    """
    
    n = 8
    
    # Define representative cosmologies for fitting
    # w0, wa pairs covering a diverse range
    cosmologies = [
        (0.0, 0.0),    # LCDM-like
        (-1.0, 0.0),   # Cosmological constant
        (-0.5, 0.5),   # Phantom-ish
        (-1.5, 1.0),   # Strongly evolving
        (0.5, -0.5),   # Quintessence-like
        (-1.0, 0.5),   # Mildly evolving
        (-0.8, -0.2),  # Near LCDM
        (1.0, 0.0),    # Exotic
        (-2.0, 2.0),   # Very strong evolution
        (0.0, 1.0),    # Matter-dominated with evolution
        (-0.3, 0.3),   # Moderate
        (-1.2, -0.3),  # Another variant
        (-0.6, 0.0),   # Close to LCDM
        (-1.4, 0.4),   # Evolving dark energy
        (0.2, -0.2),   # Mild quintessence
    ]
    
    # Test redshifts - more dense coverage
    z_values = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0])
    
    Omega_m = 0.3
    Omega_r = 0.0001
    Omega_de = 1.0 - Omega_m - Omega_r
    
    def E_of_a(a, w0, wa):
        """Compute E(a) = H(a)/H0 for CPL model."""
        de_part = Omega_de * a**(-3*(1+w0+wa)) * np.exp(3*wa*(1-a))
        matter_part = Omega_m * a**(-3)
        radiation_part = Omega_r * a**(-4)
        return np.sqrt(matter_part + radiation_part + de_part)
    
    def integrand_z(z, w0, wa):
        """The integrand for the luminosity distance integral."""
        a = 1.0/(1.0+z)
        return z / E_of_a(a, w0, wa)
    
    # Pre-compute exact integrals for all test cases
    print("Pre-computing exact integrals...")
    exact_values = {}
    for w0, wa in cosmologies:
        for z in z_values:
            if z < 1e-10:
                exact_values[(w0, wa, z)] = 0.0
            else:
                result, _ = quad(integrand_z, 0, z, args=(w0, wa), limit=200, epsabs=1e-12, epsrel=1e-12)
                exact_values[(w0, wa, z)] = result
    print("Pre-computation done.")
    
    # Pre-compute E(a) for each cosmology at candidate nodes will be done in objective
    
    def quadrature_integral(z, w0, wa, nodes, weights):
        """Compute the quadrature approximation."""
        a_vals = 1.0/(1.0 + z*nodes)
        E_vals = E_of_a(a_vals, w0, wa)
        return z * np.sum(weights * nodes / E_vals)
    
    def objective(params):
        """Objective: minimize worst-case relative error (in percent)."""
        nodes = params[:n]
        weights = params[n:]
        
        # Enforce constraints: nodes in (0,1), increasing, weights positive, sum to 1
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e6
        if np.any(np.diff(nodes) <= 0):
            return 1e6
        if np.any(weights <= 0):
            return 1e6
        if abs(np.sum(weights) - 1.0) > 1e-10:
            return 1e6
        
        max_err = 0.0
        for w0, wa in cosmologies:
            for z in z_values:
                exact = exact_values[(w0, wa, z)]
                if abs(exact) < 1e-15:
                    continue
                approx = quadrature_integral(z, w0, wa, nodes, weights)
                err = abs(approx - exact) / abs(exact)
                if err > max_err:
                    max_err = err
                if max_err > 1.0:  # Early exit if error is huge
                    return 100.0
        
        return max_err * 100.0  # Convert to percent
    
    # Initial guess: use Gauss-Legendre nodes as starting point
    x_gl = np.array([
        -0.9602898564975362,
        -0.7966664774136267,
        -0.5255324099163290,
        -0.1834346425002680,
         0.1834346425002680,
         0.5255324099163290,
         0.7966664774136267,
         0.9602898564975362
    ])
    w_gl = np.array([
        0.1012285362903763,
        0.2223810344533745,
        0.3137066458778873,
        0.3626837833783620,
        0.3626837833783620,
        0.3137066458778873,
        0.2223810344533745,
        0.1012285362903763
    ])
    
    # Map to [0,1]
    x0_nodes = (x_gl + 1.0) / 2.0
    x0_weights = w_gl / 2.0
    
    # Sort
    idx = np.argsort(x0_nodes)
    x0_nodes = x0_nodes[idx]
    x0_weights = x0_weights[idx]
    
    x0 = np.concatenate([x0_nodes, x0_weights])
    
    # Several initial guesses
    initial_guesses = [
        x0,  # Gauss-Legendre
        np.concatenate([(np.arange(n) + 0.5)/n, np.ones(n)/n]),  # Midpoint
        np.concatenate([np.linspace(0.05, 0.95, n), np.ones(n)/n]),  # Uniform
        np.concatenate([np.sqrt(np.linspace(0.01, 0.99, n)), np.ones(n)/n]),  # Sqrt-spaced
        np.concatenate([1-np.sqrt(np.linspace(0.01, 0.99, n)), np.ones(n)/n]),  # Inverse sqrt
        np.concatenate([np.array([0.02, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]), np.ones(n)/n]),
        np.concatenate([np.array([0.01, 0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.98]), np.ones(n)/n]),
        np.concatenate([np.array([0.03, 0.12, 0.25, 0.4, 0.55, 0.7, 0.82, 0.93]), np.ones(n)/n]),
    ]
    
    best_params = None
    best_score = np.inf
    
    # Use multiple optimization methods
    methods = [
        ('Nelder-Mead', {'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}),
        ('Powell', {'maxiter': 2000, 'xatol': 1e-9, 'ftol': 1e-11}),
    ]
    
    for x_init in initial_guesses:
        for method_name, options in methods:
            try:
                result = minimize(objective, x_init, method=method_name, options=options)
                if result.fun < best_score:
                    best_score = result.fun
                    best_params = result.x.copy()
            except Exception as e:
                pass
    
    if best_params is None:
        # Fallback to Gauss-Legendre
        nodes = (x_gl + 1.0) / 2.0
        weights = w_gl / 2.0
        idx = np.argsort(nodes)
        nodes = nodes[idx]
        weights = weights[idx]
    else:
        nodes = best_params[:n]
        weights = best_params[n:]
        
        # Ensure nodes are strictly in (0,1) and increasing
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        nodes = np.sort(nodes)
        
        # Ensure weights are positive and sum to 1
        weights = np.maximum(weights, 1e-10)
        weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
