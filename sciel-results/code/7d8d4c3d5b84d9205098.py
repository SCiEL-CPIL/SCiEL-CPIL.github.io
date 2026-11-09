import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit

    # Define a set of representative cosmologies for optimization
    # w0: dark energy equation of state at z=0
    # wa: dark energy equation of state evolution
    # H0: Hubble constant (doesn't affect relative error for dL/c/H0)
    # Omega_m0: matter density parameter
    cosmologies = [
        {'w0': -1.0, 'wa': 0.0, 'Om0': 0.3},   # LambdaCDM
        {'w0': -1.2, 'wa': 0.2, 'Om0': 0.3},   # Phantom
        {'w0': -0.8, 'wa': -0.4, 'Om0': 0.3},  # Quintessence
        {'w0': -1.5, 'wa': 0.5, 'Om0': 0.3},   # Strong Phantom
        {'w0': -0.5, 'wa': 0.0, 'Om0': 0.3},   # Kessner
        {'w0': -1.0, 'wa': 1.0, 'Om0': 0.3},   # Evolving
        {'w0': -1.0, 'wa': -1.0, 'Om0': 0.3},  # Evolving
        {'w0': -2.0, 'wa': 0.0, 'Om0': 0.3},   # Extreme Phantom
        {'w0': 0.0, 'wa': 0.0, 'Om0': 0.3},    # Matter-like DE
        {'w0': -1.0, 'wa': 0.0, 'Om0': 0.25},  # Different Om0
        {'w0': -1.0, 'wa': 0.0, 'Om0': 0.35},  # Different Om0
    ]

    def E_of_a(a, cosmo):
        Om0 = cosmo['Om0']
        w0 = cosmo['w0']
        wa = cosmo['wa']
        # E(a)^2 = Om_m / a^3 + Om_de / a^(3(1+w0+wa(1-a)))
        # Om_de = 1 - Om_m for flat universe
        Om_de = 1.0 - Om0
        if a <= 0:
            return np.inf
        w_eff = w0 + wa * (1.0 - a)
        return np.sqrt(Om0 / (a**3) + Om_de / (a**(3.0 * (1.0 + w_eff))))

    def exact_integral(z, cosmo):
        # Use adaptive integration for high precision
        from scipy.integrate import quad
        def integrand(z_prime):
            a = 1.0 / (1.0 + z_prime)
            return 1.0 / E_of_a(a, cosmo)
        # Integrate from 0 to z
        val, _ = quad(integrand, 0, z, limit=200, epsabs=1e-12, epsrel=1e-12)
        return z * val

    def quadrature_integral(z, nodes, weights, cosmo):
        a_vals = 1.0 / (1.0 + z * nodes)
        E_vals = E_of_a(a_vals, cosmo)
        return z * np.sum(weights / E_vals)

    def objective(params):
        if time.time() > deadline:
            return 1e10
        # Transform params to nodes and weights
        # params: 16 values, first 8 for nodes, next 8 for weights
        node_params = params[:8]
        weight_params = params[8:]
        
        # Nodes: use cumulative sum of sigmoid to ensure strictly increasing in (0,1)
        v = np.exp(node_params)
        x = np.cumsum(v) / np.sum(v)
        # Ensure x is strictly in (0,1)
        x = np.clip(x, 1e-6, 1 - 1e-6)
        # Ensure strictly increasing (cumsum of positive numbers is strictly increasing)
        
        # Weights: softmax
        w = np.exp(weight_params - np.max(weight_params))
        w = w / np.sum(w)
        
        # Compute worst-case relative error
        max_err = 0.0
        for cosmo in cosmologies:
            # Test at several redshifts
            for z in [0.1, 0.5, 1.0, 2.0, 5.0]:
                exact = exact_integral(z, cosmo)
                quad_val = quadrature_integral(z, x, w, cosmo)
                if exact == 0:
                    rel_err = 0.0
                else:
                    rel_err = abs(quad_val - exact) / exact
                if rel_err > max_err:
                    max_err = rel_err
        return max_err

    # Initial guess: Gauss-Legendre nodes and weights
    from numpy.polynomial.legendre import leggauss
    x_gl, w_gl = leggauss(8)
    x_gl = (x_gl + 1.0) / 2.0  # Map to [0,1]
    w_gl = w_gl / 2.0
    
    # Inverse transform for initial guess
    # For nodes, we need v such that cumsum(v)/sum(v) = x_gl
    # This is tricky, so we start with random positive v
    np.random.seed(42)
    v_init = np.random.uniform(0.1, 1.0, 8)
    x_init = np.cumsum(v_init) / np.sum(v_init)
    # Adjust to be close to Gauss-Legendre? Or just use random start.
    # Let's use a start that maps to uniform nodes first, or just random.
    # Better: use the Gauss-Legendre nodes directly as a starting point for a different parameterization?
    # The cumulative sum parameterization is robust. Let's just start with random v.
    
    # Weights: log(w_gl)
    w_init_log = np.log(w_gl)
    
    params_init = np.concatenate([np.log(v_init), w_init_log])
    
    # Optimize
    res = minimize(objective, params_init, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
    
    if time.time() > deadline:
        # Fallback to Gauss-Legendre
        return {"nodes": x_gl, "weights": w_gl}
    
    if res.success or res.fun < 0.1:  # If optimization improved significantly
        node_params = res.x[:8]
        weight_params = res.x[8:]
        v = np.exp(node_params)
        nodes = np.cumsum(v) / np.sum(v)
        nodes = np.clip(nodes, 1e-6, 1 - 1e-6)
        
        w = np.exp(weight_params - np.max(weight_params))
        weights = w / np.sum(w)
        
        # Final check
        if np.all(np.diff(nodes) > 0) and np.all(weights > 0) and abs(np.sum(weights) - 1.0) < 1e-12:
            return {"nodes": nodes, "weights": weights}
    
    # Fallback to Gauss-Legendre
    return {"nodes": x_gl, "weights": w_gl}
