import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

def construct():
    """
    Construct an 8-node positive quadrature rule for dimensionless luminosity distance
    in spatially flat, time-varying dark-energy models.
    
    The integrand is f(z) = z / E(a) where a = 1/(1+z), and the full distance is:
    dL/(c/H0) = (1+z) * integral_0^z f(z') dz'
    
    We are approximating integral_0^z f(z') dz' with a fixed 8-node rule on [0,1]
    where the variable is u = z'/z, so z' = z*u, a = 1/(1+z*u), and:
    integral_0^z f(z') dz' = z * integral_0^1 f(z*u) du
    
    So the quadrature approximates:
    integral_0^1 [ (z*u) / E(1/(1+z*u)) ] du
    
    with fixed nodes u_i in [0,1] and weights w_i.
    """
    start_time = time.time()
    time_limit = 85.0  # Leave 5 seconds margin
    
    # Define a family of cosmologies to optimize against.
    # We'll use a diverse set of CPL (CPL) models: w0, wa parameters.
    
    def E_of_a(a, Omega_m, w0, wa):
        """
        Compute E(a) = H(a)/H0 for flat CPL model.
        a is scale factor.
        """
        term_m = Omega_m * a**(-3)
        term_L = (1.0 - Omega_m) * a**(-3*(1+w0+wa)) * np.exp(3*wa*(1-a))
        return np.sqrt(term_m + term_L)
    
    def integrand_z(z, u, Omega_m, w0, wa):
        """
        The integrand in the variable u = z'/z.
        z' = z * u
        a = 1 / (1 + z') = 1 / (1 + z*u)
        integrand = z' / E(a) = z*u / E(1/(1+z*u))
        """
        a = 1.0 / (1.0 + z * u)
        E = E_of_a(a, Omega_m, w0, wa)
        return z * u / E
    
    def quadrature_error(nodes, weights, z, Omega_m, w0, wa):
        """
        Compute relative error of the quadrature for a given z and cosmology.
        """
        I_quad = np.sum(weights * [integrand_z(z, u, Omega_m, w0, wa) for u in nodes])
        
        def f_integrand(u):
            return integrand_z(z, u, Omega_m, w0, wa)
        
        I_exact, _ = quad(f_integrand, 0, 1, limit=200)
        
        if abs(I_exact) < 1e-15:
            return 0.0
        return abs(I_quad - I_exact) / abs(I_exact)
    
    # Define a set of validation cosmologies and redshifts.
    cosmologies = [
        (0.27, -1.0, 0.0),   # LambdaCDM
        (0.27, -0.9, 0.1),   # Mildly varying w
        (0.30, -1.1, 0.5),   # Phantom-ish, varying
        (0.25, -0.5, -0.5),  # Quintessence-like
        (0.35, -1.0, 0.3),   # Higher matter, varying
        (0.27, -1.0, -0.3),  # Varying w, negative wa
        (0.30, -0.8, 0.2),   # Moderate w0, positive wa
        (0.28, -1.2, 0.8),   # More phantom, strong variation
    ]
    
    redshifts = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    # Precompute exact integrals for speed during optimization
    # We'll cache them based on (z, Omega_m, w0, wa)
    exact_cache = {}
    
    def get_exact(z, Omega_m, w0, wa):
        key = (z, Omega_m, w0, wa)
        if key not in exact_cache:
            def f_integrand(u):
                return integrand_z(z, u, Omega_m, w0, wa)
            exact, _ = quad(f_integrand, 0, 1, limit=200)
            exact_cache[key] = exact
        return exact_cache[key]
    
    def objective(params):
        """
        params: array of 16 values: 8 nodes and 8 weights.
        We'll use a transformation to ensure nodes are in (0,1) and increasing,
        and weights are positive.
        """
        node_raw = params[0:8]
        log_w = params[8:16]
        
        # Sort nodes to ensure increasing order
        nodes = np.sort(node_raw)
        
        # Ensure nodes are in [0,1]
        nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
        
        # Weights from log
        w = np.exp(log_w)
        w = w / np.sum(w)  # Normalize to sum to 1
        
        # Compute worst-case error over a subset of cosmologies and redshifts
        # For speed, use a subset during optimization
        max_err = 0.0
        for om, w0, wa in cosmologies[:5]:  # Use subset
            for z in redshifts[:5]:  # Use subset
                exact = get_exact(z, om, w0, wa)
                I_quad = np.sum(w * [integrand_z(z, u, om, w0, wa) for u in nodes])
                err = abs(I_quad - exact) / abs(exact) if abs(exact) > 1e-15 else 0.0
                if err > max_err:
                    max_err = err
        
        return max_err
    
    # Initial guess: Gauss-Legendre nodes and weights
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes_gl = 0.5 * (x_gl + 1.0)
    w_gl = 0.5 * w_gl
    
    # Parameterize: nodes directly, log-weights
    p0 = np.concatenate([nodes_gl, np.log(w_gl)])
    
    # Bounds
    bounds = []
    for i in range(8):
        bounds.append((1e-10, 1.0 - 1e-10))  # Nodes
    for i in range(8):
        bounds.append((-10, 10))  # Log-weights
    
    # Optimize using Powell (local)
    best_params = p0.copy()
    best_obj = objective(p0)
    
    try:
        if time.time() - start_time < time_limit - 10:
            pow_res = minimize(
                objective, best_params,
                method='Powell',
                bounds=bounds,
                options={'maxiter': 200, 'xtol': 1e-12, 'ftol': 1e-14}
            )
            if pow_res.fun < best_obj:
                best_obj = pow_res.fun
                best_params = pow_res.x
    except Exception:
        pass
    
    # Extract final nodes and weights
    nodes_final = np.sort(best_params[0:8])
    nodes_final = np.clip(nodes_final, 1e-10, 1.0 - 1e-10)
    w_final = np.exp(best_params[8:16])
    w_final = w_final / np.sum(w_final)
    
    # Check time and return
    return {"nodes": nodes_final, "weights": w_final}
