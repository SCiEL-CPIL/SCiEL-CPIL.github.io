import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

def construct():
    # Define E(z) for CPL
    def E_z(z, Om, w0, wa):
        a = 1.0 / (1.0 + z)
        term_m = Om * a**-3
        term_de = (1.0 - Om) * a**(-3.0*(1.0+w0)) * (1.0-a)**(3.0*wa)
        return np.sqrt(term_m + term_de)

    # Integrand for the quadrature: f(u) = 1 / E(z*u)
    def integrand(u, z, Om, w0, wa):
        zp = z * u
        a_p = 1.0 / (1.0 + zp)
        term_m = Om * a_p**-3
        term_de = (1.0 - Om) * a_p**(-3.0*(1.0+w0)) * (1.0-a_p)**(3.0*wa)
        E_val = np.sqrt(term_m + term_de)
        return 1.0 / E_val

    # Comprehensive test cases covering the parameter space
    test_cases = []
    # Vary z, Om, w0, wa
    for z in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]:
        for Om in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for w0 in [-1.2, -1.0, -0.8, -0.6]:
                for wa in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                    test_cases.append((z, Om, w0, wa))
    
    # Pre-compute exact integrals
    exact_vals = []
    for z, Om, w0, wa in test_cases:
        val, err = quad(lambda u: integrand(u, z, Om, w0, wa), 0, 1, epsabs=1e-10, epsrel=1e-10)
        exact_vals.append(val)
    exact_vals = np.array(exact_vals)

    # Objective function: minimize worst relative error
    def objective(params):
        # params: 16 values - 8 nodes + 8 weights
        nodes = params[:8]
        weights = params[8:]
        
        # Penalty for invalid nodes (not in (0,1) or not increasing)
        if not np.all(nodes > 0) or not np.all(nodes < 1) or not np.all(np.diff(nodes) > 0):
            return 1e10
        if not np.all(weights > 0):
            return 1e10
        if abs(np.sum(weights) - 1.0) > 1e-8:
            return 1e10
        
        # Compute errors for all test cases
        max_err = 0.0
        for i, (z, Om, w0, wa) in enumerate(test_cases):
            # Compute f at each node
            zp = z * nodes
            a_p = 1.0 / (1.0 + zp)
            term_m = Om * a_p**-3
            term_de = (1.0 - Om) * a_p**(-3.0*(1.0+w0)) * (1.0-a_p)**(3.0*wa)
            E_val = np.sqrt(term_m + term_de)
            f_vals = 1.0 / E_val
            
            approx = np.dot(weights, f_vals)
            exact = exact_vals[i]
            
            err = abs(approx - exact) / exact
            if err > max_err:
                max_err = err
                
        return max_err

    # Initial guess: Gauss-Legendre
    x, w_gl = np.polynomial.legendre.leggauss(8)
    u_gl = (x + 1.0) / 2.0
    w_gl = w_gl / 2.0
    
    x0 = np.concatenate((u_gl, w_gl))
    
    # Use Nelder-Mead with many restarts from different starting points
    best_result = None
    best_obj = np.inf
    
    # Start from GL
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12})
    if res.fun < best_obj:
        best_obj = res.fun
        best_result = res.x
    
    # Try several random restarts
    np.random.seed(42)
    for _ in range(10):
        # Random nodes in (0,1), sorted
        nodes_rand = np.sort(np.random.uniform(0.05, 0.95, 8))
        # Random positive weights normalized to sum to 1
        w_rand = np.random.uniform(0.1, 1.0, 8)
        w_rand = w_rand / np.sum(w_rand)
        
        x0_rand = np.concatenate((nodes_rand, w_rand))
        
        res = minimize(objective, x0_rand, method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res.fun < best_obj:
            best_obj = res.fun
            best_result = res.x
    
    # Extract best params
    nodes = best_result[:8]
    weights = best_result[8:]
    
    # Final sanity check and normalization
    if not np.all(nodes > 0) or not np.all(nodes < 1) or not np.all(np.diff(nodes) > 0):
        # Fallback to GL
        nodes = u_gl
        weights = w_gl
    
    # Ensure weights sum to 1
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
