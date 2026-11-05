import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

def construct():
    t_start = time.time()
    deadline = t_start + 80.0  # 80s safety margin
    
    # Define test cosmologies: (Omega_m, w0, w1)
    # Include LCDM, quintessence, phantom, varying Omega_m
    test_cosmos = [
        (0.3, -1.0, 0.0),    # LCDM
        (0.3, -1.0, 0.5),    # CPL w1>0
        (0.3, -1.0, -0.5),   # CPL w1<0
        (0.3, -0.5, 0.0),    # w0=-0.5
        (0.3, -1.5, 0.0),    # w0=-1.5 (phantom)
        (0.2, -1.0, 0.0),    # Omega_m=0.2
        (0.4, -1.0, 0.0),    # Omega_m=0.4
        (0.3, -1.2, 0.3),    # mixed
        (0.3, -0.8, -0.3),   # mixed
        (0.35, -1.0, 0.2),   # another
    ]
    
    def E_of_z(z, om, w0, w1):
        """H(z)/H0 for flat CPL model"""
        a = 1.0 / (1.0 + z)
        # E^2 = Om/a^3 + Ol/a^(3(1+w0+w1))
        ol = 1.0 - om
        if w1 != 0.0:
            e2 = om / a**3 + ol / a**(3.0*(1.0+w0+w1))
        else:
            e2 = om / a**3 + ol / a**(3.0*(1.0+w0))
        return np.sqrt(np.maximum(e2, 1e-30))
    
    def exact_integral(z, om, w0, w1):
        """Exact dL/(c/H0) = (1+z) * integral_0^z dz'/E(z')"""
        if z < 1e-10:
            return 0.0
        f = lambda zp: 1.0 / E_of_z(zp, om, w0, w1)
        val, _ = quad(f, 0, z, limit=200)
        return (1.0 + z) * val
    
    def quadrature(z, nodes, weights, om, w0, w1):
        """Quadrature approximation: (1+z)*z*sum_i w_i / E(z*t_i)"""
        if z < 1e-10:
            return 0.0
        z_nodes = z * nodes
        E_vals = E_of_z(z_nodes, om, w0, w1)
        return (1.0 + z) * z * np.sum(weights / E_vals)
    
    def worst_case_error(nodes, weights, test_zs, cosmos):
        """Compute worst relative error in percent"""
        max_err = 0.0
        for om, w0, w1 in cosmos:
            for z in test_zs:
                exact = exact_integral(z, om, w0, w1)
                approx = quadrature(z, nodes, weights, om, w0, w1)
                if abs(exact) > 1e-10:
                    rel_err = abs(approx - exact) / abs(exact) * 100.0
                    if rel_err > max_err:
                        max_err = rel_err
        return max_err
    
    # Test redshifts for evaluation
    test_zs = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    # Initial guess: Gauss-Legendre-like nodes
    # Start with shifted Gauss-Legendre or just uniform
    n = 8
    # Use Gauss-Legendre nodes on [0,1] as initial guess
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n)
    nodes0 = 0.5 * (gl_nodes + 1.0)
    weights0 = 0.5 * gl_weights
    
    # Ensure positive weights (GL weights are positive)
    # Ensure nodes strictly increasing in (0,1)
    
    def objective(params):
        if time.time() > deadline - 5.0:
            return 1e10
        # params: log-spaced nodes to enforce ordering and bounds, log-weights for positivity
        log_nodes = params[:n]
        log_weights = params[n:]
        
        # Map log_nodes to (0,1) with spacing constraint
        # Use softplus-like: nodes[i] = sigmoid(cumsum) ... simpler: use exp differences
        # Actually, let's use: raw_nodes in R, then nodes = cumulative softmax-like
        # Simpler: nodes[0] = sigmoid(log_nodes[0]) * 0.99 + 0.01
        # nodes[i] = nodes[i-1] + 0.01 + sigmoid(log_nodes[i]) * (0.98 - (i-1)*0.01 - 0.01) / ...
        
        # Even simpler: use the fact that we can just optimize nodes directly with bounds
        # But SLSQP with bounds is tricky for strict ordering. Let's use a different parameterization.
        
        # Parameterization: nodes = 0.01 + 0.98 * cumsum(exp(log_nodes)) / cumsum(exp(log_nodes))[-1]
        # This ensures 0 < nodes[0] < ... < nodes[7] < 1 and strictly increasing
        exp_ln = np.exp(log_nodes)
        cum = np.cumsum(exp_ln)
        nodes = 0.01 + 0.98 * cum / cum[-1]
        # Ensure strictly within (0,1)
        nodes = np.clip(nodes, 0.001, 0.999)
        # Ensure strict increase (the cumsum guarantees this if exp_ln > 0)
        
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights)  # normalize to sum=1
        
        return worst_case_error(nodes, weights, test_zs, test_cosmos)
    
    # Initial log parameters
    log_nodes0 = np.zeros(n)  # uniform-ish
    log_weights0 = np.log(weights0)
    
    params0 = np.concatenate([log_nodes0, log_weights0])
    
    best_params = params0.copy()
    best_score = objective(params0)
    
    # Try a few optimization approaches
    methods_to_try = [
        ('Nelder-Mead', {'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}),
        ('Powell', {'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-8}),
    ]
    
    for method, options in methods_to_try:
        if time.time() > deadline - 10.0:
            break
        try:
            res = minimize(objective, best_params, method=method, options=options)
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Extract best nodes and weights
    log_nodes = best_params[:n]
    log_weights = best_params[n:]
    
    exp_ln = np.exp(log_nodes)
    cum = np.cumsum(exp_ln)
    nodes = 0.01 + 0.98 * cum / cum[-1]
    nodes = np.clip(nodes, 0.001, 0.999)
    
    weights = np.exp(log_weights)
    weights = weights / np.sum(weights)
    
    # Ensure weights sum to 1 within 1e-12
    weights = weights / np.sum(weights)
    
    # Final check: nodes strictly increasing in [0,1], weights positive
    if not np.all(np.diff(nodes) > 0):
        # Fallback to uniform
        nodes = (np.arange(n) + 0.5) / n
        weights = np.full(n, 1.0/n)
    
    return {"nodes": nodes, "weights": weights}
