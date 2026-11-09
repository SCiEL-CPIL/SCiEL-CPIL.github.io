import numpy as np
from scipy.optimize import minimize
import time

def construct():
    # Time guard
    start_time = time.time()
    time_limit = 80.0  # Leave 10s margin for overhead/return
    
    # Define CPL E(a)
    def E_cpl(a, w0, w1):
        # E(a) = (1-a)^(3(1+w0+w1)/2) * a^(-3w1/2)
        # Handle a=0 or a=1 carefully if needed, but nodes are in (0,1)
        # and a = 1/(1+zx) is in (1/(1+z), 1).
        # If z>0, a < 1.
        # a > 0 always.
        return (1.0 - a)**(1.5 * (1.0 + w0 + w1)) * a**(-1.5 * w1)
    
    def integrand(x, z, w0, w1):
        a = 1.0 / (1.0 + z * x)
        return 1.0 / E_cpl(a, w0, w1)
        
    def true_integral(z, w0, w1):
        # Use quad for high precision
        val, err = __import__('scipy.integrate', fromlist=['quad']).quad(
            lambda x: integrand(x, z, w0, w1), 0, 1, limit=100, epsabs=1e-12, epsrel=1e-12
        )
        return val

    # Generate surrogate test cases
    w0_vals = [-1.5, -1.2, -1.0, -0.8, -0.5]
    w1_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    z_vals = [0.1, 0.5, 1.0, 2.0, 3.0]
    
    test_cases = []
    for w0 in w0_vals:
        for w1 in w1_vals:
            for z in z_vals:
                # Check if model is valid (E(a) should not be negative or zero in integration range)
                # For CPL, E(a) > 0 for a in (0,1) if parameters are reasonable.
                # If w0+w1 < -1, exponent of (1-a) is negative, so E(a) -> inf as a->1.
                # a = 1/(1+zx). As x->0, a->1.
                # If exponent < 0, 1/E(a) -> 0.
                # If exponent > 0, 1/E(a) -> inf.
                # We assume standard reasonable parameters where integral converges.
                # Integral of x^(-q) converges if q < 1.
                # q = 1.5(1+w0+w1).
                # 1.5(1+w0+w1) < 1 => 1+w0+w1 < 2/3 => w0+w1 < -1/3.
                # If w0+w1 >= -1/3, the integral might diverge or be very large?
                # Wait, if q > 0, integrand ~ x^-q. Integral converges if q < 1.
                # If q >= 1, integral diverges at 0.
                # Let's check the range.
                # If w0=-0.5, w1=0, q = 1.5(0.5) = 0.75 < 1. OK.
                # If w0=-0.5, w1=1, q = 1.5(1.5) = 2.25 > 1. Diverges?
                # In physical CPL, we usually require convergence.
                # The problem statement implies valid models.
                # We will filter cases where the true integral is finite and reasonable.
                try:
                    I_true = true_integral(z, w0, w1)
                    if np.isfinite(I_true) and I_true > 1e-10:
                        test_cases.append((z, w0, w1, I_true))
                except Exception:
                    pass

    if not test_cases:
        # Fallback to Gauss-Legendre if no valid cases found (should not happen)
        n = 8
        x_old, w_old = np.polynomial.legendre.leggauss(n)
        nodes = (x_old + 1.0) / 2.0
        weights = w_old / 2.0
        return {"nodes": nodes, "weights": weights}

    # Optimization setup
    # Parameters: u (8) for nodes, v (8) for weights
    # x_i = cumsum(exp(u)) / sum(exp(u))
    # w_i = exp(v) / sum(exp(v))
    
    def get_nodes_weights(params):
        u = params[:8]
        v = params[8:]
        # Softplus or exp to ensure positivity
        # Use exp for nodes steps
        steps = np.exp(u)
        x = np.cumsum(steps)
        x = x / x[-1] # Normalize to (0, 1)
        
        w = np.exp(v)
        w = w / np.sum(w)
        
        return x, w

    def objective(params):
        x, w = get_nodes_weights(params)
        
        # Check for numerical issues
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(w)):
            return 1e10
            
        # Check strictly increasing
        if not np.all(np.diff(x) > 1e-12):
            return 1e10
            
        max_rel_err = 0.0
        for z, w0, w1, I_true in test_cases:
            a = 1.0 / (1.0 + z * x)
            E_vals = (1.0 - a)**(1.5 * (1.0 + w0 + w1)) * a**(-1.5 * w1)
            # Avoid division by zero or negative base issues
            if np.any(E_vals <= 0) or not np.all(np.isfinite(E_vals)):
                return 1e10
            I_approx = np.sum(w / E_vals)
            rel_err = abs(I_approx - I_true) / I_true
            if rel_err > max_rel_err:
                max_rel_err = rel_err
                
        return max_rel_err

    # Initialize with Gauss-Legendre
    n = 8
    x_gl, w_gl = np.polynomial.legendre.leggauss(n)
    x_gl = (x_gl + 1.0) / 2.0
    w_gl = w_gl / 2.0
    
    # Invert mapping to get u, v
    # x_i = sum_{j=1}^i exp(u_j) / S
    # This inversion is tricky. We can just initialize u, v randomly or with small values.
    # Or approximate:
    # Let's just start with u=0, v=0 which gives uniform nodes and weights.
    # Uniform is a decent start.
    x0 = np.zeros(16)
    
    # Run optimization
    best_params = x0
    best_score = objective(x0)
    
    # Use L-BFGS-B or Nelder-Mead
    # Nelder-Mead is robust for non-smooth (max) objectives
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
    
    if res.fun < best_score:
        best_params = res.x
        best_score = res.fun
        
    # Check time
    if time.time() - start_time > time_limit:
        pass # Just proceed to return best found
    
    x_opt, w_opt = get_nodes_weights(best_params)
    
    # Final verification
    # Ensure weights sum to 1
    w_opt = w_opt / np.sum(w_opt)
    
    return {"nodes": x_opt, "weights": w_opt}
