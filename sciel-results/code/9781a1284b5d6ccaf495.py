import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

def construct():
    # Define E(z) for CPL
    def E_z(z, Om, w0, wa):
        a = 1.0 / (1.0 + z)
        # H^2/H0^2 = Om * a^-3 + (1-Om) * a^(-3(1+w0)) * (1-a)^(3*wa)
        # Note: For w=-1, wa=0, this is Om*a^-3 + (1-Om).
        # General CPL: E^2 = Om a^-3 + (1-Om) a^-3(1+w0) (1-a)^(3 wa) ?
        # Standard CPL: w(a) = w0 + wa(1-a).
        # rho_DE(a) = rho_DE0 * a^{-3(1+w0)} * (1-a)^{3 wa}.
        # So E^2 = Om a^-3 + (1-Om) a^{-3(1+w0)} (1-a)^{3 wa}.
        
        term_m = Om * a**-3
        term_de = (1.0 - Om) * a**(-3.0*(1.0+w0)) * (1.0-a)**(3.0*wa)
        return np.sqrt(term_m + term_de)

    # Integrand for the quadrature: f(u) = 1 / E(z*u)
    # Note: The integral is over u in [0,1].
    # The argument to E is z' = z*u.
    # So we need E(z*u).
    def integrand(u, z, Om, w0, wa):
        # Calculate E at z' = z*u
        # a' = 1/(1+z*u)
        # We can reuse E_z logic but with z'
        zp = z * u
        a_p = 1.0 / (1.0 + zp)
        term_m = Om * a_p**-3
        term_de = (1.0 - Om) * a_p**(-3.0*(1.0+w0)) * (1.0-a_p)**(3.0*wa)
        E_val = np.sqrt(term_m + term_de)
        return 1.0 / E_val

    # Test cases: (z, Om, w0, wa)
    # We want to cover the range of possible "shapes"
    test_cases = [
        (0.5, 0.3, -1.0, 0.0),
        (1.0, 0.3, -1.0, 0.0),
        (2.0, 0.3, -1.0, 0.0),
        (1.0, 0.3, -1.0, 1.0),
        (1.0, 0.3, -1.0, -1.0),
        (1.0, 0.3, -0.8, 0.0),
        (1.0, 0.3, -1.2, 0.0),
        (1.0, 0.2, -1.0, 0.5),
        (1.0, 0.4, -1.0, -0.5),
        (5.0, 0.3, -1.0, 0.0),
    ]

    # Pre-compute exact integrals
    exact_vals = []
    for z, Om, w0, wa in test_cases:
        # Integrate f(u) from 0 to 1
        # f(u) = 1/E(z*u)
        val, err = quad(lambda u: integrand(u, z, Om, w0, wa), 0, 1, epsabs=1e-10, epsrel=1e-10)
        exact_vals.append(val)

    # Objective function
    def objective(params):
        # params: 16 values
        # 8 gaps h_i (log-sum-exp) -> u_i
        # 8 weights v_i (log-sum-exp) -> w_i
        
        h = params[:8]
        v = params[8:]
        
        # Compute gaps
        exp_h = np.exp(h - np.max(h)) # stability
        g = exp_h / np.sum(exp_h)
        
        # Compute nodes u_i = cumsum(g)
        # u_1 = g_1, u_2 = g_1+g_2, ...
        # But we need 8 nodes.
        # If we use 8 gaps, we get 8 nodes?
        # u_1 = g_1
        # u_2 = g_1 + g_2
        # ...
        # u_8 = g_1 + ... + g_8
        # Sum g_i = 1, so u_8 = 1.
        # But nodes must be in [0,1] strictly increasing.
        # If u_8 = 1, is it allowed? "Nodes must be strictly increasing in [0,1]".
        # Usually quadrature nodes are inside (0,1).
        # Gauss-Legendre nodes are inside.
        # If we use 8 gaps, the last node is 1.
        # We should probably use 9 gaps to define 8 nodes inside (0,1)?
        # Or just use 8 nodes u_i directly with bounds?
        
        # Let's stick to direct nodes with penalty for ordering?
        # No, the gap method is cleaner but forces u_8=1.
        # Let's modify: Use 9 gaps to define 8 nodes in (0,1).
        # u_1 = g_1
        # u_2 = g_1+g_2
        # ...
        # u_8 = g_1+...+g_8
        # g_9 is the remaining gap to 1.
        # Sum g_i = 1.
        
        h = params[:9] # 9 gaps
        v = params[9:] # 8 weights
        
        exp_h = np.exp(h - np.max(h))
        g = exp_h / np.sum(exp_h)
        
        # Nodes are cumulative sums of first 8 gaps
        u = np.cumsum(g[:8])
        
        # Weights
        exp_v = np.exp(v - np.max(v))
        w = exp_v / np.sum(exp_v)
        
        # Compute errors
        max_err = 0.0
        for i, (z, Om, w0, wa) in enumerate(test_cases):
            # Compute approx integral
            # f(u) for each node
            us = u
            # Vectorize integrand calculation
            zp = z * us
            a_p = 1.0 / (1.0 + zp)
            term_m = Om * a_p**-3
            term_de = (1.0 - Om) * a_p**(-3.0*(1.0+w0)) * (1.0-a_p)**(3.0*wa)
            E_val = np.sqrt(term_m + term_de)
            f_vals = 1.0 / E_val
            
            approx = np.dot(w, f_vals)
            exact = exact_vals[i]
            
            err = abs(approx - exact) / exact
            if err > max_err:
                max_err = err
                
        return max_err

    # Initial guess: Gauss-Legendre
    x, w_gl = np.polynomial.legendre.leggauss(8)
    u_gl = (x + 1.0) / 2.0
    w_gl = w_gl / 2.0
    
    # Convert u_gl to gaps
    u_gl_ext = np.concatenate(([0.0], u_gl, [1.0]))
    g_gl = np.diff(u_gl_ext)
    
    # Convert g_gl to h (log)
    # h_i = log(g_i / sum(g))
    h_gl = np.log(g_gl / np.sum(g_gl))
    
    # Convert w_gl to v (log)
    v_gl = np.log(w_gl / np.sum(w_gl))
    
    x0 = np.concatenate((h_gl, v_gl))
    
    # Optimize
    # Use Powell or Nelder-Mead
    # Powell is good for box constraints, but we have no explicit bounds here, just positivity via exp.
    # The domain is all real numbers for h and v.
    res = minimize(objective, x0, method='Powell', options={'maxiter': 200, 'disp': False})
    
    # Extract best params
    h_best = res.x[:9]
    v_best = res.x[9:]
    
    exp_h = np.exp(h_best - np.max(h_best))
    g_best = exp_h / np.sum(exp_h)
    nodes = np.cumsum(g_best[:8])
    
    exp_v = np.exp(v_best - np.max(v_best))
    weights = exp_v / np.sum(exp_v)
    
    # Sanity check
    if not np.all(np.diff(nodes) > 0):
        # Fallback to GL
        nodes = u_gl
        weights = w_gl
        
    return {"nodes": nodes, "weights": weights}
