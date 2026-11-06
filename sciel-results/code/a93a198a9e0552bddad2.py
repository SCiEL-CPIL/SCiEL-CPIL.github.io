import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

def construct():
    # Representative cosmologies: (Omega_m, w0, w1)
    # w = w0 + w1*(1-a) for CPL? No, standard CPL is w(a) = w0 + w1*(1-a).
    # H^2/H0^2 = Omega_m a^-3 + Omega_L a^(-3(1+w0+w1)) (1-a)^(3w1)
    # For flat universe, Omega_L = 1 - Omega_m.
    
    models = [
        (0.3, -1.0, 0.0),  # LCDM
        (0.3, -0.5, -0.5), # CPL
        (0.3, -1.5, 0.5),  # CPL
        (0.2, -1.0, 0.0),  # Low matter LCDM
        (0.4, -1.0, 0.0),  # High matter LCDM
    ]
    
    z_vals = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    def E_func(a, Om, w0, w1):
        OL = 1.0 - Om
        # Handle a=0 or a=1 carefully? a in (0,1) for z>0.
        term_m = Om * a**(-3.0)
        term_l = OL * a**(-3.0*(1.0+w0+w1)) * (1.0-a)**(3.0*w1)
        return np.sqrt(term_m + term_l)
    
    def true_integral(z, Om, w0, w1):
        # Integrate 1/E(1/(1+z*u)) from 0 to 1
        def f(u):
            a = 1.0 / (1.0 + z * u)
            return 1.0 / E_func(a, Om, w0, w1)
        val, _ = quad(f, 0, 1, limit=100)
        return val
    
    def approx_integral(nodes, weights, z, Om, w0, w1):
        # nodes: length 8, weights: length 8
        a_vals = 1.0 / (1.0 + z * nodes)
        e_vals = E_func(a_vals, Om, w0, w1)
        return np.sum(weights / e_vals)
    
    def objective(params):
        # params: [v1..v8, q1..q7]
        # Nodes derived from v's to ensure 0 < u1 < ... < u8 < 1
        # u_i = 0.5 * (1 + tanh(v_i))
        # To ensure ordering, let v_i be cumulative sums of positive steps?
        # Let's use a simpler mapping:
        # u_i = (i-1 + 0.5 + delta_i) / 8 ? No, that restricts range.
        
        # Better:
        # u_i = sigmoid(cumsum(s_i)) where s_i > 0?
        # Let's use:
        # v = params[:8]
        # q = params[8:15]
        
        v = params[:8]
        q = params[8:15]
        
        # Map v to nodes in (0,1) strictly increasing
        # Use softplus or exp to ensure steps are positive?
        # Let u_1 = sigmoid(v_1)
        # u_i = u_{i-1} + sigmoid(v_i) * (1 - u_{i-1}) / (8-i+1)?
        
        # Simpler robust mapping:
        # u_i = 1 / (1 + exp(-v_i))
        # Sort them? No, we want specific order.
        # Let's enforce ordering by construction:
        # u_1 = sigmoid(v_1)
        # u_i = u_{i-1} + (1 - u_{i-1}) * sigmoid(v_i) / (8 - i + 1)
        # This ensures u_i < 1 and u_i > u_{i-1} if sigmoid(v_i) > 0.
        
        nodes = np.zeros(8)
        u_prev = 0.0
        for i in range(8):
            # sigmoid
            s = 1.0 / (1.0 + np.exp(-v[i]))
            # Ensure step is small enough not to jump over 1?
            # The formula above distributes remaining space.
            u_prev = u_prev + (1.0 - u_prev) * s / (8 - i)
            nodes[i] = u_prev
        
        # Weights
        # q has 7 elements. w_8 = 1 - sum(w_1..w_7)
        # To ensure positivity, use softmax on 8 values?
        # We have 7 free params. Let's assume w_8 is the last one.
        # If we use softmax on 8 values, we need 8 params.
        # Let's use 8 params for weights, and constrain sum=1?
        # Minimize with constraints is harder.
        # Let's use 8 params for weights, normalize them.
        # params: [v1..v8, q1..q8]
        # Total 16 params.
        
        # Wait, I allocated 15 above. Let's use 16.
        # q = params[8:16]
        q = params[8:16]
        
        # Softmax
        exp_q = np.exp(q - np.max(q))
        weights = exp_q / np.sum(exp_q)
        
        # Calculate max relative error
        max_err = 0.0
        for Om, w0, w1 in models:
            for z in z_vals:
                true_val = true_integral(z, Om, w0, w1)
                approx_val = approx_integral(nodes, weights, z, Om, w0, w1)
                if true_val > 0:
                    rel_err = abs(approx_val - true_val) / true_val
                    if rel_err > max_err:
                        max_err = rel_err
        return max_err

    # Initial guess: Gauss-Legendre
    nodes_gl, weights_gl = np.polynomial.legendre.leggauss(8)
    nodes_gl = (nodes_gl + 1) / 2
    weights_gl = weights_gl * 0.5
    
    # Invert node mapping to get initial v's?
    # This is hard. Let's just start with random v's or v's that produce uniform nodes.
    # Uniform nodes: u_i = (i+0.5)/8.
    # We can solve for v's that produce these, or just start with v=0 (which gives specific nodes) and let optimizer move.
    # If v=0, s=0.5.
    # u_1 = 0 + 1*0.5/8 = 0.0625
    # u_2 = 0.0625 + 0.9375*0.5/7 = 0.0625 + 0.0669 = 0.129
    # ... This produces nodes clustered near 0?
    # Actually, the mapping `u_prev + (1-u_prev)*s/(N-i)` with s=0.5 tends to fill the space.
    
    # Let's just use a random start or the GL nodes directly if we could map them.
    # Since we can't easily invert the node mapping, let's start with v=0 and q=0.
    # q=0 -> uniform weights.
    
    x0 = np.zeros(16)
    
    # Optimize
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if res.success or res.fun < 1e-2: # If it found something reasonable
            v = res.x[:8]
            q = res.x[8:16]
            
            nodes = np.zeros(8)
            u_prev = 0.0
            for i in range(8):
                s = 1.0 / (1.0 + np.exp(-v[i]))
                u_prev = u_prev + (1.0 - u_prev) * s / (8 - i)
                nodes[i] = u_prev
            
            exp_q = np.exp(q - np.max(q))
            weights = exp_q / np.sum(exp_q)
            
            # Check constraints
            if np.all(np.diff(nodes) > 0) and np.all(weights > 0) and abs(np.sum(weights) - 1.0) < 1e-12:
                return {"nodes": nodes, "weights": weights}
                
    except Exception:
        pass
    
    # Fallback to Gauss-Legendre
    return {"nodes": nodes_gl, "weights": weights_gl}
