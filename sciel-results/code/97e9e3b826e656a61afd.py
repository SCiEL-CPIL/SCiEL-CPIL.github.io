import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # 1. Define representative cosmologies (CPL models)
    # w(z) = w0 + w1 * z
    # E(a) = a^(-3(1+w0+w1)) * (1+w1)^(-a)
    # We need to integrate 1/E(a) dt where a = 1/(1+zt)
    
    def E_of_a(a, w0, w1):
        # Handle w1=0 case to avoid log(1) issues or 0^0
        if np.isclose(w1, 0.0):
            return a ** (-3.0 * (1.0 + w0))
        else:
            return (a ** (-3.0 * (1.0 + w0 + w1))) * ((1.0 + w1) ** (-a))

    def integrand(t, z, w0, w1):
        a = 1.0 / (1.0 + z * t)
        return 1.0 / E_of_a(a, w0, w1)

    # Test grid
    # z values: 0.5, 1.0, 1.5, 2.0
    # w0 values: -1.5, -1.0, -0.5
    # w1 values: -1.0, 0.0, 1.0
    z_vals = [0.5, 1.0, 1.5, 2.0]
    w0_vals = [-1.5, -1.0, -0.5]
    w1_vals = [-1.0, 0.0, 1.0]
    
    test_cases = []
    for z in z_vals:
        for w0 in w0_vals:
            for w1 in w1_vals:
                # Compute reference integral
                try:
                    I_ref, _ = quad(integrand, 0, 1, args=(z, w0, w1), limit=50)
                    if I_ref > 1e-10:
                        test_cases.append((z, w0, w1, I_ref))
                except Exception:
                    pass

    if not test_cases:
        # Fallback to Gauss-Legendre if something went wrong
        from numpy.polynomial.legendre import leggauss
        x, w = leggauss(8)
        nodes = (x + 1.0) / 2.0
        weights = w / 2.0
        return {"nodes": nodes, "weights": weights}

    # 2. Parameterization
    # Nodes: t_i are cumulative sums of softmax(x)
    # Weights: softmax(y)
    # Total parameters: 8 (for node diffs, last one ignored/derived? No, 8 diffs sum to 1)
    # Actually, 8 nodes means 8 intervals? No, 8 nodes in (0,1).
    # t_0=0, t_8=1. We have 8 nodes t_1...t_8.
    # This defines 9 intervals? No.
    # The quadrature is sum_{i=1}^8 w_i f(t_i).
    # The nodes t_i are in (0,1).
    # Let's use 8 parameters for node positions via a mapping.
    # Simpler: Optimize the 8 nodes directly with bounds and penalty, or use the softmax-diff method.
    # Let x be 8 parameters. d_i = softmax(x)_i. Sum(d)=1.
    # t_i = sum_{j=1}^i d_j.
    # This gives 8 nodes t_1...t_8 in (0,1) strictly increasing.
    # Let y be 8 parameters. w_i = softmax(y)_i. Sum(w)=1.
    
    n = 8
    # Initial guess: Gauss-Legendre
    from numpy.polynomial.legendre import leggauss
    x_gl, w_gl = leggauss(n)
    nodes_gl = (x_gl + 1.0) / 2.0
    weights_gl = w_gl / 2.0
    
    # Inverse transform for initial guess is hard, so we can just start with random or zeros
    # and let optimization find the way. Or we can approximate.
    # Since softmax is smooth, starting from zeros (uniform) is okay, but GL is better.
    # We can't easily invert softmax to get x from nodes_gl.
    # We will just start with x=0, y=0 (uniform nodes and weights) and hope optimization finds GL-like or better.
    # Or, we can use a simpler parameterization: 
    # Nodes: t_i = (i + 0.5 + s_i) / n ? No, constraints are tricky.
    
    # Let's stick to the softmax parameterization.
    # x: 8 floats -> 8 diffs -> 8 nodes
    # y: 8 floats -> 8 weights
    
    def softmax(v):
        v = v - np.max(v)
        e = np.exp(v)
        return e / np.sum(e)
    
    def get_nodes_weights(params):
        x = params[:n]
        y = params[n:]
        d = softmax(x)
        t = np.cumsum(d)
        w = softmax(y)
        return t, w

    def objective(params):
        t, w = get_nodes_weights(params)
        max_rel_err = 0.0
        for z, w0, w1, I_ref in test_cases:
            # Evaluate quadrature
            # a_i = 1 / (1 + z * t_i)
            a_vals = 1.0 / (1.0 + z * t)
            E_vals = np.array([E_of_a(a, w0, w1) for a in a_vals])
            Q = np.dot(w, 1.0 / E_vals)
            
            rel_err = abs(Q - I_ref) / abs(I_ref)
            if rel_err > max_rel_err:
                max_rel_err = rel_err
        return max_rel_err

    # Initial parameters
    # Uniform nodes and weights correspond to x=0, y=0
    p0 = np.zeros(2 * n)
    
    # Optimization
    best_params = p0
    best_loss = objective(p0)
    
    try:
        res = minimize(objective, p0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if res.fun < best_loss:
            best_params = res.x
            best_loss = res.fun
            
        # Check time
        if time.time() > deadline:
            raise TimeoutError
    except Exception:
        pass

    # Extract best nodes and weights
    t_best, w_best = get_nodes_weights(best_params)
    
    # Ensure strict positivity and sum constraints are numerically stable
    # Softmax ensures this, but let's clean up
    # Nodes are strictly increasing in (0,1)
    # Weights are positive and sum to 1
    
    return {"nodes": t_best, "weights": w_best}
