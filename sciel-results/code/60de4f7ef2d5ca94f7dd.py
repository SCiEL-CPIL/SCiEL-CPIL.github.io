"""
Strategy: Decouple node and weight optimization.
Fix nodes to Gauss-Legendre points (optimal for smooth integrands on [0,1]).
Optimize weights using Linear Programming (Minimax) to minimize the worst-case 
relative error across a diverse set of CPL cosmologies and redshifts.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog, minimize
import time

def _E_cpl(a, w0, wa, om_m, om_r, om_k=0.0):
    """E(a) = H(a)/H0 for CPL model."""
    a_safe = np.maximum(a, 1e-10)
    # CPL: w(z) = w0 + wa * z / (1+z)
    # rho_de(a) = rho_de0 * a^{-3(1+w0+wa*(1-a)/a)}
    # Note: The standard CPL form in H(z) is usually written with z.
    # H(z)^2 = H0^2 [Om_m(1+z)^3 + Om_r(1+z)^4 + Om_k + Om_de(1+z)^{3(1+w0+wa)}]
    # In terms of a = 1/(1+z):
    # H(a)^2 = H0^2 [Om_m a^{-3} + Om_r a^{-4} + Om_k + Om_de a^{-3(1+w0+wa*(1-a)/a)}]
    # Wait, the exponent in the a-form is tricky.
    # Let's stick to the z-form definition which is standard:
    # E(z)^2 = Om_m(1+z)^3 + Om_r(1+z)^4 + Om_k + Om_de(1+z)^{3(1+w0+wa)}
    # a = 1/(1+z) => 1+z = 1/a
    # E(a)^2 = Om_m a^{-3} + Om_r a^{-4} + Om_k + Om_de a^{-3(1+w0+wa*(1-a)/a)}?
    # Actually, (1+z)^{3(1+w0+wa)} = (1/a)^{3(1+w0+wa*(1-a)/a)}? No.
    # w(z) = w0 + wa * z/(1+z).
    # Integral of 3(1+w(z))/(1+z) dz = 3(1+w0)z + 3wa ln(1+z).
    # So rho_de ~ (1+z)^{3(1+w0)} * (1+z)^{3wa} = (1+z)^{3(1+w0+wa*z/(1+z))}.
    # In terms of a: (1/a)^{3(1+w0+wa*(1-a)/a)}.
    
    # Let's use the z-form directly for safety and clarity, converting a to z.
    z = 1.0 / a_safe - 1.0
    h2 = om_m * (1 + z)**3 + om_r * (1 + z)**4 + om_k + \
         (1 - om_m - om_r - om_k) * (1 + z)**(3 * (1 + w0 + wa * z / (1 + z)))
    return np.sqrt(np.maximum(h2, 1e-30))

def _true_dL_integral(z, w0, wa, om_m, om_r, om_k, n=2000):
    """Compute the true integral value using high-resolution Simpson's rule."""
    # Integrand is 1/E(a(x)) where a(x) = 1/(1+z*x)
    # We integrate from x=0 to x=1
    x = np.linspace(0, 1, n)
    if n % 2 == 0:
        n += 1
        x = np.linspace(0, 1, n)
    
    a_vals = 1.0 / (1.0 + z * x)
    E_vals = _E_cpl(a_vals, w0, wa, om_m, om_r, om_k)
    y = 1.0 / E_vals
    
    h = 1.0 / (n - 1)
    integral = h / 3.0 * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2]))
    return integral

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Define test set: (w0, wa, om_m, om_r, om_k)
    # We want to cover a broad range of plausible cosmologies
    models = [
        (0.0, 0.0, 0.3, 0.05, 0.65),   # Lambda CDM
        (-0.5, 0.5, 0.3, 0.05, 0.65),  # Quintessence
        (-1.5, 0.5, 0.3, 0.05, 0.65),  # Phantom
        (-1.0, 1.0, 0.2, 0.05, 0.75),  # Varying DE
        (0.5, -0.5, 0.4, 0.05, 0.55),  # Decelerating early?
        (-0.9, 0.3, 0.3, 0.0, 0.70),   # No radiation
        (-1.1, 0.0, 0.25, 0.05, 0.70), # More DE
        (0.0, 1.0, 0.3, 0.05, 0.65),   # Strong wa
    ]
    
    # Test redshifts
    z_vals = [0.5, 1.0, 2.0, 3.0, 5.0]
    
    # Generate test cases
    test_cases = []
    for m in models:
        for z in z_vals:
            test_cases.append((m, z))
            
    n_cases = len(test_cases)
    
    # Pre-compute true values and node positions for all cases
    # We will try a few node sets. Start with Gauss-Legendre.
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes = (x_gl + 1.0) / 2.0
    weights_init = w_gl / 2.0
    
    # Pre-compute 1/E(a_i) for each case and node
    # A[i, j] = 1/E(a_j) for case i
    A = np.zeros((n_cases, 8))
    true_vals = np.zeros(n_cases)
    
    for i, (m, z) in enumerate(test_cases):
        w0, wa, om_m, om_r, om_k = m
        a_vals = 1.0 / (1.0 + z * nodes)
        E_vals = _E_cpl(a_vals, w0, wa, om_m, om_r, om_k)
        A[i, :] = 1.0 / E_vals
        true_vals[i] = _true_dL_integral(z, w0, wa, om_m, om_r, om_k)
        
    # We want to find weights w such that sum(w_j * A_ij) approximates true_vals_i
    # Minimize max_i | sum_j w_j A_ij - true_vals_i |
    # Let e_i = sum_j w_j A_ij - true_vals_i
    # Minimize max |e_i|
    # This is a LP:
    # Minimize t
    # Subject to:
    #   sum_j w_j A_ij - true_vals_i <= t
    #   -(sum_j w_j A_ij - true_vals_i) <= t  => -sum_j w_j A_ij + true_vals_i <= t
    #   sum_j w_j = 1
    #   w_j > 0 (We can enforce w_j >= 0, positivity is usually maintained if nodes are well placed)
    
    # Variables: w_0 ... w_7, t
    # Total 9 variables
    
    c = np.zeros(9)
    c[8] = 1.0  # Minimize t
    
    # Inequality constraints: A_ub x <= b_ub
    # For each case i:
    #   A_i . w - t <= true_vals_i
    #   -A_i . w - t <= -true_vals_i
    
    A_ub = np.zeros((2 * n_cases, 9))
    b_ub = np.zeros(2 * n_cases)
    
    for i in range(n_cases):
        # Row 2i: A_i . w - t <= true_vals_i
        A_ub[2*i, :8] = A[i, :]
        A_ub[2*i, 8] = -1.0
        b_ub[2*i] = true_vals[i]
        
        # Row 2i+1: -A_i . w - t <= -true_vals_i
        A_ub[2*i+1, :8] = -A[i, :]
        A_ub[2*i+1, 8] = -1.0
        b_ub[2*i+1] = -true_vals[i]
        
    # Equality constraints: sum w_j = 1
    A_eq = np.zeros((1, 9))
    A_eq[0, :8] = 1.0
    b_eq = np.array([1.0])
    
    # Bounds: w_j >= 0, t >= 0
    bounds = [(0, None)] * 8 + [(0, None)]
    
    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if res.success:
            weights = res.x[:8]
            # Ensure weights are strictly positive and sum to 1
            # LP might return 0 for some weights if they are not needed, 
            # but the problem structure usually keeps them positive for smooth integrands.
            # If any weight is 0, add a tiny epsilon and renormalize? 
            # The task requires weights > 0.
            if np.any(weights <= 0):
                weights = weights + 1e-6
                weights = weights / np.sum(weights)
            else:
                weights = weights / np.sum(weights)
            
            # Verify error
            quad_vals = A @ weights
            rel_errors = np.abs(quad_vals - true_vals) / true_vals
            max_err = np.max(rel_errors)
            
            # If time remains, we can try other node sets
            if time.time() < deadline - 5:
                # Try Gauss-Lobatto?
                # GL nodes are better for smooth functions. GL is usually superior for 1/E(a).
                # We could also try optimizing nodes slightly around GL using Nelder-Mead on the LP result?
                # No, LP is only for fixed nodes.
                # Let's stick to the LP result for GL nodes.
                pass
        else:
            # Fallback to Gauss-Legendre weights if LP fails
            weights = weights_init
    except Exception:
        weights = weights_init
    
    # Final sanity check
    if not np.all(nodes > 0) or not np.all(nodes < 1):
        nodes = (x_gl + 1.0) / 2.0
        weights = w_gl / 2.0
    
    if not np.all(np.diff(nodes) > 0):
        nodes = (x_gl + 1.0) / 2.0
        weights = w_gl / 2.0
        
    # Normalize weights
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}

# EVOLVE-BLOCK-END
