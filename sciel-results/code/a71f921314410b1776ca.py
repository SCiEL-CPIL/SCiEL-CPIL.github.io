import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
import time
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.chebyshev import chebgauss

def construct():
    t0 = time.time()
    deadline = 85.0  # Safety margin

    # Standard Flat Cosmology Parameters
    OMEGA_M = 0.3
    OMEGA_DE = 0.7

    def E_of_z(z, w0, wa):
        """H(z)/H0 for CPL model with standard Omega_m, Omega_de"""
        # Handle z=0 separately to avoid log issues if any, though formula is fine
        # w = w0 + wa * z
        # E^2 = Omega_m (1+z)^3 + Omega_de (1+z)^(3(1+w))
        z_safe = np.maximum(z, 0.0)
        w = w0 + wa * z_safe
        # Use log-space for stability if z is large, but z is in [0,1] so direct is fine
        term_m = OMEGA_M * (1.0 + z_safe)**3
        term_de = OMEGA_DE * (1.0 + z_safe)**(3.0 * (1.0 + w))
        E2 = term_m + term_de
        return np.sqrt(np.maximum(E2, 1e-30))

    def integrand(z, w0, wa):
        """The integrand for dL/(c/H0) = integral_0^z (1+z')/E(z') dz'"""
        # Note: The task formula is dL/(c/H0) = (1+z)*z*sum w_i/E(a_i)
        # This implies the integral is \int_0^z \frac{1+z'}{E(z')} dz'
        # Wait, let's verify the quadrature formula in the prompt.
        # dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
        # a_i = 1/(1+z*nodes[i])
        # This looks like a composite quadrature for \int_0^z f(t) dt.
        # Let t = z * x. dt = z dx.
        # \int_0^1 z * f(z*x) dx = \int_0^z f(t) dt.
        # f(t) = (1+t)/E(t).
        # So the quadrature approximates \int_0^z \frac{1+t}{E(t)} dt.
        # The formula in prompt: (1+z)*z*sum w_i/E(a_i).
        # a_i = 1/(1+z*x_i) => 1+a_i = 1 + 1/(1+z*x_i) ? No.
        # a = 1/(1+z). So 1+z = 1/a.
        # (1+z)*z*sum w_i/E(a_i) = z * (1/a_i? No, 1+z is the upper limit factor?)
        
        # Let's check the standard luminosity distance formula:
        # d_L = (c/H0) (1+z) \int_0^z \frac{dz'}{H(z')}
        # = (c/H0) (1+z) \int_0^z \frac{dz'}{H0 E(z')}
        # = (c/H0) \frac{1+z}{H0} \int_0^z \frac{dz'}{E(z')} ? No.
        # d_L = (c/H0) (1+z) \int_0^z \frac{dz'}{H(z')}
        # Let's substitute t = z'.
        # The prompt says: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
        # This implies the integral being approximated is \int_0^z \frac{1}{E(z')} dz' ?
        # If \int_0^z \frac{dz'}{E(z')} \approx z \sum w_i \frac{1}{E(z x_i)}
        # Then dL/(c/H0) = (1+z) * [ z \sum w_i \frac{1}{E(z x_i)} ]
        # This matches the prompt formula EXACTLY.
        # So the integrand to approximate is f(z') = 1/E(z').
        # The quadrature nodes x_i are in [0,1]. The points in z-space are z*x_i.
        # a_i = 1/(1+z*x_i).
        # So we need to approximate \int_0^z \frac{dz'}{E(z')} using nodes x_i and weights w_i.
        
        return 1.0 / E_of_z(z, w0, wa)

    def reference_integral(z, w0, wa):
        """High accuracy reference for \int_0^z 1/E(z') dz'"""
        # Use scipy.integrate.quad for high precision
        val, err = quad(integrand, 0, z, args=(w0, wa), limit=100)
        return val

    def compute_quadrature_value(z, nodes, weights, w0, wa):
        """Compute (1+z) * z * sum w_i / E(z * nodes_i)"""
        z_nodes = z * nodes
        E_vals = E_of_z(z_nodes, w0, wa)
        integral_approx = z * np.sum(weights / E_vals)
        return (1.0 + z) * integral_approx

    # Define training models and redshifts
    # We need to optimize for the "worst relative error".
    # The error is |Quad - Ref| / |Ref|.
    # Since the factor (1+z) is common to both Quad and Ref (for the same z),
    # the relative error of dL is the same as the relative error of the integral \int_0^z 1/E.
    # So we can just minimize the error of the integral approximation.
    
    models = [
        (-1.0, 0.0),   # LambdaCDM
        (-0.5, 0.0),   # Quintessence
        (-1.5, 0.0),   # Phantom
        (-0.9, 0.3),   # CPL
        (-1.1, -0.2),  # CPL
        (0.0, 0.0),    # w=0
        (-0.7, 0.7),   # Strong variation
    ]
    
    z_list = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]
    
    # Precompute references
    # We will evaluate the objective function over these (z, model) pairs
    # To make it fast, we precompute the reference values.
    refs = []
    for z in z_list:
        for w0, wa in models:
            refs.append(reference_integral(z, w0, wa))
    refs = np.array(refs)
    
    # Flatten z and models for vectorized evaluation in objective
    z_flat = np.repeat(z_list, len(models))
    models_flat = np.tile(models, len(z_list))
    
    def objective(params):
        """Minimize max relative error of the integral approximation."""
        if time.time() > deadline - 5:
            return 1e10
            
        node_params = params[:8]
        weight_params = params[8:16]
        
        # Map node_params to (0,1) using sigmoid, then sort
        raw_nodes = 1.0 / (1.0 + np.exp(-node_params))
        sorted_nodes = np.sort(raw_nodes)
        # Ensure strict increase by adding small epsilon
        nodes = sorted_nodes + np.arange(8) * 1e-6
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        
        # Weights via softmax
        exp_w = np.exp(weight_params - np.max(weight_params))
        weights = exp_w / np.sum(exp_w)
        
        max_err = 0.0
        # Vectorized computation for speed
        # z_flat has shape (N), nodes has shape (8)
        # We need z_flat[:, None] * nodes[None, :]
        z_nodes = z_flat[:, None] * nodes[None, :]
        
        # Compute E for all pairs
        # E_of_z expects array. 
        # We need to compute E for each (z, node) pair.
        # This might be memory heavy if N is large, but N is small (6*7=42).
        
        # Loop is fine for N=42
        for i in range(len(z_flat)):
            z_val = z_flat[i]
            w0, wa = models_flat[i]
            
            z_nodes_i = z_val * nodes
            E_vals = E_of_z(z_nodes_i, w0, wa)
            integral_approx = z_val * np.sum(weights / E_vals)
            
            ref_val = refs[i]
            
            if ref_val > 1e-10:
                rel_err = abs(integral_approx - ref_val) / abs(ref_val)
                if rel_err > max_err:
                    max_err = rel_err
                    
        return max_err

    # Initializations
    # 1. Gauss-Legendre
    gl_nodes, gl_weights = leggauss(8)
    gl_nodes = 0.5 * (gl_nodes + 1.0)
    # Invert sigmoid
    x0_gl_nodes = np.log(gl_nodes / (1.0 - gl_nodes))
    x0_gl_weights = np.log(gl_weights)
    x0_gl = np.concatenate([x0_gl_nodes, x0_gl_weights])
    
    # 2. Gauss-Chebyshev
    ch_nodes, ch_weights = chebgauss(8)
    ch_nodes = 0.5 * (ch_nodes + 1.0)
    x0_ch_nodes = np.log(ch_nodes / (1.0 - ch_nodes))
    x0_ch_weights = np.log(ch_weights)
    x0_ch = np.concatenate([x0_ch_nodes, x0_ch_weights])
    
    best_params = x0_gl.copy()
    best_err = objective(x0_gl)
    
    # Local Refinement from GL
    if time.time() < deadline - 15:
        try:
            res_gl = minimize(
                objective, x0_gl, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if res_gl.fun < best_err:
                best_err = res_gl.fun
                best_params = res_gl.x
        except Exception:
            pass

    # Local Refinement from Chebyshev
    if time.time() < deadline - 15:
        try:
            res_ch = minimize(
                objective, x0_ch, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if res_ch.fun < best_err:
                best_err = res_ch.fun
                best_params = res_ch.x
        except Exception:
            pass

    # Optional: Differential Evolution if time permits
    if time.time() < deadline - 30:
        try:
            bounds = [(-5, 5)] * 16
            de_res = differential_evolution(
                objective, bounds, seed=42, maxiter=50, popsize=15,
                tol=1e-8, polish=False
            )
            if de_res.fun < best_err:
                best_err = de_res.fun
                best_params = de_res.x
                
                # Refine DE result
                if time.time() < deadline - 5:
                    res_de = minimize(
                        objective, best_params, method='Nelder-Mead',
                        options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
                    )
                    if res_de.fun < best_err:
                        best_err = res_de.fun
                        best_params = res_de.x
        except Exception:
            pass

    # Extract final nodes and weights
    node_params = best_params[:8]
    weight_params = best_params[8:16]
    
    raw_nodes = 1.0 / (1.0 + np.exp(-node_params))
    sorted_nodes = np.sort(raw_nodes)
    nodes = sorted_nodes + np.arange(8) * 1e-6
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    
    exp_w = np.exp(weight_params - np.max(weight_params))
    weights = exp_w / np.sum(exp_w)
    
    # Final normalization check
    weights = weights / np.sum(weights)

    return {"nodes": nodes, "weights": weights}
