"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.integrate import quad
import sys

def construct():
    """
    Construct an 8-node positive quadrature rule for the dimensionless
    luminosity distance integral in flat Lambda-CDM-like models.
    
    The integral to approximate is:
        J(z) = ∫_0^1 dt / E(1/(1+zt))
    such that dL/(c/H0) = (1+z)*z*J(z).
    
    We seek nodes t_i and weights w_i to minimize the worst-case relative error
    across a set of representative cosmologies and redshifts.
    """
    
    n = 8
    
    # Representative cosmologies: (w0, wa, Om)
    # We use the CPL parametrization for Dark Energy.
    # E(a)^2 = Om*a^(-3) + (1-Om)*a^(-3*(1+w0+wa))*(1+a*wa)^(-3)
    # This is an approximation but captures the essential behavior for a wide range.
    # Note: For strict correctness in the evaluator, the evaluator uses its own integrands.
    # We just need to optimize against a diverse set of likely integrands.
    
    def make_E(w0, wa, Om):
        def E(a):
            # Avoid division by zero or negative bases if a is very small
            a_safe = np.maximum(a, 1e-10)
            term_matter = Om * a_safe**(-3)
            term_de = (1 - Om) * a_safe**(-3 * (1 + w0 + wa)) * (1 + a_safe * wa)**(-3)
            return np.sqrt(term_matter + term_de)
        return E

    # Set of test cosmologies
    # We want to cover Lambda-CDM, Phantom, Quintessence, and varying Om
    cosmos_params = [
        (-1.0, 0.0, 0.3),   # Lambda-CDM
        (-1.0, 0.0, 0.2),   # Lower Om
        (-1.0, 0.0, 0.4),   # Higher Om
        (-0.9, -0.3, 0.3),  # wCDM
        (-1.1, 0.5, 0.3),   # Quintessence
        (-0.8, -0.5, 0.3),  # Phantom
        (-1.0, 0.2, 0.3),   # Mildly evolving
        (-1.2, 0.8, 0.3),   # Strongly evolving
        (-1.0, -0.2, 0.25), # Another variant
    ]
    
    test_cosmos = [make_E(*p) for p in cosmos_params]
    
    # Test redshifts: Log-spaced from low to high
    test_zs = np.logspace(-1, np.log10(5.0), 15) # 0.1 to 5.0
    
    def objective(params):
        """
        Minimize worst-case relative error.
        params: [d1, d2, ..., d7, log_w1, ..., log_w8]
        d_i are differences for nodes: t_1 = 1e-5 + d1, t_{i+1} = t_i + d_{i+1}
        This ensures strictly increasing nodes in (0, 1).
        We need sum(d_i) < 1 - 2e-5.
        """
        # Unpack params
        diffs = params[:n-1]
        log_weights = params[n-1:]
        
        # Reconstruct nodes
        # Ensure diffs are positive and sum is within bounds
        # If any diff is <= 0, penalty
        if np.any(diffs <= 0):
            return 1e6
        
        t1 = 1e-5
        nodes = np.zeros(n)
        nodes[0] = t1
        for i in range(1, n):
            nodes[i] = nodes[i-1] + diffs[i-1]
            
        if nodes[-1] > 1.0 - 1e-5:
            return 1e6
            
        # Reconstruct weights
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights)
        
        max_rel_err = 0.0
        
        for E_func in test_cosmos:
            for z in test_zs:
                # Compute exact integral
                def integrand(t):
                    a = 1.0 / (1.0 + z * t)
                    return 1.0 / E_func(a)
                
                try:
                    exact, _ = quad(integrand, 0, 1, limit=50, epsabs=1e-10, epsrel=1e-10)
                except Exception:
                    return 1e6
                
                if exact <= 0:
                    continue
                
                # Compute quadrature approximation
                approx = 0.0
                for i in range(n):
                    a_i = 1.0 / (1.0 + z * nodes[i])
                    approx += weights[i] / E_func(a_i)
                
                rel_err = abs(approx - exact) / exact
                if rel_err > max_rel_err:
                    max_rel_err = rel_err
                    
        return max_rel_err

    # Define bounds for differential evolution
    # diffs: 7 values, each > 0. Sum < 1.
    # Let's bound each diff to [1e-6, 0.5]
    bounds_diff = [(1e-6, 0.5)] * (n - 1)
    # log_weights: 8 values. Weights sum to 1, so individual weights < 1.
    # log(w) < 0. Let's bound log_w to [-10, 5] (weights from 1e-4 to 148, normalized).
    bounds_logw = [(-10, 5)] * n
    
    bounds = bounds_diff + bounds_logw
    
    # Initial guess: Gauss-Legendre mapped to [0,1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(n)
    nodes_gl = 0.5 * (x_gl + 1)
    weights_gl = 0.5 * w_gl
    
    # Convert GL nodes to diffs
    diffs_gl = np.diff(nodes_gl)
    # Ensure diffs are within bounds
    diffs_gl = np.clip(diffs_gl, 1e-6, 0.5)
    # Adjust last diff to ensure sum < 1? 
    # Actually, DE handles this. We just provide a good start.
    # But DE start is random by default unless x0 is provided.
    # Let's provide x0 based on GL.
    
    log_weights_gl = np.log(weights_gl)
    x0 = np.concatenate([diffs_gl, log_weights_gl])
    
    # Run optimization
    # maxiter and popsize control cost. 60s limit.
    # 9 cosmologies * 15 zs = 135 evaluations per objective call.
    # quad is fast. 
    # Let's try 100 iterations, popsize 20.
    # This might be slow. Let's reduce cosmologies or zs if needed.
    # For now, keep it robust.
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=15, 
            tol=1e-6, 
            polish=True,
            x0=x0
        )
        
        # Extract solution
        best_params = result.x
        diffs = best_params[:n-1]
        log_weights = best_params[n-1:]
        
        nodes = np.zeros(n)
        nodes[0] = 1e-5
        for i in range(1, n):
            nodes[i] = nodes[i-1] + diffs[i-1]
            
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights)
        
        # Final check
        if not np.all(nodes > 0) or not np.all(nodes < 1) or not np.all(np.diff(nodes) > 0):
            # Fallback to GL
            return {"nodes": nodes_gl, "weights": weights_gl}
            
        if not np.all(weights > 0) or abs(np.sum(weights) - 1.0) > 1e-12:
            return {"nodes": nodes_gl, "weights": weights_gl}

        return {"nodes": nodes, "weights": weights}
        
    except Exception as e:
        # Fallback to Gauss-Legendre if optimization fails
        return {"nodes": nodes_gl, "weights": weights_gl}

# EVOLVE-BLOCK-END
