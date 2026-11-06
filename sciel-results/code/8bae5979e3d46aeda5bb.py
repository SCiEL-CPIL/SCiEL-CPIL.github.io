"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad

def construct():
    """
    Construct an 8-node positive quadrature rule using minimax optimization
    over a diverse set of cosmologies and redshifts.
    
    Strategy:
    1. Define a representative set of cosmologies (Omega_m, w0, wa)
    2. Precompute exact integrals using adaptive quadrature
    3. Optimize nodes and weights to minimize worst-case relative error
    4. Use a robust optimization method with time management
    """
    n = 8
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s limit
    
    def E_z(z, Omega_m, w0, wa):
        """H(z)/H0 for flat CPL model"""
        Omega_de = 1.0 - Omega_m
        z = np.asarray(z, dtype=float)
        term_m = Omega_m * (1.0 + z)**3
        term_de = Omega_de * (1.0 + z)**(3.0*(1.0 + w0 + wa*z))
        return np.sqrt(term_m + term_de)
    
    def exact_integral(z, Omega_m, w0, wa):
        """Exact value of integral_0^1 1/E(1/(1+z*t)) dt"""
        def f(t):
            a = 1.0 / (1.0 + z * t)
            return 1.0 / E_z(z * t, Omega_m, w0, wa)
        result, _ = quad(f, 0, 1, limit=200)
        return result
    
    def quadrature_value(nodes, weights, z, Omega_m, w0, wa):
        """Quadrature approximation of the integral"""
        z_nodes = z * nodes
        E_vals = E_z(z_nodes, Omega_m, w0, wa)
        return np.sum(weights / E_vals)
    
    # Define test cosmologies - diverse set covering parameter space
    test_cosmologies = [
        (0.3, -1.0, 0.0),    # Lambda CDM
        (0.3, -1.0, 0.5),    # CPL positive wa
        (0.3, -1.0, -0.5),   # CPL negative wa
        (0.2, -1.0, 0.0),    # Low Omega_m
        (0.4, -1.0, 0.0),    # High Omega_m
        (0.3, -0.8, 0.0),    # w0 > -1
        (0.3, -1.2, 0.0),    # w0 < -1
        (0.3, -1.0, 1.0),    # Large positive wa
        (0.3, -1.0, -1.0),   # Large negative wa
        (0.25, -0.9, 0.3),   # Mixed
        (0.35, -1.1, -0.2),  # Mixed
        (0.3, -0.5, 0.5),    # Quintessence-like
        (0.3, -1.5, -0.3),   # Phantom-like
        (0.2, -0.9, 0.2),    # Low Omega_m, varying w
        (0.4, -1.1, -0.3),   # High Omega_m, varying w
    ]
    
    # Test redshifts - log-spaced
    z_test_values = np.logspace(-2, 1.2, 20)  # z from 0.01 to ~16
    
    # Precompute exact integrals for all test cases
    print("Precomputing exact integrals...")
    exact_values = {}
    for i, (Om, w0, wa) in enumerate(test_cosmologies):
        for z in z_test_values:
            key = (i, z)
            exact_values[key] = exact_integral(z, Om, w0, wa)
    
    def worst_case_error(params):
        """Compute worst-case relative error"""
        if time.time() > deadline:
            return np.inf
        
        # Unpack: first n are logit-transformed nodes, next n are log-transformed weights
        node_logits = params[:n]
        weight_logs = params[n:2*n]
        
        # Convert to actual nodes in (0,1) using sigmoid
        nodes = 1.0 / (1.0 + np.exp(-node_logits))
        
        # Sort nodes and corresponding weights
        sort_idx = np.argsort(nodes)
        nodes = nodes[sort_idx]
        weight_logs = weight_logs[sort_idx]
        
        # Check strict increase with margin
        if np.any(np.diff(nodes) <= 1e-8):
            return np.inf
        
        # Check nodes in (0,1)
        if np.any(nodes <= 1e-6) or np.any(nodes >= 1 - 1e-6):
            return np.inf
        
        # Convert to actual weights
        weights = np.exp(weight_logs)
        weights = weights / np.sum(weights)
        
        # Compute worst-case error
        max_err = 0.0
        for i, (Om, w0, wa) in enumerate(test_cosmologies):
            for z in z_test_values:
                key = (i, z)
                exact = exact_values[key]
                if abs(exact) < 1e-15:
                    continue
                quad_val = quadrature_value(nodes, weights, z, Om, w0, wa)
                rel_err = abs(quad_val - exact) / abs(exact)
                if rel_err > max_err:
                    max_err = rel_err
                if time.time() > deadline:
                    return np.inf
        
        return max_err * 100.0  # in percent
    
    # Initial guess: Gauss-Legendre nodes on [0,1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(n)
    init_nodes = (x_gl + 1.0) / 2.0
    init_weights = w_gl / 2.0
    init_weights = init_weights / np.sum(init_weights)
    
    # Convert to logit and log space
    init_node_logits = np.log(init_nodes / (1.0 - init_nodes))
    init_weight_logs = np.log(init_weights)
    
    init_params = np.concatenate([init_node_logits, init_weight_logs])
    
    best_params = init_params.copy()
    best_error = worst_case_error(init_params)
    print(f"Initial error: {best_error:.6f}%")
    
    # Try multiple optimization strategies
    strategies = [
        ('Nelder-Mead', {'maxiter': 300, 'fatol': 1e-10, 'adaptive': True}),
        ('Powell', {'maxiter': 200, 'ftol': 1e-10}),
    ]
    
    for method, options in strategies:
        if time.time() > deadline:
            break
        
        try:
            print(f"Running {method}...")
            result = minimize(
                worst_case_error,
                best_params,
                method=method,
                options=options
            )
            
            if result.fun < best_error:
                best_error = result.fun
                best_params = result.x
                print(f"  {method} improved to: {best_error:.6f}%")
            else:
                print(f"  {method} did not improve: {result.fun:.6f}%")
                
        except Exception as e:
            print(f"  {method} failed: {e}")
    
    # If still have time, try differential evolution for global search
    if time.time() < deadline - 20:
        try:
            print("Running differential evolution...")
            # Define bounds
            bounds = []
            for _ in range(n):
                bounds.append((-10, 10))  # logit bounds
            for _ in range(n):
                bounds.append((-10, 10))  # log weight bounds
            
            def de_objective(params):
                if time.time() > deadline:
                    return np.inf
                return worst_case_error(params)
            
            de_result = differential_evolution(
                de_objective,
                bounds,
                maxiter=50,
                tol=1e-8,
                popsize=15,
                seed=42,
                polish=False
            )
            
            if de_result.fun < best_error:
                best_error = de_result.fun
                best_params = de_result.x
                print(f"  DE improved to: {best_error:.6f}%")
            else:
                print(f"  DE did not improve: {de_result.fun:.6f}%")
                
        except Exception as e:
            print(f"  DE failed: {e}")
    
    # Extract final nodes and weights
    node_logits = best_params[:n]
    weight_logs = best_params[n:2*n]
    
    nodes = 1.0 / (1.0 + np.exp(-node_logits))
    sort_idx = np.argsort(nodes)
    nodes = nodes[sort_idx]
    weight_logs = weight_logs[sort_idx]
    
    weights = np.exp(weight_logs)
    weights = weights / np.sum(weights)
    
    # Final validation and fallback
    valid = True
    if not (np.all(nodes > 1e-6) and np.all(nodes < 1 - 1e-6)):
        valid = False
    if not (np.all(np.diff(nodes) > 1e-8)):
        valid = False
    if not np.all(weights > 0):
        valid = False
    if abs(np.sum(weights) - 1.0) > 1e-12:
        weights = weights / np.sum(weights)
    
    if not valid:
        print("Falling back to Gauss-Legendre")
        x_gl, w_gl = np.polynomial.legendre.leggauss(n)
        nodes = (x_gl + 1.0) / 2.0
        weights = w_gl / 2.0
        weights = weights / np.sum(weights)
    
    # Ensure weights sum to 1 within tolerance
    weights = weights / np.sum(weights)
    
    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.1f}s, final error: {best_error:.6f}%")
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
