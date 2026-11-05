"""Optimized 8-node positive quadrature for luminosity distance."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    n = 8

    # Define a set of CPL (Chevallier-Polarski-Linder) cosmologies to train on.
    # E(a) = (Omega_m * a^-3 + Omega_k * a^-2 + Omega_de * a^(-3(1+w0+wa)) )^0.5
    # For flat universe, Omega_k = 0.
    # Let's use a representative set of (w0, wa) pairs.
    # The validation set is disjoint, so we want to cover a reasonable range.
    # Typical ranges: w0 in [-1.5, 0.5], wa in [-1.0, 1.0].
    # We'll use a grid of training cosmologies.

    def make_E(w0, wa):
        Omega_m = 0.3  # Standard value
        Omega_de = 1.0 - Omega_m
        def E(a):
            # a in (0, 1]
            # Handle a=0 carefully, but our nodes will be in (0,1) so a>0
            return np.sqrt(Omega_m * a**(-3) + Omega_de * a**(-3*(1+w0+wa)))
        return E

    # Training cosmologies (w0, wa)
    training_cosmos = [
        (-1.0, 0.0),   # Lambda CDM
        (-1.0, 0.5),
        (-1.0, -0.5),
        (-0.5, 0.5),
        (-0.5, -0.5),
        (-1.5, 0.5),
        (-1.5, -0.5),
        (0.0, 0.5),
        (0.0, -0.5),
        (-1.0, 1.0),
        (-1.0, -1.0),
    ]

    # Pre-compute reference integrals for a set of redshifts
    # The integral is: dL/(c/H0) = (1+z) * integral_0^z dz' / E(1/(1+z'))
    # Let u = 1/(1+z'), then z' = 1/u - 1, dz' = -1/u^2 du
    # When z'=0, u=1. When z'=z, u=1/(1+z).
    # Integral becomes: integral_{1/(1+z)}^1 (1/u^2) / E(u) du
    # = integral_{1/(1+z)}^1 1/(u^2 * E(u)) du
    
    # Alternatively, in terms of a:
    # dL/(c/H0) = (1+z) * integral_0^z dz'/E(a')
    # Let a' = 1/(1+z'). dz' = -da'/a'^2.
    # Limits: z'=0 -> a'=1, z'=z -> a'=1/(1+z)=a.
    # Integral = (1+z) * integral_a^1 (1/a'^2) / E(a') da'
    
    # The quadrature rule approximates: sum_i weights[i] / E(a_i)
    # where a_i = 1/(1+z*nodes[i])
    # So the quadrature is for the integral: integral_0^1 (1/t^2) / E(1/t) dt ? No.
    
    # Let's re-read the task carefully:
    # dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # where a_i = 1/(1+z*nodes[i])
    
    # This means the quadrature is approximating:
    # integral_0^1 f(z*t) dt = sum_i weights[i] * f(z*nodes[i])
    # where f(u) = 1/E(1/(1+u))
    
    # Wait, let's check dimensions.
    # dL/(c/H0) = (1+z) * integral_0^z dz'/E(a')
    # Let u = z'/z, so z' = z*u, dz' = z*du.
    # dL/(c/H0) = (1+z) * z * integral_0^1 du / E(a(z*u))
    # where a(z*u) = 1/(1+z*u)
    
    # So the integral to approximate is: I = integral_0^1 1/E(1/(1+z*u)) du
    # The quadrature is: sum_i weights[i] / E(1/(1+z*nodes[i]))
    
    # The relative error for a given z and cosmology is:
    # | (1+z)*z*(sum_i w_i/E(a_i)) - (1+z)*z*I_true | / ((1+z)*z*I_true)
    # = | sum_i w_i/E(a_i) - I_true | / I_true
    
    # So we want to minimize the worst-case relative error of the quadrature
    # approximating integral_0^1 g_z(u) du, where g_z(u) = 1/E(1/(1+z*u)).
    
    # Note that g_z(u) depends on z. The nodes and weights must work for all z.
    
    # Let's pre-compute reference integrals for various z values and cosmologies.
    z_test = np.linspace(0.1, 2.0, 20)  # Redshifts to test
    z_test = np.append(z_test, [0.5, 1.0, 1.5, 2.0, 0.01, 0.1, 5.0])
    z_test = np.unique(z_test)
    
    # Pre-compute true integrals
    true_integrals = np.zeros((len(training_cosmos), len(z_test)))
    for i, (w0, wa) in enumerate(training_cosmos):
        E = make_E(w0, wa)
        for j, z in enumerate(z_test):
            def integrand(u):
                a = 1.0 / (1.0 + z * u)
                return 1.0 / E(a)
            val, _ = quad(integrand, 0, 1, limit=200)
            true_integrals[i, j] = val
    
    # Now define the objective function
    def objective(params):
        # params: 16 values, first 8 are nodes, next 8 are log-weights
        nodes = params[:8]
        log_weights = params[8:]
        
        # Enforce strictly increasing nodes in (0, 1)
        # Use a soft constraint or transformation
        # Let's use a transformation that ensures nodes are in (0,1) and increasing
        # nodes = 1 / (1 + np.exp(-t)) for some t, but they need to be increasing
        # Actually, let's just use the raw values and add a penalty if violated
        if np.any(nodes <= 0) or np.any(nodes >= 1) or np.any(np.diff(nodes) <= 0):
            return 1e10
        
        weights = np.exp(log_weights)
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        # Compute worst-case relative error
        max_rel_err = 0.0
        for i, (w0, wa) in enumerate(training_cosmos):
            E = make_E(w0, wa)
            for j, z in enumerate(z_test):
                # Compute quadrature
                a_vals = 1.0 / (1.0 + z * nodes)
                quad_val = np.sum(weights / E(a_vals))
                true_val = true_integrals[i, j]
                rel_err = abs(quad_val - true_val) / true_val
                if rel_err > max_rel_err:
                    max_rel_err = rel_err
        return max_rel_err
    
    # Initial guess: midpoint rule
    nodes0 = (np.arange(n) + 0.5) / n
    weights0 = np.full(n, 1.0 / n)
    log_weights0 = np.log(weights0)
    params0 = np.concatenate([nodes0, log_weights0])
    
    # Optimize
    best_params = params0
    best_obj = objective(params0)
    
    # Use Nelder-Mead or Powell
    result = minimize(objective, params0, method='Powell', 
                      options={'maxiter': 50, 'xtol': 1e-8, 'ftol': 1e-8})
    
    if result.fun < best_obj:
        best_obj = result.fun
        best_params = result.x
    
    # Check time
    if time.time() > deadline - 10:
        pass  # Not enough time for more
    
    # Try a few more local optimizations from perturbed starts
    rng = np.random.default_rng(42)
    for _ in range(5):
        if time.time() > deadline - 15:
            break
        perturbation = rng.normal(0, 0.05, size=16)
        # Keep nodes in a reasonable range
        perturbation[:8] = np.clip(perturbation[:8], -0.1, 0.1)
        params_try = best_params + perturbation
        result = minimize(objective, params_try, method='Nelder-Mead',
                         options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x
    
    # Extract final nodes and weights
    nodes = best_params[:8]
    log_weights = best_params[8:]
    weights = np.exp(log_weights)
    weights = weights / np.sum(weights)
    
    # Ensure nodes are strictly increasing and in (0,1)
    # If not, fall back to midpoint
    if np.any(nodes <= 0) or np.any(nodes >= 1) or np.any(np.diff(nodes) <= 0):
        nodes = (np.arange(n) + 0.5) / n
        weights = np.full(n, 1.0 / n)
    
    return {"nodes": nodes, "weights": weights}
