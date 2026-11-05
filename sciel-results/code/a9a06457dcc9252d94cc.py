import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

def _E_a(a, w0, wa):
    """E(a) = H(a)/H0 for spatially flat CPL model: w = w0 + wa*(1-a)"""
    # E(z)^2 = (1+z)^3 + Omega_m * (1+z)^3 ... wait, for spatially flat:
    # E(z)^2 = (1+z)^3 + Omega_Lambda * (1+z)^{3(1+w0+wa)} ... 
    # Actually for CPL: w(z) = w0 + wa*z
    # E(z)^2 = (1+z)^3 + Omega_Lambda * exp(3*wa*(z - z/(1+z))) ... 
    # Let me use the standard form: E(z)^2 = (1+z)^3 + Omega_Lambda * (1+z)^{3+3w0+3wa} * exp(-3wa*z/(1+z))
    # For simplicity, use: E(z)^2 = (1+z)^3 + Omega_Lambda * exp(3*wa*(1 - 1/(1+z)))
    # Actually the standard CPL form is:
    # E(z) = (1+z)^{3/2} * sqrt(1 + (Omega_Lambda/Omega_m) * exp(3*wa*(1 - 1/(1+z))))
    # Let me use a simpler parametrization. For spatially flat:
    # E(z)^2 = (1+z)^3 + Omega_Lambda * (1+z)^{3(1+w)} where w is constant
    # For CPL with w(z) = w0 + wa*z:
    # E(z)^2 = (1+z)^3 + Omega_Lambda * exp(3*wa * (z - log(1+z)))
    # This is getting complex. Let me use a simpler model for optimization.
    
    # Actually, for the quadrature to be universal, I should test against various models.
    # Let me define E(a) in terms of a = 1/(1+z):
    # E(a) = a^{-3/2} * sqrt(Omega_m + Omega_Lambda * a^{3(1+w0+wa)} * exp(-3*wa*(1-a)))
    # For simplicity in optimization, use w=constant models and a few CPL models.
    
    pass

def _E_a_simple(a, w0, wa, Omega_m=0.3):
    """Simplified E(a) for CPL model in terms of a."""
    # z = 1/a - 1
    # w(z) = w0 + wa*z = w0 + wa*(1/a - 1)
    # E(z)^2 = (1+z)^3 + Omega_Lambda * exp(3*wa*(z - log(1+z)))
    # (1+z) = 1/a
    # z = (1-a)/a
    # log(1+z) = log(1/a) = -log(a)
    # z - log(1+z) = (1-a)/a + log(a)
    Omega_Lambda = 1.0 - Omega_m
    z = (1.0 - a) / a
    # E(z)^2 = (1+z)^3 + Omega_Lambda * exp(3*wa*(z - log(1+z)))
    # (1+z)^3 = (1/a)^3 = a^{-3}
    # z - log(1+z) = (1-a)/a - log(1/a) = (1-a)/a + log(a)
    exponent = 3.0 * wa * ((1.0 - a) / a + np.log(a))
    E2 = a**(-3.0) + Omega_Lambda * np.exp(exponent)
    # Wait, this doesn't look right. Let me reconsider.
    # Standard CPL: E(z)^2 = (1+z)^3 * [Omega_m + Omega_Lambda * exp(3*wa*(1 - 1/(1+z)))]
    # = a^{-3} * [Omega_m + Omega_Lambda * exp(3*wa*(1-a))]
    E2 = a**(-3.0) * (Omega_m + Omega_Lambda * np.exp(3.0 * wa * (1.0 - a)))
    return np.sqrt(E2)

def _integrand(z, w0, wa, Omega_m=0.3):
    """f(z) = z / E(1/(1+z))"""
    a = 1.0 / (1.0 + z)
    E = _E_a_simple(a, w0, wa, Omega_m)
    return z / E

def _true_integral(z, w0, wa, Omega_m=0.3):
    """Compute the true integral using adaptive quadrature."""
    if z < 1e-10:
        return 0.0
    result, _ = quad(_integrand, 0, z, args=(w0, wa, Omega_m), limit=200, epsabs=1e-14, epsrel=1e-14)
    return result

def _quadrature(z, nodes, weights, w0, wa, Omega_m=0.3):
    """Compute quadrature approximation of ∫₀^z f(z') dz'"""
    # We need to integrate from 0 to z, but our nodes are in [0,1]
    # The quadrature rule approximates ∫₀¹ g(t) dt = sum w_i * g(x_i)
    # For ∫₀^z f(z') dz', we can change variable: t = z'/z, z' = z*t
    # ∫₀^z f(z') dz' = z * ∫₀¹ f(z*t) dt
    # So we apply our [0,1] quadrature to h(t) = f(z*t)
    # h(t) = f(z*t) = (z*t) / E(1/(1+z*t))
    # Quadrature: z * sum_i w_i * h(x_i) = z * sum_i w_i * f(z*x_i)
    
    # Actually wait - the problem says: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # where a_i = 1/(1+z*nodes[i])
    # So the quadrature is: sum_i w_i / E(a_i) approximates ∫₀^1 [1/E(1/(1+z*t))] dt
    # Then dL/(c/H0) = (1+z)*z * that sum
    
    # Let me re-read: "dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)"
    # So the quadrature approximates ∫₀¹ dt/E(1/(1+z*t))
    # And the full expression is (1+z)*z times that.
    
    # But the true value should be: dL/(c/H0) = (1+z) * ∫₀^z dz'/E(1/(1+z'))
    # Let u = z'/z, dz' = z*du:
    # = (1+z) * z * ∫₀¹ du/E(1/(1+z*u))
    # So yes, the quadrature approximates ∫₀¹ du/E(1/(1+z*u))
    
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_a_simple(a_vals, w0, wa, Omega_m)
    return np.sum(weights / E_vals)

def _relative_error(z, nodes, weights, w0, wa, Omega_m=0.3):
    """Compute relative error for a given z and model."""
    true_val = _true_integral(z, w0, wa, Omega_m)
    quad_val = _quadrature(z, nodes, weights, w0, wa, Omega_m)
    # The actual dL/(c/H0) = (1+z)*z*quad_val, but relative error is the same
    # as for the integral part since (1+z)*z is common factor
    if abs(true_val) < 1e-15:
        return 0.0
    return abs(quad_val - true_val) / abs(true_val) * 100.0  # in percent

def _worst_error(nodes, weights, test_zs, models, Omega_m=0.3):
    """Compute worst-case relative error across test points and models."""
    max_err = 0.0
    for w0, wa in models:
        for z in test_zs:
            err = _relative_error(z, nodes, weights, w0, wa, Omega_m)
            if err > max_err:
                max_err = err
    return max_err

def _objective(params):
    """Objective: minimize worst-case error. params = [nodes(8), weights(8)]"""
    nodes = params[:8]
    weights = params[8:16]
    
    # Enforce constraints: nodes in (0,1) increasing, weights positive sum to 1
    # Use soft constraints via penalty
    # Node constraints
    node_penalty = 0.0
    for i in range(8):
        if nodes[i] <= 0 or nodes[i] >= 1:
            node_penalty += 1e6 * abs(nodes[i] - 0.5)
    for i in range(7):
        if nodes[i+1] <= nodes[i]:
            node_penalty += 1e6 * (nodes[i] - nodes[i+1])
    
    # Weight constraints
    weight_penalty = 0.0
    for i in range(8):
        if weights[i] <= 0:
            weight_penalty += 1e6 * abs(weights[i])
    sum_w = np.sum(weights)
    weight_penalty += 1e6 * (sum_w - 1.0)**2
    
    if node_penalty > 1e3 or weight_penalty > 1e3:
        return 1e6 + node_penalty + weight_penalty
    
    # Test points: z from 0.01 to 3
    test_zs = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0])
    # Models: various w0, wa
    models = [(-1.0, 0.0), (-1.0, 0.5), (-1.0, -0.5), (-0.5, 0.0), (-0.5, 0.5), 
              (-1.5, 0.0), (-1.0, 1.0), (-1.0, -1.0)]
    
    err = _worst_error(nodes, weights, test_zs, models)
    return err

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s
    
    # Start with Gauss-Legendre as initial guess
    nodes0, weights0 = np.polynomial.legendre.leggauss(8)
    nodes0 = (nodes0 + 1.0) / 2.0
    weights0 = weights0 / 2.0
    
    x0 = np.concatenate([nodes0, weights0])
    
    best_params = x0.copy()
    best_err = _objective(x0)
    
    # Try multiple restarts with perturbed initial guesses
    rng = np.random.default_rng(42)
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Perturb initial guess
        if restart == 0:
            x = x0.copy()
        else:
            x = x0 + rng.normal(0, 0.05, size=16)
            # Ensure nodes are in (0,1) and increasing
            x[:8] = np.clip(x[:8], 0.01, 0.99)
            x[:8] = np.sort(x[:8])
            # Ensure weights are positive
            x[8:] = np.abs(x[8:]) + 0.01
            x[8:] = x[8:] / np.sum(x[8:])
        
        # Use Nelder-Mead for optimization
        result = minimize(_objective, x, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10,
                                  'adaptive': True})
        
        if result.fun < best_err:
            best_err = result.fun
            best_params = result.x.copy()
        
        if time.time() > deadline - 10.0:
            break
    
    # Final result
    nodes = best_params[:8]
    weights = best_params[8:16]
    
    # Ensure nodes are strictly increasing in (0,1)
    nodes = np.clip(nodes, 1e-8, 1.0 - 1e-8)
    nodes = np.sort(nodes)
    # Ensure strict increasing
    for i in range(7):
        if nodes[i+1] - nodes[i] < 1e-10:
            nodes[i+1] = nodes[i] + 1e-8
    
    # Ensure weights are positive and sum to 1
    weights = np.abs(weights) + 1e-10
    weights = weights / np.sum(weights)
    
    # Final check
    assert len(nodes) == 8
    assert len(weights) == 8
    assert np.all(nodes > 0) and np.all(nodes < 1)
    assert np.all(np.diff(nodes) > 0)
    assert np.all(weights > 0)
    assert abs(np.sum(weights) - 1.0) < 1e-12
    
    return {"nodes": nodes, "weights": weights}
