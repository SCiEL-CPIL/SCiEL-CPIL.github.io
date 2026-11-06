"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time


def _E_of_a(a, w0, w1):
    """H(a)/H0 for CPL model in flat universe."""
    # E(a) = (1-a^2)^((1+w0+w1)/3) * a^(-w1)
    # But more standard: E(z) = (1+z)^(3(1+w0+w1)) * (1+z)^(-3w1) = (1+z)^(3+3w0)
    # Wait, let me be careful. For CPL: w(z) = w0 + w1*z/(1+z)
    # E(z)^2 = (1+z)^3 * (1+z)^(3*w1) * exp(-3*w1*z/(1+z)) * (1+z)^(3*w0)
    # Actually: E(z) = (1+z)^(3(1+w0)/2) * (1+z)^(3*w1/2) * exp(-3*w1*z/(2*(1+z)))
    # In terms of a = 1/(1+z): E(a) = a^(-3(1+w0)/2) * a^(-3*w1/2) * exp(3*w1*(1-a)/(2*a))
    # Simpler form: E(a) = a^(-(3+3*w0)/2) * exp(3*w1*(1-a)/(2*a))
    # But the task says E(a) = H(a)/H0, and the standard CPL form is:
    # H(z)/H0 = (1+z)^(3(1+w0+w1)/3) * (1+z)^(-w1*3/(1+z))... 
    # Let me use the standard: E(z) = (1+z)^(3+3*w0) * (1+z)^(3*w1) * exp(-3*w1*z/(1+z))
    # No. The standard CPL parametrization gives:
    # E(z) = (1+z)^(3(1+w0)/2) * (1+z)^(3w1/2) * exp(-3w1*z/(2(1+z)))
    # In terms of a: z = 1/a - 1, so:
    # E(a) = a^(-3(1+w0)/2) * a^(-3w1/2) * exp(3w1*(1-a)/(2a))
    # = a^(-(3+3w0+3w1)/2) * exp(3w1*(1-a)/(2a))
    # But wait, that doesn't look right either. Let me use the well-known form:
    # For w(z) = w0 + w1*z/(1+z):
    # E(z) = (1+z)^(3(1+w0)/2) * (1+z)^(3w1/2) * exp(-3w1*z/(2(1+z)))
    # This is equivalent to: E(z) = (1+z)^(3+3w0)/2 * exp(3w1/2 * (ln(1+z) - z/(1+z)))
    # In terms of a: ln(1+z) = -ln(a), z/(1+z) = 1-a
    # E(a) = a^(-(3+3w0)/2) * exp(3w1/2 * (-ln(a) - (1-a)))
    # = a^(-(3+3w0+3w1)/2) * exp(3w1/2 * (a-1))
    # Hmm, let me just use the direct formula.
    
    # Standard result for CPL:
    # E(z) = (1+z)^(3(1+w0)/2) * (1+z)^(3w1/2) * exp(-3w1*z/(2(1+z)))
    # With a = 1/(1+z):
    # E(a) = a^(-3(1+w0)/2) * a^(-3w1/2) * exp(3w1*(1-a)/(2*a))
    
    if a <= 0:
        return np.inf
    exp_arg = 3.0 * w1 * (1.0 - a) / (2.0 * a)
    power = -(3.0 + 3.0 * w0 + 3.0 * w1) / 2.0
    return a**power * np.exp(exp_arg)


def _integrand_val(z, x, w0, w1):
    """Compute 1/E(a) where a = 1/(1+z*x)."""
    a = 1.0 / (1.0 + z * x)
    return 1.0 / _E_of_a(a, w0, w1)


def _exact_integral(z, w0, w1):
    """Compute the exact integral using adaptive quadrature."""
    def f(x):
        a = 1.0 / (1.0 + z * x)
        return 1.0 / _E_of_a(a, w0, w1)
    result, _ = quad(f, 0, 1, limit=200, epsabs=1e-15, epsrel=1e-15)
    return (1.0 + z) * z * result


def _quadrature_value(z, nodes, weights, w0, w1):
    """Compute quadrature approximation."""
    a_vals = 1.0 / (1.0 + z * nodes)
    e_vals = np.array([_E_of_a(a, w0, w1) for a in a_vals])
    return (1.0 + z) * z * np.dot(weights, 1.0 / e_vals)


def _worst_case_error(nodes, weights, z_list, w0_list, w1_list):
    """Compute worst-case relative error in percent."""
    max_err = 0.0
    for z in z_list:
        for w0 in w0_list:
            for w1 in w1_list:
                exact = _exact_integral(z, w0, w1)
                approx = _quadrature_value(z, nodes, weights, w0, w1)
                if abs(exact) > 1e-15:
                    rel_err = abs(approx - exact) / abs(exact) * 100.0
                    max_err = max(max_err, rel_err)
    return max_err


def _objective(params):
    """Objective function for optimization."""
    nodes = np.sort(params[:8])
    # Enforce nodes in (0,1) with small margin
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    # Ensure strictly increasing
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-10
    weights = np.abs(params[8:16])
    w_sum = np.sum(weights)
    if w_sum < 1e-15:
        return 1e10
    weights = weights / w_sum
    
    # Evaluate on a representative set of cosmologies and redshifts
    z_test = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0]
    w0_test = [-1.0, -0.9, -0.8, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5]
    w1_test = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]
    
    return _worst_case_error(nodes, weights, z_test, w0_test, w1_test)


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80 second budget with margin
    
    # Initial guess: use Gauss-Legendre nodes and weights as starting point
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    # Map from [-1,1] to [0,1]
    nodes0 = (x_gl + 1.0) / 2.0
    weights0 = w_gl / 2.0
    nodes0 = np.sort(nodes0)
    weights0 = weights0[np.argsort(x_gl)]
    
    params0 = np.concatenate([nodes0, weights0])
    
    best_params = params0.copy()
    best_err = _objective(params0)
    
    # Try a few local optimizations with different starting points
    start_points = [
        params0,
        # Midpoint rule
        np.concatenate([(np.arange(8) + 0.5) / 8.0, np.full(8, 0.125)]),
        # Trapezoidal-like
        np.concatenate([np.linspace(0.05, 0.95, 8), np.full(8, 0.125)]),
        # Slightly perturbed GL
        params0 + np.random.randn(16) * 0.01,
    ]
    
    for sp in start_points:
        if time.time() > deadline - 10.0:
            break
        
        try:
            res = minimize(_objective, sp, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_err:
                best_err = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Extract best solution
    nodes = np.sort(best_params[:8])
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-10
    weights = np.abs(best_params[8:16])
    weights = weights / np.sum(weights)
    
    # Final refinement with more diverse test set if time permits
    if time.time() < deadline - 5.0:
        z_test2 = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        w0_test2 = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        w1_test2 = [-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5]
        
        current_err = _worst_case_error(nodes, weights, z_test2, w0_test2, w1_test2)
        
        # Try to refine further
        params_curr = np.concatenate([nodes, weights])
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            try:
                res = minimize(_objective, params_curr, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
                if res.fun < current_err:
                    current_err = res.fun
                    params_curr = res.x
                    nodes = np.sort(res.x[:8])
                    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
                    for i in range(1, 8):
                        if nodes[i] <= nodes[i-1]:
                            nodes[i] = nodes[i-1] + 1e-10
                    weights = np.abs(res.x[8:16])
                    weights = weights / np.sum(weights)
            except Exception:
                break
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
