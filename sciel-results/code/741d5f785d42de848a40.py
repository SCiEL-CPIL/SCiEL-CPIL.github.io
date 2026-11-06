"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 85.0  # 5s safety margin before 90s kill

# Pre-compute validation models and redshifts
# w0 in [-1.2, -0.5], wa in [-1.5, 1.5]
_w0_vals = np.array([-1.2, -1.0, -0.8, -0.6, -0.5])
_wa_vals = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
_z_vals = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

_models = [(w0, wa) for w0 in _w0_vals for wa in _wa_vals]
_zs = _z_vals


def _E(z, w0, wa):
    """E(z) = H(z)/H0 for CPL model in flat universe."""
    return (1.0 + z) ** (1.5 * (2.0 + w0 + wa * z))


def _true_distance(z, w0, wa, n_points=2000):
    """Compute dL/(c/H0) = (1+z) * integral_0^z dz'/E(z') using adaptive quadrature."""
    # Integrand: 1/E(z')
    def integrand(zz):
        return 1.0 / _E(zz, w0, wa)

    # Use Simpson's rule with many points for accuracy
    zz = np.linspace(0.0, z, n_points)
    vals = integrand(zz)
    # Simpson's rule
    if n_points % 2 == 0:
        n_points += 1
        zz = np.linspace(0.0, z, n_points)
        vals = integrand(zz)
    h = z / (n_points - 1)
    integral = h / 3.0 * (vals[0] + vals[-1] + 4.0 * np.sum(vals[1:-1:2]) + 2.0 * np.sum(vals[2:-1:2]))
    return (1.0 + z) * integral


def _quadrature_distance(z, w0, wa, nodes, weights):
    """Compute dL/(c/H0) using the 8-node quadrature rule."""
    # a_i = 1/(1+z*nodes[i])
    # integral = sum_i weights[i] / E(a_i)
    # But wait: the integral is over z' from 0 to z.
    # The formula given: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i), where a_i=1/(1+z*nodes[i])
    # This is a quadrature approximation of integral_0^z dz'/E(z')
    # Let's verify: if we change variable to a = 1/(1+z'), then z' = 1/a - 1, dz' = -da/a^2
    # When z'=0, a=1; when z'=z, a=1/(1+z)
    # integral_0^z dz'/E(z') = integral_{1/(1+z)}^1 da/(a^2 * E(1/a - 1))
    # E(1/a - 1) = (1/a)^{1.5*(2+w0+wa*(1/a-1))}
    # So the integral becomes integral_{1/(1+z)}^1 da / [a^2 * (1/a)^{1.5*(2+w0+wa*(1/a-1))}]
    # = integral_{1/(1+z)}^1 a^{1.5*(2+w0+wa*(1/a-1)) - 2} da
    
    # But the task says: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i), where a_i=1/(1+z*nodes[i])
    # This looks like a direct quadrature on the z' variable with nodes in [0,1] mapped to z' in [0,z]
    # Let z' = z * t, t in [0,1]. Then integral_0^z dz'/E(z') = z * integral_0^1 dt/E(z*t)
    # So dL/(c/H0) = (1+z) * z * integral_0^1 dt/E(z*t)
    # The quadrature approximates integral_0^1 dt/E(z*t) ≈ sum_i weights[i] / E(z * nodes[i])
    # But the task says a_i = 1/(1+z*nodes[i]), and E(a_i)... 
    
    # Wait, let me re-read: "dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i), where a_i=1/(1+z*nodes[i]) and E(a)=H(a)/H0"
    # E(a) = H(a)/H0. For CPL, H(a)/H0 = a^{-3/2} * (1+(1-a)/a)^{...}
    # Actually, H(a)/H0 = a^{-3/2} * (1 + (1-a)/a)^{3/2 * (w0 + wa*(1/a - 1) - w0)}... 
    
    # Let me just use the formula as given: E(a) where a = 1/(1+z*nodes[i])
    # For a scale factor a, z_a = 1/a - 1
    # E(a) = H(a)/H0 = (1+z_a)^{1.5*(2+w0+wa*z_a)} = (1/a)^{1.5*(2+w0+wa*(1/a-1))}
    
    a_vals = 1.0 / (1.0 + z * nodes)
    # E(a) = (1/a)^{1.5*(2+w0+wa*(1/a-1))}
    z_a = 1.0 / a_vals - 1.0
    E_vals = (1.0 + z_a) ** (1.5 * (2.0 + w0 + wa * z_a))
    # Which is (1/a)^{1.5*(2+w0+wa*(1/a-1))}
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx


def _objective(params):
    """Objective: worst relative error percent over all validation models and redshifts."""
    if time.time() > _DEADLINE:
        return 1e10
    
    # Unpack parameters
    # params: 8 nodes + 8 weights = 16 values
    nodes = params[:8]
    weights = params[8:16]
    
    # Ensure nodes are in [0,1] and strictly increasing
    if np.any(nodes < 0) or np.any(nodes > 1):
        return 1e10
    if np.any(np.diff(nodes) <= 0):
        return 1e10
    
    # Ensure weights are positive and sum to 1
    if np.any(weights <= 0):
        return 1e10
    w_sum = np.sum(weights)
    if abs(w_sum - 1.0) > 1e-6:
        return 1e10 + abs(w_sum - 1.0)
    
    # Normalize weights for evaluation
    w_norm = weights / w_sum
    
    worst_err = 0.0
    for w0, wa in _models:
        for z in _zs:
            true_d = _true_distance(z, w0, wa)
            quad_d = _quadrature_distance(z, w0, wa, nodes, w_norm)
            if true_d > 0:
                rel_err = abs(quad_d - true_d) / true_d * 100.0
                if rel_err > worst_err:
                    worst_err = rel_err
    
    return worst_err


def construct():
    """Construct an 8-node positive quadrature rule for luminosity distance."""
    # Start with midpoint rule
    n = 8
    nodes0 = (np.arange(n) + 0.5) / n
    weights0 = np.full(n, 1.0 / n)
    x0 = np.concatenate([nodes0, weights0])
    
    # Bounds: nodes in [0.01, 0.99], weights in [0.01, 0.5]
    bounds = [(0.01, 0.99)] * 8 + [(0.01, 0.5)] * 8
    
    best_x = x0.copy()
    best_obj = _objective(x0)
    
    # Try a few optimization attempts
    attempts = 0
    while time.time() < _DEADLINE - 5.0:
        attempts += 1
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
        
        # Add some random restarts
        if attempts < 3:
            # Perturb initial guess
            x0 = best_x + np.random.randn(16) * 0.01
            # Clamp to bounds
            x0[:8] = np.clip(x0[:8], 0.01, 0.99)
            x0[8:16] = np.clip(x0[8:16], 0.01, 0.5)
    
    # Use best result found
    nodes = best_x[:8]
    weights = best_x[8:16]
    
    # Ensure nodes are strictly increasing
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-4
    
    # Ensure nodes in [0,1]
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    
    # Ensure weights positive and sum to 1
    weights = np.clip(weights, 1e-6, None)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
