"""Cosmology: universal luminosity-distance quadrature (8-node positive rule)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

# Deadline guard: evaluator kills after 90s; keep a safety margin.
_START = time.time()
_DEADLINE = _START + 85.0  # 5s safety margin

def _time_left():
    return _DEADLINE - time.time()

def _E_of_a(a, w0, wa):
    """E(a) = H(a)/H0 for flat CPL dark-energy model.
    E(z) = (1+z)^(3(1+w0+wa)) * (1+z)^(3*wa) / (1+z)^(3*wa) ... 
    Actually: E(z) = (1+z)^(3(1+w0)) * (1+z)^(3*wa) / ... 
    Standard CPL: E(z) = (1+z)^(3(1+w0+wa)) * (Omega_m)^(1/3) ... no.
    
    E(z) = [Omega_m*(1+z)^3 + Omega_Lambda*(1+z)^(3(1+w0+wa))]^0.5
    where Omega_Lambda = 1 - Omega_m (flat universe, no radiation/matter for simplicity)
    """
    z = 1.0 / a - 1.0
    Om_m = 0.3
    Om_L = 1.0 - Om_m
    # E(z)^2 = Om_m*(1+z)^3 + Om_L*(1+z)^(3*(1+w0+wa))
    exp_L = 3.0 * (1.0 + w0 + wa)
    E2 = Om_m * (1.0 + z)**3 + Om_L * (1.0 + z)**exp_L
    return np.sqrt(E2)

def _integrand(z, w0, wa):
    """Integrand for dL/(c/H0) = (1+z) * integral_0^z dz'/E(z')
    The integral part: I(z) = integral_0^z dz'/E(z')
    Using substitution a = 1/(1+z), da = -dz/(1+z)^2, so dz = -(1+z)^2 da
    When z=0, a=1; when z=z, a=1/(1+z)
    I(z) = integral_{a=1}^{a=1/(1+z)} -(1+z')^2/E(a') da' where z' = 1/a' - 1
    = integral_{1/(1+z)}^{1} (1/a' - 1)^2 / E(a') da' * (1/a')^2 ... 
    
    Let me redo: dz = -da/(1-a)^2 * (1-a)^2... 
    z = 1/a - 1 = (1-a)/a, dz = -da/a^2
    I(z) = integral_0^z dz'/E(z') = integral_{a=1}^{1/(1+z)} (-da/a^2) / E(a)
         = integral_{1/(1+z)}^{1} da/(a^2 * E(a))
    
    So dL/(c/H0) = (1+z) * integral_{1/(1+z)}^1 da/(a^2 * E(a))
    
    The quadrature is on the variable in [0,1]. Let's use t = 1 - a, so a = 1-t, t in [0, z/(1+z)]... 
    Actually the task says: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i) where a_i = 1/(1+z*nodes[i])
    
    So the integral being approximated is: integral_0^1 dt / E(1/(1+z*t)) * z
    Wait: (1+z)*z*sum w_i/E(a_i) = (1+z) * [z * sum w_i/E(a_i)]
    The bracket approximates integral_0^1 dt/E(1/(1+z*t))
    
    Let u = 1+z*t, du = z*dt, so integral_0^1 dt/E(1+z*t) = (1/z) integral_1^{1+z} du/E(u)
    So (1+z)*z*(1/z) integral_1^{1+z} du/E(u) = (1+z) * integral_1^{1+z} du/E(u)
    
    But dL/(c/H0) = (1+z) * integral_0^z dz'/E(z') = (1+z) * integral_1^{1+z} du/E(u) since u=1+z'
    Yes! So the quadrature approximates integral_0^1 dt/E(1+z*t) with 8 nodes, then multiplies by (1+z)*z.
    """
    a = 1.0 / (1.0 + z)
    return 1.0 / (a**2 * _E_of_a(a, w0, wa))

def _true_distance(z, w0, wa):
    """True dL/(c/H0) via adaptive quadrature."""
    if z <= 0:
        return 0.0
    def f(t):
        return 1.0 / _E_of_a(1.0/(1.0 + z*t), w0, wa)
    val, _ = quad(f, 0.0, 1.0, limit=200)
    return (1.0 + z) * z * val

def _quad_distance(z, nodes, weights, w0, wa):
    """Quadrature approximation of dL/(c/H0)."""
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_of_a(a_vals, w0, wa)
    return (1.0 + z) * z * np.sum(weights / E_vals)

def _relative_error(z, nodes, weights, w0, wa):
    true_val = _true_distance(z, w0, wa)
    quad_val = _quad_distance(z, nodes, weights, w0, wa)
    if abs(true_val) < 1e-15:
        return 0.0
    return abs(quad_val - true_val) / abs(true_val) * 100.0  # percent

def _worst_error(nodes, weights, cosmologies, z_values):
    """Compute worst relative error across all cosmologies and z values."""
    worst = 0.0
    for w0, wa in cosmologies:
        for z in z_values:
            err = _relative_error(z, nodes, weights, w0, wa)
            if err > worst:
                worst = err
    return worst

def _objective(params):
    """Optimization objective: minimize worst error with constraints handled by penalty."""
    # params: 16 values = 8 nodes + 8 weights
    nodes = params[:8]
    weights = params[8:16]
    
    # Enforce strict increasing nodes in (0,1)
    # Use soft penalty for violations
    penalty = 0.0
    if nodes[0] <= 0 or nodes[-1] >= 1:
        penalty += 1000.0
    for i in range(7):
        if nodes[i+1] - nodes[i] <= 1e-6:
            penalty += 1000.0
    if np.any(weights <= 0):
        penalty += 1000.0
    wsum = np.sum(weights)
    penalty += 100.0 * abs(wsum - 1.0)**2
    
    if penalty > 100:
        return penalty
    
    # Normalize weights for evaluation
    weights_norm = weights / wsum
    err = _worst_error(nodes, weights_norm, _COSMOLOGIES, _Z_VALUES)
    return err + penalty

# Training cosmologies (CPL parameters w0, wa)
_COSMOLOGIES = [
    (-1.0, 0.0),      # LCDM
    (-1.0, 0.5),
    (-1.2, 0.4),
    (-0.9, -0.3),
    (-1.5, 0.8),
    (-0.8, 0.2),
    (-1.1, -0.5),
    (-1.3, 0.6),
]

# Test z values for training
_Z_VALUES = np.array([0.1, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5, 3.5, 5.0])

def _initial_guess():
    """Gauss-Legendre-like initial guess with positive weights."""
    # Use Gauss-Legendre nodes and weights as starting point
    nodes, weights = np.polynomial.legendre.leggauss(8)
    # Map from [-1,1] to [0,1]
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    return np.concatenate([nodes, weights])

def construct():
    global _COSMOLOGIES, _Z_VALUES
    best_params = _initial_guess()
    best_score = _objective(best_params)
    
    # Try multiple optimization approaches
    attempts = [
        {'method': 'Nelder-Mead', 'options': {'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}},
        {'method': 'Powell', 'options': {'maxiter': 1000, 'ftol': 1e-10}},
    ]
    
    for attempt in attempts:
        if _time_left() < 10:
            break
        try:
            result = minimize(_objective, best_params, method=attempt['method'], options=attempt['options'])
            if result.fun < best_score:
                best_params = result.x
                best_score = result.fun
        except Exception:
            pass
    
    # Extract nodes and weights, enforce constraints
    nodes = best_params[:8].copy()
    weights = best_params[8:16].copy()
    
    # Ensure nodes are strictly increasing in (0,1)
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    # Sort and ensure strict increase
    nodes = np.sort(nodes)
    for i in range(7):
        if nodes[i+1] - nodes[i] < 1e-5:
            nodes[i+1] = nodes[i] + 1e-5 + 1e-7 * i
    
    # Ensure weights are positive and sum to 1
    weights = np.abs(weights) + 1e-10
    weights = weights / np.sum(weights)
    
    # Final refinement: local optimization on normalized weights
    if _time_left() > 15:
        def _obj_refine(w):
            w = np.abs(w) + 1e-12
            w = w / np.sum(w)
            return _worst_error(nodes, w, _COSMOLOGIES, _Z_VALUES)
        
        try:
            res = minimize(_obj_refine, weights, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            w_new = np.abs(res.x) + 1e-12
            w_new = w_new / np.sum(w_new)
            if _obj_refine(w_new) < _obj_refine(weights):
                weights = w_new
        except Exception:
            pass
    
    # Final validation
    assert len(nodes) == 8
    assert len(weights) == 8
    assert np.all(np.diff(nodes) > 0)
    assert nodes[0] > 0 and nodes[-1] < 1
    assert np.all(weights > 0)
    assert abs(np.sum(weights) - 1.0) < 1e-12
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
