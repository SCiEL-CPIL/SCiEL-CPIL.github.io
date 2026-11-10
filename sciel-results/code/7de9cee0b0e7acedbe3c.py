import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

def construct():
    """
    Eight-node positive quadrature rule for dimensionless luminosity distance.
    Optimizes nodes and weights to minimize worst-case relative error across
    a range of CPL cosmological models and redshifts, starting from Gauss-Legendre.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # 1. Initial Guess: Gauss-Legendre 8-point on [0, 1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes_gl = 0.5 * (x_gl + 1.0)
    weights_gl = 0.5 * w_gl
    weights_gl = weights_gl / np.sum(weights_gl)
    
    best_nodes = nodes_gl.copy()
    best_weights = weights_gl.copy()
    best_error = np.inf
    
    # 2. Define Representative Cosmologies and Redshifts
    # CPL models: E^2(a) = Om_m a^-3 + Om_de a^(3(1+w0+wa(1-a)))
    # Flat universe: Om_de = 1 - Om_m
    Om_m = 0.3
    cpl_models = [
        (-1.0, 0.0),   # Lambda CDM
        (-1.0, 1.0),   # Quintessence
        (-0.5, 0.0),   # Mild DE
        (-1.0, -1.0),  # Phantom
        (0.0, 0.0),    # Matter dominated (effectively)
        (-1.0, 0.5),   # Evolving DE
        (-0.8, 0.2),   # Intermediate
        (-1.2, 0.3),   # Phantom with evolution
        (-0.9, -0.4),  # Near -1 with negative evolution
        (-1.5, 1.0),   # Strong phantom/evolution
        (-1.0, 2.0),   # Very strong evolution
    ]
    
    z_values = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    
    def _E(a, w0, wa):
        """Compute E(a) = H(a)/H0 for CPL model."""
        Om_de = 1.0 - Om_m
        # Handle a=0 or a=1 edge cases carefully, though nodes are in (0,1)
        a_safe = np.clip(a, 1e-15, 1.0 - 1e-15)
        
        # Matter term
        term_m = Om_m * a_safe**(-3.0)
        
        # Dark Energy term
        # Exponent: 3 * (1 + w0 + wa * (1 - a))
        exponent_de = 3.0 * (1.0 + w0 + wa * (1.0 - a_safe))
        term_de = Om_de * a_safe**(exponent_de)
        
        E_sq = term_m + term_de
        return np.sqrt(np.maximum(E_sq, 1e-30))
    
    def _true_integral(w0, wa, z):
        """Compute true integral I = integral_0^1 du / E(1/(1+zu)) using adaptive quadrature."""
        def integrand(u):
            a = 1.0 / (1.0 + z * u)
            return 1.0 / _E(a, w0, wa)
        
        # Use high precision adaptive quadrature
        val, _ = quad(integrand, 0.0, 1.0, epsabs=1e-12, epsrel=1e-12, limit=200)
        return val
    
    # Pre-compute true integrals for the training set
    # This is done once to speed up the objective function evaluation
    true_integrals = {}
    for w0, wa in cpl_models:
        for z in z_values:
            try:
                true_integrals[(w0, wa, z)] = _true_integral(w0, wa, z)
            except Exception:
                true_integrals[(w0, wa, z)] = 0.0
                
    # Check time
    if time.time() > deadline:
        return {"nodes": best_nodes, "weights": best_weights}
        
    def _quadrature_error(nodes, weights):
        """Compute worst-case relative error across all models and redshifts."""
        max_rel_err = 0.0
        for w0, wa in cpl_models:
            for z in z_values:
                true_val = true_integrals.get((w0, wa, z), 0.0)
                if true_val < 1e-10:
                    continue
                
                # Compute quadrature approximation
                a_vals = 1.0 / (1.0 + z * nodes)
                E_vals = _E(a_vals, w0, wa)
                quad_val = np.sum(weights / E_vals)
                
                rel_err = abs(quad_val - true_val) / true_val
                if rel_err > max_rel_err:
                    max_rel_err = rel_err
        return max_rel_err
    
    # 3. Optimization
    # Parameterization:
    # Nodes: 8 values in (0, 1), strictly increasing.
    # We use log-spacings to ensure positivity and ordering.
    # Let d_i be spacings. nodes[0] = d_0, nodes[i] = nodes[i-1] + d_i.
    # To map 8 nodes to [0,1], we need sum(d_i) = 1? No, nodes are in [0,1].
    # Let's use 8 positive parameters p_i representing spacings.
    # nodes = cumsum(p) / sum(p). This ensures nodes in (0,1) and increasing.
    # Wait, this forces the first node to be >0 and last <1, which is good.
    # But we have 8 nodes. If we use 8 spacings, the last node is sum(p)/sum(p)=1?
    # No, if nodes = cumsum(p)/sum(p), the last node is 1. We want nodes in [0,1] but strictly inside?
    # The task says "Nodes must be strictly increasing in [0,1]". It doesn't forbid 0 or 1,
    # but usually quadrature nodes are interior. GL nodes are interior.
    # Let's use 7 spacings between 8 nodes? Or 8 nodes defined by 8 parameters?
    # Standard approach: define 8 nodes directly with constraints?
    # L-BFGS-B supports bounds.
    # Let's try optimizing nodes directly with bounds [1e-6, 1-1e-6] and weights with bounds [1e-6, 1].
    # And enforce sum(weights)=1 by normalizing inside the objective?
    # Better: optimize 7 weights and set the 8th to 1 - sum(others).
    # And optimize 8 nodes with bounds.
    
    def _decode(x):
        """Decode parameter vector x to nodes and weights."""
        # x[0:8] are nodes
        nodes = x[:8]
        # x[8:15] are first 7 weights
        w_raw = x[8:15]
        w_raw = np.exp(w_raw)  # Ensure positive
        w_sum = np.sum(w_raw)
        w8 = 1.0 - w_sum
        if w8 <= 1e-6:
            # Fallback to uniform if constraint violated significantly
            return None, None
        weights = np.concatenate([w_raw / w_sum * (1.0 - 1e-4), [1e-4]]) 
        # Actually, simpler: just normalize the 7 optimized + 1 fixed?
        # Let's stick to 8 weights, optimize 7, fix 1.
        weights = np.zeros(8)
        weights[:7] = w_raw
        weights[7] = 1.0 - np.sum(w_raw)
        
        if np.any(weights <= 1e-8):
            return None, None
            
        # Check node ordering
        if not np.all(np.diff(nodes) > 1e-6):
            return None, None
            
        return nodes, weights

    # Initial parameters
    x0_nodes = best_nodes
    x0_weights = best_weights
    # For x0, we need to encode back to the optimization space?
    # We are optimizing nodes (8) and 7 weights (log scale?).
    # Let's just use L-BFGS-B on 15 variables: 8 nodes, 7 weights (linear).
    # Bounds for nodes: [1e-5, 1-1e-5]
    # Bounds for weights: [1e-5, 1]
    
    def _objective_opt(x):
        nodes = x[:8]
        weights = x[8:15]
        w_total = np.sum(weights)
        if w_total <= 0:
            return 1e10
        weights_norm = weights / w_total
        w_last = 1.0 - w_total
        if w_last <= 1e-8:
            return 1e10
        
        full_weights = np.append(weights_norm, w_last)
        
        # Check nodes
        if not np.all(nodes > 1e-5) or not np.all(nodes < 1-1e-5):
            return 1e10
        if not np.all(np.diff(nodes) > 1e-5):
            return 1e10
            
        return _quadrature_error(nodes, full_weights)
    
    # Initial point
    x0 = np.concatenate([best_nodes, best_weights[:7]])
    
    # Bounds
    bounds = []
    for i in range(8):
        bounds.append((1e-5, 1-1e-5))
    for i in range(7):
        bounds.append((1e-5, 1.0))
        
    try:
        res = minimize(_objective_opt, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.fun < best_error:
            best_error = res.fun
            best_nodes = res.x[:8]
            best_weights = np.append(res.x[8:15], 1.0 - np.sum(res.x[8:15]))
            
    except Exception as e:
        pass
    
    # Final sanity check and normalization
    # Ensure weights sum to 1
    sum_w = np.sum(best_weights)
    if sum_w > 0:
        best_weights = best_weights / sum_w
    else:
        best_weights = np.full(8, 1.0/8.0)
        best_nodes = nodes_gl.copy()
        
    # Verify constraints
    if not np.all(np.diff(best_nodes) > 0) or not np.all(best_weights > 0):
        best_nodes = nodes_gl.copy()
        best_weights = weights_gl.copy()
        
    # Final normalization to be safe
    best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}
