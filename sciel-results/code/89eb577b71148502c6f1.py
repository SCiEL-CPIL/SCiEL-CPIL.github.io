"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _gauss_legendre_rule(n=8):
    """Standard Gauss-Legendre rule on [0,1]."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    nodes = (nodes + 1.0) / 2.0
    weights = weights / 2.0
    return nodes, weights


def _E_cpl(a, w0, wa, om_m, om_r, om_k=0.0):
    """E(a) = H(a)/H0 for CPL model."""
    h2 = om_m * a**(-3) + om_r * a**(-4) + om_k * a**0 + (1 - om_m - om_r - om_k) * a**(-3*(1+w0+wa))
    return np.sqrt(np.maximum(h2, 1e-30))


def _integrand(x, z, w0, wa, om_m, om_r, om_k):
    """The integrand 1/E(a) where a = 1/(1+z*x)."""
    a = 1.0 / (1.0 + z * x)
    return 1.0 / _E_cpl(a, w0, wa, om_m, om_r, om_k)


def _true_dL(z, w0, wa, om_m, om_r, om_k, n=2000):
    """Compute the true integral using adaptive/simpson-like method."""
    x = np.linspace(0, 1, n)
    f = _integrand(x, z, w0, wa, om_m, om_r, om_k)
    return np.trapz(f, x)


def _quadrature_error(nodes, weights, z, w0, wa, om_m, om_r, om_k):
    """Compute relative error of quadrature vs true value."""
    x_nodes = z * nodes
    a_nodes = 1.0 / (1.0 + x_nodes)
    f_nodes = 1.0 / _E_cpl(a_nodes, w0, wa, om_m, om_r, om_k)
    q_val = np.sum(weights * f_nodes)
    t_val = _true_dL(z, w0, wa, om_m, om_r, om_k, n=2000)
    if t_val < 1e-15:
        return 0.0
    return abs(q_val - t_val) / t_val


def _loss(params, models):
    """Worst-case relative error across all models."""
    nodes = params[:8]
    log_weights = params[8:]
    weights = np.exp(log_weights)
    weights = weights / np.sum(weights)
    
    if not np.all(np.diff(nodes) > 0):
        return 1e10
    if nodes[0] < 0 or nodes[-1] > 1:
        return 1e10
    if not np.all(weights > 0):
        return 1e10
    
    max_err = 0.0
    for (z, w0, wa, om_m, om_r, om_k) in models:
        err = _quadrature_error(nodes, weights, z, w0, wa, om_m, om_r, om_k)
        max_err = max(max_err, err)
    return max_err


def _valid(nodes, weights):
    """Check validity of nodes and weights."""
    if len(nodes) != 8 or len(weights) != 8:
        return False
    if not np.all(np.diff(nodes) > 0):
        return False
    if nodes[0] < 0 or nodes[-1] > 1:
        return False
    if not np.all(weights > 0):
        return False
    if abs(np.sum(weights) - 1.0) > 1e-10:
        return False
    return True


def construct():
    """
    Constructs an 8-node positive quadrature rule on [0,1] for luminosity distance.
    
    Strategy:
    1. Start with Gauss-Legendre nodes and weights (optimal for smooth functions).
    2. Use SLSQP optimization to refine nodes and weights to minimize worst-case
       relative error across a set of representative CPL cosmologies.
    3. Return the optimized rule or fall back to GL if optimization fails.
    """
    n = 8
    deadline = time.time() + 60.0  # 60s budget
    
    # Step 1: Get Gauss-Legendre as starting point
    gl_nodes, gl_weights = _gauss_legendre_rule(n)
    
    # Build training models (representative CPL parameters)
    # z range: 0.1 to 3.0
    # w0: -0.5 to 0.5
    # wa: -0.5 to 0.5
    # om_m: 0.2 to 0.4
    # om_r: 9e-5 (standard)
    # om_k: derived from flatness
    
    models = []
    z_vals = [0.5, 1.0, 2.0]
    w0_vals = [-0.2, 0.0, 0.2]
    wa_vals = [-0.2, 0.0, 0.2]
    om_m_vals = [0.25, 0.30]
    
    for z in z_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                for om_m in om_m_vals:
                    om_r = 9e-5
                    # Flat universe: om_k = 1 - om_m - om_r - om_de
                    # For CPL, om_de evolves, but at z=0: om_de = 1 - om_m - om_r
                    # We'll use om_k=0 and let om_de adjust via the H(z) formula
                    # Actually, for flat universe: om_m + om_r + om_de = 1
                    # In E(a) formula, we use om_de0 = 1 - om_m - om_r
                    om_k = 0.0
                    models.append((z, w0, wa, om_m, om_r, om_k))
    
    # If we have time, try to optimize
    if time.time() < deadline - 10.0:  # Need at least 10s remaining
        try:
            # Initial parameters: GL nodes and log-weights
            x0 = np.concatenate([gl_nodes, np.log(gl_weights)])
            
            # Bounds: nodes in (0,1) with strict ordering, weights positive
            bounds = []
            for i in range(8):
                if i == 0:
                    bounds.append((1e-6, 1.0 - 1e-6))
                elif i == 7:
                    bounds.append((1e-6, 1.0 - 1e-6))
                else:
                    bounds.append((1e-6, 1.0 - 1e-6))
            for i in range(8):
                bounds.append((-10, 10))  # log weights
            
            # Constraints: nodes must be strictly increasing
            cons = []
            for i in range(7):
                def make_con(i):
                    def con(params):
                        return params[i+1] - params[i] - 1e-8
                    return con
                cons.append({'type': 'ineq', 'fun': make_con(i)})
            
            res = minimize(
                _loss,
                x0,
                args=(models,),
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 50, 'ftol': 1e-10}
            )
            
            if res.success and time.time() < deadline - 5.0:
                best_nodes = res.x[:8]
                best_log_weights = res.x[8:]
                best_weights = np.exp(best_log_weights)
                best_weights = best_weights / np.sum(best_weights)
                
                if _valid(best_nodes, best_weights):
                    return {
                        "nodes": best_nodes,
                        "weights": best_weights
                    }
        except Exception:
            pass
    
    # Fallback: return Gauss-Legendre
    return {
        "nodes": gl_nodes,
        "weights": gl_weights
    }
# EVOLVE-BLOCK-END
