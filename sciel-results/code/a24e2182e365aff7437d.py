"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad

def _E_of_a(a, wa, wb, w0):
    """H(a)/H0 for flat LCDM-like model with dark energy."""
    matter = (1.0 - a)**3
    de = w0 + wb * (1.0 - a)
    de_factor = a**(-3.0 * (1.0 + de))
    return np.sqrt(matter + de_factor)

def _integrand_z(z, wa, wb, w0):
    """Integrand for dL/(c/H0) = (1+z)*z*int_0^z dz'/E(a')."""
    a = 1.0 / (1.0 + z)
    return a / _E_of_a(a, wa, wb, w0)

def _exact_dL(z, wa, wb, w0):
    """High-accuracy adaptive integration of dL/(c/H0)."""
    def f(zp):
        return _integrand_z(zp, wa, wb, w0)
    val, _ = quad(f, 0, z, limit=200)
    return (1.0 + z) * z * val

def _quad_dL(z, nodes, weights, wa, wb, w0):
    """Quadrature approximation of dL/(c/H0)."""
    a = 1.0 / (1.0 + z * nodes)
    E_vals = _E_of_a(a, wa, wb, w0)
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx

def _relative_error(z, nodes, weights, wa, wb, w0):
    exact = _exact_dL(z, wa, wb, w0)
    approx = _quad_dL(z, nodes, weights, wa, wb, w0)
    if exact == 0:
        return 0.0
    return abs(approx - exact) / exact * 100.0

def _worst_error(nodes, weights, z_samples, models):
    """Compute worst-case relative error across all z and models."""
    max_err = 0.0
    for z in z_samples:
        for wa, wb, w0 in models:
            err = _relative_error(z, nodes, weights, wa, wb, w0)
            if err > max_err:
                max_err = err
    return max_err

def _gauss_legendre_nodes_weights(n):
    """Get Gauss-Legendre nodes and weights on [0,1]."""
    # Gauss-Legendre on [-1,1]
    x, w = np.polynomial.legendre.leggauss(n)
    # Transform to [0,1]
    nodes = 0.5 * (x + 1.0)
    weights = 0.5 * w
    return nodes, weights

def _optimize_from_initial(nodes_init, weights_init, z_samples, models, time_limit=30.0):
    """Optimize nodes and weights together using SLSQP with time limit."""
    start_time = time.time()
    n = len(nodes_init)
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e10
        nodes = params[:n]
        weights = params[n:]
        
        # Check constraints
        if nodes[0] < 0 or nodes[-1] > 1:
            return 1e10
        if np.any(np.diff(nodes) <= 1e-10):
            return 1e10
        if np.any(weights <= 0):
            return 1e10
        if abs(np.sum(weights) - 1.0) > 1e-8:
            return 1e10
        
        return _worst_error(nodes, weights, z_samples, models)
    
    # Constraints: sum(weights) = 1, nodes in [0,1], strictly increasing
    constraints = [
        {'type': 'eq', 'fun': lambda p: np.sum(p[n:]) - 1.0}
    ]
    
    bounds = [(1e-6, 1 - 1e-6)] * n + [(1e-10, None)] * n
    
    x0 = np.concatenate([nodes_init, weights_init])
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                         constraints=constraints, options={'maxiter': 100, 'ftol': 1e-8})
        
        if result.success and result.fun < 1e10:
            nodes_opt = result.x[:n]
            weights_opt = result.x[n:]
            # Ensure weights sum to 1 and are positive
            weights_opt = np.maximum(weights_opt, 1e-15)
            weights_opt = weights_opt / np.sum(weights_opt)
            return nodes_opt, weights_opt, result.fun
        else:
            return nodes_init, weights_init, objective(x0)
    except Exception:
        return nodes_init, weights_init, objective(x0)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline (5s margin)
    
    # Diverse cosmology sample for optimization
    models = [
        (0.0, 0.0, -1.0),       # LambdaCDM
        (0.1, 0.0, -0.9),       # Mild DE
        (0.0, 0.3, -1.0),       # CPL with w_a>0
        (0.0, -0.5, -1.0),      # CPL with w_a<0
        (0.2, 0.5, -0.8),       # Stronger matter, mild DE
        (0.0, 0.0, -0.5),       # Quintessence-like
        (0.1, -0.3, -1.2),      # Phantom-like
        (0.0, 0.8, -0.7),       # Strong w_a
        (0.0, 0.0, -1.5),       # Phantom
        (0.0, 0.0, -0.8),       # Mild quintessence
    ]
    
    # Z samples for evaluation - include more spread
    z_samples = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0])
    
    n = 8
    best_nodes = None
    best_weights = None
    best_score = np.inf
    
    # Try multiple initializations
    initializations = []
    
    # 1. Gauss-Legendre
    gl_nodes, gl_weights = _gauss_legendre_nodes_weights(n)
    initializations.append((gl_nodes, gl_weights))
    
    # 2. Midpoint rule
    mid_nodes = (np.arange(n) + 0.5) / n
    mid_weights = np.full(n, 1.0 / n)
    initializations.append((mid_nodes, mid_weights))
    
    # 3. Chebyshev-like (cosine spaced)
    cos_nodes = 0.5 * (1.0 - np.cos(np.pi * (np.arange(n) + 0.5) / n))
    cos_weights = np.full(n, 1.0 / n)
    initializations.append((cos_nodes, cos_weights))
    
    # 4. Clustered toward low z (since integrand may be more structured there)
    cluster_nodes = np.array([0.01, 0.03, 0.06, 0.1, 0.15, 0.25, 0.4, 0.7])
    cluster_weights = np.array([0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.12])
    cluster_weights = cluster_weights / np.sum(cluster_weights)
    initializations.append((cluster_nodes, cluster_weights))
    
    # 5. Uniformly spaced but with edge emphasis
    edge_nodes = np.array([0.005, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.95])
    edge_weights = np.array([0.08, 0.1, 0.12, 0.15, 0.18, 0.18, 0.15, 0.04])
    edge_weights = edge_weights / np.sum(edge_weights)
    initializations.append((edge_nodes, edge_weights))
    
    # Evaluate and optimize from each initialization
    for i, (nodes_init, weights_init) in enumerate(initializations):
        if time.time() > deadline - 10:
            break
        
        # First evaluate the initial point
        score = _worst_error(nodes_init, weights_init, z_samples, models)
        if score < best_score:
            best_score = score
            best_nodes = nodes_init.copy()
            best_weights = weights_init.copy()
        
        # Optimize from this starting point
        time_left = deadline - time.time() - 5
        if time_left < 5:
            continue
        
        nodes_opt, weights_opt, opt_score = _optimize_from_initial(
            nodes_init, weights_init, z_samples, models, time_limit=time_left
        )
        
        if opt_score < best_score:
            best_score = opt_score
            best_nodes = nodes_opt.copy()
            best_weights = weights_opt.copy()
    
    # If we still have time, try a broader global search
    if time.time() < deadline - 20 and best_nodes is not None:
        try:
            def global_objective(params):
                if time.time() > deadline - 10:
                    return 1e10
                nodes = params[:n]
                weights = params[n:]
                
                if nodes[0] < 0 or nodes[-1] > 1:
                    return 1e10
                if np.any(np.diff(nodes) <= 1e-10):
                    return 1e10
                if np.any(weights <= 0):
                    return 1e10
                if abs(np.sum(weights) - 1.0) > 1e-6:
                    return 1e10
                
                return _worst_error(nodes, weights, z_samples, models)
            
            bounds = [(1e-5, 1 - 1e-5)] * n + [(1e-10, None)] * n
            x0 = np.concatenate([best_nodes, best_weights])
            
            result = differential_evolution(
                global_objective, bounds,
                maxiter=30, popsize=12,
                tol=1e-5, seed=42,
                polish=True,
                workers=1,
                time_limit=min(25, deadline - time.time() - 10)
            )
            
            if result.success and result.fun < best_score:
                best_nodes = result.x[:n]
                best_weights = result.x[n:]
                best_weights = np.maximum(best_weights, 1e-15)
                best_weights = best_weights / np.sum(best_weights)
                best_score = result.fun
        except Exception:
            pass
    
    # Final polish if time allows
    if time.time() < deadline - 15 and best_nodes is not None:
        nodes_opt, weights_opt, opt_score = _optimize_from_initial(
            best_nodes, best_weights, z_samples, models, time_limit=10
        )
        if opt_score < best_score:
            best_nodes = nodes_opt
            best_weights = weights_opt
            best_score = opt_score
    
    # Ensure valid output
    if best_nodes is None:
        best_nodes, best_weights = _gauss_legendre_nodes_weights(n)
    
    best_nodes = np.sort(best_nodes)
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)
    
    # Final validation
    final_score = _worst_error(best_nodes, best_weights, z_samples, models)
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
