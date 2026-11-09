"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad


def _time_left(deadline):
    return max(0.0, deadline - time.time())


def _E_cpl(a, w0, wa):
    """Hubble parameter H(a)/H0 for CPL model in flat universe."""
    z = 1.0 / a - 1.0
    base = 1.0 + w0 + wa * z
    base = np.maximum(base, 1e-10)
    
    denom = 1.0 + w0
    if abs(denom) < 1e-10:
        exponent = 0.0
    else:
        exponent = wa / denom
        
    return a**(-3.0) * base**exponent


def _integrand(z, w0, wa):
    """1/E(z) for CPL model."""
    a = 1.0 / (1.0 + z)
    return 1.0 / _E_cpl(a, w0, wa)


def _exact_integral(z, w0, wa):
    """Compute exact integral_0^z dz'/E(z') using adaptive quadrature."""
    result, _ = quad(lambda zp: _integrand(zp, w0, wa), 0, z, limit=300, epsabs=1e-12, epsrel=1e-12)
    return result


def _quadrature_approx(z, nodes, weights, w0, wa):
    """Compute quadrature approximation for given z, nodes, weights, cosmology."""
    z_nodes = z * nodes
    a_nodes = 1.0 / (1.0 + z_nodes)
    z_calc = 1.0 / a_nodes - 1.0
    base = 1.0 + w0 + wa * z_calc
    base = np.maximum(base, 1e-10)
    denom = 1.0 + w0
    if abs(denom) < 1e-10:
        exponent = 0.0
    else:
        exponent = wa / denom
    E_vals = a_nodes**(-3.0) * base**exponent
    vals = 1.0 / E_vals
    return z * np.dot(weights, vals)


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline, 5s safety margin

    # Representative CPL models - expanded for better coverage
    models = [
        (0.0, 0.0),
        (-1.0, 0.0),
        (-0.5, -0.5),
        (-1.0, 0.5),
        (-0.8, -0.5),
        (-1.2, 0.3),
        (-0.3, -0.3),
        (-1.0, -0.5),
        (-0.7, 0.2),
        (-1.1, -0.2),
        (-0.6, 0.4),
        (-0.9, -0.3),
    ]

    # Test redshifts - dense grid
    z_test = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])

    # Pre-compute exact integrals
    exact_integrals = np.zeros((len(models), len(z_test)))
    for i, (w0, wa) in enumerate(models):
        for j, z in enumerate(z_test):
            try:
                exact_integrals[i, j] = _exact_integral(z, w0, wa)
            except Exception:
                exact_integrals[i, j] = 1.0
    
    model_params = models
    z_vals = z_test
    n_models = len(model_params)
    n_z = len(z_vals)

    def objective(params):
        if _time_left(deadline) < 3.0:
            return 1e10
            
        # Decode nodes using cumsum of exp for strict monotonicity
        raw_nodes = np.exp(params[:8])
        nodes = np.cumsum(raw_nodes) / np.sum(raw_nodes)
        
        # Decode weights using softmax for positivity
        log_weights = params[8:]
        weights = np.exp(log_weights - np.max(log_weights))
        weights = weights / np.sum(weights)
        
        max_err = 0.0
        for i in range(n_models):
            w0, wa = model_params[i]
            for j in range(n_z):
                z = z_vals[j]
                approx = _quadrature_approx(z, nodes, weights, w0, wa)
                exact = exact_integrals[i, j]
                if abs(exact) < 1e-15:
                    err = 0.0
                else:
                    err = abs((approx - exact) / exact)
                if err > max_err:
                    max_err = err
                    
        return max_err

    # Initial guess: Gauss-Chebyshev nodes (better for smooth integrands)
    # Gauss-Chebyshev nodes on [-1,1]: x_k = cos((2k-1)pi/(2n))
    n = 8
    k = np.arange(1, n + 1)
    gc_nodes_neg = np.cos((2 * k - 1) * np.pi / (2 * n))
    gc_weights_neg = np.full(n, np.pi / n)
    # Map to [0,1]
    nodes0 = 0.5 * (gc_nodes_neg + 1.0)
    weights0 = 0.5 * gc_weights_neg
    # Sort nodes in increasing order
    sort_idx = np.argsort(nodes0)
    nodes0 = nodes0[sort_idx]
    weights0 = weights0[sort_idx]
    
    # Convert to parameter space
    diffs = np.diff(nodes0)
    diffs[0] = nodes0[0]
    raw_nodes0 = np.log(diffs)
    log_weights0 = np.log(weights0)
    
    x0 = np.concatenate([raw_nodes0, log_weights0])
    
    # Multi-start local optimization
    best_x = x0
    best_obj = objective(x0)
    
    # Try several starting points
    np.random.seed(42)
    n_starts = 10
    for s in range(n_starts):
        if _time_left(deadline) < 5.0:
            break
        
        if s == 0:
            x_start = x0
        else:
            # Perturb the initial guess
            perturbation = np.random.randn(16) * 0.3
            x_start = x0 + perturbation
        
        try:
            result = minimize(
                objective,
                x_start,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_x = result.x
                
        except Exception:
            pass
    
    # Final polish with differential evolution if time permits
    if _time_left(deadline) > 15.0:
        bounds = []
        for _ in range(8):
            bounds.append((-5.0, 5.0))
        for _ in range(8):
            bounds.append((-10.0, 10.0))
        
        try:
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=50,
                popsize=10,
                tol=1e-6,
                seed=42,
                polish=True,
                init='sohs'
            )
            
            if de_result.fun < best_obj:
                best_obj = de_result.fun
                best_x = de_result.x
                
        except Exception:
            pass
    
    # Decode final result
    raw_nodes = np.exp(best_x[:8])
    best_nodes = np.cumsum(raw_nodes) / np.sum(raw_nodes)
    log_weights = best_x[8:]
    best_weights = np.exp(log_weights)
    best_weights = best_weights / np.sum(best_weights)
    
    # Validate
    if np.all(np.diff(best_nodes) > 0) and np.all(best_nodes > 0) and np.all(best_nodes < 1) and np.all(best_weights > 0):
        return {"nodes": best_nodes, "weights": best_weights}
    
    # Fallback: Gauss-Chebyshev
    return {"nodes": nodes0, "weights": weights0}
# EVOLVE-BLOCK-END
