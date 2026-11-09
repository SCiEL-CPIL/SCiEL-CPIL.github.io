"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution
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


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline, 5s safety margin

    # Representative CPL models
    models = [
        (0.0, 0.0),
        (-1.0, 0.0),
        (-0.5, -0.5),
        (-1.0, 0.5),
        (-0.8, -0.5),
        (-1.2, 0.3),
        (-0.3, -0.3),
        (-1.0, -0.5),
    ]

    # Test redshifts
    z_test = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0])

    # Pre-compute exact integrals to speed up objective function
    exact_integrals = []
    for w0, wa in models:
        for z in z_test:
            try:
                val = _exact_integral(z, w0, wa)
                exact_integrals.append(val)
            except Exception:
                exact_integrals.append(1.0) # Fallback if quad fails
    exact_integrals = np.array(exact_integrals)
    
    model_params = [(w0, wa) for w0, wa in models]
    z_vals = z_test
    
    def objective(params):
        if _time_left(deadline) < 5.0:
            return 1e10
            
        # Decode nodes
        raw_nodes = np.exp(params[:8])
        nodes = np.cumsum(raw_nodes) / np.sum(raw_nodes)
        
        # Decode weights
        log_weights = params[8:]
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights)
        
        max_err = 0.0
        idx = 0
        for i_model, (w0, wa) in enumerate(model_params):
            for i_z, z in enumerate(z_vals):
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
                
                approx = z * np.dot(weights, vals)
                
                exact = exact_integrals[idx]
                idx += 1
                
                if abs(exact) < 1e-15:
                    err = 0.0
                else:
                    err = abs((approx - exact) / exact)
                
                if err > max_err:
                    max_err = err
                    
        return max_err

    # Initial guess: Gauss-Legendre
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    nodes0 = 0.5 * (gl_nodes + 1.0)
    weights0 = 0.5 * gl_weights
    
    bounds = []
    for _ in range(8):
        bounds.append((-5.0, 5.0))
    for _ in range(8):
        bounds.append((-10.0, 10.0))

    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=True
        )
        
        if result.fun < 1e10:
            raw_nodes = np.exp(result.x[:8])
            best_nodes = np.cumsum(raw_nodes) / np.sum(raw_nodes)
            log_weights = result.x[8:]
            best_weights = np.exp(log_weights)
            best_weights = best_weights / np.sum(best_weights)
            
            if np.all(np.diff(best_nodes) > 0) and np.all(best_nodes > 0) and np.all(best_nodes < 1):
                return {"nodes": best_nodes, "weights": best_weights}
                
    except Exception:
        pass

    # Fallback: Gauss-Legendre
    return {"nodes": nodes0, "weights": weights0}
# EVOLVE-BLOCK-END
