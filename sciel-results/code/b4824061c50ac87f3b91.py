"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # Safety margin for 90s limit

    # Define a diverse set of cosmologies for training/validation
    # Covering a wide range of w0, wa, Om
    cosmologies = [
        {'Om': 0.3, 'Or': 1e-4, 'Ode': 0.7, 'w0': -1.0, 'wa': 0.0},   # LambdaCDM
        {'Om': 0.3, 'Or': 1e-4, 'Ode': 0.7, 'w0': -0.9, 'wa': 0.3},
        {'Om': 0.3, 'Or': 1e-4, 'Ode': 0.7, 'w0': -1.1, 'wa': -0.3},
        {'Om': 0.3, 'Or': 1e-4, 'Ode': 0.7, 'w0': -0.8, 'wa': 0.5},
        {'Om': 0.3, 'Or': 1e-4, 'Ode': 0.7, 'w0': -0.7, 'wa': -0.5},
        {'Om': 0.2, 'Or': 1e-4, 'Ode': 0.8, 'w0': -1.0, 'wa': 0.0},
        {'Om': 0.4, 'Or': 1e-4, 'Ode': 0.6, 'w0': -1.0, 'wa': 0.0},
        {'Om': 0.3, 'Or': 1e-4, 'Ode': 0.7, 'w0': -0.95, 'wa': 0.2},
        {'Om': 0.3, 'Or': 1e-4, 'Ode': 0.7, 'w0': -1.05, 'wa': -0.2},
        {'Om': 0.3, 'Or': 1e-4, 'Ode': 0.7, 'w0': -0.85, 'wa': 0.4},
        {'Om': 0.3, 'Or': 1e-4, 'Ode': 0.7, 'w0': -0.75, 'wa': -0.4},
        {'Om': 0.35, 'Or': 1e-4, 'Ode': 0.65, 'w0': -1.0, 'wa': 0.1},
        {'Om': 0.25, 'Or': 1e-4, 'Ode': 0.75, 'w0': -1.0, 'wa': -0.1},
    ]
    
    # z values for testing. Include small and large z.
    z_tests = np.logspace(-2, 1, 20) # 0.01 to 10

    def E_of_a(a, cosmo):
        Om, Or, Ode, w0, wa = cosmo['Om'], cosmo['Or'], cosmo['Ode'], cosmo['w0'], cosmo['wa']
        E2 = Om/a**3 + Or/a**4
        if abs(wa) < 1e-12:
            E2 += Ode * a**(-3.0*(1.0+w0))
        else:
            E2 += Ode * a**(-3.0*(1.0+w0+wa*(1.0-a)))
        return np.sqrt(np.maximum(E2, 1e-30))
    
    def true_integral(z, cosmo):
        if z < 1e-10:
            return 0.0
        def integrand(zp):
            a = 1.0/(1.0+zp)
            return 1.0/E_of_a(a, cosmo)
        val, _ = quad(integrand, 0, z, limit=100, epsabs=1e-10, epsrel=1e-10)
        return (1.0+z)*z*val
    
    # Precompute true values for the test set
    true_vals = np.zeros((len(cosmologies), len(z_tests)))
    for i, cosmo in enumerate(cosmologies):
        for j, z in enumerate(z_tests):
            true_vals[i, j] = true_integral(z, cosmo)

    def fast_worst_error(nodes, weights):
        max_e = 0.0
        # Vectorize over cosmologies and z for speed
        # nodes shape (8,), weights shape (8,)
        # z_tests shape (Nz,)
        # cosmologies shape (Nc,)
        
        # We need to compute sum_i weights[i] / E(a_i) for each (cosmo, z)
        # a_i = 1 / (1 + z * nodes[i])
        
        for i_cosmo, cosmo in enumerate(cosmologies):
            for j_z, z in enumerate(z_tests):
                a_vals = 1.0 / (1.0 + z * nodes)
                E_vals = E_of_a(a_vals, cosmo)
                qv = (1.0 + z) * z * np.sum(weights / E_vals)
                tv = true_vals[i_cosmo, j_z]
                if abs(tv) < 1e-15:
                    continue
                e = abs(qv - tv) / abs(tv) * 100.0
                if e > max_e:
                    max_e = e
        return max_e

    def objective(params):
        if time.time() > deadline:
            return 1e10
        
        # params: 8 u's (gaps) + 8 log_weights
        u = params[:8]
        log_w = params[8:]
        
        # Ensure u is positive
        u = np.maximum(u, 1e-6)
        
        # Compute nodes from u
        cumsum_u = np.cumsum(u)
        nodes = cumsum_u / cumsum_u[-1]
        
        # Ensure nodes are strictly inside (0,1)
        # The transformation guarantees this if u > 0.
        
        # Compute weights
        w = np.exp(log_w)
        w = w / np.sum(w)
        
        return fast_worst_error(nodes, w)

    # Initial guess: Gauss-Legendre nodes
    # Convert Gauss-Legendre nodes to u (gaps)
    gl_nodes, _ = np.polynomial.legendre.leggauss(8)
    # gl_nodes are in (-1, 1). Map to (0, 1)
    gl_nodes_01 = (gl_nodes + 1) / 2
    gl_nodes_01 = np.sort(gl_nodes_01)
    
    # Compute gaps u from gl_nodes_01
    # t_i = (u_1 + ... + u_i) / S
    # t_1 = u_1 / S => u_1 = t_1 * S
    # t_2 = (u_1 + u_2) / S => u_2 = (t_2 - t_1) * S
    # We can just set u_i proportional to differences, with u_1 proportional to t_1.
    # Let S = 1. Then u_1 = t_1, u_i = t_i - t_{i-1} for i>1.
    # But we need 8 u's. The sum of u's is 1.
    # t_1 = u_1
    # t_2 = u_1 + u_2
    # ...
    # t_8 = sum(u) = 1
    # So u_1 = t_1, u_2 = t_2 - t_1, ..., u_8 = 1 - t_7.
    u0 = np.diff(np.concatenate([[0], gl_nodes_01, [1]]))
    
    w0 = np.zeros(8) # log(1) = 0, so weights start equal
    
    x0 = np.concatenate([u0, w0])
    
    best_x = x0.copy()
    best_obj = objective(x0)
    
    # Optimization
    # Powell is good for non-smooth objectives
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, x0, method='Powell',
                          options={'maxiter': 100, 'maxfev': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
        except Exception as e:
            pass

    # Local refinement with Nelder-Mead
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, best_x, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
        except:
            pass

    # Extract nodes and weights
    u = np.maximum(best_x[:8], 1e-6)
    cumsum_u = np.cumsum(u)
    nodes = cumsum_u / cumsum_u[-1]
    
    log_w = best_x[8:]
    w = np.exp(log_w)
    w = w / np.sum(w)
    
    # Final validation and fallback
    if not (np.all(nodes > 0) and np.all(nodes < 1) and np.all(np.diff(nodes) > 0)):
        nodes = (np.arange(8) + 0.5) / 8.0
        w = np.full(8, 0.125)
        
    # Ensure weights sum to 1
    w = w / np.sum(w)
    
    return {"nodes": nodes, "weights": w}
# EVOLVE-BLOCK-END
