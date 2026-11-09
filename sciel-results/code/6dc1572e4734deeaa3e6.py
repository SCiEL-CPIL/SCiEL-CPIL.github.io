import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Define CPL E(z)
    def E_cpl(z, w0, w1):
        # E(z) = (1+z)^(3(1+w0)) * (1+z)^(3w1(1 - 1/(1+w1)))
        # Careful with w1 = -1
        if np.isclose(w1, -1.0):
            return (1.0 + z) ** (3.0 * (1.0 + w0))
        exponent = 3.0 * w1 * (1.0 - 1.0 / (1.0 + w1))
        return (1.0 + z) ** (3.0 * (1.0 + w0) + exponent)
    
    def integrand(u, z, w0, w1):
        # f(u) = 1 / E(1/(1+zu))
        # a = 1 / (1 + z*u)
        # E(a) in terms of z_eff = z*u
        # The prompt uses E(a) where a is scale factor.
        # In standard CPL, E is a function of redshift z.
        # E(z_eff) = H(z_eff)/H0.
        # The integral derived was int_0^1 du / E(z*u).
        # So we need E(z*u).
        return 1.0 / E_cpl(z * u, w0, w1)
        
    def true_integral(z, w0, w1):
        # int_0^1 du / E(z*u)
        val, _ = quad(integrand, 0, 1, args=(z, w0, w1), limit=100)
        return val

    def quadrature_approx(nodes, weights, z, w0, w1):
        # sum w_i / E(z * nodes_i)
        # nodes are u_i
        z_u = z * nodes
        # E(z_u)
        e_vals = E_cpl(z_u, w0, w1)
        return np.sum(weights / e_vals)

    # Test Grid
    models = [
        (-1.0, 0.0),
        (-0.5, -0.5),
        (-1.5, 1.0),
        (-0.8, 0.5),
        (-1.2, -0.8)
    ]
    zs = [0.1, 0.5, 1.0, 2.0, 4.0]
    
    # Pre-compute true values
    true_vals = []
    params_list = []
    for w0, w1 in models:
        for z in zs:
            val = true_integral(z, w0, w1)
            true_vals.append(val)
            params_list.append((z, w0, w1))
            
    true_vals = np.array(true_vals)
    
    def objective(params):
        # params: [u1..u8, w1..w8]
        nodes = params[:8]
        weights = params[8:]
        
        # Check constraints (should be handled by optimizer, but safe to check)
        if np.any(nodes <= 0) or np.any(nodes >= 1) or np.any(weights <= 0):
            return 1e10
        if not np.all(np.diff(nodes) > 1e-6):
            return 1e10
        if abs(np.sum(weights) - 1.0) > 1e-6:
            return 1e10
            
        errors = []
        for i, (z, w0, w1) in enumerate(params_list):
            q_val = quadrature_approx(nodes, weights, z, w0, w1)
            t_val = true_vals[i]
            # Relative error
            err = abs(q_val - t_val) / t_val
            errors.append(err)
            
        errors = np.array(errors)
        # Use sum of powers to emphasize worst errors
        # Power 10 is a good compromise for smoothness and focus
        return np.sum(errors ** 10)

    # Initial guess: Gauss-Legendre or Midpoint
    # Let's use Midpoint as start
    u_init = (np.arange(8) + 0.5) / 8.0
    w_init = np.full(8, 1.0/8.0)
    x0 = np.concatenate([u_init, w_init])
    
    # Bounds
    bounds = [(1e-4, 1-1e-4)] * 8 + [(1e-4, 1.0)] * 8
    
    # Constraints
    # Sum of weights = 1
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x[8:]) - 1.0}]
    
    # Inequality constraints for node ordering
    # u_{i+1} - u_i - 1e-4 >= 0
    ineqs = []
    for i in range(7):
        ineqs.append({'type': 'ineq', 'fun': lambda x, i=i: x[i+1] - x[i] - 1e-4})
    cons.extend(ineqs)
    
    best_x = x0
    best_obj = objective(x0)
    
    # Optimize
    # Use SLSQP
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                       options={'maxiter': 500, 'ftol': 1e-12, 'disp': False})
        if res.fun < best_obj:
            best_x = res.x
            best_obj = res.fun
    except Exception:
        pass
        
    # Check time
    if time.time() > deadline - 10:
        # Not enough time for more, return current best
        pass
    else:
        # Try a few restarts with different initial guesses if time permits
        # Gauss-Legendre
        from numpy.polynomial.legendre import leggauss
        x_gl, w_gl = leggauss(8)
        u_gl = (x_gl + 1.0) / 2.0
        w_gl = w_gl / 2.0
        x_gl_init = np.concatenate([u_gl, w_gl])
        
        try:
            res_gl = minimize(objective, x_gl_init, method='SLSQP', bounds=bounds, constraints=cons, 
                              options={'maxiter': 500, 'ftol': 1e-12, 'disp': False})
            if res_gl.fun < best_obj:
                best_x = res_gl.x
                best_obj = res_gl.fun
        except Exception:
            pass

        # Random restarts
        if time.time() < deadline - 20:
            for _ in range(3):
                # Perturb initial guess
                u_rand = np.sort(np.random.uniform(0.05, 0.95, 8))
                w_rand = np.random.dirichlet(np.ones(8))
                x_rand = np.concatenate([u_rand, w_rand])
                try:
                    res_rand = minimize(objective, x_rand, method='SLSQP', bounds=bounds, constraints=cons, 
                                        options={'maxiter': 300, 'ftol': 1e-12, 'disp': False})
                    if res_rand.fun < best_obj:
                        best_x = res_rand.x
                        best_obj = res_rand.fun
                except Exception:
                    pass

    nodes = best_x[:8]
    weights = best_x[8:]
    
    # Final cleanup: ensure strictly increasing and sum to 1
    # Sort nodes if needed (though constraints should prevent it)
    order = np.argsort(nodes)
    nodes = nodes[order]
    weights = weights[order]
    
    # Normalize weights to sum to 1 exactly
    weights = weights / np.sum(weights)
    
    # Check validity
    if not np.all(np.diff(nodes) > 0) or np.any(weights <= 0):
        # Fallback to midpoint
        nodes = (np.arange(8) + 0.5) / 8.0
        weights = np.full(8, 1.0/8.0)

    return {"nodes": nodes, "weights": weights}
