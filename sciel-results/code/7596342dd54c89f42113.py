import numpy as np
import scipy.optimize as so
import scipy.integrate as si
import time

def construct():
    start_time = time.time()
    timeout = 90.0
    safety_margin = 5.0

    # Default: Gauss-Legendre
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes_gl = (x_gl + 1) / 2
    weights_gl = w_gl / 2

    if time.time() - start_time > timeout - safety_margin:
        return {"nodes": nodes_gl, "weights": weights_gl}

    # Training cosmologies
    # (Omega_m, w0, wa)
    cosmologies = [
        (0.3, -1.0, 0.0), # LCDM
        (0.3, -0.5, 0.0),
        (0.3, -1.5, 0.0),
        (0.3, -1.0, 1.0),
        (0.3, -1.0, -1.0),
        (0.2, -0.8, 0.5),
        (0.4, -1.2, -0.5),
        (0.3, -0.9, 0.2),
        (0.3, -1.1, -0.2),
        (0.25, -1.0, 0.5),
        (0.35, -1.0, -0.5),
        (0.3, -0.7, 0.3),
        (0.3, -1.3, -0.3),
    ]

    # Redshift grid
    z_grid = np.linspace(0.1, 5.0, 30)

    def E_of_a_vec(a, Om, w0, wa):
        # a is array
        Ow = 1.0 - Om
        term1 = Om * a**(-3)
        exp_arg = -3.0 * wa * (a - 1.0)
        power = 3.0 * (1.0 + w0 + wa)
        term2 = Ow * (a**power) * np.exp(exp_arg)
        return np.sqrt(term1 + term2)

    def integrand(x, z, Om, w0, wa):
        a = 1.0 / (1.0 + z * x)
        return 1.0 / E_of_a_vec(a, Om, w0, wa)

    # Precompute true integrals
    true_vals = np.zeros((len(cosmologies), len(z_grid)))
    for i, (Om, w0, wa) in enumerate(cosmologies):
        for j, z in enumerate(z_grid):
            val, err = si.quad(lambda x: integrand(x, z, Om, w0, wa), 0, 1, limit=100)
            true_vals[i, j] = val

    def objective(params):
        x = params[:8]
        w = params[8:]

        if np.any(x <= 0) or np.any(x >= 1):
            return 1e6
        if not np.all(np.diff(x) > 0):
            return 1e6
        if np.any(w <= 0):
            return 1e6
        if abs(np.sum(w) - 1.0) > 1e-6:
            return 1e6

        errors = []
        for i, (Om, w0, wa) in enumerate(cosmologies):
            a_i = 1.0 / (1.0 + z_grid * x) # shape (8, 30)
            E_vals = E_of_a_vec(a_i, Om, w0, wa)
            quad_sum = np.sum(w[:, np.newaxis] / E_vals, axis=0) # shape (30,)
            rel_err = np.abs(quad_sum - true_vals[i, :]) / true_vals[i, :]
            errors.extend(rel_err.tolist())

        errors = np.array(errors)
        tau = 50.0
        return np.log(np.sum(np.exp(tau * errors))) / tau

    x0 = np.concatenate([nodes_gl, weights_gl])

    cons = [{'type': 'eq', 'fun': lambda p: np.sum(p[8:]) - 1.0}]
    bounds = [(1e-4, 1-1e-4)] * 8 + [(1e-4, 1.0)] * 8

    for i in range(7):
        cons.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[i+1] - p[i] - 1e-5
        })

    try:
        res = so.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 200, 'ftol': 1e-12})
    except Exception:
        return {"nodes": nodes_gl, "weights": weights_gl}

    if time.time() - start_time > timeout - safety_margin:
        return {"nodes": nodes_gl, "weights": weights_gl}

    if res.success:
        x_opt = res.x[:8]
        w_opt = res.x[8:]
        w_opt = w_opt / np.sum(w_opt)
        
        if np.all(x_opt > 0) and np.all(x_opt < 1) and np.all(np.diff(x_opt) > 0) and np.all(w_opt > 0):
             return {"nodes": x_opt, "weights": w_opt}
             
    return {"nodes": nodes_gl, "weights": weights_gl}
