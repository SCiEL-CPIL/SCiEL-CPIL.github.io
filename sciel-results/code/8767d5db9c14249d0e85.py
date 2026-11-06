import numpy as np
import time
import scipy.optimize as opt
from scipy.integrate import quad

def _E_cpl(z, w0, w1, Om):
    one_plus_z = 1.0 + z
    exp_out = 3.0 * (1.0 + w0) / 2.0
    exp_in = 3.0 * (1.0 - w0 - w1)
    term_m = Om * one_plus_z ** exp_in
    term_de = 1.0 - Om
    return (one_plus_z ** exp_out) * np.sqrt(term_m + term_de)

def _exact_integral(z, w0, w1, Om):
    f = lambda zp: 1.0 / _E_cpl(zp, w0, w1, Om)
    result, _ = quad(f, 0, z, limit=100)
    return result

def _quad_value(nodes, weights, z, w0, w1, Om):
    z_vals = z * nodes
    E_vals = _E_cpl(z_vals, w0, w1, Om)
    return z * np.dot(weights, 1.0 / E_vals)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    models = [
        (-1.0, 0.0, 0.3),
        (-1.2, 0.0, 0.3),
        (-0.8, 0.0, 0.3),
        (-1.0, -0.5, 0.3),
        (-1.0, 0.5, 0.3),
        (-0.8, -1.0, 0.3),
        (-1.2, 1.0, 0.3),
        (-1.0, 0.0, 0.2),
        (-1.0, 0.0, 0.35),
        (-0.9, 0.5, 0.25),
    ]
    z_test = np.array([0.5, 1.0, 2.0, 5.0])

    exact = {}
    for (w0, w1, Om) in models:
        for z in z_test:
            exact[(w0, w1, Om, z)] = _exact_integral(z, w0, w1, Om)

    def worst_err(nodes, weights):
        if not np.all((nodes > 0) & (nodes < 1)):
            return np.inf
        if not np.all(np.diff(nodes) > 1e-10):
            return np.inf
        if not np.all(weights > 1e-12):
            return np.inf
        if abs(np.sum(weights) - 1.0) > 1e-8:
            return np.inf
        max_e = 0.0
        for (w0, w1, Om) in models:
            for z in z_test:
                e = exact[(w0, w1, Om, z)]
                a = _quad_value(nodes, weights, z, w0, w1, Om)
                r = abs(a - e) / abs(e)
                if r > max_e:
                    max_e = r
        return max_e

    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    best_n = (x_gl + 1.0) / 2.0
    best_w = w_gl / 2.0
    best_w /= np.sum(best_w)
    best_e = worst_err(best_n, best_w)

    for iteration in range(5):
        if time.time() > deadline - 5:
            break

        def obj_w(vars):
            w = vars
            if not np.all(w > 1e-12):
                return np.inf
            if abs(np.sum(w) - 1.0) > 1e-8:
                return np.inf
            return worst_err(best_n, w)

        try:
            res_w = opt.minimize(obj_w, best_w, method='SLSQP',
                                  bounds=[(1e-6, 1.0)]*8,
                                  constraints=[{'type':'eq','fun':lambda v: np.sum(v)-1.0}],
                                  options={'maxiter':50,'ftol':1e-10})
            if res_w.success and res_w.fun < best_e:
                best_w = res_w.x / np.sum(res_w.x)
                best_e = res_w.fun
        except:
            pass

        def obj_n(vars):
            n = vars
            if not np.all((n > 0) & (n < 1)):
                return np.inf
            if not np.all(np.diff(n) > 1e-10):
                return np.inf
            return worst_err(n, best_w)

        try:
            res_n = opt.minimize(obj_n, best_n, method='SLSQP',
                                  bounds=[(1e-4, 1-1e-4)]*8,
                                  options={'maxiter':50,'ftol':1e-10})
            if res_n.success and res_n.fun < best_e:
                best_n = res_n.x
                sort_idx = np.argsort(best_n)
                best_n = best_n[sort_idx]
                best_w = best_w[sort_idx]
                best_e = res_n.fun
        except:
            pass

    if time.time() < deadline - 3:
        try:
            def obj_full(vars):
                n = vars[:8]
                w = vars[8:]
                return worst_err(n, w)

            def con_s(vars):
                return np.sum(vars[8:]) - 1.0

            x0 = np.concatenate([best_n, best_w])
            res = opt.minimize(obj_full, x0, method='SLSQP',
                                bounds=[(1e-4,1-1e-4)]*8 + [(1e-6,1.0)]*8,
                                constraints=[{'type':'eq','fun':con_s}],
                                options={'maxiter':100,'ftol':1e-12})
            if res.success and res.fun < best_e:
                best_n = res.x[:8]
                best_w = res.x[8:]
                sort_idx = np.argsort(best_n)
                best_n = best_n[sort_idx]
                best_w = best_w[sort_idx]
                best_e = res.fun
        except:
            pass

    eps = 1e-10
    best_n = np.clip(best_n, eps, 1.0-eps)
    sort_idx = np.argsort(best_n)
    best_n = best_n[sort_idx]
    best_w = best_w[sort_idx]
    best_w = best_w / np.sum(best_w)

    return {"nodes": best_n, "weights": best_w}
