"""QAOA parameter design: vectorized state simulation with analytic gradients + multi-start optimization."""
import numpy as np
import time
import scipy.optimize


def _precompute_cut_values(n, edges):
    """Precompute the cut value for every possible state mask."""
    dim = 1 << n
    cut_vals = np.zeros(dim, dtype=np.float64)
    for mask in range(dim):
        val = 0
        for u, v in edges:
            if ((mask >> u) ^ (mask >> v)) & 1:
                val += 1
        cut_vals[mask] = val
    return cut_vals


def _apply_mixer(psi, n, beta):
    """Apply exp(-i * beta * sum(X_i)) to the state vector psi."""
    cb = np.cos(beta)
    sb = -1j * np.sin(beta)
    
    shape = (2,) * n
    psi_t = psi.reshape(shape)
    
    for i in range(n):
        psi_moved = np.moveaxis(psi_t, i, 0)
        h0 = psi_moved[0].copy()
        h1 = psi_moved[1].copy()
        psi_moved[0] = cb * h0 + sb * h1
        psi_moved[1] = sb * h0 + cb * h1
        psi_t = np.moveaxis(psi_moved, 0, i)
    
    return psi_t.reshape(-1)


def _qaoa_expectation(n, gammas, betas, cut_vals):
    """Compute the expected cut size for given QAOA parameters."""
    dim = 1 << n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    p = len(gammas)
    for k in range(p):
        psi *= np.exp(-1j * gammas[k] * cut_vals)
        psi = _apply_mixer(psi, n, betas[k])
    
    probs = np.abs(psi) ** 2
    return float(np.dot(probs, cut_vals))


def _qaoa_gradient(n, params, cut_vals):
    """
    Compute gradient of expected cut w.r.t. params = [g0, b0, g1, b1]
    using the parameter-shift rule.
    """
    shift = np.pi / 2
    grad = np.zeros(4, dtype=np.float64)
    
    for idx in range(4):
        g_plus = np.array([params[0], params[2]])
        b_plus = np.array([params[1], params[3]])
        g_minus = np.array([params[0], params[2]])
        b_minus = np.array([params[1], params[3]])
        
        if idx == 0:
            g_plus[0] += shift
            g_minus[0] -= shift
        elif idx == 1:
            b_plus[0] += shift
            b_minus[0] -= shift
        elif idx == 2:
            g_plus[1] += shift
            g_minus[1] -= shift
        else:
            b_plus[1] += shift
            b_minus[1] -= shift
        
        exp_plus = _qaoa_expectation(n, g_plus, b_plus, cut_vals)
        exp_minus = _qaoa_expectation(n, g_minus, b_minus, cut_vals)
        
        grad[idx] = (exp_plus - exp_minus) / 2.0
    
    return grad


def _optimize_angles(n, edges, p, start_time, timeout):
    """Optimize angles using multi-start L-BFGS-B with parameter-shift gradients."""
    cut_vals = _precompute_cut_values(n, edges)
    
    def objective(params):
        try:
            gammas = np.array([params[0], params[2]])
            betas = np.array([params[1], params[3]])
            return -_qaoa_expectation(n, gammas, betas, cut_vals)
        except Exception:
            return 1e9

    def gradient(params):
        try:
            return -_qaoa_gradient(n, params, cut_vals)
        except Exception:
            return np.zeros(4)

    best_score = -np.inf
    best_params = None
    
    # Estimate evaluation time
    t0 = time.time()
    _qaoa_expectation(n, np.array([0.5, 0.5]), np.array([0.5, 0.5]), cut_vals)
    t1 = time.time()
    eval_time = max(t1 - t0, 0.0001)
    
    # Budget-aware number of restarts
    time_per_restart = 900 * eval_time
    available_time = max(timeout - (time.time() - start_time) - 10, 1.0)
    num_restarts = max(8, min(200, int(available_time / max(time_per_restart, 0.001))))
    
    np.random.seed(12345)
    
    # Initial points based on known QAOA behavior and graph structure
    base_inits = [
        [0.5, 0.5, 0.3, 0.3],
        [0.7, 0.7, 0.4, 0.4],
        [0.3, 0.4, 0.2, 0.3],
        [0.5, 0.3, 0.5, 0.3],
        [0.4, 0.6, 0.4, 0.6],
        [0.6, 0.2, 0.6, 0.2],
        [0.55, 0.45, 0.35, 0.35],
        [0.65, 0.55, 0.45, 0.45],
        [0.45, 0.55, 0.25, 0.45],
        [0.5, 0.5, 0.5, 0.5],
        [0.8, 0.3, 0.2, 0.2],
        [0.3, 0.8, 0.6, 0.4],
        [0.9, 0.4, 0.3, 0.5],
        [0.2, 0.6, 0.7, 0.3],
        [0.6, 0.6, 0.6, 0.6],
    ]
    
    bounds = [(0.0, np.pi)] * 4
    
    for r in range(num_restarts):
        if time.time() - start_time > timeout - 15:
            break
            
        if r < len(base_inits):
            init = np.array(base_inits[r], dtype=np.float64)
        else:
            init = np.random.uniform(0.1, 1.5, size=4)
        
        try:
            res = scipy.optimize.minimize(
                objective, init, jac=gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-7, 'maxls': 30, 'disp': False}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            continue
    
    # Polish with Nelder-Mead from the best L-BFGS-B result
    if best_params is not None and time.time() - start_time < timeout - 8:
        try:
            res_polish = scipy.optimize.minimize(
                objective, best_params, method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            polish_score = -res_polish.fun
            if polish_score > best_score:
                best_score = polish_score
                best_params = res_polish.x
        except Exception:
            pass
    
    # Second polish pass with even tighter tolerance if time permits
    if best_params is not None and time.time() - start_time < timeout - 4:
        try:
            res_polish2 = scipy.optimize.minimize(
                objective, best_params, method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-11, 'fatol': 1e-11}
            )
            polish_score2 = -res_polish2.fun
            if polish_score2 > best_score:
                best_score = polish_score2
                best_params = res_polish2.x
        except Exception:
            pass
    
    if best_params is not None:
        return np.array([[best_params[0], best_params[1]],
                         [best_params[2], best_params[3]]])
    else:
        return np.array([[0.35, 0.25], [0.35, 0.25]])


def angles(n, edges, p):
    start_time = time.time()
    timeout = 140  # Safety margin from 150s limit
    
    optimal_angles = _optimize_angles(n, edges, p, start_time, timeout)
    
    if time.time() - start_time > timeout:
        return np.array([[0.35, 0.25]] * p)
        
    return optimal_angles
