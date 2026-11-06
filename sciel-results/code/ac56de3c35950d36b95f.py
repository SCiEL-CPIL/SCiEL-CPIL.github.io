import numpy as np
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_TIME_LIMIT = 130.0

def _remaining_time():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _qaoa_state(n, edges, gammas, betas):
    N = 2 ** n
    psi = np.ones(N, dtype=np.complex128) / np.sqrt(N)
    k_vals = np.arange(N)
    
    cut_vals = np.zeros(N)
    for (i, j) in edges:
        z_i = np.where((k_vals >> i) & 1, -1.0, 1.0)
        z_j = np.where((k_vals >> j) & 1, -1.0, 1.0)
        cut_vals += (1.0 - z_i * z_j) / 2.0
    
    for layer in range(len(gammas)):
        gamma = gammas[layer]
        beta = betas[layer]
        psi *= np.exp(-1j * gamma * cut_vals)
        
        gc = np.cos(beta)
        gs = -1j * np.sin(beta)
        for i in range(n):
            bit = 1 << i
            psi = gc * psi + gs * psi[k_vals ^ bit]
    
    return psi

def _expected_cut(psi, n, edges):
    probs = np.abs(psi) ** 2
    N = 2 ** n
    k_vals = np.arange(N)
    exp_cut = 0.0
    for (i, j) in edges:
        z_i = np.where((k_vals >> i) & 1, -1.0, 1.0)
        z_j = np.where((k_vals >> j) & 1, -1.0, 1.0)
        zz = z_i * z_j
        exp_cut += (1.0 - np.dot(probs, zz)) / 2.0
    return float(np.real(exp_cut))

def _exact_maxcut(n, edges):
    N = 2 ** n
    k_vals = np.arange(N)
    total_cut = np.zeros(N)
    for (i, j) in edges:
        z_i = np.where((k_vals >> i) & 1, -1.0, 1.0)
        z_j = np.where((k_vals >> j) & 1, -1.0, 1.0)
        total_cut += (1.0 - z_i * z_j) / 2.0
    return int(np.max(total_cut))

def _optimize_angles(n, edges, p):
    exact_opt = _exact_maxcut(n, edges)
    if exact_opt == 0:
        return np.zeros((p, 2))
    
    def objective(params):
        gammas = params[:p]
        betas = params[p:]
        psi = _qaoa_state(n, edges, gammas, betas)
        return -_expected_cut(psi, n, edges)
    
    best_params = None
    best_val = -np.inf
    n_starts = 10
    rng = np.random.RandomState(42)
    
    for start in range(n_starts):
        if _remaining_time() < 5.0:
            break
        if start == 0:
            x0 = np.array([0.5, 0.5, 0.5, 0.5])
        elif start == 1:
            x0 = np.array([0.35, 0.25, 0.35, 0.25])
        else:
            x0 = rng.uniform(0, np.pi, size=4)
        
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    
    if best_params is None:
        return np.array([[0.5, 0.5], [0.5, 0.5]])
    
    if _remaining_time() > 5.0:
        res2 = minimize(objective, best_params, method='BFGS',
                       options={'maxiter': 100, 'gtol': 1e-10})
        if res2.fun < best_val:
            best_params = res2.x
    
    gammas = best_params[:p]
    betas = best_params[p:]
    return np.column_stack([gammas, betas])

def angles(n, edges, p):
    if _remaining_time() < 10.0:
        return np.tile([0.5, 0.5], (p, 1))
    result = _optimize_angles(n, edges, p)
    return result
