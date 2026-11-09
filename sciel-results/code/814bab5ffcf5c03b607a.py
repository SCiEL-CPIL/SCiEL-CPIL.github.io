import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_herm
import time

def _compute_score(a0, a4, a8):
    H0_0 = 1.0
    H4_0 = -12.0
    H8_0 = 1008.0
    H12_0 = -103950.0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    N_fft = 2**18
    x = np.linspace(-50, 50, N_fft, endpoint=False)
    dx = x[1] - x[0]
    
    H0 = eval_herm(0, x)
    H4 = eval_herm(4, x)
    H8 = eval_herm(8, x)
    H12 = eval_herm(12, x)
    
    g = (a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12) * np.exp(-x**2 / 2)
    
    l1 = np.sum(np.abs(g)) * dx
    l2 = np.sqrt(np.sum(g**2) * dx)
    
    G = np.fft.fft(g)
    l1_hat = np.sum(np.abs(G)) / N_fft
    l2_hat = np.sqrt(np.sum(np.abs(G)**2) * dx / N_fft)
    
    if l1 == 0 or l1_hat == 0 or l2 == 0 or l2_hat == 0:
        return np.inf
    
    return (l1 * l1_hat) / (l2 * l2_hat)

def construct():
    start_time = time.time()
    deadline = start_time + 180
    
    a0_seed = -113 / 100
    a4_seed = 1 / 25
    a8_seed = 1 / 3240
    
    best_score = _compute_score(a0_seed, a4_seed, a8_seed)
    best_params = np.array([a0_seed, a4_seed, a8_seed])
    
    def objective(params):
        return _compute_score(params[0], params[1], params[2])
    
    x0 = np.array([a0_seed, a4_seed, a8_seed])
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
    except:
        pass
    
    if time.time() > deadline - 30:
        return best_params
    
    rng = np.random.default_rng(123)
    for scale in [0.1, 0.5, 1.0, 2.0]:
        if time.time() > deadline - 30:
            break
        for i in range(5):
            if time.time() > deadline - 30:
                break
            x0_try = x0 + rng.normal(0, scale, 3)
            try:
                result = minimize(objective, x0_try, method='Nelder-Mead',
                                 options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
                if result.fun < best_score:
                    best_score = result.fun
                    best_params = result.x
            except:
                pass
    
    return best_params
