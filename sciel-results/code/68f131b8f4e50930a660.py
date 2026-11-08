import numpy as np
import time
from scipy.optimize import minimize

N = 600

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    # We search for a probability vector p (sum=1, p>=0)
    # that minimizes max(conv(p, p))
    
    # Parameterization with K=6 harmonics for more flexibility
    K = 6
    n_params = 2 * K
    
    i = np.arange(N)
    cos_bases = np.array([np.cos(2 * np.pi * k * i / N) for k in range(1, K+1)])
    sin_bases = np.array([np.sin(2 * np.pi * k * i / N) for k in range(1, K+1)])
    
    def get_p(params):
        # params: [A1, B1, A2, B2, ..., AK, BK]
        p = np.ones(N)
        for k in range(K):
            p += params[2*k] * cos_bases[k]
            p += params[2*k+1] * sin_bases[k]
        p = np.clip(p, 0, None)
        s = np.sum(p)
        if s < 1e-10:
            p = np.ones(N) / N
        else:
            p = p / s
        return p
    
    def objective(params):
        p = get_p(params)
        c = np.convolve(p, p, mode='full')
        return c.max()
    
    # Initial guesses
    best_p = np.ones(N) / N
    best_val = np.convolve(best_p, best_p, mode='full').max()
    
    # Try multiple starts with varying perturbation sizes
    np.random.seed(42)
    starts = [
        np.zeros(n_params),
        np.random.uniform(-0.08, 0.08, n_params),
        np.random.uniform(-0.05, 0.05, n_params),
        np.random.uniform(-0.03, 0.03, n_params),
        np.array([0.08, 0, 0.04, 0, 0.02, 0, 0.01, 0, 0.005, 0, 0.002, 0]),
        np.array([-0.08, 0, 0.04, 0, -0.02, 0, 0.01, 0, -0.005, 0, 0.002, 0]),
        np.array([0.0, 0.08, 0.0, 0.04, 0.0, 0.02, 0.0, 0.01, 0.0, 0.005, 0.0, 0.002]),
        np.array([0.05, 0.03, -0.03, 0.02, 0.01, -0.01, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]
    
    for x0 in starts:
        if time.time() > deadline - 15:
            break
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 500})
            if res.fun < best_val:
                best_val = res.fun
                best_p = get_p(res.x)
                best_p = np.clip(best_p, 0, None)
                s = np.sum(best_p)
                if s > 1e-10:
                    best_p = best_p / s
        except Exception:
            continue
    
    return [float(x) for x in best_p]
