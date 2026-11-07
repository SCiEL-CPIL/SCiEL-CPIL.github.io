import numpy as np
from scipy.optimize import minimize
import time

N = 100
TARGET_SUM = N / 2.0

def _correlation_max(h):
    corr = np.correlate(h, 1.0 - h, mode='full')
    return np.max(corr) / N * 2.0

def _project_feasible(h):
    h = np.asarray(h, dtype=float).copy()
    h = np.clip(h, 0.0, 1.0)
    for _ in range(1000):
        diff = TARGET_SUM - np.sum(h)
        if abs(diff) < 1e-12:
            break
        mask = (h > 1e-10) & (h < 1.0 - 1e-10)
        n_free = np.sum(mask)
        if n_free == 0:
            break
        h[mask] += diff / n_free
        h = np.clip(h, 0.0, 1.0)
    return h

def _soft_max_objective(h, beta=100.0):
    corr = np.correlate(h, 1.0 - h, mode='full')
    c_max = np.max(corr)
    if c_max == 0:
        return 0.0
    scaled = beta * (corr - c_max)
    scaled = np.clip(scaled, -500, 0)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return c_max + log_sum / beta

def _generate_seeds():
    seeds = []
    # Uniform
    seeds.append(np.full(N, 0.5))
    
    # Random Binary
    rng = np.random.RandomState(42)
    for _ in range(20):
        h = rng.choice([0.0, 1.0], size=N, p=[0.5, 0.5])
        seeds.append(_project_feasible(h))
        
    # Random Smooth
    for _ in range(20):
        noise = rng.randn(N)
        kernel = np.ones(5) / 5.0
        smooth = np.convolve(noise, kernel, mode='same')
        h = 0.5 + 0.4 * smooth
        seeds.append(_project_feasible(h))
        
    # Sine Waves
    x = np.arange(N)
    for freq in range(1, 15):
        for amp in [0.3, 0.45, 0.49]:
            h = 0.5 + amp * np.sin(2 * np.pi * freq * x / N)
            seeds.append(_project_feasible(h))
            
    return seeds

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_max(best_h)
    
    seeds = _generate_seeds()
    
    for h0 in seeds:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(
                lambda x: _soft_max_objective(x, beta=100.0),
                h0,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - TARGET_SUM,
                              'jac': lambda x: np.ones(N)}],
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            h_res = _project_feasible(res.x)
            val = _correlation_max(h_res)
            if val < best_val:
                best_val = val
                best_h = h_res.copy()
        except Exception:
            pass
            
    # Final polish
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda x: _soft_max_objective(x, beta=200.0),
                best_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - TARGET_SUM,
                              'jac': lambda x: np.ones(N)}],
                options={'maxiter': 1000, 'ftol': 1e-14, 'disp': False}
            )
            h_res = _project_feasible(res.x)
            val = _correlation_max(h_res)
            if val < best_val:
                best_val = val
                best_h = h_res.copy()
        except Exception:
            pass

    return _project_feasible(best_h)
