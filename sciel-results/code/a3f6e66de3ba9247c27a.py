import numpy as np
from scipy.optimize import minimize

N = 50

def compute_value(h):
    h = np.asarray(h)
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        return -np.inf
    S = np.sum(h)
    Q = np.sum(h**2)
    if S == 0 or Q == 0:
        return -np.inf
    g = np.convolve(h, h, mode='full')
    g_sq_sum = np.sum(g**2)
    cross_sum = np.sum(g[:-1] * g[1:])
    A = 2 * g_sq_sum + cross_sum
    return A / (3 * S**2 * Q)

def construct():
    # Heuristic seeds
    seeds = []
    # 1. Box
    seeds.append(np.ones(N))
    # 2. Raised Cosine
    x = np.linspace(0, 1, N)
    seeds.append(np.cos(np.pi * x / 2)**2)
    # 3. Gaussian
    t = np.linspace(-1, 1, N)
    seeds.append(np.exp(-t**2 * 5))
    # 4. Triangular
    seeds.append(1 - np.abs(t))
    
    best_h = seeds[0]
    best_val = compute_value(best_h)
    
    for h0 in seeds:
        # Normalize to unit norm for optimization stability
        h0_norm = h0 / np.linalg.norm(h0)
        
        def objective(x):
            h = np.abs(x)
            if np.sum(h) == 0: return 0
            h = h / np.linalg.norm(h)
            return -compute_value(h)
        
        res = minimize(objective, h0_norm, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        h_opt = np.abs(res.x)
        h_opt = h_opt / np.linalg.norm(h_opt)
        val_opt = compute_value(h_opt)
        
        if val_opt > best_val:
            best_val = val_opt
            best_h = h_opt
            
    return list(best_h)
