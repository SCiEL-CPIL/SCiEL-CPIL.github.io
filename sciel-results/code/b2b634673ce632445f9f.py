import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_value(h):
    h = np.array(h)
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        return 0.0
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1) # spacing w = 0.5 - (-0.5) / (m+1) ? 
    # np.linspace(-0.5, 0.5, m + 2) has m+2 points.
    # Spacing w = (0.5 - (-0.5)) / (m + 2 - 1) = 1.0 / (m + 1).
    # Correct.
    
    # l2 = sum over i in 0..m of (w / 3) * (y[i]^2 + y[i] * y[i+1] + y[i+1]^2)
    # i goes from 0 to m. y has indices 0 to m+1.
    # y[0] to y[m+1].
    # Terms: y[0], y[1] ... y[m], y[m+1].
    # The sum is over i=0 to m.
    # y[i] and y[i+1].
    # So pairs: (y0,y1), (y1,y2), ..., (ym, ym+1).
    
    terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = np.sum(terms) * (w / 3.0)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
        
    return l2 / (norm_1 * norm_inf)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    def objective(h):
        # Normalize to sum 1 to remove scale invariance for optimization stability
        s = np.sum(h)
        if s == 0:
            return 0.0
        h_norm = h / s
        return -compute_value(h_norm)
    
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}
    ]
    bounds = [(0.0, None) for _ in range(N)]
    
    initializations = []
    # 1. Uniform
    initializations.append(np.ones(N))
    # 2. Gaussian
    x = np.linspace(-1, 1, N)
    initializations.append(np.exp(-x**2 * 5))
    # 3. Raised Cosine
    initializations.append(0.5 * (1 + np.cos(np.pi * x)))
    # 4. Random
    np.random.seed(42)
    for _ in range(5):
        initializations.append(np.random.rand(N))
        
    for h0 in initializations:
        if time.time() > deadline:
            break
            
        # Normalize h0
        h0 = h0 / np.sum(h0)
        
        try:
            res = minimize(objective, h0, method='SLSQP', bounds=bounds, constraints=constraints, 
                           options={'maxiter': 200, 'ftol': 1e-10})
            if res.success:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_h = res.x
        except Exception:
            pass
            
    # Final check
    if time.time() > deadline:
        # Return best found so far
        pass
        
    return list(best_h)
