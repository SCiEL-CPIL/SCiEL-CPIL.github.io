import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    def objective(h):
        h = np.clip(h, 0, 1)
        overlaps = np.array([np.sum(h * (1 - np.roll(h, t))) for t in range(N)])
        beta = 20.0
        m = np.max(overlaps)
        return m + np.log(np.sum(np.exp(beta * (overlaps - m)))) / beta
    
    def sum_constraint(h):
        return np.sum(h) - N / 2
    
    x0 = np.full(N, 0.5)
    bounds = [(0.0, 1.0)] * N
    cons = [{'type': 'eq', 'fun': sum_constraint}]
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': 200, 'ftol': 1e-12})
        h = np.clip(res.x, 0, 1)
    except Exception:
        h = np.full(N, 0.5)
    
    # Ensure sum is exactly N/2
    s = np.sum(h)
    if s > 0:
        h = h * (N / 2.0) / s
    h = np.clip(h, 0, 1)
    # Fine-tune sum
    diff = N / 2.0 - np.sum(h)
    idx = np.argsort(h)[::-1]
    for i in idx:
        if diff > 0:
            add = min(diff, 1.0 - h[i])
            h[i] += add
            diff -= add
        elif diff < 0:
            sub = min(-diff, h[i])
            h[i] -= sub
            diff += sub
        else:
            break
    
    return h
