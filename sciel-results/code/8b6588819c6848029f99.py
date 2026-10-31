import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    def objective(vars):
        h = vars[:N]
        z = vars[N]
        return -z
        
    def constraints(vars):
        h = vars[:N]
        z = vars[N]
        cons = []
        
        # Sum constraint
        cons.append({'type': 'eq', 'fun': lambda v: np.sum(v[:N]) - N/2.0, 'jac': lambda v: np.concatenate([np.ones(N), [0]])})
        
        for t in range(N):
            def corr_fun(v, t=t):
                h = v[:N]
                z = v[N]
                h_shifted = np.roll(h, t)
                K_t = np.sum(h * h_shifted)
                return K_t - z
                
            def corr_jac(v, t=t):
                h = v[:N]
                grad_h = np.roll(h, -t) + np.roll(h, t)
                grad_z = -1.0
                return np.concatenate([grad_h, [grad_z]])
                
            cons.append({'type': 'ineq', 'fun': corr_fun, 'jac': corr_jac})
            
        return cons
        
    bounds = [(0, 1)] * N + [(0, N/2)]
    
    best_h = np.full(N, 0.5)
    best_z = -np.inf
    
    # Restart 1: Uniform
    x0_1 = np.zeros(N + 1)
    x0_1[:N] = 0.5
    
    # Restart 2: Alternating 0,1
    x0_2 = np.zeros(N + 1)
    x0_2[:N] = (np.arange(N) % 2)
    
    # Restart 3: Random
    np.random.seed(42)
    x0_3 = np.random.rand(N + 1)
    x0_3[:N] = x0_3[:N] * (N/2.0) / np.sum(x0_3[:N]) # Scale to sum N/2
    x0_3[N] = 0.0
    
    for x0 in [x0_1, x0_2, x0_3]:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 2000, 'ftol': 1e-10})
            if res.success:
                h = res.x[:N]
                z = res.x[N]
                # Calculate actual min K_t to verify
                K_vals = [np.sum(h * np.roll(h, t)) for t in range(N)]
                min_K = np.min(K_vals)
                # We want to maximize min_K
                if min_K > best_z:
                    best_z = min_K
                    best_h = h.copy()
        except Exception:
            pass
            
    h = best_h
    
    # Ensure sum is exactly N/2
    s = np.sum(h)
    if s > 0:
        h = h * (N / 2.0) / s
    h = np.clip(h, 0, 1)
    
    # Fine-tune sum to be exactly N/2
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
